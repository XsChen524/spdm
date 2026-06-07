import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_TIME_FEAT_DIM = {
    "ETTh1": 4,
    "ETTh2": 4,
    "ETTm1": 5,
    "ETTm2": 5,
    "custom": 4,
    "Solar": 0,
    "PEMS": 0,
}


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layer_norm=False, dropout=0.0):
        super().__init__()
        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim) if layer_norm else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        h = F.relu(self.dense1(x))
        out = self.dense2(h)
        if self.drop is not None:
            out = self.drop(out)
        out = out + self.skip(x)
        if self.ln is not None:
            out = self.ln(out)
        return out


def _make_residual_stack(input_dim, hidden_dims, layer_norm=False, dropout=0.0):
    if len(hidden_dims) < 2:
        return nn.Linear(input_dim, hidden_dims[-1])
    layers = []
    current_in = input_dim
    for i in range(len(hidden_dims) - 1):
        layers.append(
            ResidualBlock(current_in, hidden_dims[i], hidden_dims[i + 1], layer_norm, dropout)
        )
        current_in = hidden_dims[i + 1]
    return nn.Sequential(*layers)


class TideCore(nn.Module):
    def __init__(
        self,
        hist_len,
        pred_len,
        time_feat_dim,
        num_ts,
        hidden_dims,
        time_encoder_dims,
        decoder_output_dim,
        final_decoder_hidden,
        layer_norm=False,
        dropout=0.0,
        ts_emb_dim=16,
        transform=False,
    ):
        super().__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.decoder_output_dim = decoder_output_dim
        self.transform = transform
        self.encoded_time_dim = time_encoder_dims[-1] if time_feat_dim > 0 else 0

        if time_feat_dim > 0:
            self.time_encoder = _make_residual_stack(
                time_feat_dim, time_encoder_dims, layer_norm, dropout
            )
        else:
            self.time_encoder = None

        self.ts_embedding = nn.Embedding(num_ts, ts_emb_dim)

        encoder_input_dim = (
            hist_len
            + hist_len * self.encoded_time_dim
            + pred_len * self.encoded_time_dim
            + ts_emb_dim
        )
        self.encoder = _make_residual_stack(encoder_input_dim, hidden_dims, layer_norm, dropout)

        decoder_output_flat = decoder_output_dim * pred_len
        decoder_dims = hidden_dims[:-1] + [decoder_output_flat]
        self.decoder = _make_residual_stack(hidden_dims[-1], decoder_dims, layer_norm, dropout)

        self.linear_residual = nn.Linear(hist_len, pred_len)

        final_input_dim = decoder_output_dim + self.encoded_time_dim
        self.final_decoder = ResidualBlock(
            final_input_dim, final_decoder_hidden, 1, layer_norm, dropout
        )

        if transform:
            self.affine_weight = nn.Parameter(torch.ones(num_ts))
            self.affine_bias = nn.Parameter(torch.zeros(num_ts))

    def forward(self, past_ts, past_feats=None, future_feats=None, ts_idx=None):
        batch_size = past_ts.shape[0]
        affine_weight = None
        affine_bias = None
        batch_mean = None
        batch_std = None

        if self.transform:
            affine_weight = self.affine_weight[ts_idx]
            affine_bias = self.affine_bias[ts_idx]
            batch_mean = past_ts.mean(dim=1)
            batch_std = past_ts.std(dim=1)
            batch_std = torch.where(
                batch_std == 0.0, torch.ones_like(batch_std), batch_std
            )
            past_ts = (past_ts - batch_mean[:, None]) / batch_std[:, None]
            past_ts = affine_weight[:, None] * past_ts + affine_bias[:, None]

        encoded_future_3d = None

        if self.time_encoder is not None and past_feats is not None:
            B, T, F_dim = past_feats.shape
            encoded_past = self.time_encoder(
                past_feats.reshape(B * T, F_dim)
            ).reshape(B, T, -1)
            encoded_past_flat = encoded_past.reshape(batch_size, -1)

            B, T, F_dim = future_feats.shape
            encoded_future = self.time_encoder(
                future_feats.reshape(B * T, F_dim)
            ).reshape(B, T, -1)
            encoded_future_flat = encoded_future.reshape(batch_size, -1)
            encoded_future_3d = encoded_future.permute(0, 2, 1)
        else:
            encoded_past_flat = past_ts.new_zeros(batch_size, 0)
            encoded_future_flat = past_ts.new_zeros(batch_size, 0)

        ts_embs = self.ts_embedding(ts_idx)

        encoder_input = torch.cat(
            [past_ts, encoded_past_flat, encoded_future_flat, ts_embs], dim=-1
        )

        encoder_output = self.encoder(encoder_input)
        decoder_output = self.decoder(encoder_output)
        decoder_output = decoder_output.reshape(
            batch_size, self.decoder_output_dim, self.pred_len
        )

        if encoded_future_3d is not None:
            final_in = torch.cat([decoder_output, encoded_future_3d], dim=1)
        else:
            final_in = decoder_output

        output = self.final_decoder(final_in.permute(0, 2, 1))
        output = output.permute(0, 2, 1)

        linear_res = self.linear_residual(past_ts).unsqueeze(1)
        output = output + linear_res

        if self.transform:
            EPS = 1e-7
            output = output.squeeze(1)
            output = (output - affine_bias[:, None]) / (affine_weight[:, None] + EPS)
            output = output * batch_std[:, None] + batch_mean[:, None]
            output = output.unsqueeze(1)

        return output


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        hidden_size = getattr(configs, "hidden_size", 256)
        num_layers = getattr(configs, "num_layers", 2)
        decoder_output_dim = getattr(configs, "decoder_output_dim", 8)
        final_decoder_hidden = getattr(configs, "final_decoder_hidden", 64)
        layer_norm = getattr(configs, "tide_layer_norm", False)
        tide_dropout = getattr(configs, "tide_dropout", 0.0)
        transform = getattr(configs, "tide_transform", False)
        time_encoder_hidden = getattr(configs, "time_encoder_hidden", 64)

        data_type = getattr(configs, "data", "custom")
        time_feat_dim = DATA_TIME_FEAT_DIM.get(data_type, 4)

        hidden_dims = [hidden_size] * num_layers
        time_encoder_dims = [time_encoder_hidden, 4]

        self.tide = TideCore(
            hist_len=configs.seq_len,
            pred_len=configs.pred_len,
            time_feat_dim=time_feat_dim,
            num_ts=configs.enc_in,
            hidden_dims=hidden_dims,
            time_encoder_dims=time_encoder_dims,
            decoder_output_dim=decoder_output_dim,
            final_decoder_hidden=final_decoder_hidden,
            layer_norm=layer_norm,
            dropout=tide_dropout,
            ts_emb_dim=16,
            transform=transform,
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, T, N = x_enc.shape

        past_ts = x_enc.permute(0, 2, 1).reshape(B * N, T)

        if x_mark_enc is not None:
            past_feats = (
                x_mark_enc.unsqueeze(1).expand(B, N, T, -1).reshape(B * N, T, -1)
            )
        else:
            past_feats = None

        if x_mark_dec is not None:
            future_feats = (
                x_mark_dec[:, -self.pred_len :, :]
                .unsqueeze(1)
                .expand(B, N, self.pred_len, -1)
                .reshape(B * N, self.pred_len, -1)
            )
        else:
            future_feats = None

        ts_idx = (
            torch.arange(N, device=x_enc.device).unsqueeze(0).expand(B, -1).reshape(B * N)
        )

        output = self.tide(past_ts, past_feats, future_feats, ts_idx)

        output = output.squeeze(1).reshape(B, N, self.pred_len).permute(0, 2, 1)

        return output
