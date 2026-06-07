import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.Embed import DataEmbedding_inverted
from src.layers.SPDTangentProjection import SPDTangentProjection
from src.layers.GeometryMamba import GeometryMamba
from src.layers.ManiMambaEncoderLayer import ManiMambaEncoderLayer


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        cov_window = getattr(configs, "cov_window", 16)
        cov_stride = getattr(configs, "cov_stride", 8)
        epsilon = getattr(configs, "epsilon", 1e-4)
        cov_rank = getattr(configs, "cov_rank", 0)
        geo_d_model = getattr(configs, "geo_d_model", 64)
        geo_d_state = getattr(configs, "geo_d_state", 16)
        geo_d_conv = getattr(configs, "geo_d_conv", 4)
        geo_expand = getattr(configs, "geo_expand", 1)
        expand = getattr(configs, "expand", 1)
        d_state = getattr(configs, "d_state", 16)
        self.cov_stride = cov_stride
        self.ablation = getattr(configs, "ablation", None)
        self.auxiliary_losses = []

        self.spd_proj = SPDTangentProjection(
            enc_in=configs.enc_in,
            cov_window=cov_window,
            cov_stride=cov_stride,
            epsilon=epsilon,
            cov_rank=cov_rank,
        )

        self.geo_encoder = GeometryMamba(
            tri_dim=self.spd_proj.tri_dim,
            geo_d_model=geo_d_model,
            geo_d_state=geo_d_state,
            geo_d_conv=geo_d_conv,
            geo_expand=geo_expand,
        )

        d_inner = expand * configs.d_model
        self.geo_delta_proj = nn.Linear(geo_d_model, d_inner)

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.encoder_layers = nn.ModuleList(
            [
                ManiMambaEncoderLayer(
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    d_state=d_state,
                    d_conv=2,
                    expand=expand,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    enc_in=configs.enc_in,
                    geo_inject_threshold=getattr(configs, "geo_inject_threshold", 100),
                    ablation=self.ablation,
                )
                for _ in range(configs.e_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(configs.d_model)
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        if self.ablation == "geo_smooth_reg":
            self._geo_smooth_weight = 0.01

        self._capture = False

    def set_capture(self, mode: bool):
        self._capture = mode
        self.spd_proj._capture = mode
        if mode:
            self.spd_proj._captured = {}
        for layer in self.encoder_layers:
            for mod in (layer.mamba_fwd, layer.mamba_rev):
                mod._capture = mode
                if mode:
                    mod._captured = {}

    def _compute_geo_smooth_reg(self, x_enc):
        with torch.cuda.amp.autocast(enabled=False):
            x_enc_f = x_enc.float()
            B, L, N = x_enc_f.shape
            x_w = x_enc_f.unfold(1, self.spd_proj.cov_window, self.spd_proj.cov_stride)
            if self.spd_proj.cov_rank > 0:
                x_w = x_w.permute(0, 1, 3, 2)
                x_w = self.spd_proj.cov_proj(x_w)
                x_w = x_w.permute(0, 1, 3, 2)
            D = self.spd_proj.cov_dim
            xc = x_w - x_w.mean(dim=-1, keepdim=True)
            cov = (xc @ xc.mT) / max(
                self.spd_proj.cov_window - 1, 1
            ) + self.spd_proj.epsilon * self.spd_proj._eye
            log_cov = self.spd_proj._matrix_log(cov)
            diff = log_cov[:, 1:] - log_cov[:, :-1]
            reg = (diff**2).mean()
        return reg

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, N = x_enc.shape

        self.auxiliary_losses = []
        if self.ablation == "geo_smooth_reg":
            geo_reg = self._compute_geo_smooth_reg(x_enc)
            self.auxiliary_losses.append((self._geo_smooth_weight, geo_reg))

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x_enc = x_enc / stdev

        v_t = self.spd_proj(x_enc)
        h_geo = self.geo_encoder(v_t)
        geo_delta = self.geo_delta_proj(h_geo)

        N_prime = N + (x_mark_enc.shape[-1] if x_mark_enc is not None else 0)

        if self.ablation == "linear_interp":
            geo_delta_perm = geo_delta.permute(0, 2, 1)
            geo_delta_interp = F.interpolate(
                geo_delta_perm, size=N_prime, mode="linear", align_corners=True
            )
            geo_delta_expanded = geo_delta_interp.permute(0, 2, 1)
        else:
            geo_delta_expanded = torch.zeros(
                B,
                N_prime,
                geo_delta.shape[-1],
                device=geo_delta.device,
                dtype=geo_delta.dtype,
            )
            T_w = geo_delta.shape[1]
            positions = torch.arange(T_w, device=geo_delta.device) * self.cov_stride
            positions = positions.clamp(max=N_prime - 1)
            idx = positions.unsqueeze(0).unsqueeze(-1).expand(B, -1, geo_delta.shape[-1])
            geo_delta_expanded.scatter_(1, idx, geo_delta)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, geo_delta=geo_delta_expanded)
        enc_out = self.encoder_norm(enc_out)

        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]
