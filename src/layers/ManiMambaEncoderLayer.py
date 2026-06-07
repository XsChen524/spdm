import torch.nn as nn
import torch.nn.functional as F

from src.layers.GeometryModulatedMamba import GeometryModulatedMamba


class ManiMambaEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        d_state=16,
        d_conv=2,
        expand=1,
        dropout=0.1,
        activation="gelu",
        enc_in=0,
        geo_inject_threshold=100,
        ablation=None,
    ):
        super().__init__()
        self.mamba_fwd = GeometryModulatedMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            enc_in=enc_in,
            geo_inject_threshold=geo_inject_threshold,
            ablation=ablation,
        )
        self.mamba_rev = GeometryModulatedMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            enc_in=enc_in,
            geo_inject_threshold=geo_inject_threshold,
            ablation=ablation,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, geo_delta=None):
        h = self.norm1(x)
        new_x = self.mamba_fwd(h, geo_delta=geo_delta) + self.mamba_rev(
            h.flip(1), geo_delta=geo_delta.flip(1) if geo_delta is not None else None,
        ).flip(1)
        x = x + new_x

        h = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(h.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return x + y
