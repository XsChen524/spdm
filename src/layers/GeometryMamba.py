import torch.nn as nn
from mamba_ssm import Mamba


class GeometryMamba(nn.Module):
    def __init__(self, tri_dim, geo_d_model, geo_d_state=16, geo_d_conv=4, geo_expand=1):
        super().__init__()
        self.proj_in = nn.Linear(tri_dim, geo_d_model)
        self.mamba = Mamba(
            d_model=geo_d_model,
            d_state=geo_d_state,
            d_conv=geo_d_conv,
            expand=geo_expand,
        )
        self.norm = nn.LayerNorm(geo_d_model)

    def forward(self, v_t):
        h = self.proj_in(v_t)
        h = self.norm(self.mamba(h) + h)
        return h
