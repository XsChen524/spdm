import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class GeometryModulatedMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        enc_in=0,
        geo_inject_threshold=100,
        ablation=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.ablation = ablation

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        if ablation == "tanh_alpha":
            self.geo_delta_scale = nn.Parameter(torch.tensor(1.0))
        elif ablation == "w_dt":
            self.geo_delta_scale = nn.Parameter(torch.tensor(1.0))
            self.geo_dt_proj = nn.Sequential(
                nn.Linear(self.d_inner, self.d_inner),
                nn.GELU(),
                nn.Linear(self.d_inner, self.d_inner),
            )
            for m in self.geo_dt_proj:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)
        else:
            self.geo_delta_scale = None

        self.geo_b_proj = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner),
            nn.GELU(),
            nn.Linear(self.d_inner, d_state),
        )
        for m in self.geo_b_proj:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

        self.enc_in = enc_in
        self.geo_inject_threshold = geo_inject_threshold
        self._large_var = enc_in > geo_inject_threshold
        self.ablation = ablation

        self.geo_c_proj = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner),
            nn.GELU(),
            nn.Linear(self.d_inner, d_state),
        )
        for m in self.geo_c_proj:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

        self._capture = False
        self._captured = {}

    def forward(self, hidden_states, geo_delta=None):
        batch, seqlen, _ = hidden_states.shape

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :seqlen])

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)

        B = rearrange(B, "(b l) ds -> b ds l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) ds -> b ds l", l=seqlen).contiguous()

        if self._capture:
            self._captured["B_data"] = B.float().detach().cpu().numpy()
            self._captured["C_data"] = C.float().detach().cpu().numpy()

        if geo_delta is not None:
            geo_b = rearrange(
                self.geo_b_proj(geo_delta), "b l ds -> b ds l"
            ).contiguous()
            geo_c = rearrange(
                self.geo_c_proj(geo_delta), "b l ds -> b ds l"
            ).contiguous()

            if self._capture:
                self._captured["geo_b"] = geo_b.float().detach().cpu().numpy()
                self._captured["geo_c"] = geo_c.float().detach().cpu().numpy()

            if self.ablation == "no_bc":
                pass
            elif self.ablation == "tanh_alpha":
                B = B + (self.geo_delta_scale * torch.tanh(geo_b)).to(dtype=B.dtype)
                C = C + (self.geo_delta_scale * torch.tanh(geo_c)).to(dtype=C.dtype)
            elif self.ablation == "w_dt":
                B = B + geo_b.to(dtype=B.dtype)
                C = C + geo_c.to(dtype=C.dtype)
                geo_dt = rearrange(
                    self.geo_dt_proj(geo_delta), "b l d -> b d l"
                )
                dt = dt + (self.geo_delta_scale * torch.tanh(geo_dt)).to(dtype=dt.dtype)
            else:
                B = B + geo_b.to(dtype=B.dtype)
                C = C + geo_c.to(dtype=C.dtype)

        A = -torch.exp(self.A_log.float())

        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        y = rearrange(y, "b d l -> b l d")

        return self.out_proj(y)
