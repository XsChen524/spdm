import torch
import torch.nn as nn


class SPDTangentProjection(nn.Module):
    def __init__(self, enc_in, cov_window=16, cov_stride=8, epsilon=1e-4, cov_rank=0):
        super().__init__()
        self.cov_window = cov_window
        self.cov_stride = cov_stride
        self.epsilon = epsilon
        self.cov_rank = cov_rank
        self.cov_dim = cov_rank if cov_rank > 0 else enc_in
        if cov_rank > 0:
            self.cov_proj = nn.Linear(enc_in, cov_rank, bias=False)
        self.tri_dim = self.cov_dim * (self.cov_dim + 1) // 2
        tri_idx = torch.triu_indices(self.cov_dim, self.cov_dim, offset=0)
        self.register_buffer("_triu_row", tri_idx[0])
        self.register_buffer("_triu_col", tri_idx[1])
        self.register_buffer("_eye", torch.eye(self.cov_dim))

        self._capture = False
        self._captured = {}

    def _matrix_log(self, mat):
        egg, evv = torch.linalg.eigh(mat)
        egg = torch.clamp(egg, min=self.epsilon)
        return evv @ torch.diag_embed(torch.log(egg)) @ evv.mT

    def _matrix_exp(self, mat):
        egg, evv = torch.linalg.eigh(mat)
        exp_egg = torch.exp(egg)
        return evv @ torch.diag_embed(exp_egg) @ evv.mT, exp_egg, evv

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, x_enc):
        with torch.amp.autocast("cuda", enabled=False):
            B, L, N = x_enc.shape

            x_w = x_enc.unfold(1, self.cov_window, self.cov_stride)
            T_w = x_w.shape[1]

            if self.cov_rank > 0:
                x_w = x_w.permute(0, 1, 3, 2)
                x_w = self.cov_proj(x_w)
                x_w = x_w.permute(0, 1, 3, 2)
            D = self.cov_dim

            xc = x_w - x_w.mean(dim=-1, keepdim=True)
            cov = (xc @ xc.mT) / max(self.cov_window - 1, 1) + self.epsilon * self._eye

            log_cov = self._matrix_log(cov)

            base_log = log_cov.mean(dim=1, keepdim=True)
            base_point, egg_base, evv_base = self._matrix_exp(base_log)
            egg_base = torch.clamp(egg_base, min=self.epsilon)
            base_sqrt = evv_base @ torch.diag_embed(torch.sqrt(egg_base)) @ evv_base.mT
            base_inv_sqrt = (
                evv_base @ torch.diag_embed(1.0 / torch.sqrt(egg_base)) @ evv_base.mT
            )

            mid = base_inv_sqrt @ cov @ base_inv_sqrt
            log_mid = self._matrix_log(mid)
            tangent = base_sqrt @ log_mid @ base_sqrt

            v_t = tangent[:, :, self._triu_row, self._triu_col]

            if self._capture:
                self._captured = {
                    "cov": cov.float().detach().cpu().numpy(),
                    "log_cov": log_cov.float().detach().cpu().numpy(),
                    "tangent": tangent.float().detach().cpu().numpy(),
                    "v_t": v_t.float().detach().cpu().numpy(),
                }

            return v_t
