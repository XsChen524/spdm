import numpy as np
import torch

_ENABLED = False


def cuda_accel_enabled():
    return _ENABLED


def enable_cuda_accel(enabled=True):
    global _ENABLED
    _ENABLED = enabled


_TORCH_METRIC_EPS = 1e-8


def _torch_mae(pred, true):
    return torch.mean(torch.abs(pred - true))


def _torch_mse(pred, true):
    return torch.mean((pred - true) ** 2)


def _torch_rmse(pred, true):
    return torch.sqrt(_torch_mse(pred, true))


def _torch_mape(pred, true):
    denom = torch.clamp(torch.abs(true), min=_TORCH_METRIC_EPS)
    return torch.mean(torch.abs(pred - true) / denom)


def _torch_mspe(pred, true):
    denom = torch.clamp(torch.abs(true), min=_TORCH_METRIC_EPS)
    return torch.mean(((pred - true) / denom) ** 2)


def torch_metric(pred, true):
    mae = _torch_mae(pred, true)
    mse = _torch_mse(pred, true)
    rmse = _torch_rmse(pred, true)
    mape = _torch_mape(pred, true)
    mspe = _torch_mspe(pred, true)
    return mae.item(), mse.item(), rmse.item(), mape.item(), mspe.item()


def finalise_metrics_gpu(gpu_preds, gpu_trues):
    preds = torch.cat(gpu_preds, dim=0)
    trues = torch.cat(gpu_trues, dim=0)
    preds_np = preds.cpu().numpy()
    trues_np = trues.cpu().numpy()
    mae, mse, rmse, mape, mspe = torch_metric(preds, trues)
    return (mae, mse, rmse, mape, mspe), preds_np, trues_np


class TorchStandardScaler:
    def __init__(self, sklearn_scaler, device="cuda"):
        self.mean = torch.tensor(
            sklearn_scaler.mean_, dtype=torch.float32, device=device
        )
        self.scale = torch.tensor(
            sklearn_scaler.scale_, dtype=torch.float32, device=device
        )

    def inverse_transform(self, x):
        return x * self.scale + self.mean

    @classmethod
    def from_dataset(cls, dataset, device="cuda"):
        if not hasattr(dataset, "scaler"):
            return None
        return cls(dataset.scaler, device=device)
