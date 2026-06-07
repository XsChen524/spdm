import argparse
import gc
import glob
import os
import random
import sys
import traceback

import numpy as np
import optuna
import torch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from src.hpo.search_space import (
    build_search_space,
    build_training_space,
    get_overrides,
)
from src.hpo.gpu_budget import (
    GPUBudget,
    MultiGPUBudget,
    _NoopBudget,
    estimate_cuda_accel_test_peak_mb,
    estimate_manimamba_memory_mb,
    should_disable_cuda_accel,
    TOTAL_GPU_MEMORY_MB,
)

_ARCH_SIG_KEYS = [
    "d_model",
    "d_ff",
    "e_layers",
    "d_state",
    "expand",
    "epsilon",
    "cov_window",
    "cov_stride",
    "cov_rank",
    "geo_d_model",
    "geo_d_state",
    "geo_d_conv",
    "geo_expand",
]


def _suggest_params(trial, space: dict) -> dict:
    params = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif t == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "loguniform":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=True
            )
        elif t == "uniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
    return params


def _resolve_metrics(
    test_result,
    setting: str,
    args,
    exp,
    trial_number: int,
) -> tuple[float, float]:
    if test_result is not None:
        mse = test_result.get("mse")
        mae = test_result.get("mae")
        if mse is not None and mae is not None:
            if np.isfinite(mse) and np.isfinite(mae):
                return float(mse), float(mae)
            print(
                f"[WARN] Trial {trial_number}: test() returned non-finite "
                f"mse={mse}, mae={mae}"
            )

    folder_path = exp._get_results_path(setting)
    metrics_file = os.path.join(
        folder_path,
        f"{args.model}_{args.seq_len}_{args.pred_len}_metrics.npy",
    )
    if os.path.exists(metrics_file):
        metrics = np.load(metrics_file)
        mae, mse, rmse, mape, mspe = metrics
        if np.isfinite(mse) and np.isfinite(mae):
            print(
                f"[FALLBACK-1] Trial {trial_number}: recovered metrics from file "
                f"(test returned {test_result})"
            )
            return float(mse), float(mae)
        print(f"[WARN] Trial {trial_number}: metrics.npy has non-finite values")

    pattern = os.path.join(
        "./temp/results",
        f"{args.model_id}_{args.model}_*" f"_l{args.e_layers}_itr0",
        f"{args.model}_{args.seq_len}_{args.pred_len}_metrics.npy",
    )
    candidates = sorted(glob.glob(pattern))
    if candidates:
        metrics = np.load(candidates[-1])
        mae, mse, rmse, mape, mspe = metrics
        if np.isfinite(mse) and np.isfinite(mae):
            print(
                f"[FALLBACK-2] Trial {trial_number}: found metrics via glob: "
                f"{candidates[-1]}"
            )
            return float(mse), float(mae)

    print(f"[DIAG] Trial {trial_number}: metrics not found")
    print(f"  expected path: {metrics_file}")
    print(f"  test_result: {test_result}")
    results_base = "./temp/results"
    if os.path.isdir(results_base):
        matching = [
            d
            for d in os.listdir(results_base)
            if args.model_id in d and args.model in d
        ]
        print(f"  dirs matching model_id: {matching[:5]}")
    else:
        print(f"  results dir does not exist: {results_base}")

    print(f"[INF] Trial {trial_number}: marking as Complete Inf to inform sampler")
    return float("inf"), float("inf")


def _build_args(fixed: dict, suggested: dict) -> argparse.Namespace:
    defaults = {
        "is_training": 1,
        "model": "ManiMamba",
        "features": "M",
        "target": "OT",
        "freq": "h",
        "checkpoints": "./temp/checkpoints/",
        "checkpoint": 0,
        "label_len": 48,
        "d_layers": 1,
        "moving_avg": 25,
        "factor": 1,
        "distil": True,
        "embed": "timeF",
        "output_attention": False,
        "do_predict": False,
        "num_workers": 10,
        "itr": 1,
        "train_epochs": 15,
        "optim": "AdamW",
        "loss": "MSE",
        "lradj": "type1",
        "cosine_eta_min": 1e-7,
        "use_amp": True,
        "exp_name": "MTSF",
        "use_norm": True,
        "class_strategy": "projection",
        "inverse": False,
        "channel_independence": False,
        "efficient_training": False,
        "partial_start_index": 0,
        "revin": 1,
        "affine": 0,
        "use_gpu": True,
        "gpu": 0,
        "use_multi_gpu": False,
        "devices": "0",
        "noise_level": 0.0,
        "max_grad_norm": 0.0,
        "train_noise_level": 0.0,
        "use_cuda_accel": 1,
        "seed": 2023,
        "batch_size": 64,
        "warmup_epochs": 3,
        "patience": 5,
        "dropout": 0.1,
        "n_heads": 4,
        "save_tmp": 1,
        "weight_decay": 1e-6,
        "d_state": 16,
        "expand": 2,
        "epsilon": 1e-4,
        "cov_window": 16,
        "cov_stride": 8,
        "cov_rank": 0,
        "geo_d_model": 64,
        "geo_d_state": 16,
        "geo_d_conv": 4,
        "geo_expand": 2,
        "activation": "gelu",
    }
    merged = {**defaults, **fixed, **suggested}
    merged.setdefault("d_ff", merged.get("d_model", 512) * 2)
    args = argparse.Namespace(**merged)
    args.use_gpu = True
    return args


def _build_setting(args) -> str:
    setting = "{}_{}|{}|batch:{}|epochs:{}|d_state:{}|expand:{}|eps:{}|cw:{}|cs:{}|cr:{}|gdm:{}|gds:{}|gdc:{}|ge:{}|e_layers:{}|dm:{}|df:{}|lr:{}|wd:{}|dropout:{}|pat:{}|wu:{}|itr:0".format(
        args.model_id,
        args.model,
        args.des,
        args.batch_size,
        args.train_epochs,
        args.d_state,
        args.expand,
        args.epsilon,
        args.cov_window,
        args.cov_stride,
        args.cov_rank,
        args.geo_d_model,
        args.geo_d_state,
        args.geo_d_conv,
        args.geo_expand,
        args.e_layers,
        args.d_model,
        args.d_ff,
        args.learning_rate,
        args.weight_decay,
        args.dropout,
        args.patience,
        args.warmup_epochs,
    )
    return setting


def _estimate(enc_in, suggested, seq_len, pred_len, batch_size):
    return estimate_manimamba_memory_mb(
        enc_in=enc_in,
        d_model=suggested["d_model"],
        d_ff=suggested.get("d_ff", suggested.get("d_model", 512) * 2),
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size,
        e_layers=suggested["e_layers"],
        d_state=suggested.get("d_state", 16),
        expand=suggested.get("expand", 1),
        cov_window=suggested.get("cov_window", 16),
        cov_stride=suggested.get("cov_stride", 8),
        cov_rank=suggested.get("cov_rank", 0),
        geo_d_model=suggested.get("geo_d_model", 64),
        geo_d_state=suggested.get("geo_d_state", 16),
        geo_d_conv=suggested.get("geo_d_conv", 4),
        geo_expand=suggested.get("geo_expand", 1),
    )


def create_objective(
    fixed_params: dict,
    metric: str = "mse",
    gpu_ids: list[int] | None = None,
    dedicated_gpu: bool = False,
    stop_check=None,
):
    enc_in = fixed_params.get("enc_in", 7)
    pred_len = fixed_params.get("pred_len", 96)
    seq_len = fixed_params.get("seq_len", 96)
    dataset_name = fixed_params.get("dataset", "")

    model_space = build_search_space(
        enc_in=enc_in, pred_len=pred_len, dataset_name=dataset_name
    )
    train_space = build_training_space(
        dataset_name=dataset_name, pred_len=pred_len
    )

    overrides = get_overrides(dataset_name, pred_len)

    full_space = {**model_space, **train_space}
    override_fixed = overrides["fixed"]

    if dedicated_gpu:
        budget = _NoopBudget()
    elif gpu_ids and len(gpu_ids) > 1:
        budget = MultiGPUBudget(gpu_ids)
    else:
        budget = GPUBudget()

    def objective(trial) -> float:
        trial_id = f"trial_{trial.number}"

        if stop_check and stop_check():
            raise optuna.TrialPruned("stop requested")

        seed = suggested_seed if (suggested_seed := override_fixed.get("seed", 2023)) else 2023
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        exp = None
        try:
            suggested = _suggest_params(trial, full_space)
            suggested.update(
                {k: v for k, v in override_fixed.items() if k not in full_space}
            )

            batch_size = suggested.get("batch_size", 64)
            downgrade_bs = suggested.pop("oom_downgrade_bs", None)

            est_mb = _estimate(enc_in, suggested, seq_len, pred_len, batch_size)

            if est_mb > TOTAL_GPU_MEMORY_MB and downgrade_bs is not None:
                est_down = _estimate(enc_in, suggested, seq_len, pred_len, downgrade_bs)
                if est_down <= TOTAL_GPU_MEMORY_MB:
                    batch_size = downgrade_bs
                    trial.set_user_attr("auto_batch_size", downgrade_bs)
                else:
                    print(
                        f"[OOM] Trial {trial.number} skipped: "
                        f"even bs={downgrade_bs} exceeds budget ({est_down:.0f} MB)"
                    )
                    trial.set_user_attr("status", "skipped_over_budget")
                    return float("inf")

            est_mb = _estimate(enc_in, suggested, seq_len, pred_len, batch_size)

            if est_mb > TOTAL_GPU_MEMORY_MB:
                print(
                    f"[OOM] Trial {trial.number} skipped: "
                    f"estimated {est_mb:.0f} MB > limit {TOTAL_GPU_MEMORY_MB} MB"
                )
                trial.set_user_attr("status", "skipped_over_budget")
                trial.set_user_attr("est_mb", est_mb)
                return float("inf")

            suggested["batch_size"] = batch_size
            trial.set_user_attr("batch_size", batch_size)
            trial.set_user_attr("seed", 2023)

            cuda_accel = suggested.get("use_cuda_accel", 1)
            total_rows = fixed_params.get("total_rows", 0)
            if cuda_accel:
                if should_disable_cuda_accel(
                    enc_in, pred_len, total_rows, seq_len
                ):
                    suggested["use_cuda_accel"] = 0
                    cuda_accel = 0
                    trial.set_user_attr("cuda_accel_auto_disabled", True)
                else:
                    overhead = estimate_cuda_accel_test_peak_mb(
                        enc_in, pred_len, total_rows, seq_len
                    )
                    if est_mb + overhead > TOTAL_GPU_MEMORY_MB:
                        suggested["use_cuda_accel"] = 0
                        cuda_accel = 0
                        trial.set_user_attr("cuda_accel_auto_disabled", True)
                    else:
                        est_mb += overhead
                        trial.set_user_attr("cuda_accel_overhead_mb", overhead)

            acquired = budget.acquire(trial_id, est_mb, timeout=1800)
            if not acquired:
                trial.set_user_attr("status", "timeout_waiting_gpu")
                raise optuna.TrialPruned("timeout_waiting_gpu")

            try:
                arch_sig = "|".join(str(suggested.get(k, 0)) for k in _ARCH_SIG_KEYS)

                args = _build_args(fixed_params, suggested)
                if args.des == "HPO_OPTUNA":
                    args.des = f"HPO_OPTUNA_{trial.number}"

                setting = _build_setting(args)

                exp = Exp_Long_Term_Forecast(args)

                exp.train(setting)

                if hasattr(exp, "_nan_detected") and exp._nan_detected:
                    raise optuna.TrialPruned("nan_loss_during_final_epoch")

                test_result = exp.test(setting)

                _lr_seq = getattr(exp, "_lr_checkpoint_to_cleanup", None)
                if _lr_seq is not None:
                    from src.utils.checkpoint import cleanup_checkpoint

                    cleanup_checkpoint(args, _lr_seq)
                    exp._last_checkpoint_seq = None

                if test_result is not None:
                    mse_val = test_result.get("mse")
                    mae_val = test_result.get("mae")
                elif hasattr(exp, "_last_test_result") and exp._last_test_result:
                    mse_val = exp._last_test_result.get("mse")
                    mae_val = exp._last_test_result.get("mae")
                    test_result = exp._last_test_result
                else:
                    mse_val = mae_val = None

                if (
                    mse_val is not None
                    and mae_val is not None
                    and np.isfinite(mse_val)
                    and np.isfinite(mae_val)
                ):
                    mse, mae = float(mse_val), float(mae_val)
                else:
                    mse, mae = _resolve_metrics(
                        test_result, setting, args, exp, trial.number
                    )

                trial.set_user_attr("mse", mse)
                trial.set_user_attr("mae", mae)
                trial.set_user_attr("setting", setting)
                trial.set_user_attr("arch_signature", arch_sig)
                trial.set_user_attr(
                    "override_fixed",
                    {k: v for k, v in override_fixed.items() if k != "oom_downgrade_bs"},
                )

                if hasattr(exp, "_last_checkpoint_seq"):
                    trial.set_user_attr("checkpoint", exp._last_checkpoint_seq)

                return mse if metric == "mse" else mae

            except RuntimeError as e:
                err_str = str(e).lower()
                if "out of memory" in err_str:
                    print(
                        f"[OOM] Trial {trial.number}: CUDA out of memory during training"
                    )
                    trial.set_user_attr("status", "OOM")
                    if dedicated_gpu:
                        return float("inf")
                    else:
                        raise optuna.TrialPruned("OOM_contention")
                if "linalg.eigh" in err_str or "ill-conditioned" in err_str:
                    print(
                        f"[LINALG] Trial {trial.number}: eigh convergence failure — "
                        f"marking as Inf"
                    )
                    trial.set_user_attr("status", "linalg_eigh_failure")
                    return float("inf")
                trial.set_user_attr("status", "runtime_error")
                raise optuna.TrialPruned(str(e))
        except optuna.TrialPruned:
            raise
        except Exception as e:
            err_str = str(e).lower()
            if "linalg.eigh" in err_str or "ill-conditioned" in err_str:
                print(
                    f"[LINALG] Trial {trial.number}: eigh convergence failure "
                    f"(generic handler) — marking as Inf"
                )
                trial.set_user_attr("status", "linalg_eigh_failure")
                return float("inf")
            error_tb = traceback.format_exc()
            trial.set_user_attr("status", "error")
            trial.set_user_attr("error_trace", error_tb)
            print(
                f"[ERROR] Trial {trial.number} unhandled exception: {type(e).__name__}: {e}\n{error_tb}"
            )
            raise optuna.TrialPruned(f"error: {type(e).__name__}: {e}")
        finally:
            budget.release(trial_id)
            try:
                del exp
            except NameError:
                pass
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return objective
