import os
import json
import optuna
from optuna.samplers import TPESampler

from src.hpo.search_space import _FIXED_PARAMS

TEMP_OPTUNA_DIR = "./temp/optuna"
OUTPUT_OPTUNA_DIR = "./output/optuna"
STORAGE_TEMPLATE = "sqlite:///./temp/optuna/{}.db"


def _db_path(study_name: str) -> str:
    return os.path.join(TEMP_OPTUNA_DIR, f"{study_name}.db")


def study_db_exists(study_name: str) -> bool:
    return os.path.isfile(_db_path(study_name))


def create_study(
    study_name: str,
    direction: str = "minimize",
    n_startup_trials: int = 30,
) -> optuna.Study:
    os.makedirs(TEMP_OPTUNA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_OPTUNA_DIR, exist_ok=True)
    storage = STORAGE_TEMPLATE.format(study_name)

    sampler = TPESampler(
        n_startup_trials=n_startup_trials, seed=2023, multivariate=True
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        load_if_exists=True,
    )
    return study


def load_study(study_name: str) -> optuna.Study:
    storage = STORAGE_TEMPLATE.format(study_name)
    return optuna.load_study(study_name=study_name, storage=storage)


def get_best_params(study_name: str) -> dict:
    study = load_study(study_name)
    return study.best_params


_PARAMS_ORDER = [
    "e_layers",
    "batch_size",
    "d_model",
    "d_ff",
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
    "weight_decay",
    "learning_rate",
]

_USER_ATTRS_ORDER = ["mse", "mae", "checkpoint"]

_USER_ATTR_KEYS = ["batch_size", "seed"]

_USER_ATTRS_EXCLUDE = set(_USER_ATTR_KEYS) | {"setting", "override_fixed"}

_SEARCH_ONLY_DEFAULTS = {
    "d_state": 16,
    "epsilon": 1e-4,
    "cov_window": 16,
    "cov_stride": 8,
    "geo_d_model": 128,
    "geo_d_state": 16,
    "geo_d_conv": 4,
}

_DEFAULT_FILL = {**_SEARCH_ONLY_DEFAULTS, **_FIXED_PARAMS}


def _ordered(src: dict, order: list) -> dict:
    result = {}
    for key in order:
        if key in src:
            result[key] = src[key]
    for key in src:
        if key not in result:
            result[key] = src[key]
    return result


def _parse_study_name(study_name: str) -> tuple[str, int]:
    dataset_name = ""
    pred_len = 0
    prefix = "ManiMamba_"
    if study_name.startswith(prefix):
        rest = study_name[len(prefix) :]
        pl_idx = rest.rfind("_pl")
        if pl_idx > 0:
            dataset_name = rest[:pl_idx]
            try:
                pred_len = int(rest[pl_idx + 3 :])
            except ValueError:
                pass
    return dataset_name, pred_len


def export_results(study_name: str):
    import pandas as pd

    study = load_study(study_name)
    df = study.trials_dataframe()

    csv_dir = os.path.join(OUTPUT_OPTUNA_DIR, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{study_name}_results.csv")
    df.to_csv(csv_path, index=False)

    try:
        best_trial = study.best_trial
    except ValueError:
        print(f"No completed trials in study {study_name}, skipping best params export")
        return df

    user_attrs = dict(best_trial.user_attrs)

    best_params = dict(best_trial.params)
    for key, val in _DEFAULT_FILL.items():
        best_params.setdefault(key, val)
    override_fixed = user_attrs.get("override_fixed", {})
    suggested_keys = set(best_trial.params.keys())
    for key, val in override_fixed.items():
        if key in _DEFAULT_FILL and key not in suggested_keys:
            best_params[key] = val
    for key in _USER_ATTR_KEYS:
        if key in user_attrs:
            best_params[key] = user_attrs[key]

    best_params = _ordered(best_params, _PARAMS_ORDER)

    ordered_attrs = _ordered(user_attrs, _USER_ATTRS_ORDER)
    ordered_attrs = {
        k: v for k, v in ordered_attrs.items() if k not in _USER_ATTRS_EXCLUDE
    }

    best_path = os.path.join(OUTPUT_OPTUNA_DIR, f"{study_name}_best_params.json")

    best_data = {
        "study_name": study_name,
        "version": "v4",
        "best_value": study.best_value,
        "best_params": best_params,
        "best_trial_number": best_trial.number,
        "best_trial_user_attrs": ordered_attrs,
    }
    with open(best_path, "w") as f:
        json.dump(best_data, f, indent=4, default=str)

    return df


def make_best_callback(study_name: str, metric: str = "mse"):
    def _callback(study, trial):
        try:
            if study.best_trial.number == trial.number:
                print(
                    f" ★ New best {metric.upper()}: {study.best_value:.6f} (trial #{trial.number}) — results exported"
                )
                export_results(study_name)
        except ValueError:
            pass

    return _callback
