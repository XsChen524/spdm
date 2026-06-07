_BASE_SEARCH_SPACE = {
    "d_model": {"type": "categorical", "choices": [128, 256, 512]},
    "e_layers": {"type": "categorical", "choices": [2, 3]},
    "d_state": {"type": "categorical", "choices": [2, 8, 16]},
    "epsilon": {"type": "categorical", "choices": [1e-5, 1e-4, 1e-3]},
    "cov_window": {"type": "categorical", "choices": [8, 16, 32]},
    "cov_stride": {"type": "categorical", "choices": [4, 8, 16]},
    "cov_rank": {"type": "categorical", "choices": [0]},
    "geo_d_model": {"type": "categorical", "choices": [128, 256, 512]},
    "geo_d_state": {"type": "categorical", "choices": [4, 8, 16]},
    "geo_d_conv": {"type": "categorical", "choices": [2, 4]},
    "learning_rate": {
        "type": "categorical",
        "choices": [
            1e-5,
            1.17e-5,
            1.37e-5,
            1.61e-5,
            1.89e-5,
            2.21e-5,
            2.59e-5,
            3.04e-5,
            3.56e-5,
            4.18e-5,
            4.89e-5,
            5.74e-5,
            6.72e-5,
            7.88e-5,
            9.24e-5,
            1.083e-4,
            1.269e-4,
            1.487e-4,
            1.743e-4,
            2.043e-4,
            2.395e-4,
            2.807e-4,
            3.29e-4,
            3.857e-4,
            4.52e-4,
            5.298e-4,
            6.21e-4,
            7.279e-4,
            8.532e-4,
            1e-3,
        ],
    },
}

_FIXED_PARAMS = {
    "weight_decay": 1e-6,
    "train_epochs": 15,
    "batch_size": 64,
    "warmup_epochs": 3,
    "patience": 5,
    "expand": 2,
    "geo_expand": 2,
    "dropout": 0.2,
    "cov_rank": 0,
}

_BASE_TRAINING_SPACE = {}

_DATASET_PL_CONFIGS = {
    "ETTh1": {
        "*": {
            "search": {
                "d_model": {"type": "categorical", "choices": [128, 256]},
                "d_state": {"type": "categorical", "choices": [8, 16]},
                "epsilon": {"type": "categorical", "choices": [1e-4, 1e-3]},
                "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
                "cov_rank": {"type": "categorical", "choices": [0, 8, 16]},
            },
            "fixed": {
                "lradj": "type1",
            },
        },
    },
    "ETTh2": {
        "*": {
            "search": {
                "d_model": {"type": "categorical", "choices": [128, 256]},
            },
        },
    },
    "ETTm1": {
        "*": {
            "search": {
                "geo_d_model": {"type": "categorical", "choices": [64, 128, 256]},
                "d_state": {"type": "categorical", "choices": [8, 16]},
                "epsilon": {"type": "categorical", "choices": [1e-4, 1e-3]},
                "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
                "expand": {"type": "categorical", "choices": [1]},
                "geo_expand": {"type": "categorical", "choices": [1, 2]},
                "cov_rank": {"type": "categorical", "choices": [0, 8, 16]},
            },
            "fixed": {
                "batch_size": 32,
                "lradj": "type1",
            },
        },
    },
    "ETTm2": {
        "*": {
            "search": {
                "geo_d_model": {"type": "categorical", "choices": [64, 128, 256, 512]},
                "d_state": {"type": "categorical", "choices": [2, 8]},
                "learning_rate": {"type": "loguniform", "low": 5e-6, "high": 5e-4},
                "expand": {"type": "categorical", "choices": [1, 2]},
                "geo_expand": {"type": "categorical", "choices": [1, 2]},
            },
            "fixed": {
                "batch_size": 16,
                "optim": "Adam",
                "weight_decay": 0,
                "lradj": "type1",
            },
        },
    },
    "Weather": {
        "*": {
            "search": {
                "epsilon": {"type": "categorical", "choices": [1e-4, 1e-3]},
            },
            "fixed": {
                "expand": 1,
                "geo_expand": 1,
                "batch_size": 32,
                "max_grad_norm": 1.0,
                "dropout": 0.2,
            },
        },
    },
    "ECL": {
        "*": {
            "search": {
                "cov_rank": {"type": "categorical", "choices": [16]},
                "d_model": {"type": "categorical", "choices": [128, 256]},
                "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
            },
            "fixed": {
                "batch_size": 32,
                "oom_downgrade_bs": 16,
                "use_cuda_accel": 0,
            },
        },
    },
    "Traffic": {
        "*": {
            "search": {
                "d_model": {"type": "categorical", "choices": [128, 256]},
                "epsilon": {"type": "categorical", "choices": [1e-4, 5e-4]},
                "cov_window": {"type": "categorical", "choices": [32]},
                "cov_stride": {"type": "categorical", "choices": [4, 8, 16]},
                "cov_rank": {"type": "categorical", "choices": [20]},
                "geo_d_model": {"type": "categorical", "choices": [128, 256]},
                "learning_rate": {"type": "loguniform", "low": 5e-6, "high": 5e-4},
            },
            "fixed": {
                "batch_size": 16,
                "use_cuda_accel": 0,
            },
        },
        96: {
            "search": {
                "d_state": {"type": "categorical", "choices": [8, 16]},
                "e_layers": {"type": "categorical", "choices": [3]},
                "cov_window": {"type": "categorical", "choices": [16]},
                "epsilon": {"type": "categorical", "choices": [1e-5]},
                "geo_d_model": {"type": "categorical", "choices": [128, 256, 512]},
                "learning_rate": {"type": "loguniform", "low": 5e-5, "high": 1e-3},
            },
            "fixed": {
                "batch_size": 16,
                "use_cuda_accel": 0,
            },
        },
        192: {
            "search": {
                "d_state": {"type": "categorical", "choices": [8, 16]},
                "e_layers": {"type": "categorical", "choices": [3]},
                "cov_window": {"type": "categorical", "choices": [16]},
                "epsilon": {"type": "categorical", "choices": [1e-5]},
                "geo_d_model": {"type": "categorical", "choices": [128, 256, 512]},
                "learning_rate": {"type": "loguniform", "low": 5e-5, "high": 1e-3},
            },
            "fixed": {
                "batch_size": 16,
                "use_cuda_accel": 0,
            },
        },
    },
    "Exchange": {
        "*": {
            "search": {
                "d_state": {"type": "categorical", "choices": [2, 8, 16]},
                "expand": {"type": "categorical", "choices": [1]},
                "cov_rank": {"type": "categorical", "choices": [0, 8, 16]},
                "epsilon": {"type": "categorical", "choices": [1e-4, 1e-3]},
                "geo_d_model": {"type": "categorical", "choices": [64, 128, 256]},
                "geo_expand": {"type": "categorical", "choices": [1, 2]},
                "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
            },
            "fixed": {
                "batch_size": 32,
                "lradj": "type1",
            },
        },
    },
    "illness": {
        "*": {
            "search": {
                "learning_rate": {"type": "loguniform", "low": 5e-6, "high": 1e-3},
                "cov_rank": {"type": "categorical", "choices": [0, 8, 16]},
                "expand": {"type": "categorical", "choices": [1, 2]},
                "geo_expand": {"type": "categorical", "choices": [1, 2]},
            },
            "fixed": {
                "batch_size": 32,
                "weight_decay": 1e-6,
                "lradj": "type1",
            },
        },
    },
    "Solar": {
        "*": {
            "search": {
                "cov_rank": {"type": "categorical", "choices": [32]},
                "d_model": {"type": "categorical", "choices": [128, 256]},
                "expand": {"type": "categorical", "choices": [1, 2]},
                "geo_d_model": {"type": "categorical", "choices": [64, 128]},
                "geo_expand": {"type": "categorical", "choices": [1, 2]},
                "learning_rate": {"type": "loguniform", "low": 5e-6, "high": 5e-4},
            },
            "fixed": {
                "batch_size": 32,
                "oom_downgrade_bs": 16,
            },
        },
        720: {
            "search": {
                "cov_rank": {"type": "categorical", "choices": [16]},
                "d_model": {"type": "categorical", "choices": [128, 256]},
                "expand": {"type": "categorical", "choices": [1]},
                "geo_d_model": {"type": "categorical", "choices": [64, 128]},
                "geo_expand": {"type": "categorical", "choices": [1]},
                "learning_rate": {"type": "loguniform", "low": 5e-6, "high": 5e-4},
            },
            "fixed": {
                "use_cuda_accel": 0,
            },
        },
    },
    "PEMS03": {
        "*": {
            "search": {
                "cov_rank": {"type": "categorical", "choices": [8, 16]},
                "epsilon": {"type": "categorical", "choices": [1e-4, 1e-3, 1e-5]},
                "cov_window": {"type": "categorical", "choices": [16, 32]},
                "cov_stride": {"type": "categorical", "choices": [8, 16]},
                "d_model": {"type": "categorical", "choices": [128, 256, 512]},
                "geo_d_model": {"type": "categorical", "choices": [64, 128, 256]},
                "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 2e-3},
            },
            "fixed": {
                "batch_size": 32,
                "oom_downgrade_bs": 16,
                "use_cuda_accel": 0,
            },
        },
    },
    "PEMS04": {
        "*": {
            "search": {
                "cov_rank": {"type": "categorical", "choices": [16]},
                "epsilon": {"type": "categorical", "choices": [1e-5, 1e-4]},
                "d_state": {"type": "categorical", "choices": [8, 16]},
                "cov_window": {"type": "categorical", "choices": [16, 32]},
                "cov_stride": {"type": "categorical", "choices": [8, 16]},
                "d_model": {"type": "categorical", "choices": [128, 256, 512]},
                "geo_d_model": {"type": "categorical", "choices": [128, 256, 512]},
                "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 2e-3},
            },
            "fixed": {
                "batch_size": 32,
                "oom_downgrade_bs": 16,
                "use_cuda_accel": 0,
            },
        },
    },
    "PEMS07": {
        "*": {
            "search": {
                "cov_rank": {"type": "categorical", "choices": [8, 16]},
                "epsilon": {"type": "categorical", "choices": [1e-4, 1e-3, 1e-5]},
                "cov_window": {"type": "categorical", "choices": [16, 32]},
                "cov_stride": {"type": "categorical", "choices": [8, 16]},
                "d_model": {"type": "categorical", "choices": [128, 256, 512]},
                "geo_d_model": {"type": "categorical", "choices": [64, 128, 256]},
                "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 2e-3},
            },
            "fixed": {
                "batch_size": 32,
                "oom_downgrade_bs": 16,
                "use_cuda_accel": 0,
            },
        },
    },
    "PEMS08": {
        "*": {
            "search": {
                "cov_rank": {"type": "categorical", "choices": [16]},
                "d_model": {"type": "categorical", "choices": [128, 256]},
                "geo_d_model": {"type": "categorical", "choices": [64, 128]},
            },
            "fixed": {
                "expand": 1,
                "geo_expand": 1,
                "batch_size": 32,
                "oom_downgrade_bs": 16,
            },
        },
        96: {
            "search": {
                "epsilon": {"type": "categorical", "choices": [1e-6, 1e-5]},
                "learning_rate": {"type": "loguniform", "low": 5e-5, "high": 1e-3},
            },
        },
    },
}


def _get_config(dataset_name: str, pred_len: int) -> dict:
    ds = _DATASET_PL_CONFIGS.get(dataset_name, {})
    wildcard = ds.get("*", {})
    specific = ds.get(pred_len, {})
    if not wildcard and not specific:
        return {}
    merged = {}
    for section in ("search", "training", "fixed"):
        wc = wildcard.get(section, {})
        sp = specific.get(section, {})
        if wc or sp:
            merged[section] = {**wc, **sp}
    return merged


def _apply_overrides(space: dict, overrides: dict):
    for k, v in overrides.items():
        space[k] = v


def build_search_space(enc_in: int, pred_len: int, dataset_name: str = "") -> dict:
    space = dict(_BASE_SEARCH_SPACE)
    config = _get_config(dataset_name, pred_len)
    search_overrides = config.get("search", {})
    _apply_overrides(space, search_overrides)
    return space


def build_training_space(dataset_name: str = "", pred_len: int = 0) -> dict:
    space = dict(_BASE_TRAINING_SPACE)
    config = _get_config(dataset_name, pred_len)
    training_overrides = config.get("training", {})
    _apply_overrides(space, training_overrides)
    return space


def get_fixed_params() -> dict:
    return dict(_FIXED_PARAMS)


def get_overrides(data_name: str, pred_len: int) -> dict:
    config = _get_config(data_name, pred_len)
    result = {"fixed": dict(_FIXED_PARAMS)}
    search = config.get("search", {})
    if search:
        result["search_space"] = dict(search)
    training = config.get("training")
    if training:
        result["training_space"] = dict(training)
    dataset_fixed = config.get("fixed")
    if dataset_fixed:
        result["fixed"].update(dataset_fixed)
    return result
