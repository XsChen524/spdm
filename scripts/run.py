import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from src.experiments.exp_long_term_forecasting_partial import (
    Exp_Long_Term_Forecast_Partial,
)
from configs.model_parser_dict import model_parser_dict


def _build_setting(args, robustness_str, ii):
    setting = "{}_{}|{}|lr:{}|ft:{}|sl:{}|ll:{}|pl:{}|dm:{}|nh:{}|el:{}|dl:{}|df:{}|fc:{}|eb:{}|dt:{}|cs:{}{}|itr:{}".format(
        args.model_id,
        args.model,
        args.des,
        args.learning_rate,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.class_strategy,
        robustness_str,
        ii,
    )
    return setting


if __name__ == "__main__":
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="Time Series Forecasting", conflict_handler="resolve")

    if "__all__" in model_parser_dict:
        for common_parser in model_parser_dict["__all__"]:
            common_parser(parser=parser)

    temp_args, _ = parser.parse_known_args()
    if temp_args.model and temp_args.model in model_parser_dict:
        for model_parser in model_parser_dict[temp_args.model]:
            model_parser(parser=parser)

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)

    if args.exp_name == "partial_train":
        Exp = Exp_Long_Term_Forecast_Partial
    else:
        Exp = Exp_Long_Term_Forecast

    torch.multiprocessing.set_sharing_strategy("file_system")

    if getattr(args, "checkpoint", 0) > 0:
        from src.utils.checkpoint import checkpoint_exists, _ckpt_prefix, cleanup_checkpoint

        prefix = _ckpt_prefix(args)
        if not checkpoint_exists(args, args.checkpoint):
            print(f"ERROR: Checkpoint {prefix}_{args.checkpoint}.pth not found")
            sys.exit(1)
        setting = "{}_{}|{}|checkpoint:{}|itr:0".format(
            args.model_id, args.model, args.des, args.checkpoint
        )
        exp = Exp(args)
        print(
            ">>>>>>>continued training from checkpoint {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(
                args.checkpoint
            )
        )
        exp.train(setting)
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting)
        if args.do_predict:
            print(
                ">>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(
                    setting
                )
            )
            exp.predict(setting, True)

        _lr_seq = getattr(exp, "_lr_checkpoint_to_cleanup", None)
        if _lr_seq is not None:
            cleanup_checkpoint(args, _lr_seq)
            exp._last_checkpoint_seq = None

        torch.cuda.empty_cache()
    elif args.is_training == 1:
        for ii in range(args.itr):
            robustness_str = ""
            if args.noise_level > 0:
                robustness_str = "|nl:{}".format(args.noise_level)

            setting = _build_setting(args, robustness_str, ii)

            exp = Exp(args)
            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            exp.train(setting)

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            exp.test(setting)

            if args.do_predict:
                print(
                    ">>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(
                        setting
                    )
                )
                exp.predict(setting, True)

            if getattr(exp, "_lr_checkpoint_to_cleanup", None) is not None:
                from src.utils.checkpoint import cleanup_checkpoint
                cleanup_checkpoint(args, exp._lr_checkpoint_to_cleanup)
                exp._last_checkpoint_seq = None

            torch.cuda.empty_cache()
    elif args.is_training == 2:
        for ii in range(args.itr):
            robustness_str = ""
            if args.noise_level > 0:
                robustness_str = "|nl:{}".format(args.noise_level)
            setting = _build_setting(args, robustness_str, ii)
            exp = Exp(args)
        ii = 0
        robustness_str = ""
        if args.noise_level > 0:
            robustness_str = "|nl:{}".format(args.noise_level)
        setting = _build_setting(args, robustness_str, ii)

        exp = Exp(args)
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
