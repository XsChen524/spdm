#!/usr/bin/env python3
"""
Legacy efficiency experiment runner for non-Hi-Mamba models.
Replaces run_efficiency_exp.sh + parse_efficiency_metrics.py.

Supports: S_Mamba, UniMamba, BiMamba4TS, PatchTST, DLinear, iTransformer,
          Transformer, Autoformer, Flowformer, Informer, Reformer,
          iFlashformer, iFlowformer, and other models registered in run.py.

Usage:
    python scripts/efficiency/legacy/run_efficiency_legacy.py --model S_Mamba --dataset ETTm1 --d_state 8
    python scripts/efficiency/legacy/run_efficiency_legacy.py --model PatchTST --dataset ECL --patch_len 16 --stride 8

Results are written to output/efficiency_exp/efficiency_results_{Dataset}.txt.
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "efficiency_exp")

DATASET_CONFIGS = {
    "ETTm1": {
        "root_path": "./data/ETT-small/",
        "data_path": "ETTm1.csv",
        "data": "ETTm1",
        "freq": "t",
        "enc_in": 7, "dec_in": 7, "c_out": 7,
        "seq_len": 96, "pred_len": 96,
        "e_layers": 2, "d_model": 256, "d_ff": 256,
        "learning_rate": 7e-5,
        "d_state": 8,
    },
    "ECL": {
        "root_path": "./data/electricity/",
        "data_path": "electricity.csv",
        "data": "custom",
        "freq": "h",
        "enc_in": 321, "dec_in": 321, "c_out": 321,
        "seq_len": 96, "pred_len": 336,
        "e_layers": 2, "d_model": 512, "d_ff": 512,
        "learning_rate": 1e-4,
        "d_state": 8,
    },
    "weather": {
        "root_path": "./data/weather/",
        "data_path": "weather.csv",
        "data": "custom",
        "freq": "h",
        "enc_in": 21, "dec_in": 21, "c_out": 21,
        "seq_len": 96, "pred_len": 96,
        "e_layers": 2, "d_model": 512, "d_ff": 512,
        "learning_rate": 5e-5,
        "d_state": 8,
    },
    "PEMS08": {
        "root_path": "./data/PEMS/",
        "data_path": "PEMS08.npz",
        "data": "PEMS",
        "freq": "h",
        "enc_in": 170, "dec_in": 170, "c_out": 170,
        "seq_len": 96, "pred_len": 12,
        "e_layers": 2, "d_model": 512, "d_ff": 512,
        "learning_rate": 1e-4,
        "d_state": 8,
    },
}

MAMBA_MODELS = {"S_Mamba", "UniMamba", "BiMamba4TS", "MODE",
                "Autoformer_M", "Flowformer_M", "Flashformer_M"}

DATASET_FILE_MAP = {
    "ETTm1": "ETTm1",
    "ECL": "ECL",
    "PEMS08": "PEMS08",
    "weather": "Weather",
}


def parse_training_output(log_content):
    """Parse training log and extract MSE, MAE, training speed, GPU memory."""
    metrics = {
        "mse": 0.0,
        "mae": 0.0,
        "train_time_ms_per_iter": 0.0,
        "avg_gpu_mem_allocated_mb": 0.0,
    }

    m = re.search(r"mse:([0-9\.e\-]+),\s*mae:([0-9\.e\-]+)", log_content)
    if m:
        metrics["mse"] = float(m.group(1))
        metrics["mae"] = float(m.group(2))

    speeds = re.findall(r"speed: ([0-9\.]+)s/iter", log_content)
    if speeds:
        times_ms = [float(t) * 1000 for t in speeds]
        metrics["train_time_ms_per_iter"] = sum(times_ms) / len(times_ms)

    gpu_matches = re.findall(r"allocated_memory: ([0-9\.]+)", log_content)
    if gpu_matches:
        vals = [float(g) * 1024 for g in gpu_matches]
        metrics["avg_gpu_mem_allocated_mb"] = sum(vals) / len(vals)

    return metrics


def write_results(results_file, model_name, dataset, model_id, timestamp,
                  config, metrics):
    """Append efficiency results to the dataset-specific results file."""
    with open(results_file, "a") as f:
        f.write("=" * 80 + "\n")
        f.write("Model: {}, Dataset: {}, Model ID: {}, Timestamp: {}\n".format(
            model_name, dataset, model_id, timestamp))
        f.write("-" * 80 + "\n")
        f.write("Configuration:\n")
        f.write("  - Model: {}\n".format(model_name))
        f.write("  - Dataset: {}\n".format(dataset))
        f.write("  - Model ID: {}\n".format(model_id))
        f.write("  - Batch Size: {}\n".format(config.get("batch_size")))
        f.write("  - Train Epochs: {}\n".format(config.get("train_epochs")))
        f.write("  - Learning Rate: {}\n".format(config.get("learning_rate")))
        f.write("  - D Model: {}\n".format(config.get("d_model")))
        f.write("  - E Layers: {}\n".format(config.get("e_layers")))
        f.write("  - Seq Len: {}\n".format(config.get("seq_len")))
        f.write("  - Pred Len: {}\n".format(config.get("pred_len")))
        f.write("  - Low Rank: {}\n".format(config.get("low_rank", 0)))
        f.write("-" * 80 + "\n")
        f.write("Results:\n")
        f.write("  - MSE: {:.8f}\n".format(metrics["mse"]))
        f.write("  - MAE: {:.8f}\n".format(metrics["mae"]))
        f.write("  - Training Time: {:.2f} ms/iter\n".format(metrics["train_time_ms_per_iter"]))
        f.write("  - Avg Allocated GPU Memory: {:.2f} MB\n".format(metrics["avg_gpu_mem_allocated_mb"]))
        f.write("=" * 80 + "\n\n")


def build_command(args, ds_config):
    """Build the subprocess command list for ``scripts/run.py``."""
    model_name = args.model
    dataset = args.dataset

    learning_rate = args.learning_rate if args.learning_rate is not None else ds_config["learning_rate"]
    d_model = args.d_model if args.d_model is not None else ds_config["d_model"]
    d_ff = args.d_ff if args.d_ff is not None else ds_config["d_ff"]
    e_layers = args.e_layers if args.e_layers is not None else ds_config["e_layers"]
    d_state = args.d_state if args.d_state is not None else ds_config.get("d_state")

    train_epochs = args.train_epochs
    batch_size = args.batch_size
    n_heads = 8
    patience = 3

    model_id = "{}_{}_{}".format(dataset, ds_config["seq_len"], ds_config["pred_len"])

    cmd = [
        sys.executable, "-u",
        os.path.join(PROJECT_ROOT, "scripts", "run.py"),
        "--is_training", "1",
        "--root_path", ds_config["root_path"],
        "--data_path", ds_config["data_path"],
        "--model_id", model_id,
        "--model", model_name,
        "--data", ds_config["data"],
        "--freq", ds_config.get("freq", "h"),
        "--features", "M",
        "--seq_len", str(ds_config["seq_len"]),
        "--label_len", "48",
        "--pred_len", str(ds_config["pred_len"]),
        "--e_layers", str(e_layers),
        "--enc_in", str(ds_config["enc_in"]),
        "--dec_in", str(ds_config["dec_in"]),
        "--c_out", str(ds_config["c_out"]),
        "--des", "Efficiency-Exp",
        "--batch_size", str(batch_size),
        "--train_epochs", str(train_epochs),
        "--learning_rate", str(learning_rate),
        "--d_model", str(d_model),
        "--d_ff", str(d_ff),
        "--n_heads", str(n_heads),
        "--itr", "1",
        "--patience", str(patience),
    ]

    if args.use_amp:
        cmd.append("--use_amp")

    if args.use_8bit:
        cmd.extend(["--use_8bit", str(args.use_8bit)])

    if d_state is not None and model_name in MAMBA_MODELS:
        cmd.extend(["--d_state", str(d_state)])

    if model_name == "UniMamba":
        cmd.extend([
            "--use_laplace", "default",
            "--use_tcn", "default",
            "--use_attention", "st",
            "--tcn_num_levels", "2",
            "--expand", "2",
        ])
        if args.tcn_kernel_size is not None:
            cmd.extend(["--tcn_kernel_size", str(args.tcn_kernel_size)])
        if args.tcn_dropout is not None:
            cmd.extend(["--tcn_dropout", str(args.tcn_dropout)])
        if args.d_conv is not None:
            cmd.extend(["--d_conv", str(args.d_conv)])
        if args.st_attention_dim is not None:
            cmd.extend(["--st_attention_dim", str(args.st_attention_dim)])
        if args.st_dropout is not None:
            cmd.extend(["--st_dropout", str(args.st_dropout)])
        if args.low_rank is not None and args.low_rank > 0:
            cmd.extend(["--low_rank", str(args.low_rank)])

    if model_name == "BiMamba4TS":
        if args.e_fact is not None:
            cmd.extend(["--e_fact", str(args.e_fact)])
        if args.bi_dir is not None:
            cmd.extend(["--bi_dir", str(args.bi_dir)])
        if args.residual is not None:
            cmd.extend(["--residual", str(args.residual)])
        if args.ch_ind is not None:
            cmd.extend(["--ch_ind", str(args.ch_ind)])
        if args.embed_type is not None:
            cmd.extend(["--embed_type", str(args.embed_type)])
        if args.d_conv is not None:
            cmd.extend(["--d_conv", str(args.d_conv)])
        if args.SRA:
            cmd.append("--SRA")

    if model_name == "interPDN":
        if args.ma_type is not None:
            cmd.extend(["--ma_type", args.ma_type])
        if args.alpha is not None:
            cmd.extend(["--alpha", str(args.alpha)])
        if args.beta is not None:
            cmd.extend(["--beta", str(args.beta)])
        if args.con_cls_1 is not None:
            cmd.extend(["--con_cls_1", str(args.con_cls_1)])
        if args.con_cls_2 is not None:
            cmd.extend(["--con_cls_2", str(args.con_cls_2)])
        if args.con_time is not None:
            cmd.extend(["--con_time", str(args.con_time)])

    if args.individual:
        cmd.append("--individual")

    if args.patch_len is not None:
        cmd.extend(["--patch_len", str(args.patch_len)])
    if args.stride is not None:
        cmd.extend(["--stride", str(args.stride)])
    if args.padding_patch is not None:
        cmd.extend(["--padding_patch", args.padding_patch])

    if args.fc_dropout is not None:
        cmd.extend(["--fc_dropout", str(args.fc_dropout)])
    if args.head_dropout is not None:
        cmd.extend(["--head_dropout", str(args.head_dropout)])
    if args.pct_start is not None:
        cmd.extend(["--pct_start", str(args.pct_start)])
    if args.lradj is not None:
        cmd.extend(["--lradj", args.lradj])
    if args.dropout is not None:
        cmd.extend(["--dropout", str(args.dropout)])
    if args.use_norm is not None:
        cmd.extend(["--use_norm", str(args.use_norm)])

    cmd.extend(args.extra)

    run_config = {
        "model": model_name,
        "dataset": dataset,
        "model_id": model_id,
        "batch_size": batch_size,
        "train_epochs": train_epochs,
        "learning_rate": learning_rate,
        "d_model": d_model,
        "d_ff": d_ff,
        "e_layers": e_layers,
        "seq_len": ds_config["seq_len"],
        "pred_len": ds_config["pred_len"],
        "d_state": d_state,
        "low_rank": args.low_rank if args.low_rank is not None else 0,
    }

    return cmd, model_id, run_config


def main():
    parser = argparse.ArgumentParser(
        description="Legacy efficiency experiment runner for non-Hi-Mamba models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/efficiency/legacy/run_efficiency_legacy.py --model S_Mamba --dataset ETTm1 --d_state 8\n"
            "  python scripts/efficiency/legacy/run_efficiency_legacy.py --model BiMamba4TS --dataset ETTm1 --SRA\n"
            "  python scripts/efficiency/legacy/run_efficiency_legacy.py --model PatchTST --dataset ECL --patch_len 16 --stride 8\n"
        ),
    )

    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset: ETTm1, ECL, weather, PEMS08")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--d_state", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--e_layers", type=int, default=None)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--train_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_8bit", type=int, default=0)

    parser.add_argument("--tcn_kernel_size", type=int, default=None)
    parser.add_argument("--tcn_dropout", type=float, default=None)
    parser.add_argument("--d_conv", type=int, default=None)
    parser.add_argument("--st_attention_dim", type=int, default=None)
    parser.add_argument("--st_dropout", type=float, default=None)
    parser.add_argument("--low_rank", type=int, default=None)

    parser.add_argument("--e_fact", type=int, default=None)
    parser.add_argument("--bi_dir", type=int, default=None)
    parser.add_argument("--residual", type=int, default=None)
    parser.add_argument("--ch_ind", type=int, default=None)
    parser.add_argument("--embed_type", type=int, default=None)
    parser.add_argument("--SRA", action="store_true", default=False)

    parser.add_argument("--patch_len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--padding_patch", type=str, default=None)

    parser.add_argument("--individual", action="store_true", default=False)

    parser.add_argument("--fc_dropout", type=float, default=None)
    parser.add_argument("--head_dropout", type=float, default=None)
    parser.add_argument("--pct_start", type=float, default=None)
    parser.add_argument("--lradj", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--use_norm", type=int, default=None)

    parser.add_argument("--ma_type", type=str, default=None, help="interPDN: reg, ema, dema")
    parser.add_argument("--alpha", type=float, default=None, help="interPDN: EMA alpha")
    parser.add_argument("--beta", type=float, default=None, help="interPDN: DEMA beta")
    parser.add_argument("--con_cls_1", type=float, default=None, help="interPDN: consistency loss weight")
    parser.add_argument("--con_cls_2", type=float, default=None, help="interPDN: consistency loss weight")
    parser.add_argument("--con_time", type=float, default=None, help="interPDN: cross-scale consistency weight")

    args, extra = parser.parse_known_args()
    args.extra = extra

    dataset = args.dataset
    if dataset not in DATASET_CONFIGS:
        print("ERROR: Unknown dataset '{}'".format(dataset))
        print("  Available: {}".format(list(DATASET_CONFIGS.keys())))
        sys.exit(1)

    ds_config = DATASET_CONFIGS[dataset]
    cmd, model_id, run_config = build_command(args, ds_config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Parameters: Batch {}, Epochs {}, LR {}".format(
        run_config["batch_size"], run_config["train_epochs"], run_config["learning_rate"]))
    print("Model ID: {}".format(model_id))
    print("=" * 80)
    print()
    print("Starting training...")
    print()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=PROJECT_ROOT,
    )

    output_lines = []
    for line in iter(proc.stdout.readline, b""):
        decoded = line.decode("utf-8", errors="replace")
        print(decoded, end="")
        output_lines.append(decoded)

    proc.wait()

    if proc.returncode != 0:
        print("\nERROR: Training failed for {} on {}".format(args.model, dataset))
        sys.exit(1)

    log_content = "".join(output_lines)
    print("\nParsing metrics...")

    metrics = parse_training_output(log_content)

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    ds_file = DATASET_FILE_MAP.get(dataset, dataset)
    results_file = os.path.join(DEFAULT_OUTPUT_DIR,
                                "efficiency_results_{}.txt".format(ds_file))

    write_results(results_file, args.model, dataset, model_id, timestamp,
                  run_config, metrics)

    print("Successfully parsed metrics for {} on {}".format(args.model, dataset))
    print("  Model ID: {}".format(model_id))
    print("  MSE: {:.8f}".format(metrics["mse"]))
    print("  MAE: {:.8f}".format(metrics["mae"]))
    print("  Training Time: {:.2f} ms/iter".format(metrics["train_time_ms_per_iter"]))
    print("  Avg Allocated GPU Memory: {:.2f} MB".format(metrics["avg_gpu_mem_allocated_mb"]))
    print("Results appended to {}".format(results_file))
    print("=" * 80)


if __name__ == "__main__":
    main()
