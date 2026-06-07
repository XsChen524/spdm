#!/usr/bin/env python
import argparse
import gc
import os
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import optuna
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

sys.path.append(
	os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.hpo.objective import create_objective
from src.hpo.study_manager import create_study, export_results, make_best_callback
from src.hpo.suggest_overrides import get_dataset_config, PRED_LENS_BY_DATASET
from src.hpo.gpu_budget import MultiGPUBudget, estimate_manimamba_memory_mb, TOTAL_GPU_MEMORY_MB

_stop_requested = False
_current_study = None


def _handle_sigterm(signum, frame):
	global _stop_requested, _current_study
	_stop_requested = True
	if _current_study is not None:
		_current_study.stop()


signal.signal(signal.SIGTERM, _handle_sigterm)


class ConsecutivePrunedStopper:
	def __init__(self, max_consecutive: int = 3, max_inf: int = 10):
		self.max_consecutive = max_consecutive
		self.max_inf = max_inf

	def __call__(self, study, trial):
		pruned_count = 0
		inf_count = 0
		for t in reversed(study.trials):
			if t.state == optuna.trial.TrialState.PRUNED:
				pruned_count += 1
				inf_count = 0
			elif (
				t.state == optuna.trial.TrialState.COMPLETE
				and t.value is not None
				and np.isinf(t.value)
			):
				inf_count += 1
			else:
				break
			if pruned_count >= self.max_consecutive:
				print(
					f"[STOP] {pruned_count} consecutive trials pruned — "
					f"stopping study {study.study_name}"
				)
				study.stop()
				break
			if inf_count >= self.max_inf:
				print(
					f"[STOP] {inf_count} consecutive Complete-Inf trials — "
					f"stopping study {study.study_name}"
				)
				study.stop()
				break


def _get_max_hyperparams() -> dict:
	return {
		"d_model": 512,
		"d_ff": 1024,
		"e_layers": 3,
		"d_state": 32,
		"expand": 2,
		"batch_size": 32,
		"epsilon": 1e-3,
		"cov_window": 32,
		"cov_stride": 16,
		"cov_rank": 0,
		"geo_d_model": 128,
		"geo_d_state": 32,
		"geo_d_conv": 4,
		"geo_expand": 2,
	}


def main():
	parser = argparse.ArgumentParser(description="ManiMamba HPO with Optuna")
	parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
	parser.add_argument(
		"--pred_len", type=str, default="all", help="Prediction length, or 'all'"
	)
	parser.add_argument("--n_trials", type=int, default=50, help="Number of trials")
	parser.add_argument(
		"--metric",
		type=str,
		default="mse",
		choices=["mse", "mae"],
		help="Optimization target metric",
	)
	parser.add_argument(
		"--resume", action="store_true", default=True, help="Resume from existing Study"
	)
	parser.add_argument(
		"--timeout", type=int, default=None, help="Total timeout (seconds)"
	)
	parser.add_argument("--gpu", type=int, default=0, help="Starting physical GPU index")
	parser.add_argument(
		"--nodes",
		type=int,
		default=1,
		help="Number of GPUs (1=single GPU, >1=multi GPU mode)",
	)
	parser.add_argument(
		"--train_epochs",
		type=int,
		default=0,
		help="Override training epochs (0=use default 50)",
	)
	parser.add_argument(
		"--patience",
		type=int,
		default=5,
		help="Early stopping patience (default: 5)",
	)
	parser.add_argument(
		"--bg",
		action="store_true",
		default=False,
		help="Redirect output to log/{dataset}_{pl}_study.log instead of console",
	)
	parser.add_argument(
		"--save_tmp",
		type=int,
		default=1,
		help="Save temp .npy files (1=yes, 0=no)",
	)
	args = parser.parse_args()

	gpu_manager = None

	if args.nodes > 1:
		result = subprocess.run(
			["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
			capture_output=True, text=True,
		)
		system_gpu_count = len(result.stdout.strip().split("\n"))

		physical_ids = list(range(args.gpu, args.gpu + args.nodes))
		if max(physical_ids) >= system_gpu_count:
			print(f"ERROR: Requested GPUs {physical_ids} but system has {system_gpu_count}")
			sys.exit(1)

		os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in physical_ids)

		import torch
		torch.cuda.init()
		for g in range(args.nodes):
			_ = torch.cuda.get_device_name(g)

		gpu_ids = list(range(args.nodes))

		gpu_manager = MultiGPUBudget(gpu_ids)
		print(f"Multi-GPU mode: {args.nodes} GPUs (physical {physical_ids}, runtime {gpu_ids})")
	else:
		if "CUDA_VISIBLE_DEVICES" not in os.environ:
			os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
		gpu_ids = None
		print(f"Single GPU mode: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

	dataset_config = get_dataset_config(args.dataset)

	if args.pred_len == "all":
		pred_lens = PRED_LENS_BY_DATASET.get(args.dataset, [96, 192, 336, 720])
	else:
		pred_lens = [int(args.pred_len)]

	for pred_len in pred_lens:
		if _stop_requested:
			break

		_log_file = None
		if args.bg:
			os.makedirs("log", exist_ok=True)
			_log_path = os.path.join("log", f"{args.dataset}_{pred_len}_study.log")
			_log_file = open(_log_path, "a", buffering=1)
			sys.stdout = _log_file
			sys.stderr = _log_file

		reserved_gpu = None
		study_tag = f"{args.dataset}_pl{pred_len}"

		try:
			if gpu_manager is not None:
				max_params = _get_max_hyperparams()
				study_mb = estimate_manimamba_memory_mb(
					enc_in=dataset_config["enc_in"],
					d_model=max_params["d_model"],
					d_ff=max_params["d_ff"],
					seq_len=dataset_config.get("seq_len", 96),
					pred_len=pred_len,
					batch_size=max_params["batch_size"],
					e_layers=max_params["e_layers"],
					d_state=max_params["d_state"],
					expand=max_params["expand"],
					cov_window=max_params["cov_window"],
					cov_stride=max_params["cov_stride"],
					cov_rank=max_params["cov_rank"],
					geo_d_model=max_params["geo_d_model"],
					geo_d_state=max_params["geo_d_state"],
					geo_d_conv=max_params["geo_d_conv"],
					geo_expand=max_params["geo_expand"],
				)
				study_mb = min(study_mb * 1.1, TOTAL_GPU_MEMORY_MB)
				reserved_gpu = gpu_manager.reserve(study_tag, study_mb, timeout=7200)
				if reserved_gpu is None:
					print(f"SKIP {study_tag}: GPU reservation timeout")
					continue
				gpu_for_study = reserved_gpu
			else:
				gpu_for_study = args.gpu

			study_name = f"ManiMamba_{args.dataset}_pl{pred_len}"
			print(f"\n{'='*60}")
			print(f"Study: {study_name}")
			print(
				f"Dataset: {args.dataset} | pred_len: {pred_len} | Optimization Target: {args.metric.upper()}"
			)
			print(f"{'='*60}")

			seq_len = dataset_config.get("seq_len", 96)
			label_len = dataset_config.get("label_len", 48)

			fixed_params = {
				**dataset_config,
				"dataset": args.dataset,
				"seq_len": seq_len,
				"label_len": label_len,
				"pred_len": pred_len,
				"model_id": f"{args.dataset}_{seq_len}_{pred_len}",
				"model": "ManiMamba",
				"des": "HPO_OPTUNA",
				"gpu": gpu_for_study,
			}

			if args.train_epochs > 0:
				fixed_params["train_epochs"] = args.train_epochs

			fixed_params["patience"] = args.patience
			fixed_params["save_tmp"] = args.save_tmp

			study = create_study(
				study_name=study_name,
				direction="minimize",
			)

			global _current_study
			_current_study = study

			objective = create_objective(
				fixed_params=fixed_params,
				metric=args.metric,
				gpu_ids=None,
				dedicated_gpu=True,
				stop_check=lambda: _stop_requested,
			)

			callbacks = [make_best_callback(study_name, args.metric), ConsecutivePrunedStopper(max_consecutive=3, max_inf=10)]

			if args.resume:
				completed = len(
					[t for t in study.trials if t.state.is_finished()]
				)
				remaining = max(0, args.n_trials - completed)
				print(
					f"Resuming Study, {completed} trials completed, {remaining} trials remaining"
				)
				if remaining == 0:
					print("Study completed, skipping.")
					continue
				study.optimize(objective, n_trials=remaining, timeout=args.timeout,
							   callbacks=callbacks)
			else:
				study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
							   callbacks=callbacks)

			completed_trials = [
				t for t in study.trials
				if t.state == optuna.trial.TrialState.COMPLETE
			]
			if completed_trials:
				print(f"\nBest Trial:")
				print(f"  Value ({args.metric}): {study.best_value:.6f}")
				best_params_display = dict(study.best_params)
				user_attrs = dict(study.best_trial.user_attrs)
				for key in ["batch_size", "seed"]:
					if key in user_attrs:
						best_params_display[key] = user_attrs[key]
				print(f"  Params:")
				for k, v in best_params_display.items():
					print(f"    {k}: {v}")

				export_results(study_name)
			else:
				print("\nNo completed trials — skipping best trial summary.")
		except Exception as e:
			print(f"ERROR in {study_tag}: {e}")
			traceback.print_exc()
		finally:
			_current_study = None
			if reserved_gpu is not None and gpu_manager is not None:
				gpu_manager.unreserve(study_tag)
			gc.collect()

			try:
				import torch
				torch.cuda.empty_cache()
			except Exception:
				pass

			if _log_file is not None:
				sys.stdout = sys.__stdout__
				sys.stderr = sys.__stderr__
				_log_file.close()

	print("\nAll Study completed.")


if __name__ == "__main__":
	main()
