#!/usr/bin/env python
import argparse
import sys
import os

sys.path.append(
	os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.hpo.study_manager import export_results, load_study, study_db_exists
from src.hpo.suggest_overrides import PRED_LENS_BY_DATASET


def main():
	parser = argparse.ArgumentParser(description="Dump current best Optuna results")
	parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
	parser.add_argument("--pl", type=str, required=True, help="Prediction length or 'all'")
	args = parser.parse_args()

	if args.pl == "all":
		pred_lens = PRED_LENS_BY_DATASET.get(args.dataset, [96, 192, 336, 720])
	else:
		pred_lens = [int(args.pl)]

	for pred_len in pred_lens:
		study_name = f"ManiMamba_{args.dataset}_pl{pred_len}"

		if not study_db_exists(study_name):
			print(f"[SKIP] {study_name}: database file not found")
			continue

		try:
			study = load_study(study_name)
		except Exception as e:
			print(f"[SKIP] {study_name}: study not found ({e})")
			continue

		try:
			best = study.best_trial
		except ValueError:
			print(f"[SKIP] {study_name}: no completed trials yet")
			continue

		df = export_results(study_name)
		print(f"[OK]   {study_name}: best_mse={study.best_value:.6f}, "
			  f"trial #{best.number}, "
			  f"{len(df)} trials exported")

	print("\nDone.")


if __name__ == "__main__":
	main()
