#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-}"

if [ -z "$DATASET" ]; then
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	rm -rf temp/results/* temp/test_results/* temp/checkpoints/*
else
	rm -rf "temp/checkpoints/${DATASET}/"
	for dir in temp/results temp/test_results; do
		if [ -d "$dir" ]; then
			find "$dir" -maxdepth 1 -name "${DATASET}_*" -exec rm -rf {} +
		fi
	done
fi
