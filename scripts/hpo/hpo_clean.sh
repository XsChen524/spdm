#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Usage:
	bash scripts/hpo/hpo_clean.sh --dataset ETTh1 --pl 96
	bash scripts/hpo/hpo_clean.sh --dataset ETTh1 --pl 96 --confirm
	bash scripts/hpo/hpo_clean.sh --dataset ETTh1 --pl 96 --dry-run

Options:
	--dataset, -d   Dataset name (ETTh1, Weather, ECL, ...)
	--pl, -p        Prediction length (96, 192, 336, 720, ...)
	--confirm, -y   Skip confirmation prompt
	--dry-run       Show what would be deleted without deleting
	--help, -h      Show this help
EOF
}

DATASET=""
PL=""
CONFIRM=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
	case "$1" in
		--dataset|-d)
			DATASET="$2"
			shift 2
			;;
		--pl|-p)
			PL="$2"
			shift 2
			;;
		--confirm|-y)
			CONFIRM=true
			shift
			;;
		--dry-run)
			DRY_RUN=true
			shift
			;;
		--help|-h)
			usage
			exit 0
			;;
		*)
			echo "Unknown option: $1"
			usage
			exit 1
			;;
	esac
done

if [[ -z "$DATASET" ]]; then
	echo "Error: --dataset is required"
	echo ""
	usage
	exit 1
fi

if [[ -z "$PL" ]]; then
	echo "Error: --pl is required"
	echo ""
	usage
	exit 1
fi

STUDY="ManiMamba_${DATASET}_pl${PL}"

DB_FILE="./temp/optuna/${STUDY}.db"

if [[ ! -f "$DB_FILE" ]]; then
	echo "No database found for study: $STUDY"
	exit 0
fi

echo "Study: $STUDY"
echo ""
echo "SQLite database to delete:"
echo "  - $DB_FILE"
echo ""

if [[ "$DRY_RUN" == true ]]; then
	echo "[dry-run] No files deleted."
	exit 0
fi

if [[ "$CONFIRM" == false ]]; then
	read -rp "Delete this database? [y/N] " answer
	if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
		echo "Aborted."
		exit 0
	fi
fi

rm -f "$DB_FILE"
echo "  Deleted: $DB_FILE"

echo ""
echo "Done. Study $STUDY database cleaned (best_params.json and CSV preserved)."
