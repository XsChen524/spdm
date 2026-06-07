#!/bin/bash
set -euo pipefail

MERGED_DB="./temp/optuna/_all_studies.db"

cleanup_merged() {
	rm -f "$MERGED_DB"
}

usage() {
	cat <<'EOF'
Usage:
	bash scripts/hpo/hpo_dashboard.sh --dataset ETTh1 --pl 96
	bash scripts/hpo/hpo_dashboard.sh --dataset ETTh1 --pl 96 --port 59102
	bash scripts/hpo/hpo_dashboard.sh --study ManiMamba_ETTh1_pl96
	bash scripts/hpo/hpo_dashboard.sh --study ManiMamba_ETTh1_pl96 --port 59102
	bash scripts/hpo/hpo_dashboard.sh --all
	bash scripts/hpo/hpo_dashboard.sh --all --port 59102

Options:
	--dataset, -d   Dataset name (ETTh1, Weather, ECL, ...)
	--pl, -p        Prediction length (96, 192, 336, 720, ...)
	--study, -s     Full study name (overrides --dataset/--pl)
	--all, -a       Show all studies in one dashboard
	--port          Dashboard port (default: 58231)
	--help, -h      Show this help
EOF
}

DATASET=""
PL=""
STUDY=""
ALL_STUDIES=false
STORAGE=""
PORT=${OPTUNA_PORT:-58231}

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
		--study|-s)
			STUDY="$2"
			shift 2
			;;
		--all|-a)
			ALL_STUDIES=true
			shift
			;;
		--port)
			PORT="$2"
			shift 2
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

if [[ "$ALL_STUDIES" == true ]]; then
	if [[ -n "$STUDY" || -n "$DATASET" || -n "$PL" ]]; then
		echo "Error: --all cannot be used with --study, --dataset, or --pl"
		echo ""
		usage
		exit 1
	fi

	DB_FILES=(./temp/optuna/ManiMamba_*.db)
	if [[ ! -f "${DB_FILES[0]}" ]]; then
		echo "Error: no study databases found in ./temp/optuna/"
		exit 1
	fi

	cleanup_merged

	echo "Merging all studies into temporary database..."
	python3 -c "
import optuna, glob, sys

dst = 'sqlite:///./temp/optuna/_all_studies.db'
count = 0
for db in sorted(glob.glob('./temp/optuna/ManiMamba_*.db')):
    src = f'sqlite:///{db}'
    try:
        for s in optuna.study.get_all_study_summaries(src):
            optuna.copy_study(from_study_name=s.study_name, from_storage=src, to_storage=dst)
            print(f'  {s.study_name} ({s.n_trials} trials)')
            count += 1
    except Exception as e:
        print(f'  Skipped {db}: {e}', file=sys.stderr)
print(f'\n  Total: {count} studies merged')
"

	STUDY="(all studies)"
	STORAGE="sqlite:///./temp/optuna/_all_studies.db"

	trap cleanup_merged EXIT
elif [[ -n "$STUDY" ]]; then
	STORAGE="sqlite:///./temp/optuna/${STUDY}.db"
elif [[ -n "$DATASET" && -n "$PL" ]]; then
	STUDY="ManiMamba_${DATASET}_pl${PL}"
	STORAGE="sqlite:///./temp/optuna/${STUDY}.db"
else
	echo "Error: specify --all, --study, OR (--dataset AND --pl)"
	echo ""
	usage
	exit 1
fi

if [[ "$ALL_STUDIES" == false ]]; then
	DB_FILE="./temp/optuna/${STUDY}.db"
	if [[ ! -f "$DB_FILE" ]]; then
		echo "Warning: database not found: $DB_FILE"
		echo "The dashboard will start, but the study may not exist yet."
		echo ""
	fi
fi

echo "Launching Optuna Dashboard..."
echo "  Study:      $STUDY"
echo "  Storage:    $STORAGE"
echo "  Access URL: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop."

optuna-dashboard "$STORAGE" --port "$PORT"
