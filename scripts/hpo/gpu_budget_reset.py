import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import fcntl

from src.hpo.gpu_budget import (
	BUDGET_JSON_FILE,
	MULTI_BUDGET_FILE,
	BUDGET_LOCK_FILE,
	GPUBudget,
	MultiGPUBudget,
	_entry_mb,
	_entry_pid,
	release_budget_by_pid,
	release_multi_budget_by_pid,
	release_study_reservation,
)
import optuna

_SEP = "─" * 60
_INDENT = "  "
_INDENT2 = _INDENT * 2
_INDENT3 = _INDENT * 3


def _fmt_mb(mb: float) -> str:
	if mb >= 1024:
		return f"{mb:.1f} MB ({mb / 1024:.2f} GB)"
	return f"{mb:.1f} MB"


def _is_pid_alive(pid: int) -> bool:
	try:
		os.kill(pid, 0)
		return True
	except (OSError, ProcessLookupError):
		return False


def _running_hpo_daemons() -> list[str]:
	pid_dir = "./temp/optuna/pids"
	if not os.path.isdir(pid_dir):
		return []
	alive = []
	for pf in os.listdir(pid_dir):
		if not pf.endswith(".pid"):
			continue
		try:
			with open(os.path.join(pid_dir, pf)) as f:
				pid = int(f.read().strip())
			if _is_pid_alive(pid):
				alive.append(pf[:-4])
		except (ValueError, OSError):
			pass
	return alive


def _reset_interactive():
	daemons = _running_hpo_daemons()
	if daemons:
		print(f"\n{_SEP}")
		print(f"WARNING: {len(daemons)} HPO daemon(s) are running:")
		for d in daemons:
			print(f"{_INDENT}- {d}")
		print(f"{_SEP}")
		print("Resetting GPU budget while daemons are running may cause issues.")
		answer = input("Continue anyway? [y/N] ").strip().lower()
		if answer not in ("y", "yes"):
			print("Aborted. Stop daemons first with: poe hpo-stop-all")
			return

	has_any = False

	if os.path.exists(BUDGET_JSON_FILE):
		has_any = True
		with open(BUDGET_JSON_FILE, "r") as f:
			budget = json.load(f)

		used_mb = budget.get("used_mb", 0)
		trials = budget.get("trials", {})

		print(f"\n{_SEP}")
		print("Single-GPU Budget")
		print(_SEP)
		print(f"{_INDENT}Used   : {_fmt_mb(used_mb)}")
		print(f"{_INDENT}Trials : {len(trials)} active")

		if not trials:
			print(f"\n{_INDENT}(clean — no reset needed)")
		else:
			alive_trials = {}
			dead_trials = {}

			for tid, entry in trials.items():
				pid = _entry_pid(entry)
				mb = _entry_mb(entry)
				if pid is not None and _is_pid_alive(int(pid)):
					alive_trials[tid] = (mb, pid)
				else:
					dead_trials[tid] = (mb, pid)

			if dead_trials:
				print(f"\n{_INDENT}Stale entries ({len(dead_trials)}):")
				for tid, (mb, pid) in dead_trials.items():
					pid_str = f"PID={pid}" if pid is not None else "no PID"
					print(f"{_INDENT2}- {tid}")
					print(f"{_INDENT3}Size : {_fmt_mb(mb)}")
					print(f"{_INDENT3}PID  : {pid_str} (dead)")

			if alive_trials:
				print(f"\n{_INDENT}Active entries ({len(alive_trials)}):")
				for tid, (mb, pid) in alive_trials.items():
					print(f"{_INDENT2}- {tid}")
					print(f"{_INDENT3}Size : {_fmt_mb(mb)}")
					print(f"{_INDENT3}PID  : {pid} (alive)")
	else:
		print(f"\n{_SEP}")
		print("Single-GPU Budget: no file found")

	if os.path.exists(MULTI_BUDGET_FILE):
		has_any = True
		with open(MULTI_BUDGET_FILE, "r") as f:
			multi_state = json.load(f)

		print(f"\n{_SEP}")
		print("Multi-GPU Budget")
		print(_SEP)

		total_reserved = 0.0
		total_studies = 0
		for gid, gpu in multi_state.get("gpus", {}).items():
			reservations = gpu.get("reservations", {})
			n_res = len(reservations)
			res_mb = gpu.get("reserved_mb", 0.0)
			total_reserved += res_mb
			total_studies += n_res

			print(f"\n{_INDENT}GPU {gid}:")
			print(f"{_INDENT2}Reserved : {_fmt_mb(res_mb)}")
			print(f"{_INDENT2}Studies  : {n_res}")

			if reservations:
				for tag, info in reservations.items():
					alive = _is_pid_alive(info["pid"])
					status = "alive" if alive else "DEAD"
					print(f"{_INDENT2}- {tag}")
					print(f"{_INDENT3}Size   : {_fmt_mb(info['mb'])}")
					print(f"{_INDENT3}PID    : {info['pid']} ({status})")
			else:
				print(f"{_INDENT2}(idle)")

		print(f"\n{_INDENT}Total: {_fmt_mb(total_reserved)} across {total_studies} studies")
	else:
		print(f"\n{_SEP}")
		print("Multi-GPU Budget: no file found")

	if not has_any:
		print(f"\n{_SEP}")
		print("No budget files found — nothing to reset.")
		return

	dead_single = []
	dead_multi = []

	if os.path.exists(BUDGET_JSON_FILE):
		with open(BUDGET_JSON_FILE, "r") as f:
			budget = json.load(f)
		for tid, entry in budget.get("trials", {}).items():
			pid = _entry_pid(entry)
			if pid is None or not _is_pid_alive(int(pid)):
				dead_single.append(tid)

	if os.path.exists(MULTI_BUDGET_FILE):
		with open(MULTI_BUDGET_FILE, "r") as f:
			multi_state = json.load(f)
		for gid, gpu in multi_state.get("gpus", {}).items():
			for tag, info in gpu.get("reservations", {}).items():
				if not _is_pid_alive(info["pid"]):
					dead_multi.append((gid, tag))

	if not dead_single and not dead_multi:
		print(f"\n{_SEP}")
		print("All entries belong to live processes — no reset needed.")
		return

	parts = []
	if dead_single:
		parts.append(f"{len(dead_single)} stale single-GPU entries")
	if dead_multi:
		parts.append(f"{len(dead_multi)} stale multi-GPU reservations")
	print(f"\n{_SEP}")
	print(f"Will remove {' and '.join(parts)}")

	answer = input("\nProceed? [y/N] ").strip().lower()
	if answer not in ("y", "yes"):
		print("Aborted.")
		return

	if dead_single:
		with open(BUDGET_LOCK_FILE, "w") as lock_f:
			fcntl.flock(lock_f, fcntl.LOCK_EX)
			try:
				with open(BUDGET_JSON_FILE, "r") as f:
					budget = json.load(f)
				freed = 0.0
				for tid in dead_single:
					freed += _entry_mb(budget["trials"][tid])
					del budget["trials"][tid]
				budget["used_mb"] = max(0.0, budget["used_mb"] - freed)
				with open(BUDGET_JSON_FILE, "w") as f:
					json.dump(budget, f, indent=4, ensure_ascii=False)
				print(f"\nSingle-GPU: removed {len(dead_single)} stale entries ({_fmt_mb(freed)})")
			finally:
				fcntl.flock(lock_f, fcntl.LOCK_UN)

	if dead_multi:
		with open(BUDGET_LOCK_FILE, "w") as lock_f:
			fcntl.flock(lock_f, fcntl.LOCK_EX)
			try:
				with open(MULTI_BUDGET_FILE, "r") as f:
					multi_state = json.load(f)
				freed = 0.0
				for gid, tag in dead_multi:
					gpu = multi_state["gpus"][gid]
					freed += gpu["reservations"][tag]["mb"]
					gpu["reserved_mb"] -= gpu["reservations"][tag]["mb"]
					del gpu["reservations"][tag]
					gpu["reserved_mb"] = max(0.0, gpu["reserved_mb"])
				with open(MULTI_BUDGET_FILE, "w") as f:
					json.dump(multi_state, f, indent=4, ensure_ascii=False)
				print(f"Multi-GPU : removed {len(dead_multi)} stale reservations ({_fmt_mb(freed)})")
			finally:
				fcntl.flock(lock_f, fcntl.LOCK_UN)


def reset_stale_auto():
	daemons = _running_hpo_daemons()
	if daemons:
		print(f"\n{_SEP}")
		print(f"WARNING: {len(daemons)} HPO daemon(s) still running:")
		for d in daemons:
			print(f"{_INDENT}- {d}")
		print(f"{_SEP}")
		print("Skipping auto-reset. Stop daemons first with: poe hpo-stop-all")
		return

	print(f"\n{_SEP}")
	print("Auto-reset: scanning for stale entries ...")
	print(_SEP)

	budget = GPUBudget()
	removed = budget.release_stale()
	if removed > 0:
		print(f"{_INDENT}Single-GPU: removed {removed} stale entries")
	else:
		print(f"{_INDENT}Single-GPU: no stale entries found")

	if os.path.exists(MULTI_BUDGET_FILE):
		try:
			with open(BUDGET_LOCK_FILE, "w") as lock_f:
				fcntl.flock(lock_f, fcntl.LOCK_EX)
				try:
					with open(MULTI_BUDGET_FILE, "r") as f:
						state = json.load(f)
					stale_count = 0
					for gid, gpu in state.get("gpus", {}).items():
						stale = []
						for tag, info in gpu.get("reservations", {}).items():
							if not _is_pid_alive(info["pid"]):
								stale.append(tag)
						for tag in stale:
							gpu["reserved_mb"] -= gpu["reservations"][tag]["mb"]
							del gpu["reservations"][tag]
							stale_count += 1
						gpu["reserved_mb"] = max(0.0, gpu["reserved_mb"])
					if stale_count > 0:
						with open(MULTI_BUDGET_FILE, "w") as f:
							json.dump(state, f, indent=4, ensure_ascii=False)
						print(f"{_INDENT}Multi-GPU : removed {stale_count} stale reservations")
					else:
						print(f"{_INDENT}Multi-GPU : no stale reservations found")
				finally:
					fcntl.flock(lock_f, fcntl.LOCK_UN)
		except Exception as e:
			print(f"{_INDENT}Multi-GPU budget reset failed: {e}")

	print(_SEP)


def release_by_pid(pid: int):
	pid = int(pid)
	freed_single = release_budget_by_pid(pid)
	freed_multi = release_multi_budget_by_pid(pid)
	total = freed_single + freed_multi
	if total > 0:
		print(f"Released {_fmt_mb(total)} for PID {pid}")
		if freed_single > 0:
			print(f"{_INDENT}Single-GPU : {_fmt_mb(freed_single)}")
		if freed_multi > 0:
			print(f"{_INDENT}Multi-GPU  : {_fmt_mb(freed_multi)}")
	else:
		print(f"No budget entries found for PID {pid}")


def prune_study(study_name: str):
	from src.hpo.study_manager import study_db_exists, load_study

	if not study_db_exists(study_name):
		print(f"Study not found: {study_name}")
		return

	study = load_study(study_name)
	pruned = 0
	for trial in study.trials:
		if trial.state == optuna.trial.TrialState.RUNNING:
			study.tell(trial.number, state=optuna.trial.TrialState.PRUNED)
			pruned += 1
	print(f"Pruned {pruned} RUNNING trials in study {study_name}")


def main():
	args = sys.argv[1:]

	if "--release-pid" in args:
		idx = args.index("--release-pid")
		if idx + 1 < len(args):
			release_by_pid(args[idx + 1])
		else:
			print("Usage: gpu_budget_reset.py --release-pid <pid>")
			sys.exit(1)
		return

	if "--prune-study" in args:
		idx = args.index("--prune-study")
		if idx + 1 < len(args):
			prune_study(args[idx + 1])
		else:
			print("Usage: gpu_budget_reset.py --prune-study <study_name>")
			sys.exit(1)
		return

	if "--release-study" in args:
		idx = args.index("--release-study")
		if idx + 1 < len(args):
			freed = release_study_reservation(args[idx + 1])
			if freed > 0:
				print(f"Released {_fmt_mb(freed)} for study {args[idx + 1]}")
			else:
				print(f"Study {args[idx + 1]} not found in multi-GPU budget")
		else:
			print("Usage: gpu_budget_reset.py --release-study <study_tag>")
			sys.exit(1)
		return

	if "--yes" in args or "--force" in args:
		reset_stale_auto()
		return

	_reset_interactive()


if __name__ == "__main__":
	main()
