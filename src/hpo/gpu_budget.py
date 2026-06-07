import json
import os
import subprocess
import time
import fcntl

TEMP_OPTUNA_DIR = "./temp/optuna"
BUDGET_LOCK_FILE = os.path.join(TEMP_OPTUNA_DIR, "gpu_budget.lock")
BUDGET_JSON_FILE = os.path.join(TEMP_OPTUNA_DIR, "gpu_budget.json")
MULTI_BUDGET_FILE = os.path.join(TEMP_OPTUNA_DIR, "gpu_budget_multi.json")

_JSON_INDENT = 4


def _detect_gpu_memory_mb() -> int:
	try:
		result = subprocess.run(
			["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
			capture_output=True,
			text=True,
			timeout=10,
		)
		if result.returncode == 0:
			return int(result.stdout.strip().split("\n")[0].strip())
	except Exception:
		pass
	return 24 * 1024


TOTAL_GPU_MEMORY_MB = _detect_gpu_memory_mb()


def _entry_mb(entry) -> float:
	if isinstance(entry, dict):
		return entry.get("mb", 0)
	return float(entry)


def _entry_pid(entry):
	if isinstance(entry, dict):
		return entry.get("pid")
	return None


def _ifft_skeleton_params(seq_len: int, pred_len: int) -> float:
	n_freq = seq_len // 2 + 1
	target_freq_len = pred_len // 2 + 1
	h = max(n_freq, target_freq_len)
	mlp_params = (n_freq * h + h) + (h * target_freq_len + target_freq_len)
	return mlp_params * 2


def estimate_memory_mb(
	enc_in: int,
	d_model: int,
	seq_len: int,
	pred_len: int,
	batch_size: int,
	e_layers: int,
	expand: int,
	low_rank: int = 0,
	freq_dim: int = 0,
	d_state_n: int = 2,
	d_state_t: int = 16,
	tcn_layers: int = 5,
	tcn_ks: int = 3,
	skeleton_checkpoint: bool = True,
	skeleton: str = "Laplace",
) -> float:
	d_state_avg = (d_state_n + d_state_t) / 2
	mamba_params = (
		e_layers
		* 2
		* (d_model * expand * d_state_avg * 4 + d_model * expand * 16 + d_model * 2)
	)
	ffn_params = e_layers * 2 * (d_model * d_model * expand * 2)
	embed_params = enc_in * d_model + seq_len * d_model + d_model

	is_ifft = skeleton == "iFFT"

	if is_ifft:
		skeleton_params = _ifft_skeleton_params(seq_len, pred_len)
	else:
		if freq_dim <= 0:
			freq_dim = max(128, min(256, pred_len // 2))
		if low_rank > 0:
			skeleton_params = enc_in * (pred_len + freq_dim) * low_rank * 4
		else:
			skeleton_params = enc_in * freq_dim * pred_len * 4

	d_inner = d_model * expand
	tcn_params = enc_in * d_inner * tcn_layers * tcn_ks
	attn_params = e_layers * 2 * (d_model * d_model * 4)
	fusion_params = pred_len * 6 * pred_len + 6 * pred_len * pred_len

	total_params = mamba_params + ffn_params + embed_params + skeleton_params + tcn_params + attn_params + fusion_params

	bytes_per_param = 4
	param_mem = total_params * bytes_per_param / 1024 / 1024
	optimizer_mem = param_mem * 2
	grad_mem = param_mem
	streams = 2
	activation_mem = batch_size * seq_len * d_model * e_layers * streams * bytes_per_param / 1024 / 1024

	B = batch_size
	N = enc_in
	pl = pred_len
	fd = freq_dim if not is_ifft else 0
	lr = low_rank

	if is_ifft:
		skeleton_act = 4 * B * N * pl * bytes_per_param / 1024 / 1024
	elif low_rank > 0:
		lr_tensor_elems = 8 * B * N * (pl + fd) * lr
		if skeleton_checkpoint:
			skeleton_act = lr_tensor_elems * bytes_per_param / 1024 / 1024
		else:
			full_tensor_elems = 4 * B * N * pl * fd
			skeleton_act = (lr_tensor_elems + full_tensor_elems) * bytes_per_param / 1024 / 1024
	else:
		skeleton_act = 4 * B * N * pl * fd * bytes_per_param / 1024 / 1024
	activation_mem += skeleton_act

	cuda_overhead = 512

	total = (
		param_mem + optimizer_mem + grad_mem + activation_mem + cuda_overhead
	) * 1.05
	return total


def estimate_manimamba_memory_mb(
    enc_in: int,
    d_model: int,
    d_ff: int,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    e_layers: int,
    d_state: int = 16,
    expand: int = 1,
    cov_window: int = 16,
    cov_stride: int = 8,
    cov_rank: int = 0,
    geo_d_model: int = 64,
    geo_d_state: int = 16,
    geo_d_conv: int = 4,
    geo_expand: int = 1,
) -> float:
    cov_dim = cov_rank if cov_rank > 0 else enc_in
    tri_dim = cov_dim * (cov_dim + 1) // 2
    n_windows = max(1, (seq_len - cov_window) // cov_stride + 1)

    if cov_rank > 0:
        cov_proj_params = enc_in * cov_rank
    else:
        cov_proj_params = 0

    geo_in_proj = tri_dim * geo_d_model
    geo_d_inner = geo_expand * geo_d_model
    geo_mamba_params = 2 * (geo_d_inner * geo_d_state + geo_d_inner * geo_d_conv + geo_d_inner * geo_d_model)
    geo_out_params = geo_d_inner * geo_d_model
    geo_total = cov_proj_params + geo_in_proj + geo_mamba_params + geo_out_params

    d_inner = expand * d_model
    mamba_params_per_layer = d_inner * d_state * 4 + d_inner * 16 + d_model * 2
    mamba_params = e_layers * 2 * mamba_params_per_layer
    ffn_params = e_layers * 2 * (d_model * d_ff + d_ff * d_model)
    embed_params = enc_in * d_model + seq_len * d_model + d_model
    geo_delta_proj = geo_d_model * d_inner
    projector_params = d_model * pred_len
    attn_pool_params = 1 * 1 * d_inner

    total_params = geo_total + mamba_params + ffn_params + embed_params + geo_delta_proj + projector_params + attn_pool_params

    bytes_per_param = 4
    param_mem = total_params * bytes_per_param / 1024 / 1024
    spd_eye_buffer_mb = cov_dim * cov_dim * bytes_per_param / 1024 / 1024
    param_mem += spd_eye_buffer_mb
    optimizer_mem = param_mem * 2
    grad_mem = param_mem

    B = batch_size
    geo_act = B * n_windows * tri_dim * bytes_per_param / 1024 / 1024
    geo_mamba_act = B * n_windows * geo_d_model * 2 * bytes_per_param / 1024 / 1024
    temporal_act = B * enc_in * seq_len * d_model * e_layers * 2 * bytes_per_param / 1024 / 1024
    activation_mem = geo_act + geo_mamba_act + temporal_act

    spd_intermediate_factor = 8
    cov_act = B * n_windows * cov_dim * cov_dim * bytes_per_param / 1024 / 1024 * spd_intermediate_factor
    activation_mem += cov_act

    cuda_overhead = 512

    total = (
        param_mem + optimizer_mem + grad_mem + activation_mem + cuda_overhead
    ) * 1.05
    return total


def estimate_cuda_accel_test_peak_mb(
	enc_in: int,
	pred_len: int,
	total_rows: int,
	seq_len: int = 96,
	test_ratio: float = 0.2,
) -> float:
	n_test = int(total_rows * test_ratio) - seq_len - pred_len + 1
	n_test = max(n_test, 0)
	if n_test == 0:
		return 0.0
	elem_bytes = 4
	per_elem = n_test * pred_len * enc_in
	preds_trues_bytes = 2 * per_elem * elem_bytes
	batch_overhead_bytes = 4 * per_elem * elem_bytes
	peak_bytes = preds_trues_bytes + batch_overhead_bytes
	return peak_bytes / (1024 * 1024)


def should_disable_cuda_accel(
	enc_in: int,
	pred_len: int = 96,
	total_rows: int = 0,
	seq_len: int = 96,
	available_mb: float | None = None,
) -> bool:
	if total_rows > 0 and available_mb is not None:
		overhead = estimate_cuda_accel_test_peak_mb(
			enc_in, pred_len, total_rows, seq_len
		)
		if overhead > available_mb:
			return True
	return False


def check_memory_feasibility(
	enc_in,
	d_model,
	seq_len,
	pred_len,
	batch_size,
	e_layers,
	expand,
	freq_dim=0,
	d_state_n=2,
	d_state_t=16,
	tcn_layers=5,
	tcn_ks=3,
	skeleton_checkpoint=True,
	skeleton="Laplace",
) -> int:
	if skeleton == "iFFT":
		est = estimate_memory_mb(
			enc_in, d_model, seq_len, pred_len, batch_size,
			e_layers, expand, low_rank=0,
			d_state_n=d_state_n, d_state_t=d_state_t,
			tcn_layers=tcn_layers, tcn_ks=tcn_ks,
			skeleton_checkpoint=skeleton_checkpoint,
			skeleton=skeleton,
		)
		return 0 if est <= TOTAL_GPU_MEMORY_MB else -1

	if freq_dim <= 0:
		freq_dim = max(128, min(256, pred_len // 2))
	est = estimate_memory_mb(
		enc_in, d_model, seq_len, pred_len, batch_size,
		e_layers, expand, low_rank=0, freq_dim=freq_dim,
		d_state_n=d_state_n, d_state_t=d_state_t,
		tcn_layers=tcn_layers, tcn_ks=tcn_ks,
		skeleton_checkpoint=skeleton_checkpoint,
		skeleton=skeleton,
	)
	if est <= TOTAL_GPU_MEMORY_MB:
		return 0

	for lr in [32, 64, 128, 256]:
		if lr >= freq_dim:
			continue
		est_lr = estimate_memory_mb(
			enc_in, d_model, seq_len, pred_len, batch_size,
			e_layers, expand, low_rank=lr, freq_dim=freq_dim,
			d_state_n=d_state_n, d_state_t=d_state_t,
			tcn_layers=tcn_layers, tcn_ks=tcn_ks,
			skeleton_checkpoint=skeleton_checkpoint,
			skeleton=skeleton,
		)
		if est_lr <= TOTAL_GPU_MEMORY_MB:
			return lr

	return -1


class _NoopBudget:
	def acquire(self, *args, **kwargs):
		return True

	def release(self, *args, **kwargs):
		pass


def _migrate_gpu_state(gpu_state: dict) -> dict:
	if "reservations" in gpu_state and "reserved_by" not in gpu_state:
		return gpu_state
	reserved_mb = 0.0
	reservations = {}
	if gpu_state.get("reserved_by") is not None:
		reservations[gpu_state["reserved_by"]] = {
			"pid": gpu_state.get("reserve_pid"),
			"mb": float(gpu_state.get("used_mb", 0)),
		}
		reserved_mb = float(gpu_state.get("used_mb", 0))
	return {"reserved_mb": reserved_mb, "reservations": reservations}


class GPUBudget:
	def __init__(self, total_mb: int | None = None):
		self.total_mb = total_mb or TOTAL_GPU_MEMORY_MB
		self._pid = os.getpid()
		os.makedirs(TEMP_OPTUNA_DIR, exist_ok=True)
		if not os.path.exists(BUDGET_JSON_FILE):
			self._write_budget({"used_mb": 0, "trials": {}})

	def _read_budget(self) -> dict:
		try:
			with open(BUDGET_JSON_FILE, "r") as f:
				return json.load(f)
		except (json.JSONDecodeError, ValueError):
			default = {"used_mb": 0, "trials": {}}
			self._write_budget(default)
			return default

	def _write_budget(self, data: dict):
		with open(BUDGET_JSON_FILE, "w") as f:
			json.dump(data, f, indent=_JSON_INDENT, ensure_ascii=False)

	def _cleanup_stale(self, budget: dict) -> dict:
		stale_ids = []
		for tid, entry in budget["trials"].items():
			pid = _entry_pid(entry)
			if pid is not None:
				try:
					os.kill(pid, 0)
				except (OSError, ProcessLookupError):
					stale_ids.append(tid)
			else:
				stale_ids.append(tid)
		for tid in stale_ids:
			del budget["trials"][tid]
		used = sum(_entry_mb(e) for e in budget["trials"].values())
		budget["used_mb"] = used
		return budget

	def acquire(self, trial_id: str, needed_mb: float, timeout: int = 3600) -> bool:
		start = time.time()
		while time.time() - start < timeout:
			with open(BUDGET_LOCK_FILE, "w") as lock_f:
				fcntl.flock(lock_f, fcntl.LOCK_EX)
				try:
					budget = self._read_budget()
					budget = self._cleanup_stale(budget)
					available = self.total_mb - budget["used_mb"]
					if needed_mb <= available:
						budget["used_mb"] += needed_mb
						budget["trials"][trial_id] = {"pid": self._pid, "mb": needed_mb}
						self._write_budget(budget)
						return True
				finally:
					fcntl.flock(lock_f, fcntl.LOCK_UN)
			time.sleep(5)
		return False

	def release(self, trial_id: str):
		with open(BUDGET_LOCK_FILE, "w") as lock_f:
			fcntl.flock(lock_f, fcntl.LOCK_EX)
			try:
				budget = self._read_budget()
				if trial_id in budget["trials"]:
					mb = _entry_mb(budget["trials"][trial_id])
					budget["used_mb"] -= mb
					del budget["trials"][trial_id]
					budget["used_mb"] = max(0.0, budget["used_mb"])
					self._write_budget(budget)
			finally:
				fcntl.flock(lock_f, fcntl.LOCK_UN)

	def release_stale(self) -> int:
		with open(BUDGET_LOCK_FILE, "w") as lock_f:
			fcntl.flock(lock_f, fcntl.LOCK_EX)
			try:
				budget = self._read_budget()
				budget = self._cleanup_stale(budget)
				removed = len([tid for tid in budget["trials"]
							   if _entry_pid(budget["trials"][tid]) is None])
				self._write_budget(budget)
			finally:
				fcntl.flock(lock_f, fcntl.LOCK_UN)
		return removed

	def release_by_prefix(self, prefix: str) -> int:
		with open(BUDGET_LOCK_FILE, "w") as lock_f:
			fcntl.flock(lock_f, fcntl.LOCK_EX)
			try:
				budget = self._read_budget()
				removed = 0
				for tid in list(budget["trials"].keys()):
					if tid.startswith(prefix):
						mb = _entry_mb(budget["trials"][tid])
						budget["used_mb"] -= mb
						del budget["trials"][tid]
						removed += 1
				budget["used_mb"] = max(0.0, budget["used_mb"])
				self._write_budget(budget)
			finally:
				fcntl.flock(lock_f, fcntl.LOCK_UN)
		return removed

	def get_available_mb(self) -> float:
		budget = self._read_budget()
		return self.total_mb - budget["used_mb"]


class MultiGPUBudget:
	def __init__(self, gpu_ids: list[int], total_mb: int = TOTAL_GPU_MEMORY_MB):
		self.gpu_ids = gpu_ids
		self.total_mb = total_mb
		self._pid = os.getpid()
		os.makedirs(TEMP_OPTUNA_DIR, exist_ok=True)
		default = {"gpus": {str(g): {"reserved_mb": 0.0, "reservations": {}} for g in gpu_ids}}
		if not os.path.exists(MULTI_BUDGET_FILE):
			self._write_budget(default)
		else:
			with open(BUDGET_LOCK_FILE, "w") as lock_f:
				fcntl.flock(lock_f, fcntl.LOCK_EX)
				try:
					budget = self._read_budget()
					for g in gpu_ids:
						gk = str(g)
						if gk not in budget["gpus"]:
							budget["gpus"][gk] = {"reserved_mb": 0.0, "reservations": {}}
						else:
							budget["gpus"][gk] = _migrate_gpu_state(budget["gpus"][gk])
					self._write_budget(budget)
				finally:
					fcntl.flock(lock_f, fcntl.LOCK_UN)

	def _read_budget(self) -> dict:
		try:
			with open(MULTI_BUDGET_FILE, "r") as f:
				return json.load(f)
		except (json.JSONDecodeError, ValueError):
			default = {"gpus": {str(g): {"reserved_mb": 0.0, "reservations": {}} for g in self.gpu_ids}}
			self._write_budget(default)
			return default

	def _write_budget(self, data: dict):
		with open(MULTI_BUDGET_FILE, "w") as f:
			json.dump(data, f, indent=_JSON_INDENT, ensure_ascii=False)

	def _cleanup_stale(self, budget: dict) -> dict:
		for gpu_key, gpu_state in budget["gpus"].items():
			stale_tags = []
			for tag, entry in gpu_state.get("reservations", {}).items():
				pid = entry.get("pid") if isinstance(entry, dict) else None
				if pid is not None:
					try:
						os.kill(pid, 0)
					except (OSError, ProcessLookupError):
						stale_tags.append(tag)
			for tag in stale_tags:
				mb = gpu_state["reservations"][tag].get("mb", 0)
				gpu_state["reserved_mb"] = max(0.0, gpu_state.get("reserved_mb", 0) - mb)
				del gpu_state["reservations"][tag]
		return budget

	def reserve(self, study_tag: str, needed_mb: float, timeout: int = 7200) -> int | None:
		start = time.time()
		while time.time() - start < timeout:
			with open(BUDGET_LOCK_FILE, "w") as lock_f:
				fcntl.flock(lock_f, fcntl.LOCK_EX)
				try:
					budget = self._read_budget()
					budget = self._cleanup_stale(budget)
					best_gpu = None
					best_available = -1
					for g in self.gpu_ids:
						gpu_key = str(g)
						gpu_state = budget["gpus"].get(gpu_key, {"reserved_mb": 0.0, "reservations": {}})
						used = gpu_state.get("reserved_mb", 0.0)
						available = self.total_mb - used
						if available >= needed_mb and available > best_available:
							best_gpu = g
							best_available = available
					if best_gpu is not None:
						gpu_key = str(best_gpu)
						if gpu_key not in budget["gpus"]:
							budget["gpus"][gpu_key] = {"reserved_mb": 0.0, "reservations": {}}
						budget["gpus"][gpu_key]["reserved_mb"] += needed_mb
						budget["gpus"][gpu_key].setdefault("reservations", {})[study_tag] = {
							"pid": self._pid, "mb": needed_mb
						}
						self._write_budget(budget)
						return best_gpu
				finally:
					fcntl.flock(lock_f, fcntl.LOCK_UN)
			time.sleep(2)
		return None

	def unreserve(self, study_tag: str):
		with open(BUDGET_LOCK_FILE, "w") as lock_f:
			fcntl.flock(lock_f, fcntl.LOCK_EX)
			try:
				budget = self._read_budget()
				for gpu_key, gpu_state in budget["gpus"].items():
					reservations = gpu_state.get("reservations", {})
					if study_tag in reservations:
						mb = reservations[study_tag].get("mb", 0)
						gpu_state["reserved_mb"] = max(0.0, gpu_state.get("reserved_mb", 0) - mb)
						del reservations[study_tag]
						break
				self._write_budget(budget)
			finally:
				fcntl.flock(lock_f, fcntl.LOCK_UN)

	def get_available_per_gpu(self) -> dict[int, float]:
		budget = self._read_budget()
		result = {}
		for g in self.gpu_ids:
			gpu_state = budget["gpus"].get(str(g), {"reserved_mb": 0.0})
			result[g] = self.total_mb - gpu_state.get("reserved_mb", 0.0)
		return result


def release_budget_by_pid(pid: int) -> float:
	if not os.path.exists(BUDGET_JSON_FILE):
		return 0.0
	with open(BUDGET_LOCK_FILE, "w") as lock_f:
		fcntl.flock(lock_f, fcntl.LOCK_EX)
		try:
			with open(BUDGET_JSON_FILE, "r") as f:
				budget = json.load(f)
			trials = budget.get("trials", {})
			to_remove = [tid for tid, entry in trials.items()
						 if isinstance(entry, dict) and entry.get("pid") == pid]
			if not to_remove:
				return 0.0
			freed = sum(_entry_mb(trials[tid]) for tid in to_remove)
			for tid in to_remove:
				del trials[tid]
			budget["used_mb"] = max(0.0, budget.get("used_mb", 0) - freed)
			budget["trials"] = trials
			with open(BUDGET_JSON_FILE, "w") as f:
				json.dump(budget, f, indent=_JSON_INDENT, ensure_ascii=False)
			return freed
		finally:
			fcntl.flock(lock_f, fcntl.LOCK_UN)


def release_multi_budget_by_pid(pid: int) -> float:
	if not os.path.exists(MULTI_BUDGET_FILE):
		return 0.0
	with open(BUDGET_LOCK_FILE, "w") as lock_f:
		fcntl.flock(lock_f, fcntl.LOCK_EX)
		try:
			with open(MULTI_BUDGET_FILE, "r") as f:
				budget = json.load(f)
			total_freed = 0.0
			for gpu_key, gpu_state in budget.get("gpus", {}).items():
				reservations = gpu_state.get("reservations", {})
				to_remove = [tag for tag, entry in reservations.items()
							 if isinstance(entry, dict) and entry.get("pid") == pid]
				if not to_remove:
					continue
				freed = sum(reservations[tag].get("mb", 0) for tag in to_remove)
				for tag in to_remove:
					del reservations[tag]
				gpu_state["reserved_mb"] = max(0.0, gpu_state.get("reserved_mb", 0) - freed)
				total_freed += freed
			if total_freed > 0:
				with open(MULTI_BUDGET_FILE, "w") as f:
					json.dump(budget, f, indent=_JSON_INDENT, ensure_ascii=False)
			return total_freed
		finally:
			fcntl.flock(lock_f, fcntl.LOCK_UN)


def release_study_reservation(study_tag: str) -> float:
	if not os.path.exists(MULTI_BUDGET_FILE):
		return 0.0
	with open(BUDGET_LOCK_FILE, "w") as lock_f:
		fcntl.flock(lock_f, fcntl.LOCK_EX)
		try:
			with open(MULTI_BUDGET_FILE, "r") as f:
				budget = json.load(f)
			for gid_str, gpu in budget["gpus"].items():
				if study_tag in gpu.get("reservations", {}):
					freed = gpu["reservations"][study_tag].get("mb", 0)
					gpu["reserved_mb"] = max(0.0, gpu.get("reserved_mb", 0) - freed)
					del gpu["reservations"][study_tag]
					with open(MULTI_BUDGET_FILE, "w") as f:
						json.dump(budget, f, indent=_JSON_INDENT, ensure_ascii=False)
					return freed
			return 0.0
		finally:
			fcntl.flock(lock_f, fcntl.LOCK_UN)
