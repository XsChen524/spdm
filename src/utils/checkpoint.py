import os
import torch


_CHECKPOINT_ROOT = "./temp/checkpoints"


def _get_dataset_name(args):
	name = getattr(args, "data", "custom")
	if name == "PEMS":
		data_path = getattr(args, "data_path", "")
		base = os.path.splitext(os.path.basename(data_path))[0]
		if base:
			return base
	if name == "custom":
		model_id = getattr(args, "model_id", "custom")
		parts = model_id.split("_")
		for i in range(len(parts), 0, -1):
			candidate = "_".join(parts[:i])
			if not candidate.isdigit():
				return candidate
		return model_id
	return name


def _ckpt_prefix(args):
	dataset = _get_dataset_name(args)
	seq_len = getattr(args, "seq_len", 0)
	pred_len = getattr(args, "pred_len", 0)
	des = getattr(args, "des", "")
	low_rank = getattr(args, "low_rank", 0)
	suffix = f"_{des}" if des else ""
	lr_tag = f"_lr{low_rank}" if low_rank > 0 else ""
	return f"{dataset}_{args.model}_sl{seq_len}_pl{pred_len}{lr_tag}{suffix}"


def _get_checkpoint_dir(args):
	dataset = _get_dataset_name(args)
	return os.path.join(_CHECKPOINT_ROOT, dataset)


def _get_log_path(args):
	dataset = _get_dataset_name(args)
	return os.path.join(_CHECKPOINT_ROOT, dataset, f"checkpoints_{dataset}.txt")


def _get_next_seq(args):
	ckpt_dir = _get_checkpoint_dir(args)
	if not os.path.isdir(ckpt_dir):
		return 1
	prefix = f"{_ckpt_prefix(args)}_"
	max_seq = 0
	for f in os.listdir(ckpt_dir):
		if f.endswith(".pth") and f.startswith(prefix):
			seq_str = f[len(prefix):-4]
			try:
				max_seq = max(max_seq, int(seq_str))
			except ValueError:
				pass
	return max_seq + 1


def _collect_hyperparams(args):
	_ARCH_PARAMS = [
		("enc_in", "enc_in"),
		("dec_in", "dec_in"),
		("c_out", "c_out"),
		("d_model", "d_model"),
		("d_ff", "d_ff"),
		("n_heads", "n_heads"),
		("e_layers", "e_layers"),
		("d_layers", "d_layers"),
		("dropout", "dropout"),
		("activation", "activation"),
		("factor", "factor"),
		("embed", "embed"),
		("moving_avg", "moving_avg"),
		("use_norm", "use_norm"),
		("class_strategy", "class_strategy"),
		("batch_size", "batch_size"),
	]
	_UNIMAMBA_PARAMS = [
		("d_state_n", "d_state_n"),
		("d_state_t", "d_state_t"),
		("d_conv_n", "d_conv_n"),
		("d_conv_t", "d_conv_t"),
		("expand", "expand"),
		("use_tcn", "use_tcn"),
		("use_attention", "use_attention"),
		("tcn_ks", "tcn_ks"),
		("tcn_layers", "tcn_layers"),
		("tcn_dropout", "tcn_dropout"),
		("st_attention_dim", "st_attention_dim"),
		("st_dropout", "st_dropout"),
		("low_rank", "low_rank"),
		("dense_mode", "dense_mode"),
	]
	result = []
	for attr, label in _ARCH_PARAMS + _UNIMAMBA_PARAMS:
		val = getattr(args, attr, None)
		if val is not None:
			result.append((label, val))
	return result


def save_checkpoint(args, model, seq=None):
	ckpt_dir = _get_checkpoint_dir(args)
	os.makedirs(ckpt_dir, exist_ok=True)
	if seq is None:
		seq = _get_next_seq(args)
	prefix = _ckpt_prefix(args)
	pth_path = os.path.join(ckpt_dir, f"{prefix}_{seq}.pth")
	torch.save(model.state_dict(), pth_path)
	_append_log(args, seq)
	print(f"Checkpoint saved: {pth_path}")
	return seq


def _append_log(args, seq):
	log_path = _get_log_path(args)
	os.makedirs(os.path.dirname(log_path), exist_ok=True)
	params = _collect_hyperparams(args)
	timestamp = _now_str()
	seq_len = getattr(args, "seq_len", "?")
	pred_len = getattr(args, "pred_len", "?")
	seed = getattr(args, "seed", "?")
	with open(log_path, "a") as f:
		f.write("=" * 60 + "\n")
		f.write(f"  Checkpoint Seq : {seq}\n")
		f.write(f"  Param          : Seq:{seq_len} | Pred:{pred_len} | Seed:{seed}\n")
		f.write(f"  Timestamp      : {timestamp}\n")
		f.write("-" * 60 + "\n")
		for label, val in params:
			f.write(f"  {label:<20s}: {val}\n")
		f.write("=" * 60 + "\n\n")


def load_checkpoint(args, model, seq):
	prefix = _ckpt_prefix(args)
	ckpt_dir = _get_checkpoint_dir(args)
	pth_path = os.path.join(ckpt_dir, f"{prefix}_{seq}.pth")
	if not os.path.isfile(pth_path):
		raise FileNotFoundError(f"Checkpoint not found: {pth_path}")
	state_dict = torch.load(pth_path, map_location="cpu")
	model.load_state_dict(state_dict)
	print(f"Checkpoint loaded: {pth_path}")
	return model


def checkpoint_exists(args, seq):
	prefix = _ckpt_prefix(args)
	ckpt_dir = _get_checkpoint_dir(args)
	pth_path = os.path.join(ckpt_dir, f"{prefix}_{seq}.pth")
	return os.path.isfile(pth_path)


def cleanup_checkpoint(args, seq):
	"""Delete a numbered .pth checkpoint file.  Safe no-op if absent."""
	prefix = _ckpt_prefix(args)
	ckpt_dir = _get_checkpoint_dir(args)
	pth_path = os.path.join(ckpt_dir, f"{prefix}_{seq}.pth")
	if os.path.isfile(pth_path):
		os.remove(pth_path)
		print(f"Low-rank checkpoint cleaned up: {pth_path}")
		return True
	return False


def _now_str():
	from datetime import datetime
	return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
