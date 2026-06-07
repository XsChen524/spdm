import math
import os
import random
import time
import warnings
from datetime import datetime

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from src.data_provider.data_factory import data_provider
from src.experiments.exp_basic import Exp_Basic
from src.utils.checkpoint import save_checkpoint, load_checkpoint, checkpoint_exists
from src.utils.cuda_accel import (
    cuda_accel_enabled,
    enable_cuda_accel,
    finalise_metrics_gpu,
    TorchStandardScaler,
)
from src.utils.metrics import metric
from src.utils.scheduler import adjust_learning_rate, build_scheduler
from src.utils.tools import EarlyStopping, visual

warnings.filterwarnings("ignore")


def _atomic_append(filepath, text, timeout=30, poll=0.1):
    """Append *text* to *filepath* using an NFS-safe dotlock.

    Uses os.link() to atomically create a lock file — this is one of the few
    operations that is truly atomic on NFS.  Falls back to fcntl on non-NFS
    local filesystems.
    """
    lock_path = filepath + ".lock"
    tmp_path = filepath + ".tmp.{}".format(os.getpid())
    deadline = time.monotonic() + timeout

    while True:
        try:
            fd = os.open(tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, "{}".format(os.getpid()).encode())
            os.close(fd)
        except FileExistsError:
            if time.monotonic() > deadline:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                break
            time.sleep(poll)
            continue

        try:
            os.link(tmp_path, lock_path)
        except OSError:
            pass

        if os.path.exists(tmp_path):
            try:
                nlink = os.stat(tmp_path).st_nlink
            except OSError:
                nlink = 0
            if nlink >= 2:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                break

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        if time.monotonic() > deadline:
            try:
                os.unlink(lock_path)
            except OSError:
                pass
            break
        time.sleep(poll)

    try:
        with open(filepath, "a") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    finally:
        try:
            os.unlink(lock_path)
        except OSError:
            pass


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self._nan_detected = False
        enable_cuda_accel(getattr(args, "use_cuda_accel", 1) == 1)

    def add_gaussian_noise(self, batch_x, noise_level=0.0):
        """
        Add Gaussian noise to input data for robustness testing.

        Args:
            batch_x: Input tensor of shape (batch, seq_len, n_vars)
            noise_level: Standard deviation of Gaussian noise as a fraction of data range (0-1)
                        For example, 0.1 means noise with std=10% of data range

        Returns:
            Perturbed tensor with added noise
        """
        if noise_level <= 0:
            return batch_x

        # Calculate data range (max - min) for each batch
        data_min = batch_x.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        data_max = batch_x.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        data_range = data_max - data_min

        # Scale noise level by data range
        # noise_std = noise_level * data_range
        # For simplicity and stability, we use noise_level directly as std
        # assuming the data is normalized to have reasonable scale
        noise_std = noise_level

        # Generate Gaussian noise
        noise = torch.randn_like(batch_x) * noise_std

        # Add noise to entire data
        perturbed = batch_x + noise

        return perturbed

    def _build_model(self):
        model_class = self.model_dict[self.args.model]
        model = model_class(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        base_lr = self.args.learning_rate
        wd = getattr(self.args, "weight_decay", 1e-5)
        optim_type = getattr(self.args, "optim", "AdamW")

        if getattr(self.args, "use_8bit", 0) == 1:
            import bitsandbytes as bnb

            optim_cls = (
                bnb.optim.AdamW8bit if optim_type == "AdamW" else bnb.optim.Adam8bit
            )
        else:
            optim_cls = optim.AdamW if optim_type == "AdamW" else optim.Adam
        model_optim = optim_cls(self.model.parameters(), lr=base_lr, weight_decay=wd)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _add_auxiliary_loss(self, loss):
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_ref, "auxiliary_losses") and model_ref.auxiliary_losses:
            for w, aux in model_ref.auxiliary_losses:
                loss = loss + w * aux
        return loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if "PEMS" in self.args.data or "Solar" in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )
                            # Handle models that return tuple (e.g., PatchTST)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                        # Handle models that return tuple (e.g., PatchTST)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                if cuda_accel_enabled():
                    loss = criterion(outputs.detach(), batch_y.detach())
                else:
                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                    if self.args.model == "interPDN":
                        ratio = torch.tensor(
                            [
                                -math.atan(i + 1) + math.pi / 4 + 1
                                for i in range(self.args.pred_len)
                            ]
                        )
                        ratio = ratio.unsqueeze(-1).to(self.device)
                        loss = F.l1_loss(outputs * ratio, batch_y * ratio)
                        model_ref = (
                            self.model.module
                            if hasattr(self.model, "module")
                            else self.model
                        )
                        con_losses = model_ref.consistency_losses
                        loss = (
                            loss.item()
                            + self.args.con_cls_1 * con_losses[0].item()
                            + self.args.con_cls_2 * con_losses[1].item()
                            + self.args.con_time * con_losses[2].item()
                        )
                    else:
                        loss = criterion(pred, true)

                total_loss.append(loss)
        if cuda_accel_enabled():
            total_loss = torch.stack(total_loss).mean().item()
        else:
            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        save_tmp = getattr(self.args, "save_tmp", 1)

        from src.utils.checkpoint import (
            _get_checkpoint_dir,
            _get_dataset_name,
            _get_next_seq,
            _ckpt_prefix,
        )

        if save_tmp:
            ckpt_dir = _get_checkpoint_dir(self.args)
            os.makedirs(ckpt_dir, exist_ok=True)
            seq = _get_next_seq(self.args)
            prefix = _ckpt_prefix(self.args)
            best_model_path = os.path.join(ckpt_dir, f"{prefix}_{seq}.pth")
        else:
            seq = 0
            prefix = ""
            best_model_path = None

        if getattr(self.args, "checkpoint", 0) > 0:
            print(
                f"Loading checkpoint {_ckpt_prefix(self.args)}_{self.args.checkpoint}.pth for continued training"
            )
            load_checkpoint(self.args, self.model, self.args.checkpoint)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, save_path=best_model_path
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = build_scheduler(model_optim, self.args, train_steps=train_steps)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            print(f"Epoch {epoch + 1} : LR {model_optim.param_groups[0]['lr']:.8e}")
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                train_noise = getattr(self.args, "train_noise_level", 0.0)
                if train_noise > 0:
                    batch_x = self.add_gaussian_noise(batch_x, noise_level=train_noise)
                if "PEMS" in self.args.data or "Solar" in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]

                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                            self.device
                        )

                        if self.args.model == "interPDN":
                            ratio = torch.tensor(
                                [
                                    -math.atan(i + 1) + math.pi / 4 + 1
                                    for i in range(self.args.pred_len)
                                ]
                            )
                            ratio = ratio.unsqueeze(-1).to(self.device)
                            loss = F.l1_loss(outputs * ratio, batch_y * ratio)
                            model_ref = (
                                self.model.module
                                if hasattr(self.model, "module")
                                else self.model
                            )
                            con_losses = model_ref.consistency_losses
                            loss += (
                                self.args.con_cls_1 * con_losses[0]
                                + self.args.con_cls_2 * con_losses[1]
                                + self.args.con_time * con_losses[2]
                            )
                        else:
                            loss = criterion(outputs, batch_y)
                        loss = self._add_auxiliary_loss(loss)
                        if torch.isnan(loss) or torch.isinf(loss):
                            self._nan_detected = True
                            continue
                        train_loss.append(
                            loss.detach() if cuda_accel_enabled() else loss.item()
                        )
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]

                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                    if self.args.model == "interPDN":
                        ratio = torch.tensor(
                            [
                                -math.atan(i + 1) + math.pi / 4 + 1
                                for i in range(self.args.pred_len)
                            ]
                        )
                        ratio = ratio.unsqueeze(-1).to(self.device)
                        loss = F.l1_loss(outputs * ratio, batch_y * ratio)
                        model_ref = (
                            self.model.module
                            if hasattr(self.model, "module")
                            else self.model
                        )
                        con_losses = model_ref.consistency_losses
                        loss += (
                            self.args.con_cls_1 * con_losses[0]
                            + self.args.con_cls_2 * con_losses[1]
                            + self.args.con_time * con_losses[2]
                        )
                    else:
                        loss = criterion(outputs, batch_y)
                    loss = self._add_auxiliary_loss(loss)
                    if torch.isnan(loss) or torch.isinf(loss):
                        self._nan_detected = True
                        continue
                    train_loss.append(
                        loss.detach() if cuda_accel_enabled() else loss.item()
                    )

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    if self.args.use_gpu and torch.cuda.is_available():
                        allocated_memory = torch.cuda.max_memory_allocated() / (
                            1024 * 1024 * 1024
                        )
                        print("allocated_memory:", allocated_memory)
                        torch.cuda.reset_peak_memory_stats()
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    max_grad_norm = getattr(self.args, "max_grad_norm", 0.0)
                    if max_grad_norm > 0:
                        scaler.unscale_(model_optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    max_grad_norm = getattr(self.args, "max_grad_norm", 0.0)
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )
                    model_optim.step()
                if self.args.lradj == "onecycle":
                    scheduler.step()

            if scheduler is not None and self.args.lradj != "onecycle":
                scheduler.step()
            if self.args.lradj in ("type1", "type2", "sigmoid"):
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if cuda_accel_enabled():
                train_loss = torch.stack(train_loss).mean().item()
            else:
                train_loss = np.average(train_loss)
            if self._nan_detected:
                print(
                    f"NaN/Inf detected during epoch {epoch + 1}, treating as worst validation loss"
                )
                self._nan_detected = False
                vali_loss = float("inf")
            else:
                vali_loss = self.vali(vali_data, vali_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss
                )
            )
            early_stopping(vali_loss, self.model, best_model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        if best_model_path is not None and os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            save_checkpoint(self.args, self.model, seq=seq)
            self._last_checkpoint_seq = seq
        else:
            self._last_checkpoint_seq = None

        self._lr_checkpoint_to_cleanup = (
            seq
            if getattr(self.args, "low_rank", 0) > 0
            and self._last_checkpoint_seq is not None
            else None
        )

        return self.model

    def test(self, setting, test=0):
        save_tmp = getattr(self.args, "save_tmp", 1)
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            if getattr(self.args, "checkpoint", 0) > 0:
                load_checkpoint(self.args, self.model, self.args.checkpoint)
            else:
                checkpoint_path = self._get_checkpoint_path(setting)
                self.model.load_state_dict(
                    torch.load(os.path.join(checkpoint_path, "checkpoint.pth"))
                )

        preds = []
        trues = []
        _gpu_preds = []
        _gpu_trues = []
        _gpu_scaler = None
        if cuda_accel_enabled() and test_data.scale and self.args.inverse:
            _gpu_scaler = TorchStandardScaler.from_dataset(
                test_data, device=self.device
            )
        folder_path = None
        if save_tmp:
            folder_path = self._get_test_results_path(setting)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        self.model.eval()

        explain_mode = getattr(self.args, "explain", False)
        _explain_collected = False

        if explain_mode:
            model_ref = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            if hasattr(model_ref, "set_capture"):
                model_ref.set_capture(True)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # Apply Gaussian noise for robustness testing
                if self.args.noise_level > 0:
                    batch_x = self.add_gaussian_noise(
                        batch_x, noise_level=self.args.noise_level
                    )
                if "PEMS" in self.args.data or "Solar" in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )
                            # Handle models that return tuples
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]

                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                        # Handle models that return tuples
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]

                if explain_mode and not _explain_collected:
                    self._save_geometry_intermediates(setting)
                    model_ref = (
                        self.model.module if hasattr(self.model, "module") else self.model
                    )
                    if hasattr(model_ref, "set_capture"):
                        model_ref.set_capture(False)
                    _explain_collected = True

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                if cuda_accel_enabled():
                    if _gpu_scaler is not None:
                        shape = outputs.shape
                        outputs = _gpu_scaler.inverse_transform(
                            outputs.float().reshape(-1, outputs.shape[-1])
                        ).reshape(shape)
                        batch_y = _gpu_scaler.inverse_transform(
                            batch_y.float().reshape(-1, batch_y.shape[-1])
                        ).reshape(shape)
                    _gpu_preds.append(outputs.detach().float())
                    _gpu_trues.append(batch_y.detach().float())
                else:
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(
                            outputs.reshape(-1, shape[-1])
                        ).reshape(shape)
                        batch_y = test_data.inverse_transform(
                            batch_y.reshape(-1, shape[-1])
                        ).reshape(shape)

                    pred = outputs
                    true = batch_y

                    preds.append(pred)
                    trues.append(true)

            if i % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                if test_data.scale and self.args.inverse and not cuda_accel_enabled():
                    shape = input.shape
                    input = test_data.inverse_transform(input.reshape(-1, shape[-1])).reshape(shape)
                elif test_data.scale and self.args.inverse and _gpu_scaler is not None:
                    shape = input.shape
                    input_t = _gpu_scaler.inverse_transform(
                        batch_x.float().reshape(-1, batch_x.shape[-1])
                    ).reshape(shape)
                    input = input_t.cpu().numpy()
                if cuda_accel_enabled():
                    gt_np = np.concatenate(
                        (input[0, :, -1], _gpu_trues[-1][-1, :, -1].cpu().numpy()),
                        axis=0,
                    )
                    pd_np = np.concatenate(
                        (input[0, :, -1], _gpu_preds[-1][-1, :, -1].cpu().numpy()),
                        axis=0,
                    )
                else:
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    gt_np, pd_np = gt, pd
                if folder_path is not None:
                    visual(gt_np, pd_np, os.path.join(folder_path, str(i) + ".pdf"))

        if cuda_accel_enabled():
            (mae, mse, rmse, mape, mspe), preds, trues = finalise_metrics_gpu(
                _gpu_preds, _gpu_trues
            )
            print("test shape:", preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print("test shape:", preds.shape, trues.shape)
        else:
            preds = np.array(preds)
            trues = np.array(trues)
            print("test shape:", preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print("test shape:", preds.shape, trues.shape)

            mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))

        if save_tmp:
            folder_path = self._get_results_path(setting)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        # Create output directory based on dataset name
        # Check if this is a case study experiment
        if os.environ.get("CASE_STUDY_MODE"):
            output_dir = "./output/case_study"
        # Check if this is a lookback experiment
        elif os.environ.get("LOOKBACK_OUTPUT_DIR"):
            output_dir = os.environ.get("LOOKBACK_OUTPUT_DIR")
        # Check if this is an ablation experiment
        elif os.environ.get("ABLATION_OUTPUT_DIR"):
            output_dir = os.environ.get("ABLATION_OUTPUT_DIR")
        # Check if this is a robustness experiment
        elif os.environ.get("ROBUSTNESS_OUTPUT_DIR"):
            output_dir = os.environ.get("ROBUSTNESS_OUTPUT_DIR")
        else:
            output_dir = "./output/long_term_forecast"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Determine dataset name for output file
        dataset_name = self.args.data

        # For custom datasets, extract the dataset name from model_id
        if self.args.data == "custom":
            # Extract dataset name from model_id (e.g., "ECL_96_96" -> "ECL")
            model_id_parts = self.args.model_id.split("_")
            # Take the part before the first underscore(s) that looks like a dataset name
            # This handles cases like "weather_96_96", "Exchange_96_96", "Traffic_96_96", "ECL_96_96"
            if (
                len(model_id_parts) >= 3
                and model_id_parts[-2].isdigit()
                and model_id_parts[-1].isdigit()
            ):
                # Pattern: dataset_predLen_horizon (e.g., "weather_96_192")
                dataset_name = "_".join(model_id_parts[:-2])
            elif len(model_id_parts) >= 2 and model_id_parts[-1].isdigit():
                # Pattern: dataset_length (e.g., "weather_96")
                dataset_name = "_".join(model_id_parts[:-1])
            else:
                # Unknown pattern, use model_id as fallback
                dataset_name = self.args.model_id.split("_")[0]

        # For case studies and lookback experiments, output file is named after the model
        # For robustness experiments, output file is named after the dataset with _robustness suffix
        # For regular experiments, output file is named after the dataset
        if os.environ.get("CASE_STUDY_MODE"):
            output_file = os.path.join(output_dir, "{}.txt".format(self.args.model))
        elif os.environ.get("LOOKBACK_OUTPUT_DIR"):
            output_file = os.path.join(output_dir, "{}.txt".format(self.args.model))
        elif os.environ.get("ROBUSTNESS_OUTPUT_DIR"):
            output_file = os.path.join(
                output_dir, "{}_robustness.txt".format(dataset_name)
            )
        else:
            output_file = os.path.join(output_dir, "{}.txt".format(dataset_name))

        check_seq = getattr(self, "_last_checkpoint_seq", None)
        if check_seq is None:
            check_seq = getattr(self.args, "checkpoint", 0)
        result_setting = setting + "|checkpoint:{}".format(check_seq)
        result_lines = result_setting + "  \n" + "mse:{}, mae:{}".format(mse, mae) + "\n\n"
        try:
            _atomic_append(output_file, result_lines)
            print(f"Results successfully written to: {output_file}")
        except Exception as e:
            print(f"ERROR: Failed to write results to {output_file}: {str(e)}")

        file_prefix = f"{self.args.model}_{self.args.seq_len}_{self.args.pred_len}"
        if save_tmp:
            np.save(
                os.path.join(folder_path, f"{file_prefix}_metrics.npy"),
                np.array([mae, mse, rmse, mape, mspe]),
            )
            np.save(os.path.join(folder_path, f"{file_prefix}_pred.npy"), preds)
            np.save(os.path.join(folder_path, f"{file_prefix}_true.npy"), trues)

        self._last_test_result = {"mse": float(mse), "mae": float(mae)}

        return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe}

    def _save_geometry_intermediates(self, setting):
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        if not hasattr(model_ref, "spd_proj"):
            print("[explain] Model has no spd_proj, skipping geometry capture")
            return

        subdir = self._get_output_subdir()
        out_dir = os.path.join("./output/exp_plot", subdir)
        os.makedirs(out_dir, exist_ok=True)

        spd = model_ref.spd_proj._captured
        for key, arr in spd.items():
            np.save(os.path.join(out_dir, f"spd_{key}.npy"), arr)

        for li, layer in enumerate(model_ref.encoder_layers):
            for direction, mod in [("fwd", layer.mamba_fwd), ("rev", layer.mamba_rev)]:
                cap = mod._captured
                for key, arr in cap.items():
                    np.save(
                        os.path.join(out_dir, f"layer{li}_{direction}_{key}.npy"), arr
                    )

        meta = {
            "setting": subdir,
            "dataset": self.args.data,
            "model": self.args.model,
            "seq_len": self.args.seq_len,
            "pred_len": self.args.pred_len,
            "enc_in": self.args.enc_in,
            "d_model": self.args.d_model,
            "d_state": getattr(self.args, "d_state", 16),
            "e_layers": self.args.e_layers,
            "cov_window": getattr(self.args, "cov_window", 16),
            "cov_stride": getattr(self.args, "cov_stride", 8),
            "d_inner": int(getattr(self.args, "expand", 1) * self.args.d_model),
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

        print(f"[explain] Geometry intermediates saved to {out_dir}")

    def get_input(self, setting):
        test_data, test_loader = self._get_data(flag="test")
        inputs = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            input = batch_x.detach().cpu().numpy()
            inputs.append((input))
        folder_path = self._get_results_path(setting)
        np.save(folder_path + "input.npy", inputs)

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag="pred")

        if load:
            if getattr(self.args, "checkpoint", 0) > 0:
                load_checkpoint(self.args, self.model, self.args.checkpoint)
            else:
                checkpoint_path = self._get_checkpoint_path(setting)
                best_model_path = os.path.join(checkpoint_path, "checkpoint.pth")
                self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                pred_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )
                            # Handle models that return tuple (e.g., PatchTST)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]

                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                        # Handle models that return tuple (e.g., PatchTST)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]

                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(
                        shape
                    )
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = self._get_results_path(setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + "real_prediction.npy", preds)

        return
