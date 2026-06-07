import os
import time
import warnings
import random

import numpy as np
import torch
import torch.nn as nn
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
from src.utils.scheduler import adjust_learning_rate
from src.utils.tools import EarlyStopping, visual
from src.utils.metrics import metric

warnings.filterwarnings("ignore")


# train on partial variate data and test on the full variates, used for two types of experiments:
# (1) Generalize on unseen variate (Figure 5 of our paper)
# (2) Efficient training strategy  (Figure 8 of our paper)
class Exp_Long_Term_Forecast_Partial(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Partial, self).__init__(args)
        enable_cuda_accel(getattr(args, "use_cuda_accel", 1) == 1)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        optim_type = getattr(self.args, "optim", "AdamW")
        wd = getattr(self.args, "weight_decay", 1e-4)

        if getattr(self.args, "use_8bit", 0) == 1:
            import bitsandbytes as bnb
            optim_cls = (
                bnb.optim.AdamW8bit if optim_type == "AdamW" else bnb.optim.Adam8bit
            )
        else:
            optim_cls = optim.AdamW if optim_type == "AdamW" else optim.Adam

        model_optim = optim_cls(self.model.parameters(), lr=self.args.learning_rate, weight_decay=wd)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, partial_train=False):
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

                if (
                    partial_train
                ):  # we train models with only partial variates from the dataset
                    partial_start = self.args.partial_start_index
                    partial_end = min(
                        self.args.enc_in + partial_start, batch_x.shape[-1]
                    )
                    batch_x = batch_x[:, :, partial_start:partial_end]
                    batch_y = batch_y[:, :, partial_start:partial_end]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
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
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]
                    elif self.args.channel_independence:
                        B, Tx, N = batch_x.shape
                        _, Ty, _ = dec_inp.shape
                        if batch_x_mark == None:
                            outputs = (
                                self.model(
                                    batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1),
                                    batch_x_mark,
                                    dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1),
                                    batch_y_mark,
                                )
                                .reshape(B, N, -1)
                                .permute(0, 2, 1)
                            )
                        else:
                            outputs = (
                                self.model(
                                    batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1),
                                    batch_x_mark.repeat(N, 1, 1),
                                    dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1),
                                    batch_y_mark.repeat(N, 1, 1),
                                )
                                .reshape(B, N, -1)
                                .permute(0, 2, 1)
                            )
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if cuda_accel_enabled():
                    loss = criterion(outputs.detach(), batch_y.detach())
                else:
                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()
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
        test_data, test_loader = self._get_data(flag="test")

        from src.utils.checkpoint import _get_checkpoint_dir, _get_dataset_name, _get_next_seq, _ckpt_prefix
        ckpt_dir = _get_checkpoint_dir(self.args)
        os.makedirs(ckpt_dir, exist_ok=True)
        seq = _get_next_seq(self.args)
        prefix = _ckpt_prefix(self.args)
        best_model_path = os.path.join(ckpt_dir, f"{prefix}_{seq}.pth")

        if getattr(self.args, "checkpoint", 0) > 0:
            print(f"Loading checkpoint {prefix}_{self.args.checkpoint}.pth for continued training")
            load_checkpoint(self.args, self.model, self.args.checkpoint)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, save_path=best_model_path
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            print(f"【Epoch {epoch + 1} : LR {model_optim.param_groups[0]['lr']:.8e}】")
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                if "PEMS" in self.args.data or "Solar" in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                partial_start = self.args.partial_start_index
                partial_end = min(self.args.enc_in + partial_start, batch_x.shape[-1])
                batch_x = batch_x[:, :, partial_start:partial_end]
                batch_y = batch_y[:, :, partial_start:partial_end]
                if self.args.efficient_training:
                    _, _, N = batch_x.shape
                    index = np.stack(random.sample(range(N), N))[-self.args.enc_in:]
                    batch_x = batch_x[:, :, index]
                    batch_y = batch_y[:, :, index]

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

                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                            self.device
                        )
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.detach() if cuda_accel_enabled() else loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]
                    elif self.args.channel_independence:
                        B, Tx, N = batch_x.shape
                        _, Ty, _ = dec_inp.shape
                        if batch_x_mark == None:
                            outputs = (
                                self.model(
                                    batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1),
                                    batch_x_mark,
                                    dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1),
                                    batch_y_mark,
                                )
                                .reshape(B, N, -1)
                                .permute(0, 2, 1)
                            )
                        else:
                            a = batch_x.permute(0, 2, 1)
                            b = batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1)
                            outputs = (
                                self.model(
                                    batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1),
                                    batch_x_mark.repeat(N, 1, 1),
                                    dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1),
                                    batch_y_mark.repeat(N, 1, 1),
                                )
                                .reshape(B, N, -1)
                                .permute(0, 2, 1)
                            )
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )

                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.detach() if cuda_accel_enabled() else loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
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
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if cuda_accel_enabled():
                train_loss = torch.stack(train_loss).mean().item()
            else:
                train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, partial_train=True)
            test_loss = self.vali(
                test_data, test_loader, criterion, partial_train=False
            )

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, best_model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(best_model_path))
        save_checkpoint(self.args, self.model, seq=seq)
        self._last_checkpoint_seq = seq

        return self.model

    def test(self, setting, test=0):

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
            _gpu_scaler = TorchStandardScaler.from_dataset(test_data, device=self.device)
        folder_path = self._get_test_results_path(setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                # During model inference, test the obtained model directly on all variates.
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

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
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]
                    elif (
                        self.args.channel_independence
                    ):  # compare the result with channel_independence
                        B, Tx, N = batch_x.shape
                        _, Ty, _ = dec_inp.shape
                        if batch_x_mark == None:
                            outputs = (
                                self.model(
                                    batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1),
                                    batch_x_mark,
                                    dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1),
                                    batch_y_mark,
                                )
                                .reshape(B, N, -1)
                                .permute(0, 2, 1)
                            )
                        else:
                            outputs = (
                                self.model(
                                    batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1),
                                    batch_x_mark.repeat(N, 1, 1),
                                    dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1),
                                    batch_y_mark.repeat(N, 1, 1),
                                )
                                .reshape(B, N, -1)
                                .permute(0, 2, 1)
                            )
                    else:
                        # directly test the trained model on all variates without fine-tuning.
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

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
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(
                            shape
                        )
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(
                            shape
                        )

                    pred = outputs
                    true = batch_y

                    preds.append(pred)
                    trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse and not cuda_accel_enabled():
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(
                            shape
                        )
                    elif test_data.scale and self.args.inverse and _gpu_scaler is not None:
                        shape = input.shape
                        input_t = _gpu_scaler.inverse_transform(
                            batch_x.float().reshape(-1, batch_x.shape[-1])
                        ).reshape(shape)
                        input = input_t.cpu().numpy()
                    if cuda_accel_enabled():
                        gt_np = np.concatenate((input[0, :, -1], _gpu_trues[-1][-1, :, -1].cpu().numpy()), axis=0)
                        pd_np = np.concatenate((input[0, :, -1], _gpu_preds[-1][-1, :, -1].cpu().numpy()), axis=0)
                    else:
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        gt_np, pd_np = gt, pd
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

        # result save
        folder_path = self._get_results_path(setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Create output directory based on dataset name
        output_dir = "./output/long_term_forecast"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, "{}.txt".format(self.args.data))
        check_seq = getattr(self, "_last_checkpoint_seq", None)
        if check_seq is None:
            check_seq = getattr(self.args, "checkpoint", 0)
        result_setting = setting + "|checkpoint:{}".format(check_seq)
        f = open(output_file, "a")
        f.write(result_setting + "  \n")
        f.write("mse:{}, mae:{}".format(mse, mae))
        f.write("\n")
        f.write("\n")
        f.close()

        file_prefix = f"{self.args.model}_{self.args.seq_len}_{self.args.pred_len}"
        np.save(os.path.join(folder_path, f"{file_prefix}_metrics.npy"), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder_path, f"{file_prefix}_pred.npy"), preds)
        np.save(os.path.join(folder_path, f"{file_prefix}_true.npy"), trues)

        return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "mspe": mspe}

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
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
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
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
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
