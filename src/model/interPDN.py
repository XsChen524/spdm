import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from src.layers.decomp import DECOMP
from src.layers.network import Network
from src.layers.revin_interpdn import RevIN


class Model(nn.Module):
	def __init__(self, configs):
		super(Model, self).__init__()

		seq_len = configs.seq_len
		pred_len = configs.pred_len
		c_in = configs.enc_in

		patch_len = configs.patch_len
		stride = configs.stride
		padding_patch = configs.padding_patch

		self.revin = getattr(configs, "revin", 1)
		self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

		self.ma_type = configs.ma_type
		alpha = configs.alpha
		beta = configs.beta

		self.decomp = DECOMP(self.ma_type, alpha, beta)
		scale_1 = "fine"
		scale_2 = "grain"
		self.net_1 = Network(seq_len, pred_len, patch_len, stride, padding_patch, scale_1)
		self.net_2 = Network(seq_len, pred_len, patch_len, stride, padding_patch, scale_1)
		self.net_3 = Network(seq_len, pred_len, patch_len, stride, padding_patch, scale_2)
		self.net_4 = Network(seq_len, pred_len, patch_len, stride, padding_patch, scale_2)

		_xlsx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets", "idx2.xlsx")
		df = pd.read_excel(_xlsx_path, usecols=[1, 2], header=None, skiprows=1, nrows=25)
		self.register_buffer("idx_1", torch.tensor(df.iloc[:, 0].values, dtype=torch.float32).view(25, 1))
		self.register_buffer("idx_2", torch.tensor(df.iloc[:, 1].values, dtype=torch.float32).view(25, 1))

		self.consistency_losses = None

	def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
		x = x_enc

		if self.revin:
			x = self.revin_layer(x, "norm")

		if self.ma_type == "reg":
			x_1 = self.net_1(x, x)
			x_2 = self.net_2(x, x)
			x_3 = self.net_3(x, x)
			x_4 = self.net_4(x, x)
		else:
			seasonal_init, trend_init = self.decomp(x)
			x_1 = self.net_1(seasonal_init, trend_init)
			x_2 = self.net_2(seasonal_init, trend_init)
			x_3 = self.net_3(seasonal_init, trend_init)
			x_4 = self.net_4(seasonal_init, trend_init)

		x_1 = x_1.reshape(x_1.shape[0], x_1.shape[1] // 25, 25, x_1.shape[2])
		x_cls_1 = torch.softmax(x_1, dim=2)
		output_1 = torch.einsum("blec,ei->blic", x_cls_1, self.idx_1).squeeze(2)

		x_2 = x_2.reshape(x_2.shape[0], x_2.shape[1] // 25, 25, x_2.shape[2])
		x_cls_2 = torch.softmax(x_2, dim=2)
		output_2 = torch.einsum("blec,ei->blic", x_cls_2, self.idx_2).squeeze(2)

		entro_1 = x_cls_1.permute(0, 3, 1, 2)
		entro_1 = torch.max(entro_1, dim=-1)[0]
		entro_2 = x_cls_2.permute(0, 3, 1, 2)
		entro_2 = torch.max(entro_2, dim=-1)[0]
		mask = entro_1 / (entro_1 + entro_2)
		mask = mask.permute(0, 2, 1)
		output_fine = (output_1 * mask + output_2 * (1 - mask))

		output_down = output_fine.reshape(x_2.shape[0], x_2.shape[1] // 4, 4, x_2.shape[3])
		output_down_fine = output_down.mean(dim=2)

		x_3 = x_3.reshape(x_3.shape[0], x_3.shape[1] // 25, 25, x_3.shape[2])
		x_cls_3 = torch.softmax(x_3, dim=2)
		output_3 = torch.einsum("blec,ei->blic", x_cls_3, self.idx_1).squeeze(2)

		x_4 = x_4.reshape(x_4.shape[0], x_4.shape[1] // 25, 25, x_4.shape[2])
		x_cls_4 = torch.softmax(x_4, dim=2)
		output_4 = torch.einsum("blec,ei->blic", x_cls_4, self.idx_2).squeeze(2)

		entro_3 = x_cls_3.permute(0, 3, 1, 2)
		entro_3 = torch.max(entro_3, dim=-1)[0]
		entro_4 = x_cls_4.permute(0, 3, 1, 2)
		entro_4 = torch.max(entro_4, dim=-1)[0]
		mask = entro_3 / (entro_3 + entro_4)
		mask = mask.permute(0, 2, 1)
		output_grain = (output_3 * mask + output_4 * (1 - mask))

		con_loss_time = F.mse_loss(output_grain, output_down_fine)
		con_loss_cls_1 = F.mse_loss(output_2, output_1)
		cor_loss_cls_2 = F.mse_loss(output_4, output_3)

		self.consistency_losses = (con_loss_cls_1, cor_loss_cls_2, con_loss_time)

		if self.revin:
			output_fine = self.revin_layer(output_fine, "denorm")

		return output_fine, None
