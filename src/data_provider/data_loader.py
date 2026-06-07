import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from src.utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


def _resolve_pt_path(root_path, data_path):
    base, _ = os.path.splitext(data_path)
    pt_path = os.path.join(root_path, base + '.pt')
    return pt_path if os.path.exists(pt_path) else None


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        pt_path = _resolve_pt_path(self.root_path, self.data_path)
        if pt_path:
            cached = torch.load(pt_path, weights_only=False)
            all_values = cached['values']
            all_dates = cached['dates']
            numeric_cols = [c for c in cached['columns'] if c != 'date']
        else:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            all_values = df_raw[df_raw.columns[1:]].values
            all_dates = pd.to_datetime(df_raw['date']).values
            numeric_cols = list(df_raw.columns[1:])

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            data = all_values
        elif self.features == 'S':
            idx = numeric_cols.index(self.target)
            data = all_values[:, idx:idx + 1]

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        stamp_dates = all_dates[border1:border2]
        if self.timeenc == 0:
            dt = pd.DatetimeIndex(stamp_dates)
            data_stamp = np.stack([dt.month, dt.day, dt.weekday, dt.hour], axis=1).astype(np.float64)
        elif self.timeenc == 1:
            data_stamp = time_features(pd.DatetimeIndex(stamp_dates), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = torch.from_numpy(data[border1:border2]).float()
        self.data_y = torch.from_numpy(data[border1:border2]).float()
        self.data_stamp = torch.from_numpy(data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        pt_path = _resolve_pt_path(self.root_path, self.data_path)
        if pt_path:
            cached = torch.load(pt_path, weights_only=False)
            all_values = cached['values']
            all_dates = cached['dates']
            numeric_cols = [c for c in cached['columns'] if c != 'date']
        else:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            all_values = df_raw[df_raw.columns[1:]].values
            all_dates = pd.to_datetime(df_raw['date']).values
            numeric_cols = list(df_raw.columns[1:])

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            data = all_values
        elif self.features == 'S':
            idx = numeric_cols.index(self.target)
            data = all_values[:, idx:idx + 1]

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        stamp_dates = all_dates[border1:border2]
        if self.timeenc == 0:
            dt = pd.DatetimeIndex(stamp_dates)
            data_stamp = np.stack([
                dt.month, dt.day, dt.weekday, dt.hour,
                dt.minute // 15,
            ], axis=1).astype(np.float64)
        elif self.timeenc == 1:
            data_stamp = time_features(pd.DatetimeIndex(stamp_dates), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = torch.from_numpy(data[border1:border2]).float()
        self.data_y = torch.from_numpy(data[border1:border2]).float()
        self.data_stamp = torch.from_numpy(data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        pt_path = _resolve_pt_path(self.root_path, self.data_path)
        if pt_path:
            cached = torch.load(pt_path, weights_only=False)
            all_values = cached['values']
            all_dates = cached['dates']
            numeric_cols = [c for c in cached['columns'] if c != 'date']
        else:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            all_values = df_raw[df_raw.columns[1:]].values
            all_dates = pd.to_datetime(df_raw['date']).values
            numeric_cols = list(df_raw.columns[1:])

        reordered = [c for c in numeric_cols if c != self.target] + [self.target]
        reorder_idx = [numeric_cols.index(c) for c in reordered]
        all_values = all_values[:, reorder_idx]
        numeric_cols = reordered

        n_total = all_values.shape[0]
        num_train = int(n_total * 0.7)
        num_test = int(n_total * 0.2)
        num_vali = n_total - num_train - num_test
        border1s = [0, num_train - self.seq_len, n_total - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, n_total]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            data = all_values
        elif self.features == 'S':
            data = all_values[:, -1:]

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        stamp_dates = all_dates[border1:border2]
        if self.timeenc == 0:
            dt = pd.DatetimeIndex(stamp_dates)
            data_stamp = np.stack([dt.month, dt.day, dt.weekday, dt.hour], axis=1).astype(np.float64)
        elif self.timeenc == 1:
            data_stamp = time_features(pd.DatetimeIndex(stamp_dates), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = torch.from_numpy(data[border1:border2]).float()
        self.data_y = torch.from_numpy(data[border1:border2]).float()
        self.data_stamp = torch.from_numpy(data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        pt_path = _resolve_pt_path(self.root_path, self.data_path)
        if pt_path:
            cached = torch.load(pt_path, weights_only=False)
            data = cached['values']
        else:
            data = np.load(os.path.join(self.root_path, self.data_path), allow_pickle=True)
            data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.ffill().bfill().values

        self.data_x = torch.from_numpy(df).float()
        self.data_y = torch.from_numpy(df).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros(seq_x.shape[0], 1)
        seq_y_mark = torch.zeros(seq_x.shape[0], 1)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        pt_path = _resolve_pt_path(self.root_path, self.data_path)
        if pt_path:
            cached = torch.load(pt_path, weights_only=False)
            raw_values = cached['values']
        else:
            df_raw = []
            with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            raw_values = np.stack(df_raw, 0)

        n_total = raw_values.shape[0]
        num_train = int(n_total * 0.7)
        num_test = int(n_total * 0.2)
        num_valid = int(n_total * 0.1)
        border1s = [0, num_train - self.seq_len, n_total - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, n_total]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = raw_values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = torch.from_numpy(data[border1:border2]).float()
        self.data_y = torch.from_numpy(data[border1:border2]).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros(seq_x.shape[0], 1)
        seq_y_mark = torch.zeros(seq_x.shape[0], 1)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        pt_path = _resolve_pt_path(self.root_path, self.data_path)
        if pt_path:
            cached = torch.load(pt_path, weights_only=False)
            all_values = cached['values']
            all_dates = cached['dates']
            all_columns = cached['columns']
            numeric_cols = [c for c in all_columns if c != 'date']
        else:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            all_values = df_raw[df_raw.columns[1:]].values
            all_dates = pd.to_datetime(df_raw['date']).values
            numeric_cols = list(df_raw.columns[1:])

        if self.cols:
            ordered_cols = [c for c in self.cols if c != 'date']
            if self.target not in ordered_cols:
                ordered_cols = ordered_cols + [self.target]
        else:
            ordered_cols = [c for c in numeric_cols if c != self.target] + [self.target]
        reorder_idx = [numeric_cols.index(c) for c in ordered_cols]
        all_values = all_values[:, reorder_idx]

        n_total = all_values.shape[0]
        border1 = n_total - self.seq_len
        border2 = n_total

        if self.features == 'M' or self.features == 'MS':
            df_data = all_values
        elif self.features == 'S':
            target_idx = ordered_cols.index(self.target)
            df_data = all_values[:, target_idx:target_idx + 1]

        if self.scale:
            self.scaler.fit(df_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        last_date = all_dates[-1]
        pred_dates = pd.date_range(last_date, periods=self.pred_len + 1, freq=self.freq)
        stamp_dates = np.concatenate([all_dates[border1:border2], pred_dates[1:].values])

        if self.timeenc == 0:
            dt = pd.DatetimeIndex(stamp_dates)
            data_stamp = np.stack([
                dt.month, dt.day, dt.weekday, dt.hour,
                dt.minute // 15,
            ], axis=1).astype(np.float64)
        elif self.timeenc == 1:
            data_stamp = time_features(pd.DatetimeIndex(stamp_dates), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = torch.from_numpy(data[border1:border2]).float()
        if self.inverse:
            self.data_y = torch.from_numpy(df_data[border1:border2]).float()
        else:
            self.data_y = torch.from_numpy(data[border1:border2]).float()
        self.data_stamp = torch.from_numpy(data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
