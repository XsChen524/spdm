import os
import torch
from src.model import Transformer, Informer, Reformer, Flowformer, Flashformer, \
    Autoformer, S_Mamba, PatchTST, DLinear, BiMamba4TS, iTransformer, interPDN, \
    ManiMamba, Crossformer, TimesNet, TiDE


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Flowformer': Flowformer,
            'Flashformer': Flashformer,
            'Autoformer': Autoformer,
            'S_Mamba': S_Mamba,
            'PatchTST': PatchTST,
            'DLinear': DLinear,
            'BiMamba4TS': BiMamba4TS,
            'iTransformer': iTransformer,
            'interPDN': interPDN,
            'ManiMamba': ManiMamba,
            'Crossformer': Crossformer,
            'TimesNet': TimesNet,
            'TiDE': TiDE,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            if "CUDA_VISIBLE_DEVICES" not in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:0')
            print('Use GPU: cuda:0 (CUDA_VISIBLE_DEVICES={})'.format(
                os.environ.get("CUDA_VISIBLE_DEVICES", "all")))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def _get_output_subdir(self, itr=0):
        args = self.args
        model_id = getattr(args, 'model_id', 'default')
        model = getattr(args, 'model', 'unknown')
        des = getattr(args, 'des', 'default')
        e_layers = getattr(args, 'e_layers', 3)

        dir_name = "{}_{}_{}_l{}_itr{}".format(
            model_id,
            model,
            des,
            e_layers,
            itr
        )

        return dir_name

    def _extract_itr_from_setting(self, setting):
        try:
            if "|itr:" in setting:
                return int(setting.split("|itr:")[-1])
        except (ValueError, IndexError):
            pass
        return 0

    def _get_checkpoint_path(self, setting=None):
        itr = self._extract_itr_from_setting(setting) if setting else 0
        subdir = self._get_output_subdir(itr)
        return os.path.join("./temp/checkpoints", subdir)

    def _get_test_results_path(self, setting=None):
        itr = self._extract_itr_from_setting(setting) if setting else 0
        subdir = self._get_output_subdir(itr)
        return os.path.join("./temp/test_results", subdir)

    def _get_results_path(self, setting=None):
        itr = self._extract_itr_from_setting(setting) if setting else 0
        subdir = self._get_output_subdir(itr)
        return os.path.join("./temp/results", subdir)
