"""
Model parameter parser dictionary.
Maps model names to their specific parameter parser functions.
"""

from configs.common import add_common_args
from configs.custom_S_Mamba import add_s_mamba_parser
from configs.custom_PatchTST import add_patchtst_parser
from configs.custom_DLinear import add_dlinear_parser
from configs.custom_BiMamba4TS import add_bimamba4ts_parser
from configs.custom_interPDN import add_interpdn_parser
from configs.custom_ManiMamba import add_manimamba_parser
from configs.custom_Crossformer import add_crossformer_parser
from configs.custom_TimesNet import add_timesnet_parser
from configs.custom_TiDE import add_tide_parser

model_parser_dict = {
    "__all__": [add_common_args],
    "Transformer": [],
    "Informer": [],
    "Reformer": [],
    "Flowformer": [],
    "Flashformer": [],
    "Autoformer": [],
    "S_Mamba": [add_s_mamba_parser],
    "PatchTST": [add_patchtst_parser],
    "DLinear": [add_dlinear_parser],
    "BiMamba4TS": [add_bimamba4ts_parser],
    "iTransformer": [],
    "interPDN": [add_interpdn_parser],
    "ManiMamba": [add_manimamba_parser],
    "Crossformer": [add_crossformer_parser],
    "TimesNet": [add_timesnet_parser],
    "TiDE": [add_tide_parser],
}

model_aliases = {
    "Transformer": "Transformer",
    "Informer": "Informer",
    "Reformer": "Reformer",
    "Flowformer": "Flowformer",
    "Flashformer": "Flashformer",
    "Autoformer": "Autoformer",
    "S_Mamba": "S_Mamba",
    "PatchTST": "PatchTST",
    "DLinear": "DLinear",
    "BiMamba4TS": "BiMamba4TS",
    "iTransformer": "iTransformer",
    "interPDN": "interPDN",
    "ManiMamba": "ManiMamba",
    "Crossformer": "Crossformer",
    "TimesNet": "TimesNet",
    "TiDE": "TiDE",
}
