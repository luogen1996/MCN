from yacs.config import CfgNode as CN
import json
from typing import Dict, Any

def_cfg = CN(json.load(open('./config/config.json')))

config = CN(def_cfg)


key_maps = {}


def create_from_dict(dct: Dict[str, Any], prefix: str, cfg: CN):
    """
    Helper function to create yacs config from dictionary
    """
    dct_cfg = CN(dct, new_allowed=True)
    prefix_list = prefix.split('.')
    d = cfg
    for pref in prefix_list[:-1]:
        assert isinstance(d, CN)
        if pref not in d:
            setattr(d, pref, CN())
        d = d[pref]
    if hasattr(d, prefix_list[-1]):
        old_dct_cfg = d[prefix_list[-1]]
        dct_cfg.merge_from_other_cfg(old_dct_cfg)

    setattr(d, prefix_list[-1], dct_cfg)
    return cfg


def update_from_dict(cfg: CN, dct: Dict[str, Any],
                     key_maps: Dict[str, str] = None) -> CN:
    """
    Given original CfgNode (cfg) and input dictionary allows changing
    the cfg with the updated dictionary values
    Optional key_maps argument which defines a mapping between
    same keys of the cfg node. Only used for convenience
    Adapted from:
    https://github.com/rbgirshick/yacs/blob/master/yacs/config.py#L219
    """
    # Original cfg
    root = cfg

    # Change the input dictionary using keymaps
    # Now it is aligned with the cfg
    full_key_list = list(dct.keys())
    for full_key in full_key_list:
        if full_key in key_maps:
            cfg[full_key] = dct[full_key]
            new_key = key_maps[full_key]
            dct[new_key] = dct.pop(full_key)

    # Convert the cfg using dictionary input
    for full_key, v in dct.items():
        if root.key_is_deprecated(full_key):
            continue
        if root.key_is_renamed(full_key):
            root.raise_key_rename_error(full_key)
        key_list = full_key.split(".")
        d = cfg
        for subkey in key_list[:-1]:
            # Most important statement
            assert subkey in d, f'key {full_key} doesnot exist'
            d = d[subkey]

        subkey = key_list[-1]
        # Most important statement
        assert subkey in d, f'key {full_key} doesnot exist'

        value = cfg._decode_cfg_value(v)

        assert isinstance(value, type(d[subkey]))
        d[subkey] = value

    return cfg


# def get_config_after_kwargs(cfg, kwargs: Dict[str, Any]):
# ds_info = CN(json.load(open('./configs/ds_info.json')))
# def_cfg = CN(json.load(open('./configs/cfg.json')))

#     upd_cfg = update_from_dict(def_cfg, kwargs, key_maps)

#     cfg_dict = {
#         'ds_to_use': upd_cfg.ds_to_use,
#         'mdl_to_use': upd_cfg.mdl_to_use,
#         'lfn_to_use': upd_cfg.lfn_to_use,
#         'efn_to_use': upd_cfg.efn_to_use,
#         'opt_to_use': upd_cfg.opt_to_use,
#         'sfn_to_use': upd_cfg.sfn_to_use
#     }

#     cfg = CN(cfg_dict)

#     cfg = create_from_dict(ds_info[cfg.ds_to_use], 'DS', cfg)
