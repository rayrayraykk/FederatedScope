import numpy as np
from ConfigSpace.util import generate_grid

from federatedscope.core.configs.config import global_cfg
from federatedscope.autotune.utils import parse_search_space, config2cmdargs


def cfg2cmd(cfg, device):
    ...


def yaml2sh(cfg_path, num_device=8, num_trial_per_device=3):
    cfg = global_cfg.clone()
    cfg.merge_from_file(cfg_path)

    search_space = parse_search_space(cfg.hpo.ss)
    grid = generate_grid(search_space)
    num_cfg = len(grid)
    grid_ss = ...

    for device in range(num_device):
        for trial in range(num_trial_per_device):
            trial_cfg = cfg.clone()
            trial_cfg.merge_from_list(config2cmdargs(grid_ss[device][trial]))
            print(cfg2cmd(trial_cfg, device))
