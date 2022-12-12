import os
import numpy as np
from ConfigSpace.util import generate_grid

from federatedscope.core.configs.config import global_cfg
from federatedscope.autotune.utils import parse_search_space, config2cmdargs


def cfg2cmd(cfg, config, device):
    # TODO: add expname
    cmd = ''
    for key, value in config.items():
        cmd += f'{key} {value} '
    return f'python federatedscope/main.py --cfg {cfg} device {device} {cmd}'


def convert_yaml_to_cmd(cfg_path, num_device=8, num_trial_per_device=1):
    cfg = global_cfg.clone()
    cfg.merge_from_file(cfg_path)

    search_space = parse_search_space(cfg.hpo.ss)
    grid = [dict(x) for x in generate_grid(search_space)]
    print(len(grid))
    grid_device = np.array_split(grid, num_device)
    grid_ss = []
    for device in range(len(grid_device)):
        gird_trial = np.array_split(grid_device[device], num_trial_per_device)
        gird_trial = [x for x in gird_trial if len(x) > 0]
        grid_ss.append(gird_trial)
    for device in range(len(grid_ss)):
        for trial in range(len(grid_ss[device])):
            trials = grid_ss[device][trial]
            print(f'device {device}, trial {trial}')
            for i in trials:
                print(cfg2cmd(cfg_path, i, device))


if __name__ == '__main__':
    convert_yaml_to_cmd('sha.yaml')
