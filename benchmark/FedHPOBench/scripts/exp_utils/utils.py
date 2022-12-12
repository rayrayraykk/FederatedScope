import os
import numpy as np
from ConfigSpace.util import generate_grid

from federatedscope.core.configs.config import global_cfg
from federatedscope.autotune.utils import parse_search_space


def cfg2cmd(cfg, config, device):
    cmd = ''
    for key, value in config.items():
        cmd += f'{key}:{value} '
    expname = cmd.replace(' ', '+')
    cmd = cmd.replace(':', ' ')
    return f'python federatedscope/main.py --cfg {cfg} device {device} ' \
           f'outdir {cfg} expname {expname[:-1]} {cmd}'


def convert_yaml_to_cmd(cfg_path, num_device=8, num_trial_per_device=1):
    root_dir = f'run_{cfg_path}'
    os.makedirs(root_dir, exist_ok=True)

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
            file_name = f'device{device}_trial{trial}.sh'
            file_path = os.path.join(root_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f'{file_path} exists, removed.')
            with open(file_path, 'w') as f:
                f.write('set -e' + os.linesep)
                for i in trials:
                    f.write(cfg2cmd(cfg_path, i, device) + os.linesep)


if __name__ == '__main__':
    convert_yaml_to_cmd('sha.yaml')
