import os
import shutil
import numpy as np
from ConfigSpace.util import generate_grid

from federatedscope.core.configs.config import global_cfg
from federatedscope.autotune.utils import parse_search_space


def filtered_alg(config):
    """
    Filtered out configs with incompatible algorithm.
    """
    algs = ['fedopt.use', 'fedprox.use', 'nbafl.use', 'federate.method']
    config_dict = dict(config)
    if config_dict['federate.method'] == 'FedAvg':
        config_dict['federate.method'] = False
    else:
        config_dict['federate.method'] = True
    is_used = [config_dict[x] for x in algs]
    if sum(is_used) <= 1:
        return True
    else:
        return False


def cfg2cmd(cfg, config, device, seed=1):
    cmd = ''
    for key in sorted(list(config.keys())):
        cmd += f'{key}:{config[key]} '
    expname = cmd.replace(' ', '+')[:-1]
    cmd = cmd.replace(':', ' ')

    if config['federate.method'] != 'FedAvg' or config['fedopt.use'] or \
            config['fedprox.use'] or config['nbafl.use']:
        cmd += 'federate.share_local_model False federate.online_aggr False '

    return f'python federatedscope/main.py --cfg {cfg} seed {seed} device' \
           f' {device} outdir {cfg} expname {expname} {cmd}'


def convert_yaml_to_cmd(cfg_path, num_device=8, num_trial_per_device=1, rp=3):
    root_dir = f'run_{cfg_path}'

    # Avoid conflicts
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
        print(f'{root_dir} exists, removed.')

    os.makedirs(root_dir)
    cfg = global_cfg.clone()
    cfg.merge_from_file(cfg_path)

    search_space = parse_search_space(cfg.hpo.ss)
    grid = [dict(x) for x in generate_grid(search_space) if filtered_alg(x)]
    print(f'Valid configs: {len(grid)}')
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
            with open(file_path, 'w') as f:
                f.write('set -e' + os.linesep)
                for i in trials:
                    for seed in range(1, rp + 1):
                        f.write(
                            cfg2cmd(cfg_path, i, device, seed) + os.linesep)


if __name__ == '__main__':
    convert_yaml_to_cmd('femnist.yaml')
