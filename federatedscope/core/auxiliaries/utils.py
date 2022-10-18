import logging
import math
import os
import random
import signal
import pickle

import numpy as np

try:
    import torch
    import torch.distributions as distributions
except ImportError:
    torch = None
    distributions = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

logger = logging.getLogger(__name__)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    if tf is not None:
        tf.set_random_seed(seed)


def get_random(type, sample_shape, params, device):
    if not hasattr(distributions, type):
        raise NotImplementedError("Distribution {} is not implemented, "
                                  "please refer to ```torch.distributions```"
                                  "(https://pytorch.org/docs/stable/ "
                                  "distributions.html).".format(type))
    generator = getattr(distributions, type)(**params)
    return generator.sample(sample_shape=sample_shape).to(device)


def batch_iter(data, batch_size=64, shuffled=True):
    assert 'x' in data and 'y' in data
    data_x = data['x']
    data_y = data['y']
    data_size = len(data_y)
    num_batches_per_epoch = math.ceil(data_size / batch_size)

    while True:
        shuffled_index = np.random.permutation(
            np.arange(data_size)) if shuffled else np.arange(data_size)
        for batch in range(num_batches_per_epoch):
            start_index = batch * batch_size
            end_index = min(data_size, (batch + 1) * batch_size)
            sample_index = shuffled_index[start_index:end_index]
            yield {'x': data_x[sample_index], 'y': data_y[sample_index]}


def merge_dict(dict1, dict2):
    # Merge results for history
    for key, value in dict2.items():
        if key not in dict1:
            if isinstance(value, dict):
                dict1[key] = merge_dict({}, value)
            else:
                dict1[key] = [value]
        else:
            if isinstance(value, dict):
                merge_dict(dict1[key], value)
            else:
                dict1[key].append(value)
    return dict1


def move_to(obj, device):
    import torch
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


def param2tensor(param):
    import torch
    if isinstance(param, list):
        param = torch.FloatTensor(param)
    elif isinstance(param, int):
        param = torch.tensor(param, dtype=torch.long)
    elif isinstance(param, float):
        param = torch.tensor(param, dtype=torch.float)
    return param


class Timeout(object):
    def __init__(self, seconds, max_failure=5):
        self.seconds = seconds
        self.max_failure = max_failure

    def __enter__(self):
        def signal_handler(signum, frame):
            raise TimeoutError()

        if self.seconds > 0:
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        signal.alarm(0)

    def reset(self):
        signal.alarm(self.seconds)

    def block(self):
        signal.alarm(0)

    def exceed_max_failure(self, num_failure):
        return num_failure > self.max_failure


def get_resource_info(filename):
    if filename is None or not os.path.exists(filename):
        logger.info('The device information file is not provided')
        return None

    # Users can develop this loading function according to resource_info_file
    # As an example, we use the device_info provided by FedScale (FedScale:
    # Benchmarking Model and System Performance of Federated Learning
    # at Scale), which can be downloaded from
    # https://github.com/SymbioticLab/FedScale/blob/master/benchmark/dataset/
    # data/device_info/client_device_capacity The expected format is
    # { INDEX:{'computation': FLOAT_VALUE_1, 'communication': FLOAT_VALUE_2}}
    with open(filename, 'br') as f:
        device_info = pickle.load(f)
    return device_info


def calculate_time_cost(instance_number,
                        comm_size,
                        comp_speed=None,
                        comm_bandwidth=None,
                        augmentation_factor=3.0):
    # Served as an example, this cost model is adapted from FedScale at
    # https://github.com/SymbioticLab/FedScale/blob/master/fedscale/core/
    # internal/client.py#L35 (Apache License Version 2.0)
    # Users can modify this function according to customized cost model
    if comp_speed is not None and comm_bandwidth is not None:
        comp_cost = augmentation_factor * instance_number * comp_speed
        comm_cost = 2.0 * comm_size / comm_bandwidth
    else:
        comp_cost = 0
        comm_cost = 0

    return comp_cost, comm_cost


def calculate_batch_epoch_num(steps, batch_or_epoch, num_data, batch_size,
                              drop_last):
    num_batch_per_epoch = num_data // batch_size + int(
        not drop_last and bool(num_data % batch_size))
    if num_batch_per_epoch == 0:
        raise RuntimeError(
            "The number of batch is 0, please check 'batch_size' or set "
            "'drop_last' as False")
    elif batch_or_epoch == "epoch":
        num_epoch = steps
        num_batch_last_epoch = num_batch_per_epoch
        num_total_batch = steps * num_batch_per_epoch
    else:
        num_epoch = math.ceil(steps / num_batch_per_epoch)
        num_batch_last_epoch = steps % num_batch_per_epoch or \
            num_batch_per_epoch
        num_total_batch = steps
    return num_batch_per_epoch, num_batch_last_epoch, num_epoch, \
        num_total_batch


def merge_param_dict(raw_param, filtered_param):
    for key in filtered_param.keys():
        raw_param[key] = filtered_param[key]
    return raw_param
