from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging

logger = logging.getLogger(__name__)


def register(key, module, module_dict):
    """
    Register a module in the module dictionary.

    Args:
        key: write your description
        module: write your description
        module_dict: write your description
    """
    if key in module_dict:
        logger.warning(
            'Key {} is already pre-defined, overwritten.'.format(key))
    module_dict[key] = module


data_dict = {}


def register_data(key, module):
    """
    Register a data module with the given key.

    Args:
        key: write your description
        module: write your description
    """
    register(key, module, data_dict)


model_dict = {}


def register_model(key, module):
    """
    Register a model class.

    Args:
        key: write your description
        module: write your description
    """
    register(key, module, model_dict)


trainer_dict = {}


def register_trainer(key, module):
    """
    Register a trainer module.

    Args:
        key: write your description
        module: write your description
    """
    register(key, module, trainer_dict)


config_dict = {}


def register_config(key, module):
    """
    Register a configuration key with the given module.

    Args:
        key: write your description
        module: write your description
    """
    register(key, module, config_dict)


metric_dict = {}


def register_metric(key, module):
    """
    Register a metric for a given key.

    Args:
        key: write your description
        module: write your description
    """
    register(key, module, metric_dict)


criterion_dict = {}


def register_criterion(key, module):
    """
    Register criterion module for a given key.

    Args:
        key: write your description
        module: write your description
    """
    register(key, module, criterion_dict)


regularizer_dict = {}


def register_regularizer(key, module):
    """
    Register a regularizer for a specific key in a module.

    Args:
        key: write your description
        module: write your description
    """
    register(key, module, regularizer_dict)


auxiliary_data_loader_PIA_dict = {}


def register_auxiliary_data_loader_PIA(key, module):
    """
    Register a PIA auxiliary data loader module.

    Args:
        key: write your description
        module: write your description
    """
    register(key, module, auxiliary_data_loader_PIA_dict)


transform_dict = {}


def register_transform(key, module):
    """
    Register a transform function.

    Args:
        key: write your description
        module: write your description
    """
    register(key, module, transform_dict)


splitter_dict = {}


def register_splitter(key, module):
    """
    Register a splitter module.

    Args:
        key: write your description
        module: write your description
    """
    register(key, module, splitter_dict)


scheduler_dict = {}


def register_scheduler(key, module):
    """
    Register a scheduler module with the scheduler_dict.

    Args:
        key: write your description
        module: write your description
    """
    register(key, module, scheduler_dict)


optimizer_dict = {}


def register_optimizer(key, module):
    """
    Register an optimizer for a module.

    Args:
        key: write your description
        module: write your description
    """
    register(key, module, optimizer_dict)


worker_dict = {}


def register_worker(key, module):
    """
    Register a worker module.

    Args:
        key: write your description
        module: write your description
    """
    register(key, module, worker_dict)
