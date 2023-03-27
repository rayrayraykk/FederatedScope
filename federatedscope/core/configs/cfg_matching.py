import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_matching_cfg(cfg):

    # ---------------------------------------------------------------------- #
    # hpo related options
    # ---------------------------------------------------------------------- #
    cfg.matching = CN()
    cfg.matching.use = False  # If True, return similarity of each client

    cfg.matching.method = ''
    cfg.matching.target_client_idx = 1
    cfg.matching.round = 1
    cfg.matching.split = 'val'

    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_matching_cfg)


def assert_matching_cfg(cfg):
    if not cfg.matching.use:
        return

    cfg.federate.sample_client_num = -1
    cfg.federate.sample_client_rate = -1.0

    if cfg.matching.split not in cfg.eval.split:
        cfg.eval.split.append(cfg.matching.split)

    if cfg.matching.method.lower() == 'pad':
        cfg.model.out_channels = 1
        cfg.eval.freq = cfg.matching.round
        cfg.early_stop.patience = 0
        cfg.federate.make_global_eval = False
        cfg.federate.merge_test_data = False
        cfg.criterion.type = 'L1Loss'
        cfg.eval.metrics = ['loss']


register_config("matching", extend_matching_cfg)
