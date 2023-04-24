import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_cloud_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # Cloud related options
    # ---------------------------------------------------------------------- #
    cfg.cloud = CN()

    # ---------------------------------------------------------------------- #
    # OSS related options
    # ---------------------------------------------------------------------- #
    cfg.oss = CN()
    cfg.oss.access_key_id = ''
    cfg.oss.access_key_secret = ''
    cfg.oss.endpoint = ''
    cfg.oss.bucket = ''
    cfg.oss.timeout = 30

    #  OSS - Download
    cfg.oss.download = CN()
    cfg.oss.download.data_files = ''

    # OSS - Upload
    cfg.oss.upload = CN()
    cfg.oss.upload.model_path = ''

    # ---------------------------------------------------------------------- #
    # ODPS related options (Optional)
    # ---------------------------------------------------------------------- #
    cfg.odps = CN()
    cfg.odps.access_key_id = ''
    cfg.odps.access_key_secret = ''
    cfg.odps.endpoint = ''
    cfg.odps.project = ''
    cfg.odps.table_name = ''


def assert_cloud_cfg(cfg):
    ...


register_config("cloud", extend_cloud_cfg)
