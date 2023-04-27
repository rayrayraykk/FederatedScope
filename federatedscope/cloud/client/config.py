# ---------------------------------------------------------------------- #
# FS-Service Server related (global variable stored in Redis)
# ---------------------------------------------------------------------- #
server_ip = '127.0.0.1'
broker_url = f'redis://{server_ip}:6379/0'
result_backend = f'redis://{server_ip}/0'

task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'Europe/Oslo'
enable_utc = True

# ---------------------------------------------------------------------- #
# FL-Runner related (To verify and launch the virtual environment)
# ---------------------------------------------------------------------- #
from yacs.config import CfgNode as CN

# Main
cloud_cfg = CN()
cloud_cfg.timeout = 30
cloud_cfg.user = ''
cloud_cfg.password = ''

# Runner related
cloud_cfg.runner = CN()
cloud_cfg.runner.balance = 'least_load'

# Conda related
cloud_cfg.conda = CN()
cloud_cfg.conda.python = '3.9'
cloud_cfg.conda.path = '/opt/anaconda3'
cloud_cfg.conda.env_name = 'test_fabric'

# Backend related
cloud_cfg.backend = CN()
cloud_cfg.backend.type = 'torch'
cloud_cfg.backend.pytorch = '1.10.1'
cloud_cfg.backend.torchvision = '0.11.2'
cloud_cfg.backend.torchaudio = '0.10.1'
cloud_cfg.backend.torchtext = '0.11.1'
cloud_cfg.backend.cudatoolkit = '11.3'

# FS related
cloud_cfg.fs = CN()
cloud_cfg.fs.repo_url = 'https://github.com/rayrayraykk/FederatedScope.git'
cloud_cfg.fs.branch = 'cloud-dev'
cloud_cfg.fs.path = 'test_cloud'
