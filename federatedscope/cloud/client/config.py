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
CONFIG = CN()
CONFIG.timeout = 30

# Conda related
CONFIG.conda = CN()
CONFIG.conda.python = '3.9'
CONFIG.conda.path = '/opt/anaconda3'
CONFIG.conda.env_name = 'test_fabric'

# Backend related
CONFIG.backend = CN()
CONFIG.backend.type = 'torch'
CONFIG.backend.pytorch = '1.10.1'
CONFIG.backend.torchvision = '0.11.2'
CONFIG.backend.torchaudio = '0.10.1'
CONFIG.backend.torchtext = '0.11.1'
CONFIG.backend.cudatoolkit = '11.3'

# FS related
CONFIG.fs = CN()
CONFIG.fs.repo_url = 'https://github.com/alibaba/FederatedScope.git'
CONFIG.fs.branch = 'master'
CONFIG.fs.path = 'test_fabric'
