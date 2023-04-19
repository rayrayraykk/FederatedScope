import os
import pickle

from federatedscope.register import register_data
from federatedscope.core.data.utils import convert_data_mode
from federatedscope.core.auxiliaries.utils import setup_seed


def load_data_from_oss(config, client_cfgs=None):
    file_path = os.path.join(config.data.root, config.oss.download.data_file)

    if not os.path.exists(file_path):  # Download
        import oss2
        auth = oss2.Auth(config.oss.access_key_id,
                         config.oss.access_key_secret)
        bucket = oss2.Bucket(auth,
                             config.oss.endpoint,
                             config.oss.bucket,
                             connect_timeout=config.oss.timeout)
        bucket.get_object_to_file(config.oss.download.data_file, file_path)

    with open(file_path, 'br') as file:
        data = pickle.load(file)

    # TO BE FIXED
    # Convert `StandaloneDataDict` to `ClientData` when in distribute mode
    data = convert_data_mode(data, config)

    # Restore the user-specified seed after the data generation
    setup_seed(config.seed)

    return data, config


def call_oss_data(config, client_cfgs):
    if config.data.type == "oss_file":
        data, modified_config = load_data_from_oss(config, client_cfgs)
        return data, modified_config


register_data("oss_file", call_oss_data)
