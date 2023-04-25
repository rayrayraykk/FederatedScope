import os
import pickle
from pathlib import Path

from federatedscope.register import register_data
from federatedscope.core.auxiliaries.utils import setup_seed


def load_data_from_oss(config, client_cfgs=None):
    from federatedscope.contrib.data.utils import convert2cdata

    def does_bucket_exist(bucket):
        try:
            bucket.get_bucket_info()
        except oss2.exceptions.NoSuchBucket:
            return False
        except:
            raise
        return True

    file_path = os.path.join(config.data.root, config.data.file_path)
    folder_path = os.path.dirname(file_path)
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(file_path) or not config.oss.download.use_cache:
        if os.path.exists(file_path):
            os.remove(file_path)

        # Download
        import oss2
        auth = oss2.Auth(config.oss.access_key_id,
                         config.oss.access_key_secret)
        bucket = oss2.Bucket(auth,
                             config.oss.endpoint,
                             config.oss.bucket,
                             connect_timeout=config.oss.timeout)
        if not does_bucket_exist(bucket):
            raise Exception(f'Unable to access {bucket}, with check your '
                            f'auth and settings in the `cfg.oss`.')

        bucket.get_object_to_file(config.data.file_path, file_path)

    with open(file_path, 'br') as file:
        data = pickle.load(file)

    data = convert2cdata(data, config, client_cfgs)

    # Restore the user-specified seed after the data generation
    setup_seed(config.seed)

    return data, config


def call_oss_data(config, client_cfgs):
    if config.data.type == "oss_file":
        data, modified_config = load_data_from_oss(config, client_cfgs)
        return data, modified_config


register_data("oss_file", call_oss_data)
