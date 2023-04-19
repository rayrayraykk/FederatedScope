import os

from federatedscope.register import register_data
from federatedscope.core.data.utils import convert_data_mode
from federatedscope.core.auxiliaries.utils import setup_seed


def load_data_from_odps(config, client_cfgs=None):
    from odps import ODPS

    odps = ODPS(config.odps.access_key_id,
                config.odps.access_key_secret,
                config.odps.project,
                endpoint=config.odps.endpoint)
    table = odps.get_table(config.odps.table_name, project=config.odps.project)

    # TODO: Convert table to ClientData
    ...

    # TO BE FIXED
    # Convert `StandaloneDataDict` to `ClientData` when in distribute mode
    # data = convert_data_mode(data, config)
    data = table

    # Restore the user-specified seed after the data generation
    setup_seed(config.seed)

    return data, config


def call_odps_data(config, client_cfgs):
    if config.data.type == "odps_file":
        data, modified_config = load_data_from_odps(config, client_cfgs)
        return data, modified_config


register_data("odps_file", call_odps_data)
