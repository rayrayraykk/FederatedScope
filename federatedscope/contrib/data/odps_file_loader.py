import os

from federatedscope.register import register_data
from federatedscope.core.auxiliaries.utils import setup_seed


def load_data_from_odps(config, client_cfgs=None):
    from odps import ODPS
    from federatedscope.contrib.data.utils import convert2cdata, read_df

    odps = ODPS(config.odps.access_key_id,
                config.odps.access_key_secret,
                config.odps.project,
                endpoint=config.odps.endpoint)
    all_tables = [t.name for t in odps.list_tables()]
    if config.odps.table_name not in all_tables:
        raise FileNotFoundError(f'{config.odps.table_name} not in '
                                f'{all_tables}.')
    table = odps.get_table(config.odps.table_name, project=config.odps.project)
    df = table.to_df()

    data = read_df(df, config.data.splits)
    data = convert2cdata(data, config, client_cfgs)

    # Restore the user-specified seed after the data generation
    setup_seed(config.seed)

    return data, config


def call_odps_data(config, client_cfgs):
    if config.data.type == "odps_file":
        data, modified_config = load_data_from_odps(config, client_cfgs)
        return data, modified_config


register_data("odps_file", call_odps_data)
