import os
import pandas as pd
from federatedscope.register import register_data
from federatedscope.core.auxiliaries.utils import setup_seed


def load_data_from_csv(config, client_cfgs=None):
    """
        CSV file:
        x1,x2,x3,y,(split)
        1.2,3.4,5.6,1.0,train
        2.3,4.5,6.7,0.1,test
        3.4,5.6,7.8,0.3,val
        ...
    """
    from federatedscope.contrib.data.utils import convert2cdata, read_df

    file_path = os.path.join(config.data.root, config.data.file_path)
    if not os.path.exists(file_path):
        raise ValueError(f'The file {file_path} does not exist.')

    df = pd.read_csv(file_path)
    data = read_df(df, config.data.splits)
    data = convert2cdata(data, config, client_cfgs)

    # Restore the user-specified seed after the data generation
    setup_seed(config.seed)

    return data, config


def call_csv_data(config, client_cfgs):
    if config.data.type == "csv_file":
        data, modified_config = load_data_from_csv(config, client_cfgs)
        return data, modified_config


register_data("csv_file", call_csv_data)
