import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from federatedscope.register import register_data
from federatedscope.core.auxiliaries.utils import setup_seed


def read_df(df, splits=[0.6, 0.2, 0.2]):
    X = df.drop("y", axis=1)
    y = df["y"]

    if "split" not in df.columns:
        X_train, X_test, y_train, y_test = \
            train_test_split(X,
                             y,
                             test_size=splits[-1])
        X_train, X_valid, y_train, y_valid = \
            train_test_split(X_train,
                             y_train,
                             test_size=splits[1] / (splits[0] + splits[1]))
    else:
        X_train = X[df["split"] == "train"].drop("split", axis=1)
        y_train = y[df["split"] == "train"]
        X_test = X[df["split"] == "test"].drop("split", axis=1)
        y_test = y[df["split"] == "test"]
        X_valid = X[df["split"] == "val"].drop("split", axis=1)
        y_valid = y[df["split"] == "val"]

    train_data = list(
        zip(np.array(X_train.values.tolist(), dtype=np.float32),
            np.array(y_train.values.tolist(), dtype=np.float32)))
    test_data = list(
        zip(np.array(X_test.values.tolist(), dtype=np.float32),
            np.array(y_test.values.tolist(), dtype=np.float32)))
    valid_data = list(
        zip(np.array(X_valid.values.tolist(), dtype=np.float32),
            np.array(y_valid.values.tolist(), dtype=np.float32)))
    print(train_data)
    return train_data, valid_data, test_data


def load_data_from_csv(config, client_cfgs=None):
    """
        CSV file:
        x1,x2,x3,y,(split)
        1.2,3.4,5.6,1.0,train
        2.3,4.5,6.7,0.1,test
        3.4,5.6,7.8,0.3,val
        ...
    """
    from federatedscope.contrib.data.utils import convert2cdata

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
