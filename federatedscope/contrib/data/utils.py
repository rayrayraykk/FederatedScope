import numpy as np
from sklearn.model_selection import train_test_split

from federatedscope.core.data.utils import convert_data_mode
from federatedscope.core.data import ClientData, BaseDataTranslator, \
    DummyDataTranslator


def convert2cdata(data, config, client_cfgs):
    if not config.distribute.use:
        if isinstance(data, dict):
            translator = DummyDataTranslator(config, client_cfgs)
        else:
            translator = BaseDataTranslator(config, client_cfgs)
        data = translator(data)
        data = convert_data_mode(data, config)
    else:
        data = ClientData(config,
                          train=data['train'] if 'train' in data else None,
                          val=data['val'] if 'val' in data else None,
                          test=data['test'] if 'test' in data else None)
    return data


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
