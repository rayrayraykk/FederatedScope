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
