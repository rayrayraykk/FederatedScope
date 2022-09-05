import abc
from federatedscope.core.auxiliaries.splitter_builder import get_splitter
from federatedscope.core.auxiliaries.transform_builder import get_transform
from federatedscope.core.interface.base_data import ClientData, \
    StandaloneDataDict

try:
    from torch.utils.data import DataLoader
except ImportError:
    DataLoader = None


class BaseDataTranslator(StandaloneDataDict):
    def __init__(self,
                 dataset,
                 global_cfg,
                 client_cfgs=None,
                 loader=DataLoader,
                 package=None):
        self.dataset = dataset
        self.loader = loader
        self.global_cfg = global_cfg.clone()
        self.client_cfgs = client_cfgs

        self.splitter = get_splitter(global_cfg)
        if package is not None:
            self.transform_funcs = get_transform(global_cfg, package)

        train, val, test = self.split_train_val_test()
        datadict = self.split_to_client(train, val, test)
        super(BaseDataTranslator, self).__init__(datadict, global_cfg)

    @abc.abstractmethod
    def split_train_val_test(self):
        """
        Split dataset to train, val, test.
        This method should be overridden.

        Returns:
            split_data (List): List of split dataset, [train, val, test]

        """
        from torch.utils.data.dataset import random_split

        dataset, splits = self.dataset, self.global_cfg.data.splits
        train_size = int(splits[0] * len(dataset))
        val_size = int(splits[1] * len(dataset))
        test_size = len(dataset) - train_size - val_size
        split_data = random_split(dataset, [train_size, val_size, test_size])
        return split_data

    def split_to_client(self, train, val, test):
        """
        Split dataset to clients.

        Returns:
            datadict (dict): dict of `ClientData` with client_idx as key.

        """
        split_train = self.splitter(train)
        train_label_distribution = [[j[1] for j in x] for x in split_train]
        split_val = self.splitter(val, prior=train_label_distribution)
        split_test = self.splitter(test, prior=train_label_distribution)

        # Build data dict
        datadict = {}
        for client_id in range(1, self.global_cfg.federate.client_num + 1):
            if self.client_cfgs is not None:
                client_cfg = self.global_cfg.clone().defrost()
                client_cfg.merge_from_other_cfg(
                    self.client_cfgs.get(f'client_{client_id}'))
            else:
                client_cfg = self.global_cfg
            datadict[client_id] = ClientData(self.loader,
                                             client_cfg,
                                             train=split_train[client_id - 1],
                                             val=split_val[client_id - 1],
                                             test=split_test[client_id - 1])
        return datadict
