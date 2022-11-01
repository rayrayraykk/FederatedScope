import numpy as np
from federatedscope.core.splitters import BaseSplitter


class IIDSplitter(BaseSplitter):
    def __init__(self, client_num):
        """
        Splitter class initialization.

        Args:
            self: write your description
            client_num: write your description
        """
        super(IIDSplitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None):
        """
        Split dataset into chunks of client_num clients.

        Args:
            self: write your description
            dataset: write your description
            prior: write your description
        """
        dataset = [ds for ds in dataset]
        np.random.shuffle(dataset)
        length = len(dataset)
        prop = [1.0 / self.client_num for _ in range(self.client_num)]
        prop = (np.cumsum(prop) * length).astype(int)[:-1]
        data_list = np.split(dataset, prop)
        data_list = [x.tolist() for x in data_list]
        return data_list
