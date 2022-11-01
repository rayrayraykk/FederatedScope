import zipfile
import os
import torch

import numpy as np
import os.path as osp

from torch.utils.data import Dataset

LEAF_NAMES = [
    'femnist', 'celeba', 'synthetic', 'shakespeare', 'twitter', 'subreddit'
]


def is_exists(path, names):
    """
    Check if a file or directory exists

    Args:
        path: write your description
        names: write your description
    """
    exists_list = [osp.exists(osp.join(path, name)) for name in names]
    return False not in exists_list


class LEAF(Dataset):
    """Base class for LEAF dataset from "LEAF: A Benchmark for Federated Settings"

    Arguments:
        root (str): root path.
        name (str): name of dataset, in `LEAF_NAMES`.
        transform: transform for x.
        target_transform: transform for y.

    """
    def __init__(self, root, name, transform, target_transform):
        """
        Initialize a new LEAF file.

        Args:
            self: write your description
            root: write your description
            name: write your description
            transform: write your description
            target_transform: write your description
        """
        self.root = root
        self.name = name
        self.data_dict = {}
        if name not in LEAF_NAMES:
            raise ValueError(f'No leaf dataset named {self.name}')
        self.transform = transform
        self.target_transform = target_transform
        self.process_file()

    @property
    def raw_file_names(self):
        """
        Return a list of file names.

        Args:
            self: write your description
        """
        names = ['all_data.zip']
        return names

    @property
    def extracted_file_names(self):
        """
        Return a list of file names extracted from the extracted file.

        Args:
            self: write your description
        """
        names = ['all_data']
        return names

    @property
    def raw_dir(self):
        """
        Return raw directory

        Args:
            self: write your description
        """
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        """
        Return processed directory

        Args:
            self: write your description
        """
        return osp.join(self.root, self.name, 'processed')

    def __repr__(self):
        """
        Return a string representation of the object.

        Args:
            self: write your description
        """
        return f'{self.__class__.__name__}({self.__len__()})'

    def __len__(self):
        """
        Returns the length of the data_dict.

        Args:
            self: write your description
        """
        return len(self.data_dict)

    def __getitem__(self, index):
        """
        Return the value at index.

        Args:
            self: write your description
            index: write your description
        """
        raise NotImplementedError

    def __iter__(self):
        """
        Iterator over the data.

        Args:
            self: write your description
        """
        for index in range(len(self.data_dict)):
            yield self.__getitem__(index)

    def download(self):
        """
        Download the file.

        Args:
            self: write your description
        """
        raise NotImplementedError

    def extract(self):
        """
        Extract all files from the raw zip file

        Args:
            self: write your description
        """
        for name in self.raw_file_names:
            with zipfile.ZipFile(osp.join(self.raw_dir, name), 'r') as f:
                f.extractall(self.raw_dir)

    def process_file(self):
        """
        Process the file and extract it if needed.

        Args:
            self: write your description
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        if len(os.listdir(self.processed_dir)) == 0:
            if not is_exists(self.raw_dir, self.extracted_file_names):
                if not is_exists(self.raw_dir, self.raw_file_names):
                    self.download()
                self.extract()
            self.process()

    def process(self):
        """
        Process the pipeline.

        Args:
            self: write your description
        """
        raise NotImplementedError


class LocalDataset(Dataset):
    """
        Convert data list to torch Dataset to save memory usage.
    """
    def __init__(self,
                 Xs,
                 targets,
                 pre_process=None,
                 transform=None,
                 target_transform=None):
        """
        Initialize the EM algorithm

        Args:
            self: write your description
            Xs: write your description
            targets: write your description
            pre_process: write your description
            transform: write your description
            target_transform: write your description
        """
        assert len(Xs) == len(
            targets), "The number of data and labels are not equal."
        self.Xs = np.array(Xs)
        self.targets = np.array(targets)
        self.pre_process = pre_process
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Returns the number of data points in the dataset.

        Args:
            self: write your description
        """
        return len(self.Xs)

    def __getitem__(self, idx):
        """
        Return data and target at specified index.

        Args:
            self: write your description
            idx: write your description
        """
        data, target = self.Xs[idx], self.targets[idx]
        if self.pre_process:
            data = self.pre_process(data)

        if self.transform:
            data = self.transform(data)

        if self.target_transform:
            target = self.target_transform(target)

        return data, target

    def extend(self, dataset):
        """
        Extends the dataset with another dataset.

        Args:
            self: write your description
            dataset: write your description
        """
        self.Xs = np.vstack((self.Xs, dataset.Xs))
        self.targets = np.hstack((self.targets, dataset.targets))
