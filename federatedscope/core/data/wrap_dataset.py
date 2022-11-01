import torch
import numpy as np
from torch.utils.data import Dataset


class WrapDataset(Dataset):
    """Wrap raw data into pytorch Dataset

    Arguments:
        dataset (dict): raw data dictionary contains "x" and "y"

    """
    def __init__(self, dataset):
        """
        Initializes the wrapper dataset

        Args:
            self: write your description
            dataset: write your description
        """
        super(WrapDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, idx):
        """
        Returns the specified element of the dataset.

        Args:
            self: write your description
            idx: write your description
        """
        if isinstance(self.dataset["x"][idx], torch.Tensor):
            return self.dataset["x"][idx], self.dataset["y"][idx]
        elif isinstance(self.dataset["x"][idx], np.ndarray):
            return torch.from_numpy(
                self.dataset["x"][idx]).float(), torch.from_numpy(
                    self.dataset["y"][idx]).float()
        elif isinstance(self.dataset["x"][idx], list):
            return torch.FloatTensor(self.dataset["x"][idx]), \
                   torch.FloatTensor(self.dataset["y"][idx])
        else:
            raise TypeError

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Args:
            self: write your description
        """
        return len(self.dataset["y"])
