import logging

import torch
import os
from torch_geometric.data import InMemoryDataset

logger = logging.getLogger(__name__)


class CIKMCUPDataset(InMemoryDataset):
    name = 'CIKM_CUP'

    def __init__(self, root):
        """
        Initializes the dataset from the given root object.

        Args:
            self: write your description
            root: write your description
        """
        super(CIKMCUPDataset, self).__init__(root)

    @property
    def processed_dir(self):
        """
        The directory where the file was processed.

        Args:
            self: write your description
        """
        return os.path.join(self.root, self.name)

    @property
    def processed_file_names(self):
        """
        Return the processed file names.

        Args:
            self: write your description
        """
        return ['pre_transform.pt', 'pre_filter.pt']

    def __len__(self):
        """
        Returns the number of processed files in the processed_dir.

        Args:
            self: write your description
        """
        return len([
            x for x in os.listdir(self.processed_dir)
            if not x.startswith('pre')
        ])

    def _load(self, idx, split):
        """
        Load data for a split

        Args:
            self: write your description
            idx: write your description
            split: write your description
        """
        try:
            data = torch.load(
                os.path.join(self.processed_dir, str(idx), f'{split}.pt'))
        except:
            data = None
        return data

    def process(self):
        """
        Process the command.

        Args:
            self: write your description
        """
        pass

    def __getitem__(self, idx):
        """
        Return a dictionary of data for a given index.

        Args:
            self: write your description
            idx: write your description
        """
        data = {}
        for split in ['train', 'val', 'test']:
            split_data = self._load(idx, split)
            if split_data:
                data[split] = split_data
        return data
