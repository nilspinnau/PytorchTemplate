from typing import Tuple
from torch.utils.data import Dataset, DataLoader

import torch_template.src.data.utils as utils

class TemplateDataset(Dataset):
    def __init__(self):
        super().__init__()
        # set all class variables
        self.dataset_size = 0
        self.shuffle = True
        self.batch_size = 0

        self.get_samples_and_splits()

    def __getitem__(self, idx: int):
        return idx

    def __len__(self):
        return self.dataset_size

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_dataloader = utils.generate_dataloader(
            shuffle=self.shuffle, 
            samples=self.train_samples,
            batch_size=self.batch_size)

        val_dataloader = utils.generate_dataloader(
            shuffle=self.shuffle, 
            samples=self.val_samples,
            batch_size=self.batch_size)

        test_dataloader = utils.generate_dataloader(
            shuffle=self.shuffle, 
            samples=self.test_samples,
            batch_size=self.batch_size)

        return train_dataloader, val_dataloader, test_dataloader

    def get_samples_and_splits(self):
        self.train_samples = []
        self.val_samples = []
        self.test_samples = []

    def from_config(config):
        return TemplateDataset()
