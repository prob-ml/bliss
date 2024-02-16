"""Implments a Lightning module that loads data from disk."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class SavedDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, batch_size: int):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)
