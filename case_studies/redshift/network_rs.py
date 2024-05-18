# Author: Qiaozhi Huang
# Class for network to predict redshift prediction
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from tqdm import tqdm


class Regressor(nn.Module):
    def __init__(self,
        in_dim,                             # Number of input features.
        hidden_dim,                         # Number of input features.
        out_dim,                            # Number of output features.
        dropout_rate=0.5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class LitRegressor(pl.LightningModule):
    def __init__(self,
        in_dim,                             # Number of input features.
        hidden_dim,                         # Number of input features.
        out_dim,                            # Number of output features.
        dropout_rate,
        learning_rate,
        loss_fcn,
    ):
        super().__init__()
        self.model = Regressor(in_dim, hidden_dim, out_dim, dropout_rate)
        self.learning_rate = learning_rate
        self.loss_fcn = loss_fcn

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        y_hat = self.model(x)
        if self.model.out_dim != 1:
            y = y.long()
        loss = self.loss_fcn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        y_hat = self.model(x)
        if self.model.out_dim != 1:
            y = y.long()
        loss = self.loss_fcn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        y_hat = self.model(x)
        if self.model.out_dim != 1:
            y = y.long()
        loss = self.loss_fcn(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
