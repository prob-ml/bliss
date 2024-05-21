# Author: Qiaozhi Huang
# Class for network to predict redshift prediction
import pytorch_lightning as pl
import torch


class PhotoZFromFluxes(pl.LightningModule):
    def __init__(
        self,
        in_dim,  # Number of input features.
        hidden_dim,  # Number of input features.
        out_dim,  # Number of output features.
        dropout_rate,
        learning_rate,
        loss_fcn,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.loss_fcn = loss_fcn
        # alt
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def training_step(self, batch, batch_idx):
        return self._generic_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._generic_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._generic_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _generic_step(self, batch, batch_idx, loss_type="train"):
        x, y = batch
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        y_hat = self.net(x)
        if self.out_dim != 1:
            y = y.long()
        loss = self.loss_fcn(y_hat, y)
        self.log(f"{loss_type}_loss", loss, prog_bar=True, sync_dist=True)
        return loss
