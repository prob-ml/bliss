import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.optim import Adam

from bliss.catalog import TileCatalog
from bliss.models.decoder import ImageDecoder


class LearnedBackground(pl.LightningModule):
    def __init__(self, decoder: ImageDecoder, initial=865.0, optimizer_params=None):
        super().__init__()
        self.decoder = decoder.requires_grad_(False)
        self.bg = nn.Parameter(torch.tensor(initial))
        self.optimizer_params = optimizer_params

    def forward(self, img, intensity):
        pass

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        tile_catalog = TileCatalog(
            self.decoder.tile_slen,
            {k: v for k, v in batch.items() if k not in {"images", "background"}},
        )
        recon = self.decoder.render_images(tile_catalog)
        mean_intensity = F.relu(recon + self.bg)
        loglik = Normal(mean_intensity, mean_intensity.sqrt()).log_prob(images)
        loss = -loglik.sum() / loglik.shape[0]
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.optimizer_params["lr"])
