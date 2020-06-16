import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from . import sleep


class SleepPhase(pl.LightningModule):
    def __init__(self, dataset, encoder, lr=1e-3, weight_decay=1e-5):
        """dataset is an SourceIterableDataset class"""
        super(SleepPhase, self).__init__()

        self.dataset = dataset
        self.model = encoder
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self):
        pass

    def configure_optimizers(self):
        return Adam(
            [{"params": self.model.parameters(), "lr": self.lr}],
            weight_decay=self.weight_decay,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=None)

    def training_step(self, batch, batch_idx):
        (
            loss,
            counter_loss,
            locs_loss,
            galaxy_params_loss,
            star_params_loss,
            galaxy_bool_loss,
        ) = self.get_loss(batch)

        return {
            "loss": loss,
            "logs": {
                "training_loss": loss,
                "counter_loss": counter_loss,
                "locs_loss": locs_loss,
                "galaxy_params_loss": galaxy_params_loss,
                "star_params_loss": star_params_loss,
                "galaxy_bool_loss": galaxy_bool_loss,
            },
        }

    def training_epoch_end(self, outputs):
        avg_loss = 0
        avg_counter_loss = 0
        avg_locs_loss = 0
        avg_galaxy_params_loss = 0
        avg_star_params_loss = 0
        avg_galaxy_bool_loss = 0
        tiles_per_epoch = self.dataset.n_batches * len(outputs) * self.model.n_tiles

        for output in outputs:
            avg_loss += output["loss"] * len(outputs)
            avg_counter_loss += torch.sum(output["logs"]["counter_loss"]) * len(outputs)
            avg_locs_loss += torch.sum(output["logs"]["locs_loss"]) * len(outputs)
            avg_galaxy_params_loss += torch.sum(
                output["logs"]["galaxy_params_loss"]
            ) * len(outputs)
            avg_star_params_loss += torch.sum(output["logs"]["star_params_loss"]) * len(
                outputs
            )
            avg_galaxy_bool_loss += torch.sum(output["logs"]["galaxy_bool_loss"]) * len(
                outputs
            )

        avg_loss /= self.dataset.n_batches * len(outputs)
        avg_counter_loss /= tiles_per_epoch
        avg_locs_loss /= tiles_per_epoch
        avg_galaxy_params_loss /= tiles_per_epoch
        avg_star_params_loss /= tiles_per_epoch
        avg_galaxy_bool_loss /= tiles_per_epoch

        return {
            "log": {
                "train_loss": avg_loss,
                "counter_loss": avg_counter_loss,
                "locs_loss": avg_locs_loss,
                "galaxy_params_loss": avg_galaxy_params_loss,
                "star_params_loss": avg_star_params_loss,
                "galaxy_bool_loss": avg_galaxy_bool_loss,
            }
        }

    def get_loss(self, batch):
        (images, true_locs, true_galaxy_params, true_log_fluxes, true_galaxy_bool,) = (
            batch["images"],
            batch["locs"],
            batch["galaxy_params"],
            batch["log_fluxes"],
            batch["galaxy_bool"],
        )

        # evaluate log q
        (
            loss,
            counter_loss,
            locs_loss,
            galaxy_params_loss,
            star_params_loss,
            galaxy_bool_loss,
        ) = sleep.get_inv_kl_loss(
            self.image_encoder,
            images,
            true_locs,
            true_galaxy_params,
            true_log_fluxes,
            true_galaxy_bool,
        )

        return (
            loss,
            counter_loss,
            locs_loss,
            galaxy_params_loss,
            star_params_loss,
            galaxy_bool_loss,
        )