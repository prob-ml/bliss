from typing import Optional

import torch
from matplotlib import pyplot as plt
from torchmetrics import MetricCollection

from bliss.catalog import BaseTileCatalog
from bliss.encoder.encoder import Encoder
from bliss.encoder.variational_dist import VariationalDist
from case_studies.weak_lensing.lensing_convnet import WeakLensingCatalogNet, WeakLensingFeaturesNet


class WeakLensingEncoder(Encoder):
    def __init__(
        self,
        survey_bands: list,
        tile_slen: int,
        n_tiles: int,
        nch_hidden: int,
        image_normalizers: list,
        var_dist: VariationalDist,
        sample_image_renders: MetricCollection,
        mode_metrics: MetricCollection,
        sample_metrics: Optional[MetricCollection] = None,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
        reference_band: int = 2,
        **kwargs,
    ):
        self.n_tiles = n_tiles
        self.nch_hidden = nch_hidden

        super().__init__(
            survey_bands=survey_bands,
            tile_slen=tile_slen,
            image_normalizers=image_normalizers,
            var_dist=var_dist,
            matcher=None,
            sample_image_renders=sample_image_renders,
            mode_metrics=mode_metrics,
            sample_metrics=sample_metrics,
            optimizer_params=optimizer_params,
            scheduler_params=scheduler_params,
            use_double_detect=False,
            use_checkerboard=False,
            reference_band=reference_band,
        )

        self.initialize_networks()
        self.epoch_train_losses = []  # List to store epoch losses
        self.current_epoch_train_loss = 0.0  # Variable to accumulate batch losses
        self.current_epoch_train_batches = 0  # Variable to count batches in the current epoch
        self.current_epochs = 0
        self.train_loss_location = kwargs["train_loss_location"]
        self.epoch_val_losses = []
        self.current_epoch_val_loss = 0.0
        self.current_epoch_val_batches = 0

    # override
    def initialize_networks(self):
        num_features = 512
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        self.features_net = WeakLensingFeaturesNet(
            n_bands=len(self.survey_bands),
            ch_per_band=ch_per_band,
            num_features=num_features,
            tile_slen=self.tile_slen,
            nch_hidden=self.nch_hidden,
        )

        self.catalog_net = WeakLensingCatalogNet(
            in_channels=num_features,
            out_channels=self.var_dist.n_params_per_source,
            n_tiles=self.n_tiles,
        )

    def sample(self, batch, use_mode=True):
        # multiple image normalizers
        input_lst = [inorm.get_input_tensor(batch) for inorm in self.image_normalizers]
        inputs = torch.cat(input_lst, dim=2)

        x_features = self.features_net(inputs)
        x_cat_marginal = self.catalog_net(x_features)
        # est cat
        return self.var_dist.sample(x_cat_marginal, use_mode=use_mode, return_base_cat=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            return self.sample(batch, use_mode=True)

    def _compute_loss(self, batch, logging_name):
        batch_size, _, _, _ = batch["images"].shape[0:4]

        target_cat = BaseTileCatalog(batch["tile_catalog"])

        # multiple image normalizers
        input_lst = [inorm.get_input_tensor(batch) for inorm in self.image_normalizers]
        inputs = torch.cat(input_lst, dim=2)
        pred = {}
        x_features = self.features_net(inputs)
        pred["x_cat_marginal"] = self.catalog_net(x_features)

        loss = self.var_dist.compute_nll(pred["x_cat_marginal"], target_cat)
        loss = loss.sum() / loss.numel()

        self.log(f"{logging_name}/_loss", loss, batch_size=batch_size, sync_dist=True)

        if logging_name == "train":
            # Accumulate batch loss
            self.current_epoch_train_loss += loss.item()
            self.current_epoch_train_batches += 1
        if logging_name == "val":
            self.current_epoch_val_loss += loss.item()
            self.current_epoch_val_batches += 1

        return loss

    def on_after_backward(self):
        # Calculate and log the gradient norms
        total_grad_norm = 0.0
        for _, param in self.named_parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += param_grad_norm**2
        total_grad_norm = total_grad_norm**0.5

    def update_metrics(self, batch, batch_idx):
        target_cat = BaseTileCatalog(batch["tile_catalog"])

        mode_cat = self.sample(batch, use_mode=True)
        self.mode_metrics.update(target_cat, mode_cat, None)

        sample_cat_no_mode = self.sample(batch, use_mode=False)
        self.sample_metrics.update(target_cat, sample_cat_no_mode, None)

        self.sample_image_renders.update(
            batch,
            target_cat,
            mode_cat,
            None,
            self.current_epoch,
            batch_idx,
        )

    def on_train_epoch_end(self):
        # Compute the average loss for the epoch and reset counters
        avg_epoch_train_loss = self.current_epoch_train_loss / self.current_epoch_train_batches
        self.epoch_train_losses.append(avg_epoch_train_loss)
        self.current_epoch_train_loss = 0.0
        self.current_epoch_train_batches = 0
        self.current_epochs += 1
        print(  # noqa: WPS421
            f"Average train loss for epoch {self.current_epoch}: {avg_epoch_train_loss}",
        )

    def on_validation_epoch_end(self):
        self.report_metrics(self.mode_metrics, "val/mode", show_epoch=True)
        self.mode_metrics.reset()
        if self.sample_metrics is not None:
            self.report_metrics(self.sample_metrics, "val/sample", show_epoch=True)
            self.sample_metrics.reset()
        if self.sample_image_renders is not None:
            self.report_metrics(self.sample_image_renders, "val/image_renders", show_epoch=True)
        
        # new additions
        avg_epoch_val_loss = self.current_epoch_val_loss / self.current_epoch_val_batches
        self.epoch_val_losses.append(avg_epoch_val_loss)
        self.current_epoch_val_loss = 0.0
        self.current_epoch_val_batches = 0
        print(  # noqa: WPS421
            f"Average val loss for epoch {self.current_epoch}: {avg_epoch_val_loss}",
        )

    def on_train_end(self):
        # Plot the training loss at the end of training
        plt.plot(
            range(len(self.epoch_train_losses)), self.epoch_train_losses, label="Training Loss"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.savefig(f"{self.train_loss_location}/training_loss.png")
        plt.close()

        plt.plot(range(len(self.epoch_val_losses)), self.epoch_val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Validation Loss Over Epochs")
        plt.legend()
        plt.savefig(f"{self.train_loss_location}/validation_loss.png")
        plt.close()
