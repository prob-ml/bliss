from pathlib import Path
from typing import Optional

import torch
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import MetricCollection

from bliss.catalog import BaseTileCatalog
from bliss.encoder.encoder import Encoder
from bliss.encoder.variational_dist import VariationalDist
from case_studies.weak_lensing.convnet import WeakLensingNet


class WeakLensingEncoder(Encoder):
    def __init__(
        self,
        survey_bands: list,
        tile_slen: int,
        n_pixels_per_side: int,
        n_tiles_per_side: int,
        ch_init: int,
        ch_max: int,
        initial_downsample: bool,
        more_up_layers: bool,
        num_bottleneck_layers: int,
        image_normalizers: list,
        var_dist: VariationalDist,
        sample_image_renders: MetricCollection,
        mode_metrics: MetricCollection,
        sample_metrics: Optional[MetricCollection] = None,
        optimizer_params: Optional[dict] = None,
        exp_scheduler_params: Optional[dict] = None,
        reference_band: int = 2,
        **kwargs,
    ):
        self.n_pixels_per_side = n_pixels_per_side
        self.n_tiles_per_side = n_tiles_per_side
        self.ch_init = ch_init
        self.ch_max = ch_max
        self.initial_downsample = initial_downsample
        self.more_up_layers = more_up_layers
        self.num_bottleneck_layers = num_bottleneck_layers
        self.exp_scheduler_params = exp_scheduler_params

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
            scheduler_params=None,
            use_double_detect=False,
            use_checkerboard=False,
            reference_band=reference_band,
        )

        self.initialize_networks()
        self.epoch_train_losses = []
        self.current_epoch_train_loss = 0.0
        self.current_epoch_train_batches = 0
        self.current_epochs = 0
        self.loss_plots_location = kwargs["loss_plots_location"]
        self.epoch_val_losses = []
        self.current_epoch_val_loss = 0.0
        self.current_epoch_val_batches = 0

    # override
    def initialize_networks(self):
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        self.net = WeakLensingNet(
            n_bands=len(self.survey_bands),
            n_pixels_per_side=self.n_pixels_per_side,
            n_tiles_per_side=self.n_tiles_per_side,
            ch_per_band=ch_per_band,
            ch_init=self.ch_init,
            ch_max=self.ch_max,
            initial_downsample=self.initial_downsample,
            more_up_layers=self.more_up_layers,
            num_bottleneck_layers=self.num_bottleneck_layers,
            n_var_params=self.var_dist.n_params_per_source,
        )

    def sample(self, batch, use_mode=True):
        input_lst = [inorm.get_input_tensor(batch) for inorm in self.image_normalizers]
        inputs = torch.cat(input_lst, dim=2)

        x_cat_marginal = self.net(inputs)
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
        pred["x_cat_marginal"] = self.net(inputs)

        loss = self.var_dist.compute_nll(pred["x_cat_marginal"], target_cat)
        loss = loss.sum() / loss.numel()

        self.log(f"{logging_name}/_loss", loss, batch_size=batch_size, sync_dist=True)

        if logging_name == "train":
            self.current_epoch_train_loss += loss.item()
            self.current_epoch_train_batches += 1
        if logging_name == "val":
            self.current_epoch_val_loss += loss.item()
            self.current_epoch_val_batches += 1

        return loss

    def on_after_backward(self):
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
            target_cat,
            mode_cat,
            self.current_epoch,
            batch_idx,
        )

    def report_metrics(self, metrics, logging_name):
        computed = metrics.compute()
        for k, v in computed.items():
            if torch.is_tensor(v) and v.numel() > 1:
                for i in range(v.numel()):
                    self.log(f"{logging_name}/{k}/bin_{i}", v[i].item(), sync_dist=True)
            else:
                self.log(
                    f"{logging_name}/{k}", v.item() if torch.is_tensor(v) else v, sync_dist=True
                )

        for metric_name, metric in metrics.items():
            if hasattr(metric, "plot"):  # noqa: WPS421
                try:
                    plot_or_none = metric.plot()
                except NotImplementedError:
                    continue
                name = f"Epoch:{self.current_epoch}"
                name = f"{name}/{logging_name} {metric_name}"
                if self.logger and plot_or_none:
                    fig, _axes = plot_or_none
                    self.logger.experiment.add_figure(name, fig)

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
        self.report_metrics(self.mode_metrics, "val/mode")
        self.mode_metrics.reset()
        if self.sample_metrics is not None:
            self.report_metrics(self.sample_metrics, "val/sample")
            self.sample_metrics.reset()
        if self.sample_image_renders is not None:
            self.report_metrics(self.sample_image_renders, "val/image_renders")

        avg_epoch_val_loss = self.current_epoch_val_loss / self.current_epoch_val_batches
        self.epoch_val_losses.append(avg_epoch_val_loss)
        self.current_epoch_val_loss = 0.0
        self.current_epoch_val_batches = 0
        print(  # noqa: WPS421
            f"Average val loss for epoch {self.current_epoch}: {avg_epoch_val_loss}",
        )

    def on_test_epoch_end(self):
        self.report_metrics(self.mode_metrics, "test/mode")
        if self.sample_metrics is not None:
            self.report_metrics(self.sample_metrics, "test/sample")

    def on_train_end(self):
        if not Path(self.loss_plots_location).exists():
            Path(self.loss_plots_location).mkdir(parents=True)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(13, 6))

        ax[0].plot(
            range(len(self.epoch_train_losses)), self.epoch_train_losses, label="Training loss"
        )
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].set_title("Training loss by epoch")

        ax[1].plot(
            range(len(self.epoch_val_losses)), self.epoch_val_losses, label="Validation loss"
        )
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Loss")
        ax[1].set_title("Validation loss by epoch")

        fig.tight_layout()
        fig.savefig(f"{self.loss_plots_location}/train_and_val_loss.png")

    def configure_optimizers(self):
        """Configure optimizers for training (pytorch lightning)."""
        optimizer = Adam(self.parameters(), **self.optimizer_params)
        scheduler = ExponentialLR(optimizer, **self.exp_scheduler_params)
        return [optimizer], [scheduler]
