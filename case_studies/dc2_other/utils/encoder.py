from typing import Optional

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import MetricCollection

from bliss.catalog import TileCatalog
from bliss.encoder.convnet_layers import C3, ConvBlock, Detect
from bliss.encoder.convnets import FeaturesNet
from bliss.encoder.metrics import CatalogMatcher
from bliss.global_env import GlobalEnv
from case_studies.dc2_other.utils.catalog_parser import CatalogParser


class MyEncoder(pl.LightningModule):
    def __init__(
        self,
        survey_bands: list,
        tile_slen: int,
        image_normalizers: dict,
        catalog_parser: CatalogParser,
        image_size: list,
        matcher: CatalogMatcher,
        mode_metrics: MetricCollection,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
        reference_band: int = 2,
        **kwargs,  # other args inherited from base config
    ):
        """Initializes Encoder.

        Args:
            survey_bands: all band-pass filters available for this survey
            tile_slen: dimension in pixels of a square tile
            image_normalizers: collection of objects that applies input transforms to images
            catalog_parser: encode tile cat to tensor and decode tensor to tile cat
            image_size: size of image (h, w)
            matcher: for matching predicted catalogs to ground truth catalogs
            mode_metrics: for scoring predicted mode catalogs during training
            optimizer_params: arguments passed to the Adam optimizer
            scheduler_params: arguments passed to the learning rate scheduler
            reference_band: band to use for filtering sources
        """
        super().__init__()

        self.survey_bands = survey_bands
        self.tile_slen = tile_slen
        self.image_normalizers = torch.nn.ModuleList(image_normalizers.values())
        self.catalog_parser = catalog_parser
        self.image_size = image_size
        self.mode_metrics = mode_metrics
        self.matcher = matcher
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params if scheduler_params else {"milestones": []}
        self.reference_band = reference_band

        self.initialize_networks()

    def initialize_networks(self):
        assert self.tile_slen in {2, 4}, "tile_slen must be 2 or 4"
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        num_features = 256
        self.features_net = nn.Sequential(
            FeaturesNet(
                n_bands=len(self.survey_bands),
                ch_per_band=ch_per_band,
                num_features=num_features,
                double_downsample=(self.tile_slen == 4),
            ),
            ConvBlock(num_features, num_features, kernel_size=1, gn=True),
            C3(num_features, num_features, n=3, spatial=False, gn=True),
            ConvBlock(num_features, num_features, kernel_size=1, gn=True),
        )

        self.final_process = nn.Sequential(
            ConvBlock(num_features, num_features, kernel_size=1, gn=True),
            C3(num_features, num_features, n=3, spatial=False, gn=True),
            ConvBlock(num_features, num_features, kernel_size=1, gn=True),
            Detect(num_features, self.catalog_parser.n_params_per_source),
        )

        self.cat_encoder = nn.Sequential(
            ConvBlock(
                self.catalog_parser.n_params_per_source, num_features, kernel_size=1, gn=True
            ),
            C3(num_features, num_features, n=3, spatial=False, gn=True),
            ConvBlock(num_features, num_features, kernel_size=1, gn=True),
            C3(num_features, num_features, n=3, spatial=False, gn=True),
            ConvBlock(num_features, num_features, kernel_size=1, gn=True),
        )

        self.cat_decoder = nn.Sequential(
            ConvBlock(num_features, num_features, kernel_size=1, gn=True),
            C3(num_features, num_features, n=3, spatial=False, gn=True),
            ConvBlock(num_features, num_features, kernel_size=1, gn=True),
            Detect(num_features, self.catalog_parser.n_params_per_source),
        )

    def get_features(self, batch):
        assert batch["images"].size(2) % 16 == 0, "image dims must be multiples of 16"
        assert batch["images"].size(3) % 16 == 0, "image dims must be multiples of 16"

        input_lst = [inorm.get_input_tensor(batch) for inorm in self.image_normalizers]
        inputs = torch.cat(input_lst, dim=2)

        return self.features_net(inputs)

    def sample(self, batch):
        x_features = self.get_features(batch)
        sample_tensor = self.final_process(x_features).clamp(min=-1.0, max=1.0)
        return self.catalog_parser.decode(sample_tensor)

    def _compute_cur_batch_loss(self, batch):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        encoded_catalog_tensor = self.catalog_parser.encode(target_cat1)  # (b, h, w, k)
        x_features = self.get_features(batch)

        refer_x_features = self.cat_encoder(encoded_catalog_tensor.permute([0, 3, 1, 2]))
        assert refer_x_features.shape == x_features.shape
        source_loss_gating = (
            self.catalog_parser.get_loss_gating(target_cat1).sum(dim=-1, keepdim=True) > 1
        ).permute(
            [0, 3, 1, 2]
        )  # (b, 1, h, w)
        feature_group_num = 2
        assert x_features.shape[1] % feature_group_num == 0
        gated_x_features = rearrange(
            x_features * source_loss_gating, "b (x y) h w -> b x y h w", x=feature_group_num
        )
        gated_refer_x_features = rearrange(
            refer_x_features * source_loss_gating, "b (x y) h w -> b x y h w", x=feature_group_num
        )
        features_dot_prod = (gated_x_features * gated_refer_x_features).sum(dim=2)  # (b, x, h, w)
        features_norm_prod = (
            torch.norm(gated_x_features, dim=2) * torch.norm(gated_refer_x_features, dim=2) + 1e-5
        )  # (b, x, h, w)
        feature_loss = (
            torch.acos(torch.clamp(features_dot_prod / features_norm_prod, min=-1.0, max=1.0))
            / torch.pi
        )  # (b, x, h, w)
        feature_loss = torch.where(source_loss_gating, feature_loss, 0)
        assert (not torch.isnan(feature_loss).any()) and (not torch.isinf(feature_loss).any())
        feature_loss = feature_loss.sum() / source_loss_gating.sum()

        output_catalog_tensor = self.final_process(x_features).clamp(min=-1.0, max=1.0)
        assert encoded_catalog_tensor.shape == output_catalog_tensor.shape

        refer_output_catalog_tensor = self.cat_decoder(refer_x_features).clamp(min=-1.0, max=1.0)
        assert encoded_catalog_tensor.shape == refer_output_catalog_tensor.shape

        loss = (output_catalog_tensor - encoded_catalog_tensor) ** 2  # (b, h, w, k)
        refer_loss = (refer_output_catalog_tensor - encoded_catalog_tensor) ** 2

        return (
            self.catalog_parser.gating_loss(loss, target_cat1),
            self.catalog_parser.gating_loss(refer_loss, target_cat1),
            feature_loss,
            self.catalog_parser.get_loss_gating(target_cat1),
        )

    def _compute_loss(self, batch, logging_name):
        loss, refer_loss, feature_loss, loss_gating = self._compute_cur_batch_loss(batch)
        factored_loss = self.catalog_parser.factor_param_tensor(loss)
        factored_refer_loss = self.catalog_parser.factor_param_tensor(refer_loss)
        factored_loss_gating = self.catalog_parser.factor_param_tensor(loss_gating)
        batch_size = batch["images"].size(0)

        sub_loss = []
        for l, lg in zip(factored_loss, factored_loss_gating, strict=True):
            sub_loss.append(l.sum() / lg.sum() if lg.sum() > 0 else l.sum())
        output_loss = torch.cat([sl.unsqueeze(0) for sl in sub_loss]).mean()

        self.log(f"{logging_name}/_loss", output_loss, batch_size=batch_size, sync_dist=True)
        for sl, f in zip(sub_loss, self.catalog_parser.factors, strict=True):
            self.log(f"{logging_name}/_loss_{f.name}", sl, batch_size=batch_size, sync_dist=True)

        sub_refer_loss = []
        for l, lg in zip(factored_refer_loss, factored_loss_gating, strict=True):
            sub_refer_loss.append(l.sum() / lg.sum() if lg.sum() > 0 else l.sum())
        refer_output_loss = torch.cat([sl.unsqueeze(0) for sl in sub_refer_loss]).mean()

        self.log(
            f"{logging_name}/_refer_loss", refer_output_loss, batch_size=batch_size, sync_dist=True
        )
        for sl, f in zip(sub_refer_loss, self.catalog_parser.factors, strict=True):
            self.log(
                f"{logging_name}/_refer_loss_{f.name}", sl, batch_size=batch_size, sync_dist=True
            )

        self.log(
            f"{logging_name}/_feature_loss", feature_loss, batch_size=batch_size, sync_dist=True
        )

        return output_loss + refer_output_loss + feature_loss

    def on_fit_start(self):
        GlobalEnv.current_encoder_epoch = self.current_epoch

    def on_train_epoch_start(self):
        GlobalEnv.current_encoder_epoch = self.current_epoch

    def training_step(self, batch, batch_idx):
        """Training step (pytorch lightning)."""
        return self._compute_loss(batch, "train")

    def update_metrics(self, batch, batch_idx):
        target_tile_cat = TileCatalog(batch["tile_catalog"])
        target_cat = target_tile_cat.to_full_catalog(self.tile_slen)

        mode_tile_cat = self.sample(batch)
        mode_cat = mode_tile_cat.to_full_catalog(self.tile_slen)
        mode_matching = self.matcher.match_catalogs(target_cat, mode_cat)
        self.mode_metrics.update(target_cat, mode_cat, mode_matching)

    def validation_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        self._compute_loss(batch, "val")
        self.update_metrics(batch, batch_idx)

    def report_metrics(self, metrics, logging_name, show_epoch=False):
        for k, v in metrics.compute().items():
            self.log(f"{logging_name}/{k}", v, sync_dist=True)

        for metric_name, metric in metrics.items():
            if hasattr(metric, "plot"):  # noqa: WPS421
                try:
                    plot_or_none = metric.plot()
                except NotImplementedError:
                    continue
                name = f"Epoch:{self.current_epoch}" if show_epoch else ""
                name += f"/{logging_name} {metric_name}"
                if self.logger and plot_or_none:
                    fig, _axes = plot_or_none
                    self.logger.experiment.add_figure(name, fig)

    def on_validation_epoch_end(self):
        self.report_metrics(self.mode_metrics, "val/mode", show_epoch=True)
        self.mode_metrics.reset()

    def test_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        self._compute_loss(batch, "test")
        self.update_metrics(batch, batch_idx)

    def on_test_epoch_end(self):
        # note: metrics are not reset here, to give notebooks access to them
        self.report_metrics(self.mode_metrics, "test/mode", show_epoch=False)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Pytorch lightning method."""
        with torch.no_grad():
            return self.sample(batch)

    def configure_optimizers(self):
        """Configure optimizers for training (pytorch lightning)."""
        optimizer = Adam(self.parameters(), **self.optimizer_params)
        scheduler = MultiStepLR(optimizer, **self.scheduler_params)
        return [optimizer], [scheduler]
