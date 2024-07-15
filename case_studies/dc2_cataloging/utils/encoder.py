import copy
import warnings

import torch
from einops import rearrange

from bliss.catalog import TileCatalog
from bliss.encoder.convnets import CatalogNet, FeaturesNet
from bliss.encoder.encoder import Encoder
from case_studies.dc2_cataloging.utils.dynamic_asinh_convnet import (
    FeaturesNet as DynamicAsinhFeaturesNet,
)
from case_studies.dc2_cataloging.utils.image_normalizer import DynamicAsinhImageNormalizer
from case_studies.dc2_cataloging.utils.multi_detect_convnet import (
    CatalogNet as MultiDetectCatalogNet,
)
from case_studies.dc2_cataloging.utils.variational_dist import MultiVariationalDist


class EncoderForDynamicAsinh(Encoder):
    def initialize_networks(self):
        assert isinstance(
            self.image_normalizer, DynamicAsinhImageNormalizer
        ), "wrong image normalizer"
        assert self.tile_slen in {2, 4}, "tile_slen must be 2 or 4"
        ch_per_band = self.image_normalizer.num_channels_per_band()
        num_features = 256
        self.features_net = DynamicAsinhFeaturesNet(
            6,
            ch_per_band,
            num_features,
            double_downsample=(self.tile_slen == 4),
        )
        self.catalog_net = CatalogNet(
            num_features=256,
            out_channels=self.var_dist.n_params_per_source,
        )


class EncoderAddingSourceMask(Encoder):
    @classmethod
    def _add_source_mask(cls, ori_tile_cat: TileCatalog):
        d = copy.copy(ori_tile_cat.data)
        on_mask = rearrange(ori_tile_cat.is_on_mask, "b nth ntw s -> b nth ntw s 1")
        on_mask_count = on_mask.sum(dim=(-2, -1))
        d["one_source_mask"] = rearrange(on_mask_count == 1, "b nth ntw -> b nth ntw 1 1") & on_mask
        d["two_sources_mask"] = (
            rearrange(on_mask_count == 2, "b nth ntw -> b nth ntw 1 1") & on_mask
        )
        d["more_than_two_sources_mask"] = (
            rearrange(on_mask_count > 2, "b nth ntw -> b nth ntw 1 1") & on_mask
        )

        return TileCatalog(ori_tile_cat.tile_slen, d)

    def sample(self, batch, use_mode=True):
        tile_cat = super().sample(batch, use_mode)
        return self._add_source_mask(tile_cat)


class MultiDetectEncoder(EncoderAddingSourceMask):
    def initialize_networks(self):
        assert isinstance(
            self.var_dist, MultiVariationalDist
        ), "var_dist should be MultiVariationalDist"
        assert self.tile_slen in {2, 4}, "tile_slen must be 2 or 4"
        assert not self.use_double_detect, "we disable double detect"
        assert not self.use_checkerboard, "we disable checkerboard"

        num_features = 256
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        self.features_net = FeaturesNet(
            n_bands=len(self.survey_bands),
            ch_per_band=ch_per_band,
            num_features=num_features,
            double_downsample=(self.tile_slen == 4),
        )
        self.catalog_net = MultiDetectCatalogNet(
            num_features=num_features,
            out_channels=self.var_dist.n_params_per_source,
        )

    def make_context(self, history_cat, history_mask, detection2=False):
        raise NotImplementedError()

    def sample(self, batch, use_mode=True):
        x_features = self.get_features(batch)
        x_cat = self.catalog_net(x_features)
        est_cat = self.var_dist.sample(x_cat, use_mode=use_mode)
        return self._add_source_mask(est_cat)

    def _compute_loss(self, batch, logging_name):
        batch_size = batch["images"].shape[0]

        # filter out undetectable sources and split catalog by flux
        target_cat = TileCatalog(self.tile_slen, batch["tile_catalog"])

        # TODO: move tile_cat filtering to the dataloader and from the encoder remove
        # `reference_band`, `min_flux_for_loss`, and `min_flux_for_metrics`
        # (metrics can be computed with the original full catalog if necessary)
        target_cat = target_cat.filter_by_flux(
            min_flux=self.min_flux_for_loss,
            band=self.reference_band,
        )
        # TODO: don't order the light sources by brightness; softmax instead
        target_cat = target_cat.get_brightest_sources_per_tile(
            top_k=self.var_dist.repeat_times,
            band=self.reference_band,
        )

        x_features = self.get_features(batch)
        x_cat = self.catalog_net(x_features)
        loss = self.var_dist.compute_nll(x_cat, target_cat)

        nan_mask = torch.isnan(loss)
        if nan_mask.any():
            loss = loss[~nan_mask]
            msg = f"NaN detected in loss. Ignored {nan_mask.sum().item()} NaN values."
            warnings.warn(msg)

        # could normalize by the number of tile predictions, rather than number of tiles
        loss = loss.sum() / loss.numel()
        self.log(f"{logging_name}/_loss", loss, batch_size=batch_size, sync_dist=True)

        return loss
