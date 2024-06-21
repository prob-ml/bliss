from einops import rearrange

from bliss.encoder.convnet import CatalogNet, ContextNet
from bliss.encoder.encoder import Encoder
from case_studies.dc2_cataloging.utils.dynamic_asinh_convnet import FeaturesNet
from case_studies.dc2_cataloging.utils.image_normalizer import DynamicAsinhImageNormalizer


class EncoderForDynamicAsinh(Encoder):
    def initialize_networks(self):
        assert isinstance(
            self.image_normalizer, DynamicAsinhImageNormalizer
        ), "wrong image normalizer"
        assert self.tile_slen in {2, 4}, "tile_slen must be 2 or 4"
        ch_per_band = self.image_normalizer.num_channels_per_band()
        num_features = 256
        self.features_net = FeaturesNet(
            len(self.image_normalizer.bands),
            ch_per_band,
            num_features,
            double_downsample=(self.tile_slen == 4),
        )
        n_params_per_source = self.var_dist.n_params_per_source
        self.marginal_net = CatalogNet(num_features, n_params_per_source)
        self.checkerboard_net = ContextNet(num_features, n_params_per_source)
        if self.double_detect:
            self.second_net = CatalogNet(num_features, n_params_per_source)


class EncoderAddingSourceMask(Encoder):
    def sample(self, batch, use_mode=True):
        tile_cat = super().sample(batch, use_mode)

        on_mask = rearrange(tile_cat.is_on_mask, "b nth ntw s -> b nth ntw s 1")
        on_mask_count = on_mask.sum(dim=(-2, -1))
        tile_cat["one_source_mask"] = (
            rearrange(on_mask_count == 1, "b nth ntw -> b nth ntw 1 1") & on_mask
        )
        tile_cat["two_sources_mask"] = (
            rearrange(on_mask_count == 2, "b nth ntw -> b nth ntw 1 1") & on_mask
        )
        tile_cat["more_than_two_sources_mask"] = (
            rearrange(on_mask_count > 2, "b nth ntw -> b nth ntw 1 1") & on_mask
        )

        return tile_cat
