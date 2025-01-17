from bliss.encoder.convnet import CatalogNet, ContextNet
from bliss.encoder.encoder import Encoder
from case_studies.dc2_cataloging.utils.convnet import FeaturesNet
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
