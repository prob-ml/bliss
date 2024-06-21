from bliss.catalog import BaseTileCatalog
from bliss.encoder.convnet import ContextNet
from bliss.encoder.encoder import Encoder
from case_studies.galaxy_clustering.encoder.convnet import GalaxyClusterFeaturesNet


class GalaxyClusterEncoder(Encoder):
    def __init__(self, *args, downsample_at_front: bool = True, **kwargs):
        self.downsample_at_front = downsample_at_front
        super().__init__(*args, **kwargs)

    def initialize_networks(self):
        """Load the convolutional neural networks that map normalized images to catalog parameters.
        This method can be overridden to use different network architectures.
        `checkerboard_net` and `second_net` can be left as None if not needed.
        """
        power_of_two = (self.tile_slen != 0) & (self.tile_slen & (self.tile_slen - 1) == 0)
        assert power_of_two, "tile_slen must be a power of two"
        ch_per_band = self.image_normalizer.num_channels_per_band()
        num_features = 256
        self.features_net = GalaxyClusterFeaturesNet(
            len(self.image_normalizer.bands),
            ch_per_band,
            num_features,
            tile_slen=self.tile_slen,
            downsample_at_front=self.downsample_at_front,
        )
        n_params_per_source = self.var_dist.n_params_per_source
        self.context_net = ContextNet(num_features, n_params_per_source)

    def update_metrics(self, batch, batch_idx):
        target_cat = BaseTileCatalog(self.tile_slen, batch["tile_catalog"])
        target_cat = target_cat.symmetric_crop(self.tiles_to_crop)

        mode_cat = self.sample(batch, use_mode=True)
        self.metrics.update(target_cat, mode_cat)

        sample_cat = self.sample(batch, use_mode=False)
        self.sample_metrics.update(target_cat, sample_cat)

    def on_validation_epoch_end(self):
        self.report_metrics(self.metrics, "val/mode", show_epoch=True)
        self.report_metrics(self.sample_metrics, "val/sample", show_epoch=True)
