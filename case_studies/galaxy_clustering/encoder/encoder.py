from typing import Optional

import torch
from torchmetrics import MetricCollection

from bliss.encoder.convnet import CatalogNet, ContextNet
from bliss.encoder.encoder import Encoder
from bliss.encoder.image_normalizer import ImageNormalizer
from bliss.encoder.metrics import CatalogMatcher
from bliss.encoder.variational_dist import VariationalDistSpec
from case_studies.galaxy_clustering.encoder.convnet import GalaxyClusterFeaturesNet


class GalaxyClusterEncoder(Encoder):
    def __init__(
        self,
        survey_bands: list,
        tile_slen: int,
        tiles_to_crop: int,
        image_normalizer: ImageNormalizer,
        vd_spec: VariationalDistSpec,
        metrics: MetricCollection,
        sample_image_renders: MetricCollection,
        matcher: CatalogMatcher,
        min_flux_threshold: float = 0,
        min_flux_threshold_during_test: float = 0,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
        do_data_augmentation: bool = False,
        compile_model: bool = False,
        double_detect: bool = False,
        use_checkerboard: bool = True,
        reference_band: int = 2,
        downsample_at_front: bool = True,
    ):
        """Initializes Encoder.

        Args:
            survey_bands: all band-pass filters available for this survey
            tile_slen: dimension in pixels of a square tile
            tiles_to_crop: margin of tiles not to use for computing loss
            image_normalizer: object that applies input transforms to images
            vd_spec: object that makes a variational distribution from raw convnet output
            sample_image_renders: for plotting relevant images (overlays, shear maps)
            metrics: for scoring predicted catalogs during training
            matcher: for matching predicted catalogs to ground truth catalogs
            min_flux_threshold: Sources with a lower flux will not be considered when computing loss
            min_flux_threshold_during_test: filter sources by flux during test
            optimizer_params: arguments passed to the Adam optimizer
            scheduler_params: arguments passed to the learning rate scheduler
            do_data_augmentation: used for determining whether or not do data augmentation
            compile_model: compile model for potential performance improvements
            double_detect: whether to make up to two detections per tile rather than one
            use_checkerboard: whether to use dependent tiling
            reference_band: band to use for filtering sources
            downsample_at_front: whether to downsample at front of network
        """

        self.downsample_at_front = downsample_at_front
        super().__init__(
            survey_bands,
            tile_slen,
            tiles_to_crop,
            image_normalizer,
            vd_spec,
            metrics,
            sample_image_renders,
            matcher,
            min_flux_threshold,
            min_flux_threshold_during_test,
            optimizer_params,
            scheduler_params,
            do_data_augmentation,
            compile_model,
            double_detect,
            use_checkerboard,
            reference_band,
        )

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
        n_params_per_source = self.vd_spec.n_params_per_source
        self.marginal_net = CatalogNet(num_features, n_params_per_source)
        self.checkerboard_net = ContextNet(num_features, n_params_per_source)
        if self.double_detect:
            self.second_net = CatalogNet(num_features, n_params_per_source)

        if self.compile_model:
            self.features_net = torch.compile(self.features_net)
            self.marginal_net = torch.compile(self.marginal_net)
            if self.use_checkerboard:
                self.checkerboard_net = torch.compile(self.checkerboard_net)
            if self.double_detect:
                self.second_net = torch.compile(self.second_net)
