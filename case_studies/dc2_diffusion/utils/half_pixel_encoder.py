from torch import nn
# from einops.layers.torch import Rearrange

from bliss.encoder.convnet_layers import C3, ConvBlock
from case_studies.dc2_diffusion.utils.half_pixel_features_net import FeaturesNet
from case_studies.dc2_diffusion.utils.detection_net import DetectionNet
from case_studies.dc2_diffusion.utils.diffusion import DiffusionModel
from case_studies.dc2_diffusion.utils.encoder import DiffusionEncoder


class HalfPixelDiffusionEncoder(DiffusionEncoder):
    def initialize_networks(self):
        assert self.tile_slen == 0.5
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        num_features = 64
        upscale_factor = 2
        self.features_net = nn.Sequential(
            FeaturesNet(
                n_bands=len(self.survey_bands),
                ch_per_band=ch_per_band,
                num_features=num_features,
            ),
            ConvBlock(num_features, num_features, kernel_size=1, gn=True),
            C3(num_features, num_features, n=3, spatial=False, gn=True),
            ConvBlock(num_features, num_features, kernel_size=1, gn=True),
            nn.PixelShuffle(upscale_factor=upscale_factor),
        )

        self.detection_net = DetectionNet(
            xt_in_ch=self.catalog_parser.n_params_per_source,
            xt_out_ch=4,
            extracted_feats_ch=num_features // (upscale_factor ** 2),
            use_self_cond=self.ddim_self_cond,
        )
        self.detection_diffusion = DiffusionModel(
            model=self.detection_net,
            target_size=(
                int(self.image_size[0] // self.tile_slen),
                int(self.image_size[1] // self.tile_slen),
                self.catalog_parser.n_params_per_source,
            ),
            ddim_steps=self.ddim_steps,
            objective=self.ddim_objective,
            beta_schedule=self.ddim_beta_schedule,
            self_condition=self.ddim_self_cond,
        )
