from torch import nn

from bliss.encoder.convnets import FeaturesNet
from bliss.encoder.convnet_layers import C3, ConvBlock

from case_studies.dc2_diffusion.utils.encoder import DiffusionEncoder
from case_studies.dc2_diffusion.utils.detection_net import SimpleDetectionNet, DetectionNet
from case_studies.dc2_diffusion.utils.diffusion import DiffusionModel

class SlimDiffusionEncoder(DiffusionEncoder):
     def initialize_networks(self):
        # assert self.tile_slen in {2, 4}, "tile_slen must be 2 or 4"
        # ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        # fn_num_features = 256
        # num_features = fn_num_features // 2
        # self.features_net = nn.Sequential(
        #     FeaturesNet(
        #         n_bands=len(self.survey_bands),
        #         ch_per_band=ch_per_band,
        #         num_features=fn_num_features,
        #         double_downsample=(self.tile_slen == 4),
        #     ),
        #     ConvBlock(fn_num_features, num_features, kernel_size=1, gn=True),
        #     C3(num_features, num_features, n=3, spatial=False, gn=True),
        #     ConvBlock(num_features, num_features, kernel_size=1, gn=True),
        # )

        # self.detection_net = DetectionNet(
        #     xt_in_ch=self.catalog_parser.n_params_per_source,
        #     xt_out_ch=32,
        #     extracted_feats_ch=num_features,
        #     use_self_cond=self.ddim_self_cond,
        # )
        # self.detection_diffusion = DiffusionModel(
        #     model=self.detection_net,
        #     target_size=(
        #         self.image_size[0] // self.tile_slen,
        #         self.image_size[1] // self.tile_slen,
        #         self.catalog_parser.n_params_per_source,
        #     ),
        #     catalog_parser=self.catalog_parser,
        #     ddim_steps=self.ddim_steps,
        #     objective=self.ddim_objective,
        #     beta_schedule=self.ddim_beta_schedule,
        #     self_condition=self.ddim_self_cond,
        # )

        assert self.tile_slen in {2, 4}, "tile_slen must be 2 or 4"
        assert not self.ddim_self_cond

        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        num_features = 128
        self.features_net = FeaturesNet(
                n_bands=len(self.survey_bands),
                ch_per_band=ch_per_band,
                num_features=num_features,
                double_downsample=(self.tile_slen == 4),
        )

        self.detection_net = SimpleDetectionNet(
            xt_in_ch=self.catalog_parser.n_params_per_source,
            xt_out_ch=32,
            extracted_feats_ch=num_features
        )
        self.detection_diffusion = DiffusionModel(
            model=self.detection_net,
            target_size=(
                self.image_size[0] // self.tile_slen,
                self.image_size[1] // self.tile_slen,
                self.catalog_parser.n_params_per_source,
            ),
            catalog_parser=self.catalog_parser,
            ddim_steps=self.ddim_steps,
            objective=self.ddim_objective,
            beta_schedule=self.ddim_beta_schedule,
            self_condition=self.ddim_self_cond,
            correct_bits=self.correct_bits,
            empty_tile_random_noise=self.empty_tile_random_noise,
            add_fake_tiles=self.add_fake_tiles,
        )