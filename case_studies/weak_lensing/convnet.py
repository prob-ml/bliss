import math

from torch import nn

from case_studies.weak_lensing.convnet_layers import Map, RN2Block


class WeakLensingNet(nn.Module):
    def __init__(
        self,
        n_bands,
        ch_per_band,
        n_pixels_per_side,
        n_tiles_per_side,
        ch_init,
        ch_max,
        n_var_params,
        initial_downsample,
        more_up_layers,
        num_bottleneck_layers,
    ):
        super().__init__()

        ch_final = 2 ** math.ceil(math.log2(n_var_params))

        n_image_downsamples = int(math.log2(n_pixels_per_side)) - int(math.log2(n_tiles_per_side))

        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, ch_init, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.GroupNorm(num_groups=32, num_channels=ch_init),
            nn.SiLU(),
        )

        layers = self._make_layers(
            ch_init,
            ch_max,
            ch_final,
            n_image_downsamples,
            initial_downsample,
            more_up_layers,
            num_bottleneck_layers,
        )

        self.resnet_blocks = nn.ModuleList(layers)

        self.final_layer = Map(ch_final, n_var_params)

    def _make_layers(
        self,
        ch_init,
        ch_max,
        ch_final,
        n_image_downsamples,
        initial_downsample,
        more_up_layers,
        num_bottleneck_layers,
    ):
        layers = [RN2Block(ch_init, ch_init, stride=2 if initial_downsample else 1)]
        cur_image_downsamples = int(initial_downsample)
        ch_current = ch_init

        # up from ch_init to ch_max
        while ch_current < ch_max and cur_image_downsamples < n_image_downsamples:
            ch_prev = ch_current
            ch_current = min(ch_prev * 2, ch_max)
            stride = 2 if cur_image_downsamples < 3 else 1

            if more_up_layers:
                layers.append(RN2Block(ch_prev, ch_prev))
            layers.append(RN2Block(ch_prev, ch_current, stride=stride))
            cur_image_downsamples += stride == 2
            if more_up_layers:
                layers.append(RN2Block(ch_current, ch_current))

        # additional bottleneck layers
        for _ in range(num_bottleneck_layers):
            layers.append(RN2Block(ch_current, ch_current))

        # down from ch_max to ch_final
        while ch_current > ch_final or cur_image_downsamples < n_image_downsamples:
            ch_prev = ch_current
            ch_current = max(ch_prev // 2, ch_final)
            stride = 2 if cur_image_downsamples < n_image_downsamples else 1
            layers.append(RN2Block(ch_prev, ch_current, stride=stride))
            cur_image_downsamples += stride == 2

        return layers

    def forward(self, x):
        x = self.preprocess3d(x).squeeze(2)

        for block in self.resnet_blocks:
            x = block(x)

        x = self.final_layer(x)

        return x
