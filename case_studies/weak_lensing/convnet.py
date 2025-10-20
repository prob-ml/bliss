import math

from torch import nn

from case_studies.weak_lensing.convnet_layers import Map, ResNetBlock


class WeakLensingNet(nn.Module):
    def __init__(
        self,
        n_bands,
        ch_per_band,
        n_pixels_per_side,
        n_tiles_per_side,
        ch_init,
        ch_max,
        ch_final,
        initial_downsample,
        more_up_layers,
        num_bottleneck_layers,
        map_to_var_params=True,
        n_var_params=None,
    ):
        super().__init__()

        if n_var_params is not None:
            ch_final = max(ch_final, 2 ** math.ceil(math.log2(n_var_params)))

        res_midpoint = int(math.sqrt(n_pixels_per_side * n_tiles_per_side))

        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, ch_init, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.GroupNorm(num_groups=32, num_channels=ch_init),
            nn.SiLU(),
        )

        layers = self._make_layers(
            ch_init,
            ch_max,
            ch_final,
            n_pixels_per_side,
            res_midpoint,
            n_tiles_per_side,
            initial_downsample,
            more_up_layers,
            num_bottleneck_layers,
        )

        self.resnet_blocks = nn.ModuleList(layers)

        if map_to_var_params:
            self.final_layer = Map(ch_final, n_var_params)
        else:
            self.final_layer = None

    def _make_layers(
        self,
        ch_init,
        ch_max,
        ch_final,
        res_init,
        res_midpoint,
        res_final,
        initial_downsample,
        more_up_layers,
        num_bottleneck_layers,
    ):
        layers = [ResNetBlock(ch_init, ch_init, stride=2 if initial_downsample else 1)]
        if more_up_layers:
            layers.append(ResNetBlock(ch_init, ch_init))

        ch_current = ch_init
        res_current = res_init // 2 if initial_downsample else res_init

        # ch_init -> ch_max, res_init -> res_midpoint
        while ch_current < ch_max or res_current > res_midpoint:
            ch_prev = ch_current
            ch_current = min(ch_prev * 2, ch_max)

            stride = 2 if res_current > res_midpoint else 1
            res_prev = res_current
            res_current = max(res_prev // 2, res_midpoint)

            layers.append(ResNetBlock(ch_prev, ch_current, stride=stride))
            if more_up_layers:
                layers.append(ResNetBlock(ch_current, ch_current))

        # additional bottleneck layers
        for _ in range(num_bottleneck_layers):
            layers.append(ResNetBlock(ch_current, ch_current))

        # ch_max -> ch_final, res_midpoint -> res_final
        num_res_downsamples = int(math.log2(res_midpoint) - math.log2(res_final))
        num_ch_downsamples = int(math.log2(ch_max) - math.log2(ch_final))
        res_current = res_midpoint
        while num_res_downsamples > num_ch_downsamples:
            res_prev = res_current
            res_current = max(res_prev // 2, res_final)
            num_res_downsamples -= 1
            layers.append(ResNetBlock(ch_current, ch_current, stride=2))
        while ch_current > ch_final or res_current > res_final:
            ch_prev = ch_current
            ch_current = max(ch_prev // 2, ch_final)

            stride = 2 if res_current > res_final else 1
            res_prev = res_current
            res_current = max(res_prev // 2, res_final)

            layers.append(ResNetBlock(ch_prev, ch_current, stride=stride))

        return layers

    def forward(self, x):
        x = self.preprocess3d(x).squeeze(2)

        for block in self.resnet_blocks:
            x = block(x)

        if self.final_layer is not None:
            x = self.final_layer(x)

        return x
