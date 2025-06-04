import math

import torch
from torch import nn

from bliss.encoder.convnet_layers import Detect
from case_studies.weak_lensing.convnet_layers import RN2Block


class WeakLensingNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, n_pixels_per_side, n_tiles_per_side, ch_init, ch_max, n_var_params):
        super().__init__()

        ch_final = 2 ** math.ceil(math.log2(n_var_params))
        
        n_image_downsamples = int(math.log2(n_pixels_per_side)) - int(math.log2(n_tiles_per_side))
        cur_image_downsamples = 0
        
        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, ch_init, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.GroupNorm(
                num_groups=32, num_channels=ch_init
            ),
            nn.SiLU(),
        )
        
        layers = [RN2Block(ch_init, ch_init)]
        
        ch_current = ch_init
        
        # up from ch_init to ch_max
        while ch_current < ch_max and cur_image_downsamples < n_image_downsamples:
            ch_prev = ch_current
            ch_current = min(ch_prev * 2, ch_max)
            
            stride = 2 if cur_image_downsamples < 3 else 1
            
            layers.append(RN2Block(ch_prev, ch_prev))
            
            layers.append(RN2Block(ch_prev, ch_current, stride=stride))
            cur_image_downsamples += 1 * (stride == 2)
            
            layers.append(RN2Block(ch_current, ch_current))
        
        # additional bottleneck layers
        layers.append(RN2Block(ch_current, ch_current))
        layers.append(RN2Block(ch_current, ch_current))
        layers.append(RN2Block(ch_current, ch_current))
        
        # down from ch_max to ch_final
        while ch_current > ch_final or cur_image_downsamples < n_image_downsamples:
            ch_prev = ch_current
            ch_current = max(ch_prev // 2, ch_final)
            
            stride = 2 if cur_image_downsamples < n_image_downsamples else 1
            
            layers.append(RN2Block(ch_prev, ch_current, stride=stride))
            cur_image_downsamples += 1 * (stride == 2)
        
        self.net = nn.ModuleList(layers)
        self.detect = Detect(ch_final, n_var_params)

    def forward(self, x):
        x = self.preprocess3d(x).squeeze(2)
        
        for i, layer in enumerate(self.net):
            x = layer(x)
        
        x = self.detect(x)

        return x


class WeakLensingCatalogNet(nn.Module):  # TODO: get the dimensions down to n_tiles
    def __init__(self, in_channels, out_channels, n_tiles):
        super().__init__()

        net_layers = []

        n_blocks2 = int(math.log2(in_channels)) - int(math.ceil(math.log2(out_channels)))
        last_out_dim = -1
        for i in range(n_blocks2):
            in_dim = in_channels // (2**i)
            out_dim = in_dim // 2
            if i < ((n_blocks2 + 4) // 2):
                net_layers.append(RN2Block(in_dim, out_dim, stride=2))
            else:
                net_layers.append(RN2Block(in_dim, out_dim))
            last_out_dim = out_dim

        # Final detection layer to reduce channels
        self.detect = Detect(last_out_dim, out_channels)
        self.net = nn.ModuleList(net_layers)

    def forward(self, x):
        for _i, m in enumerate(self.net):
            x = m(x)

        # Final detection layer
        x = self.detect(x)

        return x  # noqa: WPS331
