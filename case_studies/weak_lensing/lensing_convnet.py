import math

from torch import nn

from bliss.encoder.convnet_layers import Detect
from case_studies.weak_lensing.lensing_convnet_layers import RN2Block


class WeakLensingFeaturesNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, num_features, tile_slen, nch_hidden):
        super().__init__()

        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, nch_hidden, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.GroupNorm(
                num_groups=32, num_channels=nch_hidden
            ),  # sqrt of num channels, get rid of it, even shallower
            nn.SiLU(),
        )

        n_blocks2 = int(math.log2(num_features)) - int(math.log2(nch_hidden))
        module_list = [RN2Block(nch_hidden, nch_hidden), RN2Block(nch_hidden, nch_hidden)]
        for i in range(n_blocks2):
            in_dim = nch_hidden * (2**i)
            out_dim = in_dim * 2

            module_list.append(RN2Block(in_dim, out_dim, stride=2))
            module_list.append(RN2Block(out_dim, out_dim))

        self.net = nn.ModuleList(module_list)

    def forward(self, x):
        x = self.preprocess3d(x).squeeze(2)
        for _idx, layer in enumerate(self.net):
            x = layer(x)
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
