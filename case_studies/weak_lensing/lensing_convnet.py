import math

from torch import nn

from bliss.encoder.convnet_layers import C3, Bottleneck, ConvBlock, Detect
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

        n_blocks_2 = int(math.log2(num_features)) - int(math.log2(nch_hidden))
        module_list = [RN2Block(nch_hidden, nch_hidden), RN2Block(nch_hidden, nch_hidden)]
        for i in range(n_blocks_2):
            in_dim = nch_hidden * (2**i)
            out_dim = in_dim * 2
            if i < (n_blocks_2 // 2):
                module_list.append(RN2Block(in_dim, out_dim, stride=2))
            else:
                module_list.append(RN2Block(in_dim, out_dim))
            module_list.append(RN2Block(out_dim, out_dim))

        self.net = nn.ModuleList(module_list)

    def forward(self, x):
        x = self.preprocess3d(x).squeeze(2)
        # print("after p3d", x.shape)
        for idx, layer in enumerate(self.net):
            x = layer(x)
            # print("after net layer", idx, x.shape)
        return x


class WeakLensingCatalogNet(nn.Module):  # TODO: get the dimensions down to n_tiles
    def __init__(self, in_channels, out_channels, n_tiles):
        super().__init__()

        net_layers = []

        n_blocks_2 = int(math.log2(in_channels)) - int(math.ceil(math.log2(out_channels)))
        last_out_dim = -1
        # print("n blocks 2", n_blocks_2)
        for i in range(n_blocks_2):
            in_dim = in_channels // (2**i)
            out_dim = in_dim // 2
            net_layers.append(RN2Block(in_dim, in_dim))
            net_layers.append(RN2Block(in_dim, out_dim, stride=2))
            last_out_dim = out_dim

        # Final detection layer to reduce channels
        self.detect = Detect(last_out_dim, out_channels)
        self.net = nn.ModuleList(net_layers)

    def forward(self, x):
        for i, m in enumerate(self.net):
            x = m(x)
            # print("after catalog layer", i, x.shape)

        # Final detection layer
        x = self.detect(x)  # Output shape: [batch_size, 2, 4, 4]

        return x
