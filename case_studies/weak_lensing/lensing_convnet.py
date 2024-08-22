from torch import nn

from bliss.encoder.convnet_layers import C3, ConvBlock, Detect


class WeakLensingFeaturesNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, num_features, tile_slen):
        super().__init__()

        nch_hidden = 64
        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, nch_hidden, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.GroupNorm(
                num_groups=32, num_channels=nch_hidden
            ),  # sqrt of num channels, get rid of it, even shallower
            nn.SiLU(),
        )

        # TODO: adaptive downsample
        self.n_downsample = 1

        module_list = []

        for _ in range(self.n_downsample):
            module_list.append(
                ConvBlock(nch_hidden, 2 * nch_hidden, kernel_size=5, stride=2, padding=2)
            )
            nch_hidden *= 2

        module_list.extend(
            [
                ConvBlock(nch_hidden, 64, kernel_size=5, padding=2),
                nn.Sequential(*[ConvBlock(64, 64, kernel_size=5, padding=2) for _ in range(2)]),
                ConvBlock(64, 128, stride=2),
                nn.Sequential(*[ConvBlock(128, 128) for _ in range(2)]),
                ConvBlock(128, num_features, stride=1),
            ]
        )  # 4

        self.net = nn.ModuleList(module_list)

    def forward(self, x):
        x = self.preprocess3d(x).squeeze(2)
        for _i, m in enumerate(self.net):
            x = m(x)

        return x


class WeakLensingCatalogNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        net_layers = [
            C3(in_channels, 256, n=1, shortcut=True),  # 0
            ConvBlock(256, 512, stride=2),
            C3(512, 256, n=1, shortcut=True),  # true shortcut for skip connection
            ConvBlock(
                in_channels=256, out_channels=256, kernel_size=3, stride=8, padding=1
            ),  # (1, 256, 128, 128)
            ConvBlock(
                in_channels=256, out_channels=256, kernel_size=3, stride=4, padding=1
            ),  # (1, 256, 8, 8)
            Detect(256, out_channels),
        ]
        self.net = nn.ModuleList(net_layers)

    def forward(self, x):
        for _i, m in enumerate(self.net):
            x = m(x)
        return x
