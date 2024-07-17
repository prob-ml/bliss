import torch
from torch import nn

from bliss.encoder.convnet_layers import C3, ConvBlock, Detect


class WeakLensingFeaturesNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, num_features, tile_slen):
        super().__init__()

        nch_hidden = 64
        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, nch_hidden, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.GroupNorm(num_groups=32, num_channels=nch_hidden),
            nn.SiLU(),
        )

        # downsample first for now
        n_downsample = int(torch.round(torch.log2(torch.tensor(tile_slen)))) - 1

        module_list = []

        for _ in range(n_downsample):
            module_list.append(
                ConvBlock(nch_hidden, 2 * nch_hidden, kernel_size=5, stride=2, padding=2)
            )
            nch_hidden *= 2

        module_list.extend(
            [
                ConvBlock(nch_hidden, 64, kernel_size=5, padding=2),
                nn.Sequential(*[ConvBlock(64, 64, kernel_size=5, padding=2) for _ in range(4)]),
                ConvBlock(64, 128, stride=2),
                nn.Sequential(*[ConvBlock(128, 128) for _ in range(5)]),
                ConvBlock(128, num_features, stride=1),
            ]
        )  # 4
        
        self.net = nn.ModuleList(module_list)

    def forward(self, x):
        x = self.preprocess3d(x).squeeze(2)
        for i, m in enumerate(self.net):
            x = m(x)

        return x


class WeakLensingCatalogNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        net_layers = [
            C3(in_channels, 256, n=6),  # 0
            ConvBlock(256, 512, stride=2),
            C3(512, 512, n=3, shortcut=False),
            ConvBlock(512, 256, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 4
            C3(768, 256, n=3, shortcut=False),
            Detect(256, out_channels),
        
        ]
        self.net = nn.ModuleList(net_layers)

    def forward(self, x):
        save_lst = [x]
        for i, m in enumerate(self.net):
            # print("at input layer", i, x.shape)
            x = m(x)
            # print("layer", i, x.shape)
            if i in {0, 4}:
                save_lst.append(x)
            if i == 4:
                x = torch.cat(save_lst, dim=1)
        return x
