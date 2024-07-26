import torch
from torch import nn

from case_studies.dc2_cataloging.utils.conv_layers import (
    C3,
    PSA,
    SPPF,
    C2f,
    C2fCIB,
    ConvBlock,
    Detect,
    SCDown,
)


class SimpleCatalogNet(nn.Module):
    def __init__(self, num_features, out_channels):
        super().__init__()

        n_hidden_ch = 256
        self.detection_net = nn.Sequential(
            ConvBlock(num_features, n_hidden_ch),
            ConvBlock(n_hidden_ch, n_hidden_ch),
            C3(n_hidden_ch, n_hidden_ch, n=4),
            ConvBlock(n_hidden_ch, n_hidden_ch),
            Detect(n_hidden_ch, out_channels),
        )

    def forward(self, x_features):
        return self.detection_net(x_features)


class SimpleFeaturesNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, num_features, double_downsample=False):
        super().__init__()
        assert ch_per_band == 1

        nch_hidden = 64
        self.preprocess = ConvBlock(n_bands, nch_hidden, kernel_size=5)
        self.backbone = nn.Sequential(
            ConvBlock(nch_hidden, 64, kernel_size=5, padding=2),
            nn.Sequential(*[ConvBlock(64, 64, kernel_size=5, padding=2) for _ in range(4)]),
            ConvBlock(64, 128, stride=2),
            nn.Sequential(*[ConvBlock(128, 128) for _ in range(5)]),
            ConvBlock(128, 256, stride=(2 if double_downsample else 1)),  # 4
        )
        u_net_layers = [
            C3(256, 256, n=6),  # 0
            ConvBlock(256, 512, stride=2),
            C3(512, 512, n=3, shortcut=False),
            ConvBlock(512, 256, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 4
            C3(768, num_features, n=3, shortcut=False),
        ]
        self.u_net = nn.ModuleList(u_net_layers)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.backbone(x)

        save_lst = [x]
        for i, m in enumerate(self.u_net):
            x = m(x)
            if i in {0, 4}:
                save_lst.append(x)
            if i == 4:
                x = torch.cat(save_lst, dim=1)

        return x


class V10FeaturesNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, num_features, double_downsample=False):
        super().__init__()
        assert ch_per_band == 1

        nch_hidden = 64
        self.preprocess = ConvBlock(n_bands, nch_hidden, kernel_size=5)
        self.backbone = nn.Sequential(
            ConvBlock(nch_hidden, 64, kernel_size=5, padding=2),
            nn.Sequential(*[C2f(64, 64, n=1, shortcut=True) for _ in range(2)]),
            ConvBlock(64, 128, stride=2),
            nn.Sequential(*[C2f(128, 128, n=2, shortcut=True) for _ in range(4)]),
            ConvBlock(128, 256, stride=(2 if double_downsample else 1)),  # 4
            nn.Sequential(*[C2f(256, 256, n=2, shortcut=True) for _ in range(4)]),
        )
        u_net_layers = [
            C2f(256, 256, n=2, shortcut=True),  # 0
            SCDown(256, 512, k=3, s=2),
            C2fCIB(512, 512, n=1, shortcut=True, lk=True),
            SPPF(512, 512),
            PSA(512, 512),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 4
            C2f(1024, num_features, n=2, shortcut=False),
        ]
        self.u_net = nn.ModuleList(u_net_layers)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.backbone(x)

        save_lst = [x]
        for i, m in enumerate(self.u_net):
            x = m(x)
            if i in {0, 4}:
                save_lst.append(x)
            if i == 4:
                x = torch.cat(save_lst, dim=1)
        return x


class V10CatalogNet(nn.Module):
    def __init__(self, num_features, out_channels):
        super().__init__()

        n_hidden_ch = 256
        self.detection_net = nn.Sequential(
            ConvBlock(num_features, n_hidden_ch),
            ConvBlock(n_hidden_ch, n_hidden_ch),
            C2f(n_hidden_ch, n_hidden_ch, n=4, shortcut=True),
            ConvBlock(n_hidden_ch, n_hidden_ch),
            Detect(n_hidden_ch, out_channels),
        )

    def forward(self, x_features):
        return self.detection_net(x_features)
