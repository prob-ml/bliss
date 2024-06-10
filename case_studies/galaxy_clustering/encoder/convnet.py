import torch
from torch import nn

from bliss.encoder.convnet import ConvBlock


class GalaxyClusterFeaturesNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, num_features, tile_slen=2, downsample_at_front=False):
        super().__init__()
        nch_hidden = 64
        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, nch_hidden, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(nch_hidden),
            nn.SiLU(),
        )

        log_tile_size = torch.log2(torch.tensor(tile_slen))
        num_downsample = int(torch.round(log_tile_size)) - 1
        self.backbone = nn.ModuleList()

        if downsample_at_front:
            for _ in range(num_downsample):
                self.backbone.append(
                    ConvBlock(nch_hidden, 2 * nch_hidden, kernel_size=5, stride=2, padding=2)
                )
                nch_hidden *= 2
            self.backbone.append(ConvBlock(nch_hidden, 64, kernel_size=5, padding=2))
            self.backbone.append(
                nn.Sequential(*[ConvBlock(64, 64, kernel_size=5, padding=2) for _ in range(4)])
            )
            self.backbone.append(ConvBlock(64, 128, stride=2))
            self.backbone.append(nn.Sequential(*[ConvBlock(128, 128) for _ in range(5)]))
            self.backbone.append(ConvBlock(128, num_features, stride=1))
        else:
            self.backbone.append(ConvBlock(nch_hidden, 64, kernel_size=5, padding=2))
            self.backbone.append(
                nn.Sequential(*[ConvBlock(64, 64, kernel_size=5, padding=2) for _ in range(4)])
            )
            self.backbone.append(ConvBlock(64, 128, stride=2))
            self.backbone.append(nn.Sequential(*[ConvBlock(128, 128) for _ in range(5)]))
            num_channels = 128
            for _ in range(num_downsample):
                self.backbone.append(
                    ConvBlock(num_channels, 2 * num_channels, kernel_size=5, stride=2, padding=2)
                )
                num_channels *= 2
            self.backbone.append(ConvBlock(num_channels, num_features))

    def forward(self, x):
        x = self.preprocess3d(x).squeeze(2)
        for layer in self.backbone:
            x = layer(x)
        return x
