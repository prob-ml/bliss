import torch
from torch import nn

from bliss.encoder.convnet_layers import C3, Detect


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        num_groups = min([32, out_channels // 4])
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.gn(self.conv(x)))


class GalaxyClusterCatalogNet(nn.Module):
    def __init__(self, num_features, out_channels):
        super().__init__()

        # initalization for detection head
        context_channels_in = 6
        context_channels_out = 128
        self.context_net = nn.Sequential(
            ConvBlock(context_channels_in, context_channels_out),
            ConvBlock(context_channels_out, context_channels_out),
            C3(context_channels_out, context_channels_out, n=4),
            ConvBlock(context_channels_out, context_channels_out),
        )
        n_hidden_ch = 256
        self.detection_net = nn.Sequential(
            ConvBlock(num_features + context_channels_out, n_hidden_ch),
            ConvBlock(n_hidden_ch, n_hidden_ch),
            C3(n_hidden_ch, n_hidden_ch, n=4),
            ConvBlock(n_hidden_ch, n_hidden_ch),
            Detect(n_hidden_ch, out_channels),
        )

    def forward(self, x_features, context):
        x_context = self.context_net(context)
        x = torch.cat((x_features, x_context), dim=1)
        return self.detection_net(x)


class GalaxyClusterFeaturesNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, num_features, tile_slen=2, downsample_at_front=False):
        super().__init__()
        nch_hidden = 64
        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, nch_hidden, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.GroupNorm(num_groups=32, num_channels=nch_hidden),
            nn.SiLU(),
        )

        log_tile_size = torch.log2(torch.tensor(tile_slen))
        num_total_downsample = int(torch.round(log_tile_size)) - 1
        num_downsample4, num_downsample2 = divmod(num_total_downsample, 2)
        self.backbone = nn.ModuleList()

        if downsample_at_front:
            for _ in range(num_downsample4):
                self.backbone.append(
                    ConvBlock(nch_hidden, 2 * nch_hidden, kernel_size=5, stride=4, padding=2)
                )
                nch_hidden *= 2
            if num_downsample2:
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
            for _ in range(num_downsample4):
                self.backbone.append(
                    ConvBlock(num_channels, 2 * num_channels, kernel_size=5, stride=4, padding=2)
                )
                num_channels *= 2
            if num_downsample2:
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


class GalaxyClusterContextNet(nn.Module):
    def __init__(self, num_features, out_channels):
        super().__init__()

        context_dim = 64
        self.encode_context = nn.Sequential(
            ConvBlock(2, 64),
            ConvBlock(64, 64),
            ConvBlock(64, context_dim),
        )
        self.merge = ConvBlock(num_features + context_dim, num_features)
        self.catalog_net = GalaxyClusterCatalogNet(num_features, out_channels)

    def forward(self, x_features, context):
        x_context = self.encode_context(context)
        x = torch.cat((x_features, x_context), dim=1)
        x = self.merge(x)
        return self.catalog_net(x)
