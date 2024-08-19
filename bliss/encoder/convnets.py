import torch
from torch import nn

from bliss.encoder.convnet_layers import C3, ConvBlock, Detect


class FeaturesNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, num_features, double_downsample=False):
        super().__init__()

        nch_hidden = 64
        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, nch_hidden, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(nch_hidden),
            nn.SiLU(),
        )
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
        x = self.preprocess3d(x).squeeze(2)
        x = self.backbone(x)

        save_lst = [x]
        for i, m in enumerate(self.u_net):
            x = m(x)
            if i in {0, 4}:
                save_lst.append(x)
            if i == 4:
                x = torch.cat(save_lst, dim=1)

        return x


class CatalogNet(nn.Module):
    def __init__(self, num_features, out_channels):
        super().__init__()

        context_channels_in = 6
        context_channels_out = 128
        self.color_context_net = nn.Sequential(
            ConvBlock(context_channels_in, context_channels_out),
            ConvBlock(context_channels_out, context_channels_out, kernel_size=1, padding=0),
            C3(context_channels_out, context_channels_out, n=4),
            ConvBlock(context_channels_out, context_channels_out, kernel_size=1, padding=0),
        )

        self.local_context_net = nn.Sequential(
            ConvBlock(context_channels_in, context_channels_out, kernel_size=1, padding=0),
            ConvBlock(context_channels_out, context_channels_out, kernel_size=1, padding=0),
            C3(context_channels_out, context_channels_out, n=4),
            ConvBlock(context_channels_out, context_channels_out, kernel_size=1, padding=0),
        )

        n_hidden_ch = 256
        in_ch = num_features + 2 * context_channels_out
        self.detection_net = nn.Sequential(
            ConvBlock(in_ch, n_hidden_ch, kernel_size=1, padding=0),
            ConvBlock(n_hidden_ch, n_hidden_ch, kernel_size=1, padding=0),
            C3(n_hidden_ch, n_hidden_ch, n=4),
            ConvBlock(n_hidden_ch, n_hidden_ch, kernel_size=1, padding=0),
            Detect(n_hidden_ch, out_channels),
        )

    def forward(self, x_features, color_context, local_context=None):
        if local_context is None:
            local_context = torch.zeros_like(color_context)

        x_color_context = self.color_context_net(color_context)
        x_local_context = self.local_context_net(local_context)

        x = torch.cat((x_features, x_color_context, x_local_context), dim=1)
        return self.detection_net(x)
