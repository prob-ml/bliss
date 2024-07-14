import torch
from torch import nn

from bliss.encoder.convnet_layers import C3, ConvBlock, Detect


class FeaturesNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, num_features, tile_slen):
        super().__init__()

        nch_hidden = 64
        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, nch_hidden, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(nch_hidden),
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
                ConvBlock(128, 256, stride=1),
            ]
        )  # 4
        self.backbone = nn.Sequential(*module_list)
        u_net_layers = [
            C3(256, 256, n=6),  # 0
            ConvBlock(256, 512, stride=2),
            C3(512, 512, n=3, shortcut=False),
            ConvBlock(512, 256, kernel_size=1, padding=0),
            # nn.Upsample(scale_factor=2, mode="nearest"),  # 4
            # C3(768, num_features, n=3, shortcut=False),
            C3(256, num_features, n=3, shortcut=False), # fix back after
            # nn.Linear(768, 256),
            # (768, num_features, kernel_size=1, padding=0),
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

        # initalization for detection head
        context_channels_in = 6
        context_channels_out = 256
        self.context_net = nn.Sequential(
            ConvBlock(context_channels_in, context_channels_out),
            ConvBlock(context_channels_out, context_channels_out),
            C3(context_channels_out, context_channels_out, n=512),
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
