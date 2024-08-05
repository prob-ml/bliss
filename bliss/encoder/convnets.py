import torch
from einops import rearrange
from torch import nn

from bliss.encoder.convnet_layers import C3, ConvBlock, Detect


class FeaturesNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, num_features, double_downsample=False):
        super().__init__()

        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, 64, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(64),  # remove?
            nn.SiLU(),  # remove?
        )
        u_net_layers = [
            # 1x1 tiles
            nn.Sequential(*[ConvBlock(64, 64, kernel_size=5, padding=2) for _ in range(4)]),  # 0
            # 2x2 tiles
            ConvBlock(64, 128, stride=2),
            nn.Sequential(*[ConvBlock(128, 128) for _ in range(4)]),  # 2
            # 4x4 tiles
            ConvBlock(128, 256, stride=2),
            C3(256, 256, n=4),  # 4
            # 8x8 tiles
            ConvBlock(256, 512, stride=2),
            C3(512, 512, n=3, shortcut=False),
            ConvBlock(512, 256, kernel_size=1, padding=0),
            # 4x4 tiles
            nn.Upsample(scale_factor=2, mode="nearest"),
            C3(768, 256, n=3, shortcut=False),  # 9
            # 2x2 tiles
            nn.Upsample(scale_factor=2, mode="nearest"),
            C3(384, 256, n=3, shortcut=False),  # 11
            # 1x1 tiles
            nn.Upsample(scale_factor=2, mode="nearest"),
            C3(320, 64, n=3, shortcut=False),  # 13
        ]
        self.u_net = nn.ModuleList(u_net_layers)

    def forward(self, x):
        x = self.preprocess3d(x).squeeze(2)

        save_lst = []
        for i, m in enumerate(self.u_net):
            if i == 9:
                x = torch.cat((save_lst[3], save_lst[4], x), dim=1)
            if i == 11:
                x = torch.cat((save_lst[2], x), dim=1)
            if i == 13:
                x = torch.cat((save_lst[0], x), dim=1)
            x = m(x)
            save_lst.append(x)

        return save_lst[11], save_lst[13]


class CatalogNet(nn.Module):
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
        self.tile_head = nn.Sequential(
            ConvBlock(num_features + context_channels_out, n_hidden_ch),
            ConvBlock(n_hidden_ch, n_hidden_ch),
            C3(n_hidden_ch, n_hidden_ch, n=4),
            ConvBlock(n_hidden_ch, n_hidden_ch),
            Detect(n_hidden_ch, out_channels - 64),  # subtracted for now because of hard coded locs
        )

        self.loc_context = nn.Sequential(
            ConvBlock(context_channels_out, 32),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.loc_head = nn.Sequential(
            ConvBlock(64 + 32, 64),
            C3(64, 16, n=2),
            # 0.5 x 0.5 tiles
            nn.Upsample(scale_factor=2, mode="nearest"),
            C3(16, 16, n=2),
            ConvBlock(16, 8),
            # 0.25 x 0.25 tiles
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(8, 4),
            ConvBlock(4, 2),
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x_features, context):
        x_context = self.context_net(context)

        x_tile_features = torch.cat((x_features[0], x_context), dim=1)
        x_tile = self.tile_head(x_tile_features)

        x_loc_context = self.loc_context(x_context)
        x_loc_features = torch.cat((x_features[1], x_loc_context), dim=1)
        x_loc = self.loc_head(x_loc_features)

        x_loc_patches = x_loc.squeeze(1).unfold(1, 8, 8).unfold(2, 8, 8)
        x_loc_flatter = rearrange(x_loc_patches, "b ht wt hsp wsp -> b ht wt (hsp wsp)")
        return torch.concat((x_tile, x_loc_flatter), dim=3)
