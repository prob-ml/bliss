import torch
from torch import nn

from bliss.encoder.convnet_layers import C3, ConvBlock


class FeaturesNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, num_features):
        super().__init__()
        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, 64, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(64),
            nn.SiLU(),
        )

        u_net_layers = [
            ConvBlock(64, 64, kernel_size=5),
            nn.Sequential(*[ConvBlock(64, 64, kernel_size=5) for _ in range(3)]),  # 1
            ConvBlock(64, 128, stride=2),  # 2
            C3(128, 128, n=3),
            ConvBlock(128, 256, stride=2),  # 4
            C3(256, 256, n=3),
            ConvBlock(256, 512, stride=2),
            C3(512, 256, n=3, shortcut=False),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 8
            C3(512, 256, n=3, shortcut=False),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 10
            C3(384, 128, n=3, shortcut=False),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 12
            C3(192, num_features, n=3, shortcut=False),
        ]
        self.u_net = nn.ModuleList(u_net_layers)

    def forward(self, x):
        x = self.preprocess3d(x).squeeze(2)
        save_lst = []
        for i, m in enumerate(self.u_net):
            x = m(x)
            if i in {1, 2, 5}:
                save_lst.append(x)
            if i == 8:
                x = torch.cat((x, save_lst[2]), dim=1)
            if i == 10:
                x = torch.cat((x, save_lst[1]), dim=1)
            if i == 12:
                x = torch.cat((x, save_lst[0]), dim=1)
        return x
