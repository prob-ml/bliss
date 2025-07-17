import torch
from torch import nn

# The code in this file is based on the following repository:
# https://github.com/ultralytics/yolov5/


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, gn=True, use_relu=False):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel size must be odd"
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        # seems to work about as well as BatchNorm2d
        if gn:
            assert out_channels % 8 == 0
        n_groups = out_channels // 8
        self.norm = nn.GroupNorm(n_groups, out_channels) if gn else nn.Identity()
        self.activation = nn.SiLU(inplace=True) if not use_relu else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5, gn=True, spatial=True):
        super().__init__()
        ch = int(c2 * e)
        self.cv1 = ConvBlock(c1, ch, kernel_size=1, gn=gn)
        ks = 3 if spatial else 1
        self.cv2 = ConvBlock(ch, c2, kernel_size=ks, stride=1, gn=gn)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, gn=True, spatial=True):
        super().__init__()
        ch = int(c2 * e)
        self.cv1 = ConvBlock(c1, ch, kernel_size=1, gn=gn)
        self.cv2 = ConvBlock(c1, ch, kernel_size=1, gn=gn)
        self.cv3 = ConvBlock(2 * ch, c2, kernel_size=1, gn=gn)
        self.m = nn.Sequential(
            *(Bottleneck(ch, ch, shortcut, e=1.0, spatial=spatial) for _ in range(n)),
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class FeaturesNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, num_features, double_downsample=True):
        super().__init__()
        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, 64, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(64),
            nn.SiLU(),
        )

        self.downsample_net = nn.Sequential(
            ConvBlock(64, 64, kernel_size=5),
            nn.Sequential(*[ConvBlock(64, 64, kernel_size=5) for _ in range(3)]),
            ConvBlock(64, 64, stride=2 if double_downsample else 1),  # 40x40
            C3(64, 64, n=3),
            ConvBlock(64, 128, stride=2),  # 20x20
            C3(128, num_features, n=3),
        )

    def forward(self, x):
        x = self.preprocess3d(x).squeeze(2)
        return self.downsample_net(x)


class ShortFeaturesNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, num_features):
        super().__init__()
        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, 64, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(64),
            nn.SiLU(),
        )

        self.conv_skip_layers = nn.ModuleList([
            ConvBlock(64, 64, kernel_size=5) for _ in range(4)
        ])

        self.postprocess_net = nn.Sequential(
            ConvBlock(64, 128, stride=2),
            C3(128, num_features, n=3),
        )

    def conv_skip_layers_forward(self, x):
        for m in self.conv_skip_layers:
            x = m(x) + x
        return x

    def forward(self, x):
        x = self.preprocess3d(x).squeeze(2)
        x = self.conv_skip_layers_forward(x)
        return self.postprocess_net(x)


class UShapeFeaturesNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, num_features, double_downsample=True):
        super().__init__()
        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, 64, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(64),
            nn.SiLU(),
        )

        # tile_slen is 4 (rather than 2)
        self.double_downsample = double_downsample
        self.downsample_net = nn.Sequential(
            ConvBlock(64, 64, stride=2),  # 2
            C3(64, 64, n=3),
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
            C3(384, num_features, n=3, shortcut=False),
        ]
        self.u_net = nn.ModuleList(u_net_layers)

    def forward(self, x):
        x = self.preprocess3d(x).squeeze(2)
        save_lst = []
        for i, m in enumerate(self.u_net):
            x = m(x)
            if i in {2, 5}:
                save_lst.append(x)
            if i == 8:
                x = torch.cat((x, save_lst[1]), dim=1)
            if i == 10:
                x = torch.cat((x, save_lst[0]), dim=1)
            if i == 1 and self.double_downsample:
                x = self.downsample_net(x)
        return x
