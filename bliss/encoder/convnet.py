import torch
from torch import nn

# The code in this file is based on the following repository:
# https://github.com/ultralytics/yolov5/


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_group_norm=False
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        if use_group_norm:
            self.norm = nn.GroupNorm(out_channels // 8, out_channels)
        else:
            self.norm = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class Detect(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)

    def forward(self, x):
        return self.conv(x).permute([0, 2, 3, 1])


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        ch = int(c2 * e)
        self.cv1 = ConvBlock(c1, ch, kernel_size=1, padding=0)
        self.cv2 = ConvBlock(ch, c2, kernel_size=3, padding=1, stride=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        ch = int(c2 * e)
        self.cv1 = ConvBlock(c1, ch, kernel_size=1, padding=0)
        self.cv2 = ConvBlock(c1, ch, kernel_size=1, padding=0)
        self.cv3 = ConvBlock(2 * ch, c2, kernel_size=1, padding=0)
        self.m = nn.Sequential(*(Bottleneck(ch, ch, shortcut, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class FeaturesNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, num_features, double_downsample=True):
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


class ContextNet(nn.Module):
    def __init__(self, num_features, out_channels):
        super().__init__()

        context_channels_in = 6
        context_channels_out = 64
        self.encode_context = nn.Sequential(
            ConvBlock(context_channels_in, 64, use_group_norm=True),
            ConvBlock(64, 64, use_group_norm=True),
            ConvBlock(64, context_channels_out, use_group_norm=True),
        )
        self.merge = nn.Sequential(
            ConvBlock(num_features + context_channels_out, num_features, use_group_norm=True),
            ConvBlock(num_features, num_features, use_group_norm=True),
            ConvBlock(num_features, num_features, use_group_norm=True),
            Detect(num_features, out_channels),
        )

    def forward(self, x_features, context):
        x_context = self.encode_context(context)
        x = torch.cat((x_features, x_context), dim=1)
        return self.merge(x)
