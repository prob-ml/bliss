import torch
from torch import nn

# The code in this file is based on the following repository:
# https://github.com/ultralytics/yolov5/


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


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
    def __init__(self, in_channels):
        super().__init__()

        net_layers = [
            ConvBlock(in_channels, 64, kernel_size=5, padding=2),
            nn.Sequential(*[ConvBlock(64, 64, kernel_size=5, padding=2) for _ in range(9)]),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128),
            ConvBlock(128, 256, stride=2),  # 3
        ]
        self.net = nn.Sequential(*net_layers)

    def forward(self, x):
        return self.net(x)


class CatalogNet1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        net_layers = [
            C3(256, 256, n=6),  # 0
            ConvBlock(256, 512, stride=2),
            C3(512, 512, n=3, shortcut=False),
            ConvBlock(512, 256, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 4
            C3(768, 256, n=3, shortcut=False),
            Detect(256, out_channels),
        ]
        self.net_ml = nn.ModuleList(net_layers)

    def forward(self, x):
        save_lst = [x]
        for i, m in enumerate(self.net_ml):
            x = m(x)
            if i in {0, 4}:
                save_lst.append(x)
            if i == 4:
                x = torch.cat(save_lst, dim=1)
        return x


class CatalogNet2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        net_layers = [
            ConvBlock(in_channels, 256),  # 0
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            C3(256, 256, n=6),  # 3
            ConvBlock(256, 512, stride=2),
            C3(512, 512, n=3, shortcut=False),
            ConvBlock(512, 256, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 7
            C3(768, 256, n=3, shortcut=False),
            Detect(256, out_channels),
        ]
        self.net_ml = nn.ModuleList(net_layers)

    def forward(self, x):
        save_lst = []
        for i, m in enumerate(self.net_ml):
            x = m(x)
            if i in {0, 3, 7}:
                save_lst.append(x)
            if i == 7:
                x = torch.cat(save_lst, dim=1)
        return x
