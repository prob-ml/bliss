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
