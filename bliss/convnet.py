import torch
from torch import nn

# The code in this file is based on the following repository:
# https://github.com/ultralytics/yolov5/


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        ch = int(c2 * e)
        self.cv1 = ConvBlock(c1, ch, kernel_size=1, padding=0)
        self.cv2 = ConvBlock(ch, c2, kernel_size=3, padding=1, stride=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


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


class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inplace = True
        self.backbone = nn.ModuleList()

        block0 = nn.ModuleList()
        for _ in range(10):
            block0.append(ConvBlock(in_channels, 64, kernel_size=5, padding=2))
        self.backbone.append(nn.Sequential(*block0))

        self.backbone.append(ConvBlock(64, 128, stride=2))  # 1
        self.backbone.append(ConvBlock(128, 128))  # 2
        self.backbone.append(ConvBlock(128, 256, stride=2))  # 3
        self.backbone.append(C3(256, 256, n=6))  # 4
        self.backbone.append(ConvBlock(256, 512, stride=2))  # 5
        self.backbone.append(C3(512, 512, n=3, shortcut=False))  # 6
        self.backbone.append(ConvBlock(512, 256, kernel_size=1, padding=0))  # 7
        self.backbone.append(nn.Upsample(scale_factor=2, mode="nearest"))  # 8

        self.head = nn.ModuleList()
        self.head.append(C3(768, 256, n=3, shortcut=False))
        self.head.append(ConvBlock(256, out_channels, kernel_size=1, padding=0))

    def forward(self, x):
        save_lst = []
        for i, m in enumerate(self.backbone):
            x = m(x)
            if i in {3, 4, 8}:
                save_lst.append(x)

        x = torch.cat(save_lst, dim=1)

        for m in self.head:
            x = m(x)

        return x
