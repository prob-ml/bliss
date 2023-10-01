import torch
from torch import nn

# The code in this file is based on the following repository:
# https://github.com/ultralytics/yolov5/

NUM_FEATURES = 256


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


class MarginalNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, out_channels):
        super().__init__()

        nch_hidden = 64
        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, nch_hidden, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(nch_hidden),
            nn.SiLU(),
        )
        backbone_layers = [
            ConvBlock(nch_hidden, 64, kernel_size=5, padding=2),
            nn.Sequential(*[ConvBlock(64, 64, kernel_size=5, padding=2) for _ in range(9)]),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128),
            ConvBlock(128, NUM_FEATURES, stride=2),  # 4
            C3(256, 256, n=6),  # 5
            ConvBlock(256, 512, stride=2),
            C3(512, 512, n=3, shortcut=False),
            ConvBlock(512, 256, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 9
            C3(768, 256, n=3, shortcut=False),
            Detect(256, out_channels),
        ]
        self.backbone = nn.ModuleList(backbone_layers)

    def forward(self, x):
        x = self.preprocess3d(x).squeeze(2)

        save_for_cat = []
        save_as_features = None
        for i, m in enumerate(self.backbone):
            x = m(x)
            if i == 4:
                save_as_features = x
            if i in {4, 5, 9}:
                save_for_cat.append(x)
            if i == 9:
                x = torch.cat(save_for_cat, dim=1)

        return x, save_as_features


class ConditionalNet(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        neighbor_dim = 64
        preprocess_neighbors = [
            ConvBlock(2, 64),
            ConvBlock(64, 64),
            ConvBlock(64, neighbor_dim),
        ]
        self.pn_net = nn.Sequential(*preprocess_neighbors)

        net_layers = [
            ConvBlock(NUM_FEATURES + neighbor_dim, 256),  # 0
            C3(256, 256, n=6),  # 1
            ConvBlock(256, 512, stride=2),
            C3(512, 512, n=3, shortcut=False),
            ConvBlock(512, 256, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 5
            C3(768, 256, n=3, shortcut=False),
            Detect(256, out_channels),
        ]
        self.net_ml = nn.ModuleList(net_layers)

    def forward(self, x_features, detections, mask):
        neighbors = torch.cat([detections, mask], dim=1)
        x_neighbors = self.pn_net(neighbors)

        x = torch.cat((x_features, x_neighbors), dim=1)

        save_lst = []
        for i, m in enumerate(self.net_ml):
            x = m(x)
            if i in {0, 1, 5}:
                save_lst.append(x)
            if i == 5:
                x = torch.cat(save_lst, dim=1)
        return x
