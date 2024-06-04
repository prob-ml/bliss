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
    def __init__(self, n_bands, ch_per_band, num_features, tile_slen=2, downsample_at_front=False):
        super().__init__()
        nch_hidden = 64
        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, nch_hidden, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(nch_hidden),
            nn.SiLU(),
        )

        # Currently, if tile length is 2 or 4,
        # we revert to previous version of the network for backward compatibility
        # This needs to be fixed in future iterations
        # to make the network architecture cleaner and generalizable

        log_tile_size = torch.log2(torch.tensor(tile_slen))
        num_downsample = int(torch.round(log_tile_size)) - 1
        self.backbone = nn.ModuleList()
        if tile_slen == 2:
            self.backbone.append(ConvBlock(nch_hidden, 64, kernel_size=5, padding=2))
            self.backbone.append(
                nn.Sequential(*[ConvBlock(64, 64, kernel_size=5, padding=2) for _ in range(4)])
            )
            self.backbone.append(ConvBlock(64, 128, stride=2))
            self.backbone.append(nn.Sequential(*[ConvBlock(128, 128) for _ in range(5)]))
            self.backbone.append(ConvBlock(128, num_features, stride=1))
        elif tile_slen == 4:
            self.backbone.append(ConvBlock(nch_hidden, 64, kernel_size=5, padding=2))
            self.backbone.append(
                nn.Sequential(*[ConvBlock(64, 64, kernel_size=5, padding=2) for _ in range(4)])
            )
            self.backbone.append(ConvBlock(64, 128, stride=2))
            self.backbone.append(nn.Sequential(*[ConvBlock(128, 128) for _ in range(5)]))
            self.backbone.append(ConvBlock(128, num_features, stride=2))
        else:
            if downsample_at_front:
                for _ in range(num_downsample):
                    self.backbone.append(
                        ConvBlock(nch_hidden, 2 * nch_hidden, kernel_size=5, stride=2, padding=2)
                    )
                    nch_hidden *= 2
                self.backbone.append(ConvBlock(nch_hidden, 64, kernel_size=5, padding=2))
                self.backbone.append(
                    nn.Sequential(*[ConvBlock(64, 64, kernel_size=5, padding=2) for _ in range(4)])
                )
                self.backbone.append(ConvBlock(64, 128, stride=2))
                self.backbone.append(nn.Sequential(*[ConvBlock(128, 128) for _ in range(5)]))
                self.backbone.append(ConvBlock(128, num_features, stride=1))
            else:
                self.backbone.append(ConvBlock(nch_hidden, 64, kernel_size=5, padding=2))
                self.backbone.append(
                    nn.Sequential(*[ConvBlock(64, 64, kernel_size=5, padding=2) for _ in range(4)])
                )
                self.backbone.append(ConvBlock(64, 128, stride=2))
                self.backbone.append(nn.Sequential(*[ConvBlock(128, 128) for _ in range(5)]))
                num_channels = 128
                for _ in range(num_downsample):
                    self.backbone.append(
                        ConvBlock(
                            num_channels,
                            2 * num_channels,
                            kernel_size=5,
                            stride=2,
                            padding=2,
                        )
                    )
                    num_channels *= 2
                self.backbone.append(ConvBlock(num_channels, num_features))

    def forward(self, x):
        x = self.preprocess3d(x).squeeze(2)
        for layer in self.backbone:
            x = layer(x)
        return x


class CatalogNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        net_layers = [
            C3(in_channels, 256, n=6),  # 0
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


class ContextNet(nn.Module):
    def __init__(self, num_features, out_channels):
        super().__init__()

        context_dim = 64
        self.encode_context = nn.Sequential(
            ConvBlock(2, 64),
            ConvBlock(64, 64),
            ConvBlock(64, context_dim),
        )
        self.merge = ConvBlock(num_features + context_dim, num_features)
        self.catalog_net = CatalogNet(num_features, out_channels)

    def forward(self, x_features, context):
        x_context = self.encode_context(context)
        x = torch.cat((x_features, x_context), dim=1)
        x = self.merge(x)
        return self.catalog_net(x)
