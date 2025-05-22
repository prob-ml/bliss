# adapted from the convnet.py in commit 53936977c3a87f8ba937d75bec0fd951eb5c23d9

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
        self.norm = (
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
            if not use_group_norm
            else nn.GroupNorm(out_channels // 16, out_channels)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5, use_group_norm=False):
        super().__init__()
        ch = int(c2 * e)
        self.cv1 = ConvBlock(c1, ch, kernel_size=1, padding=0, use_group_norm=use_group_norm)
        self.cv2 = ConvBlock(
            ch, c2, kernel_size=3, padding=1, stride=1, use_group_norm=use_group_norm
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, use_group_norm=False):
        super().__init__()
        ch = int(c2 * e)
        self.cv1 = ConvBlock(c1, ch, kernel_size=1, padding=0, use_group_norm=use_group_norm)
        self.cv2 = ConvBlock(c1, ch, kernel_size=1, padding=0, use_group_norm=use_group_norm)
        self.cv3 = ConvBlock(2 * ch, c2, kernel_size=1, padding=0, use_group_norm=use_group_norm)
        self.m = nn.Sequential(
            *(Bottleneck(ch, ch, shortcut, e=1.0, use_group_norm=use_group_norm) for _ in range(n)),
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class FeaturesNet(nn.Module):
    def __init__(self, n_bands, ch_per_band, num_features, double_downsample=True):
        super().__init__()

        nch_hidden = 64
        nch_hidden_for_asinh_params = 32
        self.preprocess3d = nn.Sequential(
            nn.Conv3d(n_bands, nch_hidden, [ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(nch_hidden),
            nn.SiLU(),
        )
        self.backbone = nn.Sequential(
            ConvBlock(nch_hidden + nch_hidden_for_asinh_params, 64, kernel_size=5, padding=2),
            nn.Sequential(*[ConvBlock(64, 64, kernel_size=5, padding=2) for _ in range(4)]),
            ConvBlock(64, 128, stride=2),
            nn.Sequential(*[ConvBlock(128, 128) for _ in range(5)]),
            ConvBlock(128, num_features, stride=(2 if double_downsample else 1)),  # 4
        )

        self.asinh_preprocess = nn.Sequential(
            nn.ZeroPad2d(padding=(0, 1, 1, 1)),
            C3(1, nch_hidden_for_asinh_params, n=4, use_group_norm=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(
                nch_hidden_for_asinh_params, nch_hidden_for_asinh_params, use_group_norm=True
            ),
            nn.Upsample(scale_factor=5, mode="nearest"),
            ConvBlock(
                nch_hidden_for_asinh_params,
                nch_hidden_for_asinh_params,
                kernel_size=7,
                padding=3,
                use_group_norm=True,
            ),
        )

    def forward(self, x):
        preprocessed_x = self.preprocess3d(x[0]).squeeze(2)
        asinh_params = x[1]
        preprocessed_asinh_params = self.asinh_preprocess(asinh_params.unsqueeze(0))
        expanded_asinh_params = preprocessed_asinh_params.expand(
            preprocessed_x.shape[0], -1, -1, -1
        ).clone()

        return self.backbone(torch.cat((preprocessed_x, expanded_asinh_params), dim=1))
