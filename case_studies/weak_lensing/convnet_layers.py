import math

from torch import nn


class Map(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.map = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.map(x.permute(0, 2, 3, 1))


class RN2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        out_c_sqrt = math.sqrt(out_channels)
        if out_c_sqrt.is_integer():
            n_groups = int(out_c_sqrt)
        else:
            n_groups = int(
                math.sqrt(out_channels * 2)
            )  # even powers of 2 guaranteed to be perfect squares
        self.gn1 = nn.GroupNorm(num_groups=n_groups, num_channels=out_channels)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=n_groups, num_channels=out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=n_groups, num_channels=out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.silu(out)
        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.silu(out)

        return out  # noqa: WPS331


class ResNeXtBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        mid_c_sqrt = math.sqrt(mid_channels)
        if mid_c_sqrt.is_integer():
            mid_norm_n_groups = int(mid_c_sqrt)
        else:
            mid_norm_n_groups = int(
                math.sqrt(mid_channels * 2)
            )  # even powers of 2 guaranteed to be perfect squares
        self.gn1 = nn.GroupNorm(num_groups=mid_norm_n_groups, num_channels=mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=groups
        )
        self.gn2 = nn.GroupNorm(num_groups=mid_norm_n_groups, num_channels=mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        out_c_sqrt = math.sqrt(out_channels)
        if out_c_sqrt.is_integer():
            out_norm_n_groups = int(out_c_sqrt)
        else:
            out_norm_n_groups = int(
                math.sqrt(out_channels * 2)
            )  # even powers of 2 guaranteed to be perfect squares
        self.gn3 = nn.GroupNorm(num_groups=out_norm_n_groups, num_channels=out_channels)
        self.silu = nn.SiLU(inplace=True)

        # Adjust the shortcut connection to match the output dimensions
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.GroupNorm(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.silu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.silu(out)
        out = self.conv3(out)
        out = self.gn3(out)
        if self.shortcut:
            residual = self.shortcut(x)
        out += residual
        out = self.silu(out)
        return out  # noqa: WPS331
