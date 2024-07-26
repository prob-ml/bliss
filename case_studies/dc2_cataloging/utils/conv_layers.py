# flake8: noqa
# darglint: ignore

import torch
from torch import nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        use_activation=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding, dilation),
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        # seems to work about as well as BatchNorm2d
        self.norm = nn.GroupNorm(out_channels // 8, out_channels)
        self.activation = nn.SiLU(inplace=True) if use_activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, kernel_size=((1, 1), (3, 3)), shortcut=True, e=0.5):
        super().__init__()
        ch = int(c2 * e)
        self.cv1 = ConvBlock(c1, ch, kernel_size=kernel_size[0])
        self.cv2 = ConvBlock(ch, c2, kernel_size=kernel_size[1])
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
        self.m = nn.Sequential(
            *(
                Bottleneck(ch, ch, kernel_size=((1, 1), (3, 3)), shortcut=shortcut, e=1.0)
                for _ in range(n)
            ),
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvBlock(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvBlock((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, kernel_size=((3, 3), (3, 3)), shortcut=shortcut, e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = ConvBlock(c1, c2, 1, 1)
        self.cv2 = ConvBlock(c2, c2, kernel_size=k, stride=s, groups=c2, use_activation=False)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = ConvBlock(ed, ed, 7, 1, 3, groups=ed, use_activation=False)
        self.conv1 = ConvBlock(ed, ed, 3, 1, 1, groups=ed, use_activation=False)
        self.dim = ed
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.conv1(x))


class CIB(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        super().__init__()
        ch = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            ConvBlock(c1, c1, 3, groups=c1),
            ConvBlock(c1, 2 * ch, 1),
            ConvBlock(2 * ch, 2 * ch, 3, groups=2 * ch) if not lk else RepVGGDW(2 * ch),
            ConvBlock(2 * ch, c2, 1),
            ConvBlock(c2, c2, 3, groups=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, e=0.5):
        super().__init__(c1, c2, n, shortcut, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        ch = c1 // 2  # hidden channels
        self.cv1 = ConvBlock(c1, ch, 1, 1)
        self.cv2 = ConvBlock(ch * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = ConvBlock(dim, h, 1, use_activation=False)
        self.proj = ConvBlock(dim, dim, 1, use_activation=False)
        self.pe = ConvBlock(dim, dim, 3, 1, groups=dim, use_activation=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSA(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = ConvBlock(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvBlock(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(
            ConvBlock(self.c, self.c * 2, 1), ConvBlock(self.c * 2, self.c, 1, use_activation=False)
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class Detect(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)

    def forward(self, x):
        return self.conv(x).permute([0, 2, 3, 1])
