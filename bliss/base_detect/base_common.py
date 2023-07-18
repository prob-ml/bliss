import warnings

import torch


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(torch.nn.Module):
    """Standard convolution."""

    default_act = torch.nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Wrapper for torch.nn.conv2d layer.

        Args:
            c1: input channels
            c2: output channels
            k: kernel size
            s: kernel stride length
            p: input padding amount
            g: controls connections between inputs and outputs
            d: kernel dilation value
            act: torch.nn activation function
        """
        super().__init__()
        self.conv = torch.nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = torch.nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act  # noqa: WPS509
            if isinstance(act, torch.nn.Module)
            else torch.nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Conv3d(torch.nn.Module):
    """Wrapper for torch.nn.conv3d layer.

    Args:
        c1: input channels
        c2: output channels
        k: kernel size
        s: kernel stride length
        p: input padding amount
        g: controls connections between inputs and outputs
        d: kernel dilation value
        act: torch.nn activation function
    """

    default_act = torch.nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = torch.nn.Conv3d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = torch.nn.BatchNorm3d(c2)
        self.act = (
            self.default_act
            if act is True
            else act  # noqa: WPS509
            if isinstance(act, torch.nn.Module)
            else torch.nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Conv3d_down(Conv3d):  # noqa: N801
    def forward(self, x):
        z = self.act(self.bn(self.conv(x)))
        return torch.squeeze(z, 1)


class Bottleneck(torch.nn.Module):
    """Bottleneck layer.

    Args:
        c1: input channels
        c2: output channels
        shortcut: (?)
        g: controls connections between inputs and outputs
        e: expansion
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        # hidden channels
        c_ = int(c2 * e)  # noqa: WPS120
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Concat(torch.nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class C3(torch.nn.Module):
    """CSP Bottleneck with 3 convolutions.

    Args:
        c1: input channels
        c2: output channels
        n: number of repeats
        shortcut: (?)
        g: controls connections between inputs and outputs
        e: expansion
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        # hidden channels
        c_ = int(c2 * e)  # noqa: WPS120
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = torch.nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(torch.nn.Module):
    """Spatial Pyaramid Pooling - Fast (SPPF) layer.

    Args:
        c1: input channels
        c2: output channels
        k: kernel size
    """

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        # hidden channels
        c_ = c1 // 2  # noqa: WPS120
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = torch.nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


# TODO: Add Pooling class
