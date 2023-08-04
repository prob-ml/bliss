"""Custom modules for the BLISS encoder."""

import torch
from yolov5.models.yolo import autopad


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
        # format taken from yolov5 Conv2d module
        self.conv = torch.nn.Conv3d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = torch.nn.BatchNorm3d(c2)

        # Taken from yolov5 Conv2d module
        self.act = (
            self.default_act
            if act
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
