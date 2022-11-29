import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LogBackgroundTransform:
    def __init__(self, z_threshold: float = 4.0) -> None:
        self.z_threshold = z_threshold

    def __call__(self, image_and_background: Tensor) -> Tensor:
        n_bands = image_and_background.shape[1] // 2
        image, background = torch.split(image_and_background, [n_bands, n_bands], 1)
        return torch.log1p(
            F.relu(image - background + self.z_threshold * background.sqrt(), inplace=True)
        )

    def output_channels(self, input_channels: int) -> int:
        return input_channels


class ConcatBackgroundTransform:
    def __init__(self):
        pass

    def __call__(self, image_and_background: Tensor) -> Tensor:
        return image_and_background

    def output_channels(self, input_channels: int) -> int:
        return 2 * input_channels


def make_enc_final(in_size, hidden, out_size, dropout):
    return nn.Sequential(
        nn.Flatten(1),
        nn.Linear(in_size, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_size),
    )


class EncoderCNN(nn.Module):
    def __init__(self, n_bands, channel, dropout):
        super().__init__()
        self.layer = self._make_layer(n_bands, channel, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Runs encoder CNN on inputs."""
        return self.layer(x)

    def _make_layer(self, n_bands, channel, dropout):
        layers = [
            nn.Conv2d(n_bands, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
        ]
        in_channel = channel
        for i in range(3):
            downsample = i != 0
            layers += [
                ConvBlock(in_channel, channel, dropout, downsample),
                ConvBlock(channel, channel, dropout),
                ConvBlock(channel, channel, dropout),
            ]
            in_channel = channel
            channel = channel * 2
        return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    """A Convolution Layer.

    This module is two stacks of Conv2D -> ReLU -> BatchNorm, with dropout
    in the middle, and an option to downsample with a stride of 2.

    Parameters:
        in_channel: Number of input channels
        out_channel: Number of output channels
        dropout: Dropout proportion between [0, 1]
        downsample (optional): Whether to downsample with stride of 2.
    """

    def __init__(self, in_channel: int, out_channel: int, dropout: float, downsample: bool = False):
        """Initializes the module layers."""
        super().__init__()
        self.downsample = downsample
        stride = 1
        if self.downsample:
            stride = 2
            self.sc_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)
            self.sc_bn = nn.BatchNorm2d(out_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.drop1 = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x: Tensor) -> Tensor:
        """Runs convolutional block on inputs."""
        identity = x

        x = self.conv1(x)
        x = F.relu(self.bn1(x), inplace=True)

        x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            identity = self.sc_bn(self.sc_conv(identity))

        out = x + identity
        return F.relu(out, inplace=True)
