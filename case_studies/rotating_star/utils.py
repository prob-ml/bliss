import torch
import torch.nn as nn
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        assert len(x.shape) > 1

        return x.view(x.shape[0], -1)


class UnFlatten(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        # assert len(x.shape) == 2
        return x.view(*x.shape[0:-1], *self.shape)


class ConstantDist(nn.Module):
    """
    A distribution which is constant; meant only as a placeholder for sampling
    """

    def __init__(self, X):
        super().__init__()
        self.X = X

    def sample(self):
        return self.X

    def rsample(self):
        return self.sample()


class IdentityEncoder(nn.Module):
    """
    Takes a tensor and returns a constant distribution which has all mass on that tensor.
    """

    def __init__(self):
        super().__init__()
        self.X = None

    def forward(self, X):
        return ConstantDist(X)


class ReshapeWrapper(nn.Module):
    """
    This module wraps around a module which expects tensors of a fixed dimension. For example,
    a Conv2D layer expects a 4-dimensional tensor, but we may want to use 5-dimensional or higher
    tensors with it. This flattens the higher dimensions into dimension 0, then unflattens them.
    """

    def __init__(self, f, k=None, f_dim=None):
        super().__init__()
        self.f = f
        self.k = k
        self.f_dim = f_dim

    def forward(self, X):
        if self.k is None:
            k = len(X.shape) - self.f_dim + 1
        else:
            k = self.k
        assert k > 0
        in_size = torch.Size([np.product(X.shape[:k])]) + X.shape[k:]
        Y = self.f(X.view(in_size))
        Y = Y.view(X.shape[:k] + Y.shape[(k - 1) :])
        return Y


class Conv2DAutoEncoder(nn.Module):
    """
    This module creates a stacked layer of Conv2D layers to decode an image tensor to a
    flattened representation. It simulatenously creates a corresponding stacked layer of
    Conv2d Transposes which will map that representation to an output image of the same
    dimension.
    """

    def __init__(
        self,
        size_h,
        size_w,
        conv_channels,
        kernel_sizes,
        strides,
        last_decoder_channel=2,
    ):
        super().__init__()
        self.size_h = size_h
        self.size_w = size_w
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        dummy_input = torch.randn(1, 1, self.size_h, self.size_w)
        encoder_list = []
        decoder_list = []
        for i, _ in enumerate(self.conv_channels):
            conv_net = nn.Conv2d(
                1 if i == 0 else self.conv_channels[i - 1],
                self.conv_channels[i],
                self.kernel_sizes[i],
                self.strides[i],
            )
            h_in, w_in = dummy_input.shape[2:]
            dummy_output = conv_net(dummy_input)
            h_out, w_out = dummy_output.shape[2:]
            if (
                not ((self.strides[i] * h_out + self.kernel_sizes[i] - 1) == h_in)
                or self.strides[i] == 1
            ):
                pad_h = 0
            else:
                pad_h = 1
            if (
                not ((self.strides[i] * w_out + self.kernel_sizes[i] - 1) == w_in)
                or self.strides[i] == 1
            ):
                pad_w = 0
            else:
                pad_w = 1
            conv_net_t = nn.ConvTranspose2d(
                self.conv_channels[i],
                last_decoder_channel if i == 0 else self.conv_channels[i - 1],
                self.kernel_sizes[i],
                self.strides[i],
                output_padding=(pad_h, pad_w),
            )

            encoder_list.append(conv_net)
            encoder_list.append(nn.ReLU())
            decoder_list.insert(0, conv_net_t)
            if i > 0:
                decoder_list.insert(1, nn.ReLU())

            dummy_input = dummy_output

        self.dim_rep = self.conv_channels[-1] * h_out * w_out
        self.size_h_end = h_out
        self.size_w_end = w_out
        encoder_list.append(Flatten())

        self.encoder = nn.Sequential(*encoder_list)
        self.decoder = nn.Sequential(
            UnFlatten([self.conv_channels[-1], self.size_h_end, self.size_w_end]),
            *decoder_list,
        )
