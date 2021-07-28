import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal


class MLP(nn.Sequential):
    """
    A Multi-layer perceptron of dense layers with non-linear activation layers
    """

    def __init__(self, in_features, hs, out_features, act=nn.ReLU, final=None):
        self.in_features = in_features
        self.out_features = out_features
        layers = []
        for i, h in enumerate(hs):
            layers.append(nn.Linear(in_features if i == 0 else hs[i - 1], h))
            layers.append(act())
        layers.append(nn.Linear(hs[-1], out_features))
        if final is not None:
            layers.append(final())
        super().__init__(*layers)


class SequentialVarg(nn.Sequential):
    """
    This subclass of torch.nn.Sequential allows for stacking modules which take
    and/or return multiple arguments.
    """

    def forward(self, *X):
        for module in self:
            if isinstance(X, tuple):
                X = module(*X)
            else:
                X = module(X)
        return X


class SplitLayer(nn.Module):
    """
    This layer splits the input according to the arguments to torch.split
    """

    def __init__(self, split_size_or_sections, dim):
        super().__init__()
        self.split_size_or_sections = split_size_or_sections
        self.dim = dim

    def forward(self, tensor):
        return torch.split(tensor, self.split_size_or_sections, self.dim)


class ConcatLayer(nn.Module):
    """
    Concatenates input tensors of the same size along the last dimension.
    Optionally filters out arguments based on position.
    """

    def __init__(self, input_idxs=None):
        super().__init__()
        self.input_idxs = input_idxs

    def forward(self, *args):
        ## Filter only to arguments we want to concatenate
        if self.input_idxs is not None:
            args = [args[i] for i in self.input_idxs]
        else:
            args = list(args)

        ## Get the maximum size of each tensor dimension
        ## and repeat any tensors which have a 1 in
        ## a dimension
        sizes = []
        for d, _ in enumerate(args[0].size()):
            sizes.append(max([arg.size(d) for arg in args]))
        for d in range(len(sizes) - 1):
            size = sizes[d]
            if size > 1:
                for i, _ in enumerate(args):
                    if args[i].size(d) == 1:
                        r = [1] * len(sizes)
                        r[d] = size
                        args[i] = args[i].repeat(*r)
                    elif args[i].size(d) < size:
                        raise ValueError(
                            "The sizes in ConcatLayer need to be either the same or 1."
                        )
        X = torch.cat(args, -1)
        return X


# *************************
# Probabilistic Encoders
# ************************
class NormalEncoder(nn.Module):
    """
    This module takes two tensors of equal shape, mean and logscale, which parameterize
    a Normal distribution
    """

    def __init__(self, minscale=None):
        super().__init__()
        self.minscale = minscale

    def forward(self, mean_z, logscale_z):
        if self.minscale is not None:
            logscale_z = torch.log(self.minscale + (1 - self.minscale) * F.softplus(logscale_z))
        pz = Normal(mean_z, logscale_z.exp())
        return pz


# adapt from torchvision.utils.make_grid
@torch.no_grad()
def make_grid(
    tensor,
    nrow: int = 8,
    padding: int = 2,
    scale_each: bool = False,
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        tensor = tensor.unsqueeze(0)

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t):
            norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t)
        else:
            norm_range(tensor)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full(
        (num_channels, height * ymaps + padding, width * xmaps + padding), pad_value
    )
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid
