import torch
import torch.nn.functional as F
from torch import nn
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
        # Filter only to arguments we want to concatenate
        if self.input_idxs is not None:
            args = [args[i] for i in self.input_idxs]
        else:
            args = list(args)

        # Get the maximum size of each tensor dimension
        # and repeat any tensors which have a 1 in
        # a dimension
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
