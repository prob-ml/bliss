import math
import os

import torch
from pytorch_lightning.utilities import rank_zero_only
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


def fixed_workdir(fn):
    def wrapper(cfg):
        os.chdir(cfg.paths.root)
        return fn(cfg)

    return wrapper


def empty(*args, **kwargs):
    pass


# https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/utils.py
@rank_zero_only
def log_hyperparameters(config, model, trainer) -> None:
    """Log config and num of model parameters to all Lightning loggers."""

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["mode"] = config["mode"]
    hparams["gpus"] = config["gpus"]
    hparams["training"] = config["training"]
    hparams["model"] = config["training"]["model"]
    hparams["dataset"] = config["training"]["dataset"]
    hparams["optimizer"] = config["training"]["optimizer"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # trick to disable logging any more hyperparameters for all loggers
    trainer.logger.log_hyperparams = empty


class MLP(nn.Sequential):
    """A Multi-layer perceptron of dense layers with non-linear activation layers."""

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
    """Stacks modules which take and/or return multiple arguments."""

    def forward(self, *X):
        for module in self:
            if isinstance(X, tuple):
                X = module(*X)
            else:
                X = module(X)
        return X


class SplitLayer(nn.Module):
    """Splits the input according to the arguments to torch.split."""

    def __init__(self, split_size_or_sections, dim):
        super().__init__()
        self.split_size_or_sections = split_size_or_sections
        self.dim = dim

    def forward(self, tensor):
        return torch.split(tensor, self.split_size_or_sections, self.dim)


class ConcatLayer(nn.Module):
    """Concatenates input tensors of the same size along the last dimension."""

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
        return torch.cat(args, -1)


# *************************
# Probabilistic Encoders
# ************************
class NormalEncoder(nn.Module):
    """Encodes a Normal distribution with mean and logscale."""

    def __init__(self, minscale=None):
        super().__init__()
        self.minscale = minscale

    def forward(self, mean_z, logscale_z):
        if self.minscale is not None:
            logscale_z = torch.log(self.minscale + (1 - self.minscale) * F.softplus(logscale_z))
        return Normal(mean_z, logscale_z.exp())


# adapt from torchvision.utils.make_grid
@torch.no_grad()
def make_grid(
    tensor,
    nrow: int = 8,
    padding: int = 2,
    scale_each: bool = False,
    pad_value: int = 0,
) -> torch.Tensor:

    assert tensor.dim() == 4

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    size = num_channels, height * ymaps + padding, width * xmaps + padding
    grid = tensor.new_full(size, pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(
                2,
                x * width + padding,
                width - padding,
            ).copy_(tensor[k])
            k = k + 1
    return grid
