import math

import torch
from pytorch_lightning.utilities import rank_zero_only


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
    hparams["optimizer"] = config["training"]["optimizer_params"]

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
