from pathlib import Path
import torch
import numpy as np

from celeste import use_cuda


def setup_paths(args):
    root_path = Path(args.root_dir)
    path_dict = {
        "root": root_path,
        "data": root_path.joinpath("data"),
        "results": root_path.joinpath("results"),
    }

    for p in path_dict.values():
        assert p.exists(), f"path {p.as_posix()} does not exist"

    return path_dict


def setup_device(args):
    assert (
        args.no_cuda or torch.cuda.is_available()
    ), "cuda is not available but --no-cuda is false"

    if not args.no_cuda:
        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return device


def setup_seed(args):
    if args.torch_seed:
        torch.manual_seed(args.torch_seed)

        if use_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if args.numpy_seed:
        np.random.seed(args.numpy_seed)
