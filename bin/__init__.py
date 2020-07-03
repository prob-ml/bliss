from pathlib import Path
import torch


def setup_paths(args):
    root_path = Path(args.root_dir)
    path_dict = {
        "root": root_path,
        "data": root_path.joinpath("data"),
        "logs": root_path.joinpath("logs"),
        "output": root_path.joinpath(f"logs/{args.output_dir}"),
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
