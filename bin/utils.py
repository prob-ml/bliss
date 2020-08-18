from pathlib import Path
import os
import shutil


def add_path_args(parser):
    paths_group = parser.add_argument_group("[Paths]")
    paths_group.add_argument(
        "--root",
        help="Absolute path to directory containing bin and bliss package.",
        type=str,
        default=os.path.abspath("."),
    )

    paths_group.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory name relative to root/logs path, where output will be saved.",
    )

    paths_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite if directory already exists.",
    )

    return parser


def setup_paths(args, enforce_overwrite=True):
    root_path = Path(args.root)
    paths = {
        "root": root_path,
        "data": root_path.joinpath("data"),
        "logs": root_path.joinpath("logs"),
        "output": root_path.joinpath(f"logs/{args.output}"),
    }

    if enforce_overwrite:
        assert not paths["output"].exists() or args.overwrite, "Enforcing overwrite."
        if args.overwrite and paths["output"].exists():
            shutil.rmtree(paths["output"])
    paths["output"].mkdir(parents=False, exist_ok=not enforce_overwrite)

    for p in paths.values():
        assert p.exists(), f"path {p.as_posix()} does not exist"

    return paths
