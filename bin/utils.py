from pathlib import Path
import shutil
from omegaconf import DictConfig


def setup_paths(cfg: DictConfig, enforce_overwrite=True):
    root_path = Path(cfg.general.root)
    paths = {
        "root": root_path,
        "data": root_path.joinpath("data"),
        "logs": root_path.joinpath("logs"),
        "output": root_path.joinpath(f"logs/{cfg.general.output}"),
    }

    if enforce_overwrite:
        assert (
            not paths["output"].exists() or cfg.general.overwrite
        ), "Enforcing overwrite."

        if cfg.general.overwrite and paths["output"].exists():
            shutil.rmtree(paths["output"])
    paths["output"].mkdir(parents=False, exist_ok=not enforce_overwrite)

    for p in paths.values():
        assert p.exists(), f"path {p.as_posix()} does not exist"

    return paths
