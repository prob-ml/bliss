from pathlib import Path
import shutil
from omegaconf import DictConfig, OmegaConf


def setup_paths(cfg: DictConfig, enforce_overwrite=True):
    paths = OmegaConf.to_container(cfg.paths, resolve=True)
    output = Path(paths["output"])
    if enforce_overwrite:
        assert not output.exists() or cfg.general.overwrite, "Enforcing overwrite."
        if cfg.general.overwrite and output.exists():
            shutil.rmtree(output)

    output.mkdir(parents=False, exist_ok=not enforce_overwrite)

    for p in paths.values():
        assert Path(p).exists(), f"path {p.as_posix()} does not exist"

    return paths
