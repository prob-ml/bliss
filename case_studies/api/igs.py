from importlib import resources

from omegaconf import DictConfig, OmegaConf


def base_config() -> DictConfig:
    with resources.path("bliss.conf", "base_config.yaml") as base_config_path:
        cfg = OmegaConf.load(base_config_path)
        if isinstance(cfg, DictConfig):
            return cfg

        raise ValueError("base_config.yaml should represent a dictionary")
