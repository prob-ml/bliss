import hydra


@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    if cfg.mode == "train":
        from bliss.train import main as task
    elif cfg.mode == "tune":
        from bliss.tune import main as task
    elif cfg.mode == "generate":
        from bliss.generate import main as task
    else:
        raise KeyError
    task(cfg)
