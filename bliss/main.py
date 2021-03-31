import hydra


@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    if cfg.mode == "train":
        from bliss.train import train as task
    elif cfg.mode == "tune":
        from bliss.tune import tune as task
    elif cfg.mode == "generate":
        from bliss.generate import generate as task
    elif cfg.mode == "predict":
        from bliss.predict import predict as task
    else:
        raise KeyError
    task(cfg)
