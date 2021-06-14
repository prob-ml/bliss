from hydra import initialize, compose
from bliss import train


def test_train_run():
    overrides = {
        "training": "cpu",
        "dataset": "cpu",
    }
    overrides = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(config_path="../config"):
        cfg = compose("config", overrides=overrides)
        train.train(cfg)
