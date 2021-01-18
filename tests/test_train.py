from bliss import train
from hydra.experimental import initialize, compose


def test_train_run(devices):
    overrides = {
        "training": "default" if devices.use_cuda else "cpu",
        "dataset": "default" if devices.use_cuda else "cpu",
    }
    overrides = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(config_path="../config"):
        cfg = compose("config", overrides=overrides)
        train.main(cfg)
