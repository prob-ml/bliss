import shutil

from hydra import compose, initialize

from bliss import train


def test_train_run(paths):
    overrides = {
        "training": "cpu",
        "training.experiment": "unittest",
        "training.trainer.logger": True,
        "dataset": "cpu",
    }
    overrides = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(config_path="../config"):
        cfg = compose("config", overrides=overrides)
        train.train(cfg)
    shutil.rmtree(f'{paths["output"]}/unittest')
