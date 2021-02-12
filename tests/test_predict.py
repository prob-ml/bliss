from hydra.experimental import initialize, compose
from bliss import train


def test_predict_run():
    overrides = {
        "predict": "sdss_basic",
        "predict.output_file": None,
        "predict.device": "cpu",
    }
    overrides = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(config_path="../config"):
        cfg = compose("config", overrides=overrides)
        train.main(cfg)
