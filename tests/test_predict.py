from hydra.experimental import initialize, compose
from bliss import predict


def test_predict_run(devices):
    overrides = {
        "mode": "predict",
        "predict": "sdss_basic",
        "predict.output_file": "null",
        "predict.device": f"cuda:{devices.device.index}" if devices.use_cuda else "cpu",
        "predict.testing": True,
    }
    overrides = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(config_path="../config"):
        cfg = compose("config", overrides=overrides)
        predict.predict(cfg)
