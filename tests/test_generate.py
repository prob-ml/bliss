from bliss import generate
from hydra.experimental import initialize, compose


def test_generate_run(devices):
    overrides = {
        "dataset.params.generate_device": "cuda:0" if devices.use_cuda else "cpu",
    }
    overrides = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(config_path="../config"):

        cfg = compose("config", overrides=overrides)
        generate.main(cfg)
