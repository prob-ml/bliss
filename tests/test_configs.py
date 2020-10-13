from hydra.experimental import initialize, compose
from bliss.datasets import simulated
from bliss.sleep import SleepPhase


def test_basic_configs():
    # test creating star model using config files.
    with initialize(config_path="../config"):
        cfg = compose("config", overrides=["model=basic_sleep_star"])
        dataset = simulated.SimulatedDataset(cfg)
        sleep = SleepPhase(cfg, dataset)
        image_decoder = dataset.image_decoder
        image_encoder = sleep.image_encoder

        assert image_decoder.slen == image_encoder.slen
        assert image_encoder.n_bands == image_decoder.n_bands == 1
        assert image_encoder.n_galaxy_params == image_decoder.n_galaxy_params == 8
        assert image_decoder.tile_slen == image_encoder.tile_slen
