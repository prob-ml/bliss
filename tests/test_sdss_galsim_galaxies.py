from hydra.experimental import initialize, compose
from bliss.datasets.galsim_galaxies import SDSSGalaxies, SavedGalaxies


def test_sdss_galaxies():
    # just a smoke test for now, until we get VAE working.
    overrides = {"dataset": "sdss_galaxies", "dataset.batch_size": 10, "dataset.n_batches": 1}
    overrides = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(config_path="../config"):
        cfg = compose("config", overrides=overrides)
        ds = SDSSGalaxies(cfg)
        for _ in ds.train_dataloader():
            pass
        for _ in ds.val_dataloader():
            pass
        for _ in ds.test_dataloader():
            pass


def test_saved_galaxies():
    # just a smoke test
    overrides = {"dataset": "saved_galaxies", "dataset.batch_size": 64}
    overrides = [f"{k}={v}" for k, v in overrides.items()]
    with initialize(config_path="../config"):
        cfg = compose("config", overrides=overrides)
        ds = SavedGalaxies(cfg)
        for _ in ds.train_dataloader():
            pass
        for _ in ds.val_dataloader():
            pass
        for _ in ds.test_dataloader():
            pass
