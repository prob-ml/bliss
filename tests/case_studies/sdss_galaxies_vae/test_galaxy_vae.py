def test_galaxy_autoencoder_toy_gaussian(vae_setup, devices):
    use_cuda = devices.use_cuda
    overrides = {
        "mode": "train",
        "training": "sdss_vae",
        "training.dataset": "${datasets.toy_gaussian}",
    }

    if use_cuda:
        overrides.update(
            {
                "training.n_epochs": 101,
                "training.trainer.check_val_every_n_epoch": 50,
            }
        )
    else:
        overrides.update(
            {
                "datasets.toy_gaussian.batch_size": 10,
                "datasets.toy_gaussian.n_batches": 1,
                "training.n_epochs": 2,
                "training.trainer.check_val_every_n_epoch": 1,
            }
        )

    # train galaxy_vae
    galaxy_ae = vae_setup.get_trained_model(overrides)
    results = vae_setup.test_model(overrides, galaxy_ae)

    # only expect tests to pass in cuda:
    if not devices.use_cuda:
        return

    assert results["max_residual"] < 25


def test_galaxy_autoencoder_bulge_disk(vae_setup, devices):
    use_cuda = devices.use_cuda
    overrides = {
        "mode": "train",
        "training": "sdss_vae",
    }

    if use_cuda:
        overrides.update(
            {
                "datasets.sdss_galaxies.batch_size": 128,
                "datasets.sdss_galaxies.n_batches": 5,
                "datasets.sdss_galaxies.num_workers": 10,
                "training.n_epochs": 101,
            }
        )
    else:
        overrides.update(
            {
                "datasets.sdss_galaxies.batch_size": 10,
                "datasets.sdss_galaxies.n_batches": 1,
                "datasets.sdss_galaxies.num_workers": 0,
                "training.n_epochs": 2,
            }
        )

    # train galaxy_vae
    galaxy_ae = vae_setup.get_trained_model(overrides)
    results = vae_setup.test_model(overrides, galaxy_ae)

    # only expect tests to pass in cuda:
    if not devices.use_cuda:
        return

    assert results["max_residual"] < 30
