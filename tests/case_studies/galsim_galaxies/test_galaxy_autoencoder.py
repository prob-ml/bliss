def test_galaxy_autoencoder_bulge_disk(galsim_galaxies_setup, devices):
    use_cuda = devices.use_cuda
    overrides = {
        "mode": "train",
        "training": "sdss_autoencoder",
        "datasets.sdss_galaxies.batch_size": 128 if use_cuda else 10,
        "datasets.sdss_galaxies.n_batches": 5 if use_cuda else 1,
        "datasets.sdss_galaxies.num_workers": 10 if use_cuda else 0,
        "training.n_epochs": 101 if use_cuda else 2,
    }

    # train galaxy_vae
    galaxy_ae = galsim_galaxies_setup.get_trained_model(overrides)
    results = galsim_galaxies_setup.test_model(overrides, galaxy_ae)

    # only expect tests to pass in cuda:
    if not devices.use_cuda:
        return

    assert results["max_residual"] < 50
