def test_galaxy_flow(vae_setup):
    overrides = {
        "mode": "train",
        "training": "sdss_vae_flow",
        "datasets.sdss_galaxies.batch_size": 10,
        "datasets.sdss_galaxies.n_batches": 1,
        "datasets.sdss_galaxies.num_workers": 0,
        "training.n_epochs": 2,
    }

    # train galaxy_vae
    vae_setup.get_trained_model(overrides)
