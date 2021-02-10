def test_galaxy_vae(galaxy_vae_setup, devices):
    use_cuda = devices.use_cuda
    overrides = {
        "model": "galaxy_net",
        "dataset": "toy_gaussian",
        "dataset.batch_size": 128 if use_cuda else 10,
        "dataset.n_batches": 10 if use_cuda else 1,
        "training": "unittest",
        "training.n_epochs": 101 if use_cuda else 2,
        "training.trainer.check_val_every_n_epoch": 100 if use_cuda else 1,
    }

    # train galaxy_vae
    galaxy_vae = galaxy_vae_setup.get_trained_vae(overrides)
    results = galaxy_vae_setup.test_vae(overrides, galaxy_vae)

    # only expect tests to pass in cuda:
    if not devices.use_cuda:
        return

    # check residuals from test
    assert results["max_residual"] < 5
