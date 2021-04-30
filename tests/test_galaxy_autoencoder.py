def test_galaxy_autoencoder(galaxy_ae_setup, devices):
    use_cuda = devices.use_cuda
    overrides = {
        "model": "galaxy_net",
        "dataset": "toy_gaussian",
        "dataset.batch_size": 64 if use_cuda else 10,
        "dataset.n_batches": 10 if use_cuda else 1,
        "training": "unittest",
        "training.n_epochs": 101 if use_cuda else 2,
        "training.trainer.check_val_every_n_epoch": 50 if use_cuda else 1,
        "optimizer": "adam",
    }

    # train galaxy_vae
    galaxy_ae = galaxy_ae_setup.get_trained_ae(overrides)
    results = galaxy_ae_setup.test_ae(overrides, galaxy_ae)

    # only expect tests to pass in cuda:
    if not devices.use_cuda:
        return

    assert results["max_residual"] < 25
