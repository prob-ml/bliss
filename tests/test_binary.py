def test_binary(model_setup, devices):
    overrides = {
        "model": "binary_sdss",
        "dataset": "binary",
        "dataset.n_batches": 10 if devices.use_cuda else 1,
        "dataset.batch_size": 32 if devices.use_cuda else 2,
        "dataset.generate_device": "cuda:0" if devices.use_cuda else "cpu",
        "training": "unittest" if devices.use_cuda else "cpu",
        "optimizer": "adam",
        "optimizer.kwargs.lr": 1e-4,
        "dataset.decoder.mean_sources": 0.03,
        "training.n_epochs": 50 if devices.use_cuda else 2,
    }

    trained_binary = model_setup.get_trained_model(overrides)
    results = model_setup.test_model(overrides, trained_binary)

    if devices.use_cuda:
        assert results["acc"] > 0.85
