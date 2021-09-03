def test_binary(model_setup, devices):
    overrides = {
        "model": "binary_sdss",
        "dataset": "default" if devices.use_cuda else "cpu",
        "training": "unittest" if devices.use_cuda else "cpu",
        "optimizer": "adam",
        "optimizer.kwargs.lr": 1e-4,
        "model.kwargs.decoder_kwargs.mean_sources": 0.03,
        "training.n_epochs": 300 if devices.use_cuda else 2,
    }

    trained_binary = model_setup.get_trained_model(overrides)
    results = model_setup.test_model(overrides, trained_binary)

    if devices.use_cuda:
        assert results["acc"] > 0.85
