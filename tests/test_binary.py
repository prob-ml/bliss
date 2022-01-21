def test_binary(model_setup, devices):
    overrides = {
        "mode": "train",
        "training": "sdss_binary",
    }
    if devices.use_cuda:
        overrides.update({"training.n_epochs": 50})
    else:
        overrides.update(
            {
                "dataset.n_batches": 1,
                "dataset.batch_size": 2,
                "dataset.generate_device": "cpu",
                "training.n_epochs": 2,
            }
        )

    trained_binary = model_setup.get_trained_model(overrides)
    results = model_setup.test_model(overrides, trained_binary)

    if devices.use_cuda:
        assert results["acc"] > 0.85
