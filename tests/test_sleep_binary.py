def test_binary(model_setup, devices):
    overrides = {
        "model": "sleep_1galaxy_1star",
        "dataset": "default" if devices.use_cuda else "cpu",
        "training": "unittest" if devices.use_cuda else "cpu",
        "optimizer": "m2",
    }
    trained_sleep = model_setup.get_trained_model(overrides)

    overrides.update({"testing": "default"})
    results = model_setup.test_model(overrides, trained_sleep)

    # only check testing results if GPU available
    if not devices.use_cuda:
        return

    # check testing results are sensible.
    assert results["acc_counts"] > 0.6
    assert results["locs_mae"] < 0.65
