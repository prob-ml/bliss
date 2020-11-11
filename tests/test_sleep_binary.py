def test_binary(sleep_setup, devices):
    overrides = dict(
        model="sleep_1galaxy_1star",
        dataset="default" if devices.use_cuda else "cpu",
        training="unittest" if devices.use_cuda else "cpu",
    )
    trained_sleep = sleep_setup.get_trained_sleep(overrides)

    overrides.update({"testing": "default"})
    results = sleep_setup.test_sleep(overrides, trained_sleep)

    # only check testing results if GPU available
    if not devices.use_cuda:
        return

    # check testing results are sensible.
    assert results["acc_counts"] > 0.7
    assert results["acc_gal_counts"] > 0.7
    assert results["locs_median_mse"] < 0.5
