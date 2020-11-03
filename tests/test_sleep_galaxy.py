def test_n_sources_and_locs(train_sleep, devices):
    use_cuda = devices.use_cuda
    overrides = dict(
        model="basic_sleep_galaxy",
        training="unittest" if devices.use_cuda else "cpu",
        dataset="default" if devices.use_cuda else "cpu",
    )
    _, test_results = train_sleep(overrides)
    results = test_results[0]

    if not use_cuda:
        return

    # check test results are sensible.
    assert results["acc_gal_counts"] > 0.8
    assert results["locs_mse"] < 0.5
