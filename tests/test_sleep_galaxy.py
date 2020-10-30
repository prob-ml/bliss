def test_n_sources_and_locs(train_sleep, devices):
    overrides = dict(
        model="basic_sleep_galaxy",
        training="tests_default" if devices.use_cuda else "cpu",
        dataset="default" if devices.use_cuda else "cpu",
    )

    _, test_results = train_sleep(overrides)

    # check test results are sensible.
    assert test_results is not None
