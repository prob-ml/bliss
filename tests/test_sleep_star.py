def test_star_sleep_one_tile(train_sleep, devices):
    use_cuda = devices.use_cuda
    _ = devices.device
    overrides = {
        "model": "basic_sleep_star_one_tile",
        "dataset": "single_tile" if use_cuda else "cpu",
        "training": "tests_default" if use_cuda else "cpu",
    }
    _, test_results = train_sleep(overrides)
    results = test_results[0]

    # only expect tests to pass if gpu
    if not use_cuda:
        return

    # check test results are sensible
    assert results["acc_counts"] > 0.85
    assert results["locs_mse"] < 0.75


def test_star_sleep(train_sleep, devices):
    use_cuda = devices.use_cuda
    _ = devices.device
    overrides = dict(
        model="basic_sleep_star",
        dataset="default" if use_cuda else "cpu",
        training="tests_default" if use_cuda else "cpu",
    )
    _, test_results = train_sleep(overrides)
    results = test_results[0]

    if not use_cuda:
        return

    # check test results are sensible.
    assert results["acc_counts"] > 0.85
    assert results["locs_mse"] < 0.65
