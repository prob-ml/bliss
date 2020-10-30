def test_star_sleep_one_tile(train_sleep, devices):
    use_cuda = devices.use_cuda
    _ = devices.device
    overrides = {
        "model": "basic_sleep_star_one_tile",
        "dataset": "single_tile" if use_cuda else "cpu",
        "training": "tests_default" if use_cuda else "cpu",
    }
    _, test_results = train_sleep(overrides)

    # check test results are sensible.
    assert test_results is not None


def test_star_sleep(train_sleep, devices):
    use_cuda = devices.use_cuda
    _ = devices.device
    overrides = dict(
        model="basic_sleep_star",
        dataset="default" if use_cuda else "cpu",
        training="tests_default" if use_cuda else "cpu",
    )
    _, test_results = train_sleep(overrides)

    # check test results are sensible.
    assert test_results is not None
