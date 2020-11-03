def test_plotting(get_sleep_setup):
    # just to test `make_validation_plots` works.
    overrides = dict(
        model="basic_sleep_star", training="test_plotting", dataset="test_plotting"
    )
    train_sleep(overrides)
