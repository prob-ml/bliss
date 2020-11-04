def test_plotting(sleep_setup):
    # just to test `make_validation_plots` works.
    overrides = dict(
        model="basic_sleep_star", dataset="test_plotting", training="test_plotting"
    )
    sleep_setup.get_trained_sleep(overrides)
