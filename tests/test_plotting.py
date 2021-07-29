def test_plotting(sleep_setup):
    # just to test `make_validation_plots` works.
    overrides = {
        "model": "sleep_star_basic",
        "dataset": "test_plotting",
        "training": "test_plotting",
        "model.kwargs.annotate_probs": True,
    }
    sleep_setup.get_trained_sleep(overrides)
