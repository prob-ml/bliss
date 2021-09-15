def test_plotting_sleep(model_setup):
    # just to test `make_validation_plots` works.
    overrides = {
        "model": "sleep_star_basic",
        "dataset": "test_plotting",
        "training": "test_plotting",
        "model.kwargs.annotate_probs": True,
    }
    model_setup.get_trained_model(overrides)


def test_plotting_binary(model_setup):
    # just to test `make_validation_plots` works.
    overrides = {
        "model": "binary_sdss",
        "dataset": "test_plotting",
        "training": "test_plotting",
    }
    model_setup.get_trained_model(overrides)
