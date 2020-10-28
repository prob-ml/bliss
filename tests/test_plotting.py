def test_plotting(get_dataset, get_trained_encoder):
    # just to test `make_validation_plots` works.
    overrides = dict(
        model="basic_sleep_star", training="test_plotting", dataset="test_plotting"
    )
    dataset = get_datamodule()
    _ = get_trained_encoder(dataset, overrides)
