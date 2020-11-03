def test_plotting(get_sleep_setup, get_datamodule):
    # just to test `make_validation_plots` works.
    overrides = dict(
        model="basic_sleep_star", dataset="test_plotting", training="test_plotting"
    )
    datamodule = get_datamodule(overrides)
    sleep_net, trainer = get_sleep_setup
    trainer.fit(sleep_net, datamodule=datamodule)
