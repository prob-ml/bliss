def test_plotting(get_dataset, get_trained_encoder):
    # just to test `make_validation_plots` works.
    n_epochs = 5
    batch_size = 3
    n_batches = 1
    overrides = ["model=basic_sleep_star", "training=test_plotting"]
    dataset = get_dataset(batch_size, n_batches, overrides=overrides)
    _ = get_trained_encoder(n_epochs, dataset, overrides=overrides)
