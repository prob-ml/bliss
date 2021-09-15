def test_basic_configs(model_setup):
    # test creating star model using config files.
    overrides = {"model": "sleep_star_basic", "training": "cpu", "dataset": "cpu"}
    sleep_net = model_setup.get_model(overrides)
    dataset = model_setup.get_dataset(overrides)
    image_decoder = sleep_net.image_decoder
    image_encoder = sleep_net.image_encoder
    assert image_encoder.n_bands == image_decoder.n_bands == 1
    assert image_decoder.tile_slen == image_encoder.tile_slen
    assert image_decoder.prob_galaxy == 0.0
    assert dataset.n_batches == 1
