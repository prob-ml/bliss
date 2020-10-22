def test_basic_configs(get_dataset, get_trained_encoder):
    # test creating star model using config files.
    overrides = dict(model="basic_sleep_star_one_tile", training="cpu", dataset="cpu")
    dataset = get_dataset(overrides)
    image_decoder = dataset.image_decoder
    image_encoder = get_trained_encoder(dataset, overrides)
    assert image_encoder.n_bands == image_decoder.n_bands == 1
    assert image_encoder.n_galaxy_params == image_decoder.n_galaxy_params == 8
    assert image_decoder.tile_slen == image_encoder.tile_slen
    assert image_decoder.prob_galaxy == 0.0
    assert dataset.n_batches == 1
