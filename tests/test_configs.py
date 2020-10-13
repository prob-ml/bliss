def test_basic_configs(get_dataset, get_trained_encoder):
    # test creating star model using config files.
    overrides = ["model=basic_sleep_star"]
    dataset = get_dataset(1, 1, overrides=overrides)
    image_decoder = dataset.image_decoder
    image_encoder = get_trained_encoder(1, dataset, overrides=overrides)
    assert image_decoder.slen == image_encoder.slen
    assert image_encoder.n_bands == image_decoder.n_bands == 1
    assert image_encoder.n_galaxy_params == image_decoder.n_galaxy_params == 8
    assert image_decoder.tile_slen == image_encoder.tile_slen
