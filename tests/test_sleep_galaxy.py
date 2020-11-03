def test_n_sources_and_locs(get_sleep_setup, get_datamodule, devices):
    use_cuda = devices.use_cuda
    overrides = dict(
        model="basic_sleep_galaxy",
        dataset="default" if devices.use_cuda else "cpu",
        training="unittest" if devices.use_cuda else "cpu",
        testing="default",
    )
    datamodule = get_datamodule(overrides)
    sleep_net, trainer = get_sleep_setup(overrides)

    # train!
    trainer.fit(sleep_net, datamodule=datamodule)

    # test on data produced on-the-fly.
    results1 = trainer.test(sleep_net, datamodule=datamodule)[0]

    # and stored data
    overrides.update({"testing": "galaxy_test"})
    test_module = get_datamodule(overrides)
    results2 = trainer.test(sleep_net, datamodule=test_module)

    # only check testing results if GPU available
    if not use_cuda:
        return

    # check testing results are sensible.
    assert results1["acc_gal_counts"] > 0.8
    assert results1["locs_mse"] < 0.5

    assert results2["acc_gal_counts"] > 0.8
    assert results2["locs_mse"] < 0.5
