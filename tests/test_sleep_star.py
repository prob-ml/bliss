import pytest
import torch


@pytest.mark.parametrize("n_stars", ["1", "3"])
def test_star_sleep(n_stars, get_dataset, get_trained_encoder, paths, devices):
    # create trained encoder
    device = devices.device
    use_cuda = devices.use_cuda
    overrides = dict(
        model="basic_sleep_star",
        dataset="default" if use_cuda else "cpu",
        training="tests_default" if use_cuda else "cpu",
    )
    dataset = get_dataset(overrides)
    trained_encoder = get_trained_encoder(dataset, overrides)
    test_star = torch.load(paths["data"].joinpath(f"{n_stars}_star_test.pt"))
    test_image = test_star["images"].to(device)

    with torch.no_grad():
        # get the estimated params
        trained_encoder.eval()
        (
            n_sources,
            locs,
            galaxy_params,
            log_fluxes,
            galaxy_bool,
        ) = trained_encoder.map_estimate(test_image)

    # we only expect our assert statements to be true
    # when the model is trained in full, which requires cuda
    if not use_cuda:
        return

    # test n_sources and locs
    assert n_sources == test_star["n_sources"].to(device)

    diff_locs = test_star["locs"].sort(1)[0].to(device) - locs.sort(1)[0]
    diff_locs *= test_image.size(-1)
    assert diff_locs.abs().max() <= 0.5

    # test fluxes
    diff = test_star["log_fluxes"].sort(1)[0].to(device) - log_fluxes.sort(1)[0]
    assert torch.all(diff.abs() <= log_fluxes.sort(1)[0].abs() * 0.10)
    assert torch.all(
        diff.abs() <= test_star["log_fluxes"].sort(1)[0].abs().to(device) * 0.10
    )
