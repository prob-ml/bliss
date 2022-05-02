import os

import numpy as np
import pytest
import torch
from einops import rearrange, reduce

from bliss.catalog import TileCatalog, get_images_in_tiles
from bliss.models.location_encoder import LocationEncoder


def _get_tpr_ppv(true_locs, true_mag, est_locs, est_mag, slack=1.0):

    # l-infty error in location,
    # matrix of true x est error
    locs_error = torch.abs(est_locs.unsqueeze(0) - true_locs.unsqueeze(1)).max(-1)[0]

    # worst error in either band
    mag_error = torch.abs(est_mag.unsqueeze(0) - true_mag.unsqueeze(1)).max(-1)[0]

    tpr_bool = torch.any((locs_error < slack) * (mag_error < slack), dim=1).float()
    ppv_bool = torch.any((locs_error < slack) * (mag_error < slack), dim=0).float()

    return tpr_bool.mean(), ppv_bool.mean()


@pytest.fixture(scope="module")
def trained_star_encoder_m2(m2_model_setup, devices):
    overrides = {
        "training.trainer.check_val_every_n_epoch": 9999,
    }

    if devices.use_cuda:
        overrides.update({"training.n_epochs": 60})
    else:
        overrides.update(
            {
                "training.n_epochs": 1,
                "datasets.simulated_m2.n_batches": 1,
                "datasets.simulated_m2.batch_size": 2,
                "datasets.simulated_m2.generate_device": "cpu",
                "datasets.simulated_m2.testing_file": None,
            }
        )

    location_encoder: LocationEncoder = m2_model_setup.get_trained_model(overrides)
    return location_encoder.to(devices.device)


def get_map_estimate(
    image_encoder: LocationEncoder, images, background, slen: int, wlen: int = None
):
    # return full estimate of parameters in full image.
    # NOTE: slen*wlen is size of the image without border padding

    if wlen is None:
        wlen = slen
    assert isinstance(slen, int) and isinstance(wlen, int)
    # check image compatibility
    border1 = (images.shape[-2] - slen) / 2
    border2 = (images.shape[-1] - wlen) / 2
    assert border1 == border2, "border paddings on each dimension differ."
    assert slen % image_encoder.tile_slen == 0, "incompatible slen"
    assert wlen % image_encoder.tile_slen == 0, "incompatible wlen"
    assert border1 == image_encoder.border_padding, "incompatible border"

    # obtained estimates per tile, then on full image.
    image_ptiles = get_images_in_tiles(
        torch.cat((images, background), dim=1),
        image_encoder.tile_slen,
        image_encoder.ptile_slen,
    )
    _, n_tiles_h, n_tiles_w, _, _, _ = image_ptiles.shape
    image_ptiles = rearrange(image_ptiles, "n nth ntw b h w -> (n nth ntw) b h w")
    var_params = image_encoder.encode(image_ptiles)
    tile_cutoff = 25**2
    var_params2 = image_encoder.encode(image_ptiles[:tile_cutoff])

    assert torch.allclose(
        var_params["n_source_log_probs"][:tile_cutoff],
        var_params2["n_source_log_probs"],
        atol=1e-5,
    )
    assert torch.allclose(
        var_params["per_source_params"][:tile_cutoff],
        var_params2["per_source_params"],
        atol=1e-5,
    )

    tile_map_dict = image_encoder.variational_mode(var_params)
    tile_map = TileCatalog.from_flat_dict(
        image_encoder.tile_slen, n_tiles_h, n_tiles_w, tile_map_dict
    )
    full_map = tile_map.to_full_params()
    tile_map_tilde = full_map.to_tile_params(image_encoder.tile_slen, tile_map.max_sources)
    assert tile_map.equals(tile_map_tilde, exclude=("n_source_log_probs",), atol=1e-5)

    return full_map


class TestStarSleepEncoderM2:
    def test_star_sleep_m2(self, trained_star_encoder_m2, devices, m2_model_setup):
        device = devices.device
        cfg = m2_model_setup.get_cfg({})
        # the trained star encoder
        trained_star_encoder_m2.eval()

        # load hubble parameters and SDSS image
        hubble_data = np.load(os.path.join(cfg.paths.data, "true_hubble_m2.npz"))

        # the SDSS image
        test_image = torch.from_numpy(hubble_data["sdss_image"]).unsqueeze(0).to(device)
        slen = 100

        # the true parameters
        true_locs = torch.from_numpy(hubble_data["true_locs"]).to(device)
        true_fluxes = torch.from_numpy(hubble_data["true_fluxes"]).to(device)

        # Estimated background
        background = reduce(test_image, "n c h w -> 1 c 1 1", "min")
        background = background.expand(-1, -1, *test_image.shape[2:])

        # get estimated parameters
        estimate = get_map_estimate(
            trained_star_encoder_m2, test_image.to(device), background, slen
        )

        # check metrics if cuda is true
        if not devices.use_cuda:
            return

        # summary statistics
        sleep_tpr, sleep_ppv = _get_tpr_ppv(
            true_locs * slen,
            2.5 * torch.log10(true_fluxes[:, 0:1]),
            estimate.plocs[0],
            2.5 * torch.log10(estimate["fluxes"][0, :, 0:1]),
            slack=0.5,
        )

        print("Sleep phase TPR: ", sleep_tpr)
        print("Sleep phase PPV: ", sleep_ppv)

        assert sleep_tpr > 0.39
        assert sleep_ppv > 0.4
