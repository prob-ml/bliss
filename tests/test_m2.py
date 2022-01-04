import os

import numpy as np
import pytest
import torch

from bliss.models.encoder import (
    get_full_params,
    get_images_in_tiles,
    get_params_in_batches,
    get_full_params_from_tiles,
)


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
def trained_star_encoder_m2(model_setup, devices):
    overrides = {
        "model": "sleep_m2",
        "dataset": "m2" if devices.use_cuda else "cpu",
        "training": "m2" if devices.use_cuda else "cpu",
        "training.trainer.check_val_every_n_epoch": 9999,
        "training.n_epochs": 50 if devices.use_cuda else 1,
        "optimizer": "m2",
    }

    sleep_net = model_setup.get_trained_model(overrides)
    return sleep_net.image_encoder.to(devices.device)


def get_map_estimate(image_encoder, images, slen: int, wlen: int = None):
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
    ptiles = get_images_in_tiles(images, image_encoder.tile_slen, image_encoder.ptile_slen)
    var_params = image_encoder.encode(ptiles)
    tile_map = image_encoder.max_a_post(var_params)
    tile_map = get_params_in_batches(tile_map, images.shape[0])
    tile_map["prob_n_sources"] = tile_map["prob_n_sources"].unsqueeze(-2)

    est = get_full_params(tile_map, slen, wlen)
    est2 = get_full_params_from_tiles(tile_map, image_encoder.tile_slen)
    for k in est:
        assert k in est2
        assert torch.allclose(est[k], est2[k])
    return est


class TestStarSleepEncoderM2:
    def test_star_sleep_m2(self, trained_star_encoder_m2, devices, paths):
        device = devices.device

        # the trained star encoder
        trained_star_encoder_m2.eval()

        # load hubble parameters and SDSS image
        hubble_data = np.load(os.path.join(paths["data"], "true_hubble_m2.npz"))

        # the SDSS image
        test_image = torch.from_numpy(hubble_data["sdss_image"]).unsqueeze(0).to(device)
        slen = 100

        # the true parameters
        true_locs = torch.from_numpy(hubble_data["true_locs"]).to(device)
        true_fluxes = torch.from_numpy(hubble_data["true_fluxes"]).to(device)

        # get estimated parameters
        estimate = get_map_estimate(trained_star_encoder_m2, test_image.to(device), slen)

        # check metrics if cuda is true
        if not devices.use_cuda:
            return

        # summary statistics
        sleep_tpr, sleep_ppv = _get_tpr_ppv(
            true_locs * slen,
            2.5 * torch.log10(true_fluxes[:, 0:1]),
            estimate["locs"][0] * slen,
            2.5 * torch.log10(estimate["fluxes"][0, :, 0:1]),
            slack=0.5,
        )

        print("Sleep phase TPR: ", sleep_tpr)
        print("Sleep phase PPV: ", sleep_ppv)

        assert sleep_tpr > 0.4
        assert sleep_ppv > 0.4
