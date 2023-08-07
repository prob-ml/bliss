import galsim
import torch

from bliss import reporting
from bliss.catalog import FullCatalog


def test_scene_metrics():
    true_params = FullCatalog(
        100,
        100,
        {
            "plocs": torch.tensor([[[50.0, 50.0]]]).float(),
            "galaxy_bools": torch.tensor([[[1]]]).float(),
            "n_sources": torch.tensor([1]).long(),
            "mags": torch.tensor([[[23.0]]]).float(),
        },
    )
    reporting.scene_metrics(true_params, true_params, "mags", p_max=25, slack=1.0)


def test_catalog_conversion():
    true_params = FullCatalog(
        100,
        100,
        {
            "plocs": torch.tensor([[[51.8, 49.6]]]).float(),
            "galaxy_bools": torch.tensor([[[1]]]).float(),
            "n_sources": torch.tensor([1]).long(),
            "mags": torch.tensor([[[23.0]]]).float(),
        },
    )
    true_tile_params = true_params.to_tile_params(tile_slen=4, max_sources_per_tile=1)
    tile_params_tilde = true_tile_params.to_full_params()
    assert true_params.equals(tile_params_tilde)


def test_measurements():
    slen = 40
    pixel_scale = 0.2
    psf = galsim.Gaussian(sigma=0.2)
    gal = galsim.Gaussian(sigma=1.0)
    gal_conv = galsim.Convolution(gal, psf)
    psf_image = psf.drawImage(nx=slen, ny=slen, scale=pixel_scale).array.reshape(1, slen, slen)
    image = gal_conv.drawImage(nx=slen, ny=slen, scale=pixel_scale).array.reshape(1, 1, slen, slen)
    image, psf_image = torch.from_numpy(image), torch.from_numpy(psf_image)
    reporting.get_single_galaxy_measurements(image, psf_image, pixel_scale)
