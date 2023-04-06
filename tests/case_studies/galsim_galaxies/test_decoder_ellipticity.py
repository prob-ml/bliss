from hydra.utils import instantiate

from bliss.simulator.decoder import ImageDecoder
from bliss.simulator.prior import ImagePrior


def test_get_ellips(get_galsim_galaxies_config):
    cfg = get_galsim_galaxies_config({})
    image_prior: ImagePrior = instantiate(cfg.models.prior, prob_galaxy=1.0, mean_sources=0.33)
    decoder: ImageDecoder = instantiate(cfg.models.decoder)
    tile_cat = image_prior.sample_prior(decoder.tile_slen, 1, 3, 3)
    tile_cat.set_galaxy_ellips(decoder, scale=0.393)
