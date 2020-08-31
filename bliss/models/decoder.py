import numpy as np
import warnings
from astropy.io import fits

import torch
import torch.nn as nn
from torch.nn.functional import pad
import torch.nn.functional as F
from torch.distributions import Poisson, Normal


from .. import device
from .encoder import get_is_on_from_n_sources


def get_mgrid(slen):
    offset = (slen - 1) / 2
    x, y = np.mgrid[-offset : (offset + 1), -offset : (offset + 1)]
    mgrid = torch.tensor(np.dstack((y, x))) / offset
    return mgrid.type(torch.FloatTensor).to(device)


def get_psf_params(psfield_fit_file, bands):
    psfield = fits.open(psfield_fit_file)

    psf_params = torch.zeros(len(bands), 6)

    for i in range(len(bands)):
        band = bands[i]

        sigma1 = psfield[6].data["psf_sigma1"][0][band] ** 2
        sigma2 = psfield[6].data["psf_sigma2"][0][band] ** 2
        sigmap = psfield[6].data["psf_sigmap"][0][band] ** 2

        beta = psfield[6].data["psf_beta"][0][band]
        b = psfield[6].data["psf_b"][0][band]
        p0 = psfield[6].data["psf_p0"][0][band]

        psf_params[i] = torch.log(torch.tensor([sigma1, sigma2, sigmap, beta, b, p0]))

    return psf_params


class PowerLawPSF(nn.Module):
    def __init__(self, init_psf_params, psf_slen=25, image_slen=101):

        super(PowerLawPSF, self).__init__()
        assert len(init_psf_params.shape) == 2
        assert image_slen % 2 == 1, "image_slen must be odd"
        assert psf_slen % 2 == 1, "psf_slen must be odd"

        self.n_bands = init_psf_params.shape[0]

        self.psf_slen = psf_slen
        self.image_slen = image_slen

        grid = get_mgrid(self.psf_slen) * (self.psf_slen - 1) / 2
        self.cached_radii_grid = (grid ** 2).sum(2).sqrt().to(device)

        # initial weights
        self.params = nn.Parameter(init_psf_params.clone(), requires_grad=True)

        # get normalization_constant
        self.normalization_constant = torch.zeros(self.n_bands)
        for i in range(self.n_bands):
            psf_i = self._get_psf_single_band(init_psf_params[i])
            self.normalization_constant[i] = 1 / psf_i.sum()
        self.normalization_constant = self.normalization_constant.detach()

        # initial psf norm
        self.init_psf_sum = self._get_psf().sum(-1).sum(-1).detach()

    @staticmethod
    def _psf_fun(r, sigma1, sigma2, sigmap, beta, b, p0):
        term1 = torch.exp(-(r ** 2) / (2 * sigma1))
        term2 = b * torch.exp(-(r ** 2) / (2 * sigma2))
        term3 = p0 * (1 + r ** 2 / (beta * sigmap)) ** (-beta / 2)
        return (term1 + term2 + term3) / (1 + b + p0)

    def _get_psf_single_band(self, psf_params):
        _psf_params = torch.exp(psf_params)
        return self._psf_fun(
            self.cached_radii_grid,
            _psf_params[0],
            _psf_params[1],
            _psf_params[2],
            _psf_params[3],
            _psf_params[4],
            _psf_params[5],
        )

    def _get_psf(self):
        # TODO make the psf function vectorized ...
        psf = torch.tensor([]).to(device)
        for i in range(self.n_bands):
            _psf = self._get_psf_single_band(self.params[i])
            _psf *= self.normalization_constant[i]
            psf = torch.cat((psf, _psf.unsqueeze(0)))

        assert (psf > 0).all()
        return psf

    def forward(self):
        psf = self._get_psf()
        norm = psf.sum(-1).sum(-1)
        psf *= (self.init_psf_sum / norm).unsqueeze(-1).unsqueeze(-1)
        l_pad = (self.image_slen - self.psf_slen) // 2

        # add padding so psf has length of image_slen
        return pad(psf, [l_pad] * 4)


class ImageDecoder(object):
    def __init__(
        self,
        galaxy_decoder,
        init_psf_params,
        background,
        n_bands=1,
        slen=50,
        tile_slen=2,
        prob_galaxy=0.0,
        max_sources_per_tile=2,
        mean_sources_per_tile=0.4,
        min_sources_per_tile=0,
        f_min=1e4,
        f_max=1e6,
        alpha=0.5,
        add_noise=True,
    ):

        self.slen = slen
        self.tile_slen = tile_slen
        self.ptile_slen = 3 * tile_slen
        assert (self.slen % self.tile_slen) == 0,\
            'we assume that the tiles paritition and cover the image.' + \
            ' So slen must be divisible by tile_slen'
        
        self.n_tiles_per_image = (self.slen / self.tile_slen) ** 2
        self.n_tiles_per_image = int(self.n_tiles_per_image)
        
        self.n_bands = n_bands
        self.background = background.to(device)

        assert len(background.shape) == 3
        assert self.background.shape[0] == self.n_bands
        # assert self.background.shape[1] == self.background.shape[2] == self.slen

        self.max_sources_per_tile = max_sources_per_tile
        self.mean_sources_per_tile = mean_sources_per_tile
        self.min_sources_per_tile = min_sources_per_tile
        self.prob_galaxy = float(prob_galaxy)
        self.add_noise = add_noise

        self.galaxy_decoder = galaxy_decoder
        self.latent_dim = 8
        self.gal_slen = 51
        if self.prob_galaxy > 0:
            self.gal_slen = self.galaxy_decoder.slen
            self.latent_dim = self.galaxy_decoder.latent_dim
            assert self.galaxy_decoder.n_bands == self.n_bands

        # prior parameters
        self.f_min = f_min
        self.f_max = f_max
        self.alpha = alpha  # pareto parameter.

        self.cached_grid = get_mgrid(self.ptile_slen)

        # load psf_params
        self.power_law_psf = PowerLawPSF(init_psf_params.clone())

    def _trim_psf(self, psf):
        """Crop the psf to length ptile_slen x ptile_slen,
        centered at the middle.
        """

        # if self.ptile_slen is even, we still make psf dimension odd.
        # otherwise, the psf won't have a peak in the center pixel.
        _slen = self.ptile_slen + ((self.ptile_slen % 2) == 0) * 1

        psf_slen = psf.shape[2]
        psf_center = (psf_slen - 1) / 2

        assert psf_slen >= _slen

        r = np.floor(_slen / 2)
        l_indx = int(psf_center - r)
        u_indx = int(psf_center + r + 1)

        return psf[:, l_indx:u_indx, l_indx:u_indx]

    def _expand_psf(self, psf):
        """Pad the psf with zeros so that it is size ptile_slen,
        """

        _slen = self.ptile_slen + ((self.ptile_slen % 2) == 0) * 1
        psf_slen = psf.shape[2]

        assert psf_slen <= _slen, "Should be using trim psf."

        psf_expanded = torch.zeros(self.n_bands, _slen, _slen, device=device)
        offset = int((_slen - psf_slen) / 2)

        psf_expanded[
            :, offset : (offset + psf_slen), offset : (offset + psf_slen)
        ] = psf

        return psf_expanded

    def _get_psf(self):
        # use power_law_psf and current psf parameters to forward and obtain fresh psf model.
        # first dimension of psf is number of bands
        # dimension of the psf/slen should be odd
        psf = self.power_law_psf.forward()
        psf_slen = psf.shape[2]
        assert len(psf.shape) == 3
        assert psf.shape[1] == psf_slen
        assert (psf_slen % 2) == 1
        assert self.background.shape[0] == psf.shape[0] == self.n_bands

        if self.ptile_slen >= psf.shape[-1]:
            return self._expand_psf(psf)
        else:
            return self._trim_psf(psf)

    def _sample_n_sources(self, batch_size):
        # always poisson distributed.

        m = Poisson(
            torch.full((1,), self.mean_sources_per_tile, dtype=torch.float, device=device)
        )
        n_sources = m.sample([batch_size, self.n_tiles_per_image])

        # long() here is necessary because used for indexing and one_hot encoding.
        n_sources = n_sources.clamp(max=self.max_sources_per_tile, min=self.min_sources_per_tile)
        n_sources = n_sources.long().squeeze(-1)
        return n_sources

    def _sample_locs(self, is_on_array, batch_size):

        # 2 = (x,y)
        locs = torch.rand(batch_size, self.n_tiles_per_image, self.max_sources_per_tile, 2, device=device)
        locs *= is_on_array.unsqueeze(-1)

        return locs

    def _sample_n_galaxies_and_stars(self, n_sources, is_on_array):
        batch_size = n_sources.size(0)
        uniform = torch.rand(batch_size, self.n_tiles_per_image, self.max_sources_per_tile, device=device)
        galaxy_bool = uniform < self.prob_galaxy
        galaxy_bool = (galaxy_bool * is_on_array).float()
        star_bool = (1 - galaxy_bool) * is_on_array
        n_galaxies = galaxy_bool.sum(-1)
        n_stars = star_bool.sum(-1)
        assert torch.all(n_stars <= n_sources) and torch.all(n_galaxies <= n_sources)

        return n_galaxies, n_stars, galaxy_bool, star_bool

    def _pareto_cdf(self, x):
        return 1 - (self.f_min / x) ** self.alpha

    def _draw_pareto_maxed(self, shape):
        # draw pareto conditioned on being less than f_max

        u_max = self._pareto_cdf(self.f_max)
        uniform_samples = torch.rand(*shape, device=device) * u_max
        return self.f_min / (1.0 - uniform_samples) ** (1 / self.alpha)

    @staticmethod
    def _get_log_fluxes(fluxes):
        """
        To obtain fluxes from log_fluxes.

        >> is_on_array = get_is_on_from_n_stars(n_stars, max_sources)
        >> fluxes = np.exp(log_fluxes) * is_on_array
        """
        log_fluxes = torch.where(
            fluxes > 0, fluxes, torch.ones(*fluxes.shape).to(device)
        )  # prevent log(0) errors.
        log_fluxes = torch.log(log_fluxes)

        return log_fluxes

    def _sample_fluxes(self, n_stars, star_bool, batch_size):
        """

        :return: fluxes, a shape (batch_size x max_sources x n_bands) tensor
        """
        assert n_stars.shape[0] == batch_size

        shape = (batch_size, self.n_tiles_per_image, self.max_sources_per_tile)
        base_fluxes = self._draw_pareto_maxed(shape)

        if self.n_bands > 1:
            colors = (
                torch.rand(
                    batch_size, self.n_tiles_per_image, self.max_sources_per_tile, self.n_bands - 1, device=device
                )
                * 0.15
                + 0.3
            )
            _fluxes = 10 ** (colors / 2.5) * base_fluxes.unsqueeze(-1)

            fluxes = torch.cat((base_fluxes.unsqueeze(-1), _fluxes), dim=3)
            fluxes *= star_bool.unsqueeze(-1)
        else:
            fluxes = (base_fluxes * star_bool.float()).unsqueeze(-1)

        return fluxes

    def _sample_galaxy_params(self, n_galaxies, galaxy_bool):
        # galaxy params are just Normal(0,1) variables.

        assert len(n_galaxies.shape) == 2
        batch_size = n_galaxies.size(0)

        mean = torch.zeros(1, dtype=torch.float, device=device)
        std = torch.ones(1, dtype=torch.float, device=device)
        p_z = Normal(mean, std)
        sample_shape = torch.tensor([batch_size, self.n_tiles_per_image, self.max_sources_per_tile, self.latent_dim])
        galaxy_params = p_z.rsample(sample_shape)
        galaxy_params = galaxy_params.reshape(
            batch_size, self.n_tiles_per_image, self.max_sources_per_tile, self.latent_dim
        )

        # zero out excess according to galaxy_bool.
        galaxy_params = galaxy_params * galaxy_bool.unsqueeze(-1)
        return galaxy_params

    def sample_parameters(self, batch_size=1):
        n_sources = self._sample_n_sources(batch_size)
        is_on_array = get_is_on_from_n_sources(n_sources, self.max_sources_per_tile)
        locs = self._sample_locs(is_on_array, batch_size)

        n_galaxies, n_stars, galaxy_bool, star_bool = self._sample_n_galaxies_and_stars(
            n_sources, is_on_array
        )
        galaxy_params = self._sample_galaxy_params(n_galaxies, galaxy_bool)

        fluxes = self._sample_fluxes(n_sources, star_bool, batch_size)
        log_fluxes = self._get_log_fluxes(fluxes)

        return {
            "n_sources": n_sources,
            "n_galaxies": n_galaxies,
            "n_stars": n_stars,
            "locs": locs,
            "galaxy_params": galaxy_params,
            "fluxes": fluxes,
            "log_fluxes": log_fluxes,
            "galaxy_bool": galaxy_bool,
        }

    @staticmethod
    def _apply_noise(images_mean):
        # add noise to images.

        if torch.any(images_mean <= 0):
            warnings.warn("image mean less than 0")
            images_mean = images_mean.clamp(min=1.0)

        _images = torch.sqrt(images_mean)
        images = _images * torch.randn(*images_mean.shape, device=device)
        images = images + images_mean

        return images

    def _render_one_source(self, locs, source):
        """
        :param locs: is n_ptiles x len((x,y))
        :param source: is a (n_ptiles, n_bands, slen, slen) tensor, which could either be a
                        `expanded_psf` (psf repeated multiple times) for the case of of stars.
                        Or multiple galaxies in the case of galaxies.
        :return: shape = (n_ptiles x n_bands x slen x slen)
        """
        n_ptiles = locs.shape[0]
        assert locs.shape[1] == 2
        assert source.shape[0] == n_ptiles
        
        # scale so that they land in the tile within the padded tile 
        padding = (self.ptile_slen - self.tile_slen) / 2
        locs = locs * (self.tile_slen / self.ptile_slen) + (padding / self.ptile_slen)
        # scale locs so they take values between -1 and 1 for grid sample
        locs = (locs - 0.5) * 2
        _grid = self.cached_grid.view(1, self.ptile_slen, self.ptile_slen, 2)
        grid_loc = _grid - locs[:, [1, 0]].view(n_ptiles, 1, 1, 2)
        source_rendered = F.grid_sample(source, grid_loc, align_corners=True)
        return source_rendered

    def render_multiple_stars_on_ptile(self, locs, fluxes, star_bool):
        # locs: is (n_ptiles x max_num_stars x 2)
        # fluxes: Is (n_ptiles x n_bands x max_stars)
        # star_bool: Is (n_ptiles x max_stars)
        # max_sources obtained from locs, allows for more flexibility when rendering.

        psf = self._get_psf()
        n_ptiles = locs.shape[0]
        max_sources = locs.shape[1]
        ptile_shape = (n_ptiles, self.n_bands, self.ptile_slen, self.ptile_slen)
        ptile = torch.zeros(ptile_shape, device=device)

        assert len(psf.shape) == 3  # the shape is (n_bands, ptile_slen, ptile_slen)
        assert psf.shape[0] == self.n_bands
        assert fluxes.shape[0] == star_bool.shape[0] == n_ptiles
        assert fluxes.shape[1] == star_bool.shape[1] == max_sources
        assert fluxes.shape[2] == psf.shape[0] == self.n_bands

        # all stars are just the PSF so we copy it.
        expanded_psf = psf.expand(n_ptiles, self.n_bands, -1, -1)

        # this loop plots each of the ith star in each of the (n_ptiles) images.
        for n in range(max_sources):
            star_bool_n = star_bool[:, n]
            locs_n = locs[:, n, :]
            fluxes_n = fluxes[:, n, :] * star_bool_n.unsqueeze(1)
            fluxes_n = fluxes_n.view(n_ptiles, self.n_bands, 1, 1)
            one_star = self._render_one_source(locs_n, expanded_psf)
            ptile += one_star * fluxes_n

        return ptile

    def render_multiple_galaxies_on_ptile(self, locs, galaxy_params, galaxy_bool):
        # max_sources obtained from locs, allows for more flexibility when rendering.
        n_ptiles = locs.shape[0]
        max_sources = locs.shape[1]
        ptile_shape = (n_ptiles, self.n_bands, self.ptile_slen, self.ptile_slen)
        ptile = torch.zeros(ptile_shape, device=device)

        assert galaxy_params.shape[0] == galaxy_bool.shape[0] == n_ptiles
        assert galaxy_params.shape[1] == galaxy_bool.shape[1] == max_sources
        assert galaxy_params.shape[2] == self.latent_dim

        if galaxy_bool.sum() > 0 and self.galaxy_decoder is not None:
            z = galaxy_params.reshape(-1, self.latent_dim)
            gal, _ = self.galaxy_decoder.forward(z)
            gal_shape = (n_ptiles, -1, self.n_bands, self.gal_slen, self.gal_slen)
            single_galaxies = gal.reshape(gal_shape)
            for n in range(max_sources):
                galaxy_bool_n = galaxy_bool[:, n]
                locs_n = locs[:, n, :]
                _galaxy = single_galaxies[:, n, :, :, :]
                galaxy = _galaxy * galaxy_bool_n.reshape(-1, self.n_bands, 1, 1)
                one_galaxy = self._render_one_source(locs_n, galaxy)
                ptile += one_galaxy

        return ptile

    def _render_ptiles(
        self, max_sources, n_sources, locs, galaxy_bool, galaxy_params, fluxes
    ):
        # n_sources: is (n_ptiles)
        # locs: is (n_ptiles x max_sources x 2)
        # galaxy_bool: Is (n_ptiles x max_sources)
        # galaxy_params : is (n_ptiles x max_sources x galaxy_decoder.latent_dim)
        # fluxes: Is (n_ptiles x max_sources x 2)

        
        # explicit consistent shapes.
        assert len(n_sources.shape) == 1
        assert (n_sources <= max_sources).all()
        n_ptiles = n_sources.shape[0]
        is_on_array = get_is_on_from_n_sources(n_sources, max_sources)

        locs = locs.reshape(n_ptiles, max_sources, 2)
        galaxy_bool = galaxy_bool.reshape(n_ptiles, max_sources)
        star_bool = (1 - galaxy_bool) * is_on_array
        star_bool = star_bool.reshape(n_ptiles, max_sources)
        galaxy_params = galaxy_params.reshape(n_ptiles, max_sources, self.latent_dim)
        fluxes = fluxes.reshape(n_ptiles, max_sources, self.n_bands)

        # need n_sources because `*_locs` are not necessarily ordered.
        galaxies = self.render_multiple_galaxies_on_ptile(locs, galaxy_params, galaxy_bool)
        stars = self.render_multiple_stars_on_ptile(locs, fluxes, star_bool)

        # shape = (n_images x n_bands x slen x slen)
        images = galaxies + stars

        # add background and noise
        images = images + self.background.unsqueeze(0)
        if self.add_noise:
            images = self._apply_noise(images)

        return images
    
    def render_ptiles(self, max_sources, n_sources, locs, galaxy_bool, galaxy_params, fluxes): 
        # n_sources: is (batchsize x n_tiles_per_image)
        # locs: is (batchsize x n_tiles_per_image x max_sources x 2)
        # galaxy_bool: Is (batchsize x n_tiles_per_image x max_sources)
        # galaxy_params : is (batchsize x n_tiles_per_image x max_sources x galaxy_decoder.latent_dim)
        # fluxes: Is (batchsize x n_tiles_per_image x max_sources x 2)
        
        # returns the ptiles in shape batchsize x n_tiles_per_image x ptile_slen x ptile_slen
        
        batch_size = n_sources.shape[0]
        n_ptiles = batch_size * self.n_tiles_per_image
        
        # reshape parameters
        n_sources_flattened = n_sources.reshape(n_ptiles)
        locs_flattened = locs.reshape(n_ptiles, self.max_sources_per_tile, 2)  
        galaxy_bool_flattened = galaxy_bool.reshape(n_ptiles, self.max_sources_per_tile)
        galaxy_params_flattened = galaxy_params.reshape(n_ptiles, self.max_sources_per_tile, self.latent_dim)
        fluxes_flattened = fluxes.reshape(n_ptiles, self.max_sources_per_tile, self.n_bands)
        
        image_ptiles = self._render_ptiles(max_sources,
                                           n_sources_flattened, 
                                           locs_flattened,
                                           galaxy_bool_flattened,
                                           galaxy_params_flattened,
                                           fluxes_flattened)
        return image_ptiles.reshape(batch_size, self.n_tiles_per_image, 
                                    self.n_bands, self.ptile_slen, self.ptile_slen)
    
    def render_images(
        self, max_sources, n_sources, locs, galaxy_bool, galaxy_params, fluxes
    ):
        
        image_ptiles = self.render_ptiles(self.tile_slen, max_sources, n_sources,
                                          locs, galaxy_bool, galaxy_params, fluxes)
        
        return construct_full_image_from_tiles(image_ptiles)
 
    
def construct_full_image_from_tiles(image_ptiles): 
    # image_tiles is (batchsize, n_tiles_per_image, n_bands, ptile_slen x ptile_slen)
    batchsize = image_ptiles.shape[0]
    n_tiles_per_image = image_ptiles.shape[1]
    n_bands = image_ptiles.shape[2]
    ptile_slen = image_ptiles.shape[3]
    assert image_ptiles.shape[4] == ptile_slen

    n_tiles1 = np.sqrt(n_tiles_per_image)
    # check it is an integer
    assert n_tiles1 % 1 == 0
    n_tiles1 = int(n_tiles1)
    
    image_tiles_4d = image_ptiles.view(batchsize, n_tiles1, n_tiles1, n_bands, ptile_slen, ptile_slen)
    
    # zero pad tiles, so that the number of tiles in a row (and colmn)
    # are divisible by 3. 
    n_tiles_pad = 3 - (n_tiles1 % 3)
    zero_pads1 = torch.zeros(batchsize, n_tiles_pad, n_tiles1, n_bands, ptile_slen, ptile_slen, device = device)
    zero_pads2 = torch.zeros(batchsize, n_tiles1+1, n_tiles_pad, n_bands, ptile_slen, ptile_slen, device = device)
    image_tiles_4d = torch.cat((image_tiles_4d, zero_pads1), dim = 1)
    image_tiles_4d = torch.cat((image_tiles_4d, zero_pads2), dim = 2)
    
    # construct the full image
    tile_slen = int(ptile_slen / 3)
    n_tiles = n_tiles1 + n_tiles_pad
    canvas = torch.zeros(batchsize, n_bands, (n_tiles + 2) * tile_slen, (n_tiles + 2) * tile_slen, device = device)
    for i in range(3): 
        for j in range(3): 
            indx_vec1 = torch.arange(start = i, end = n_tiles, step = 3)
            indx_vec2 = torch.arange(start = j, end = n_tiles, step = 3)
            
            canvas_len = len(indx_vec1) * ptile_slen
            
            image_tile_rows = image_tiles_4d[:, indx_vec1]
            image_tile_cols = image_tile_rows[:, :, indx_vec2]
            
            # get canvas
            start_x =  i * tile_slen
            canvas[:, :, 
                   (i * tile_slen):(i * tile_slen + canvas_len), 
                   (j * tile_slen):(j * tile_slen + canvas_len)] += \
                image_tile_cols.permute(0, 3, 1, 4, 2, 5).reshape(batchsize, n_bands, canvas_len, canvas_len)
            
    # trim to original image size
    return canvas[:, :, tile_slen:(n_tiles1 * tile_slen), tile_slen:(n_tiles1 * tile_slen)]