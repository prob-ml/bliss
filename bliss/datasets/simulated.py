import inspect

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.distributions import Poisson, Normal

from bliss.models import galaxy_net
from bliss.models.decoder import ImageDecoder, get_is_on_from_n_sources


class SimulatedDataset(IterableDataset):
    def __init__(self, n_batches: int, batch_size: int, decoder_args, decoder_kwargs):
        super(SimulatedDataset, self).__init__()

        self.n_batches = n_batches
        self.batch_size = batch_size

        self.image_decoder = ImageDecoder(*decoder_args, **decoder_kwargs)
        self.slen = self.image_decoder.slen
        self.tile_slen = self.image_decoder.tile_slen
        self.n_bands = self.image_decoder.n_bands
        self.latent_dim = self.image_decoder.latent_dim

        self.max_sources_per_tile = self.image_decoder.max_sources_per_tile
        self.n_tiles_per_image = self.image_decoder.n_tiles_per_image
        self.max_sources_per_tile = self.image_decoder.max_sources_per_tile
        self.mean_sources_per_tile = self.image_decoder.mean_sources_per_tile
        self.min_sources_per_tile = self.image_decoder.min_sources_per_tile
        self.loc_min_per_tile = self.image_decoder.loc_min_per_tile
        self.loc_max_per_tile = self.image_decoder.loc_max_per_tile
        self.prob_galaxy = self.image_decoder.prob_galaxy
        self.f_min = self.image_decoder.f_min
        self.f_max = self.image_decoder.f_max
        self.alpha = self.image_decoder.alpha  # pareto parameter.

    def __iter__(self):
        return self.batch_generator()

    def batch_generator(self):
        for i in range(self.n_batches):
            yield self.get_batch()

    def get_batch(self):
        params = self.sample_prior(batch_size=self.batch_size)

        images = self.image_decoder.render_images(
            params["n_sources"],
            params["locs"],
            params["galaxy_bool"],
            params["galaxy_params"],
            params["fluxes"],
        )
        params.update({"images": images, "background": self.image_decoder.background})

        return params

    def sample_prior(self, batch_size=1):
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

    def _sample_n_sources(self, batch_size):
        # returns number of sources for each batch x tile
        # output dimension is batchsize x n_tiles_per_image

        # always poisson distributed.
        m = Poisson(torch.full((1,), self.mean_sources_per_tile, dtype=torch.float))
        n_sources = m.sample([batch_size, self.n_tiles_per_image])

        # long() here is necessary because used for indexing and one_hot encoding.
        n_sources = n_sources.clamp(
            max=self.max_sources_per_tile, min=self.min_sources_per_tile
        )
        n_sources = n_sources.long().squeeze(-1)
        return n_sources

    def _sample_locs(self, is_on_array, batch_size):
        # output dimension is batchsize x n_tiles_per_image x max_sources_per_tile x 2

        # 2 = (x,y)
        locs = (
            torch.rand(
                batch_size,
                self.n_tiles_per_image,
                self.max_sources_per_tile,
                2,
            )
            * (self.loc_max_per_tile - self.loc_min_per_tile)
            + self.loc_min_per_tile
        )
        locs *= is_on_array.unsqueeze(-1)

        return locs

    def _sample_galaxy_params(self, n_galaxies, galaxy_bool):
        # galaxy params are just Normal(0,1) variables.

        assert len(n_galaxies.shape) == 2
        batch_size = n_galaxies.size(0)

        mean = torch.zeros(1, dtype=torch.float)
        std = torch.ones(1, dtype=torch.float)
        p_z = Normal(mean, std)
        sample_shape = torch.tensor(
            [
                batch_size,
                self.n_tiles_per_image,
                self.max_sources_per_tile,
                self.latent_dim,
            ]
        )
        galaxy_params = p_z.rsample(sample_shape)
        galaxy_params = galaxy_params.reshape(
            batch_size,
            self.n_tiles_per_image,
            self.max_sources_per_tile,
            self.latent_dim,
        )

        # zero out excess according to galaxy_bool.
        galaxy_params = galaxy_params * galaxy_bool.unsqueeze(-1)
        return galaxy_params

    def _sample_n_galaxies_and_stars(self, n_sources, is_on_array):
        # the counts returned (n_galaxies, n_stars) are of shape (batch_size x n_tiles_per_image)
        # the booleans returned (galaxy_bool, star_bool) are of shape (batch_size x n_tiles_per_image x max_detections)

        batch_size = n_sources.size(0)
        uniform = torch.rand(
            batch_size,
            self.n_tiles_per_image,
            self.max_sources_per_tile,
        )
        galaxy_bool = uniform < self.prob_galaxy
        galaxy_bool = (galaxy_bool * is_on_array).float()
        star_bool = (1 - galaxy_bool) * is_on_array
        n_galaxies = galaxy_bool.sum(-1)
        n_stars = star_bool.sum(-1)
        assert torch.all(n_stars <= n_sources) and torch.all(n_galaxies <= n_sources)

        return n_galaxies, n_stars, galaxy_bool, star_bool

    def _sample_fluxes(self, n_stars, star_bool, batch_size):
        """

        :return: fluxes, a shape (batch_size x self.max_sources_per_tile x max_sources x n_bands) tensor
        """
        assert n_stars.shape[0] == batch_size

        shape = (batch_size, self.n_tiles_per_image, self.max_sources_per_tile)
        base_fluxes = self._draw_pareto_maxed(shape)

        if self.n_bands > 1:
            colors = (
                torch.rand(
                    batch_size,
                    self.n_tiles_per_image,
                    self.max_sources_per_tile,
                    self.n_bands - 1,
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

    def _draw_pareto_maxed(self, shape):
        # draw pareto conditioned on being less than f_max

        u_max = self._pareto_cdf(self.f_max)
        uniform_samples = torch.rand(*shape) * u_max
        return self.f_min / (1.0 - uniform_samples) ** (1 / self.alpha)

    def _pareto_cdf(self, x):
        return 1 - (self.f_min / x) ** self.alpha

    @staticmethod
    def _get_log_fluxes(fluxes):
        """
        To obtain fluxes from log_fluxes.

        >> is_on_array = get_is_on_from_n_stars(n_stars, max_sources)
        >> fluxes = np.exp(log_fluxes) * is_on_array
        """
        ones = torch.ones(*fluxes.shape).type_as(fluxes)

        log_fluxes = torch.where(fluxes > 0, fluxes, ones)  # prevent log(0) errors.
        log_fluxes = torch.log(log_fluxes)

        return log_fluxes

    @staticmethod
    def get_gal_decoder_from_file(
        decoder_file, device, gal_slen=51, n_bands=1, latent_dim=8
    ):
        dec = galaxy_net.CenteredGalaxyDecoder(gal_slen, latent_dim, n_bands).to(device)
        dec.load_state_dict(torch.load(decoder_file, map_location=device))
        dec.eval()
        return dec

    @staticmethod
    def get_psf_params_from_file(psf_file, device):
        return torch.from_numpy(np.load(psf_file)).to(device)

    @staticmethod
    def get_background_from_file(background_file, slen, n_bands):
        # for numpy background that are not necessarily of the correct size.
        background = torch.load(background_file)
        assert n_bands == background.shape[0]

        # now convert background to size of scenes
        values = background.mean((1, 2))  # shape = (n_bands)
        background = torch.zeros(n_bands, slen, slen)
        for i, value in enumerate(values):
            background[i, ...] = value

        return background

    @staticmethod
    def decoder_args_from_args(args, paths: dict, device):
        slen, latent_dim, n_bands = args.slen, args.latent_dim, args.n_bands
        gal_slen = args.gal_slen
        decoder_file = paths["data"].joinpath(args.galaxy_decoder_file)
        background_file = paths["data"].joinpath(args.background_file)
        psf_file = paths["data"].joinpath(args.psf_file)

        dec = SimulatedDataset.get_gal_decoder_from_file(
            decoder_file, gal_slen, n_bands, latent_dim, device
        )
        background = SimulatedDataset.get_background_from_file(
            background_file, slen, n_bands
        )
        psf_params = SimulatedDataset.get_psf_params_from_file(psf_file, device)[
            range(n_bands)
        ]

        return dec, psf_params, background

    @classmethod
    def from_args(cls, args, paths: dict, device):
        args_dict = vars(args)
        parameters = inspect.signature(ImageDecoder).parameters

        args_names = [
            "n_batches",
            "batch_size",
            "galaxy_decoder",
            "init_psf_params",
            "background",
        ]
        decoder_args = SimulatedDataset.decoder_args_from_args(args, paths, device)
        decoder_kwargs = {
            key: value
            for key, value in args_dict.items()
            if key in parameters and key not in args_names
        }
        return cls(args.n_batches, args.batch_size, decoder_args, decoder_kwargs)

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--galaxy-decoder-file",
            type=str,
            default="galaxy_decoder_1_band.dat",
            help="File relative to data directory containing galaxy decoder state_dict.",
        )

        parser.add_argument(
            "--background-file",
            type=str,
            default="background_galaxy_single_band_i.npy",
            help="File relative to data directory containing background to be used.",
        )

        parser.add_argument(
            "--psf-file",
            type=str,
            default="fitted_powerlaw_psf_params.npy",
            help="File relative to data directory containing PSF to be used.",
        )

        # general sources
        parser.add_argument("--max-sources", type=int, default=10)
        parser.add_argument("--mean-sources", type=float, default=5)
        parser.add_argument("--min-sources", type=int, default=1)
        parser.add_argument("--loc-min", type=float, default=0.0)
        parser.add_argument("--loc-max", type=float, default=1.0)
        parser.add_argument("--prob-galaxy", type=float, default=0.0)
        parser.add_argument("--gal-slen", type=int, default=51)

        # stars.
        parser.add_argument("--f-min", type=float, default=1e4)
        parser.add_argument("--f-max", type=float, default=1e6)
        parser.add_argument("--alpha", type=float, default=0.5)
