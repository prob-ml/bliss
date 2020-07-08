import inspect

import numpy as np
import torch
from torch.utils.data import IterableDataset

from bliss import device, psf_transform
from bliss.models import galaxy_net
from bliss.models.decoder import ImageDecoder


class SimulatedDataset(IterableDataset):
    def __init__(self, n_batches: int, batch_size: int, decoder_args, decoder_kwargs):
        super(SimulatedDataset, self).__init__()

        self.n_batches = n_batches
        self.batch_size = batch_size

        self.image_decoder = ImageDecoder(*decoder_args, **decoder_kwargs)
        self.slen = self.image_decoder.slen
        self.n_bands = self.image_decoder.n_bands

    def __iter__(self):
        return self.batch_generator()

    def batch_generator(self):
        for i in range(self.n_batches):
            yield self.get_batch()

    def get_batch(self):
        (
            n_sources,
            n_galaxies,
            n_stars,
            locs,
            galaxy_params,
            single_galaxies,
            fluxes,
            log_fluxes,
            galaxy_bool,
            star_bool,
        ) = self.image_decoder.sample_parameters(batch_size=self.batch_size)

        galaxy_locs = locs * galaxy_bool.unsqueeze(2)
        star_locs = locs * star_bool.unsqueeze(2)
        images = self.image_decoder.generate_images(
            n_sources, galaxy_locs, star_locs, single_galaxies, fluxes
        )

        return {
            "n_sources": n_sources,
            "n_galaxies": n_galaxies,
            "n_stars": n_stars,
            "locs": locs,
            "galaxy_params": galaxy_params,
            "log_fluxes": log_fluxes,
            "galaxy_bool": galaxy_bool,
            "images": images,
            "background": self.image_decoder.background,
        }

    @staticmethod
    def get_gal_decoder_from_file(decoder_file, slen, n_bands, latent_dim):
        dec = galaxy_net.CenteredGalaxyDecoder(slen, latent_dim, n_bands).to(device)
        dec.load_state_dict(torch.load(decoder_file, map_location=device))
        dec.eval()
        return dec

    @staticmethod
    def get_background_from_file(background_file, slen, n_bands):
        # for numpy background that are not necessarily of the correct size.
        background = np.load(background_file)
        background = torch.from_numpy(background).float()

        assert n_bands == background.shape[0]

        # now convert background to size of scenes
        values = background.mean((1, 2))  # shape = (n_bands)
        background = torch.zeros(n_bands, slen, slen)
        for i, value in enumerate(values):
            background[i, ...] = value

        return background

    @staticmethod
    def get_psf_from_file(psf_file):
        psf_params = torch.from_numpy(np.load(psf_file)).to(device)
        power_law_psf = psf_transform.PowerLawPSF(psf_params)
        psf = power_law_psf.forward().detach()
        assert psf.size(0) == 2 and psf.size(1) == psf.size(2) == 101
        return psf

    @staticmethod
    def decoder_args_from_args(args, paths: dict):
        slen, latent_dim, n_bands = args.slen, args.latent_dim, args.n_bands
        decoder_file = paths["data"].joinpath(args.decoder_state_file)
        background_file = paths["data"].joinpath(args.background_file)
        psf_file = paths["data"].joinpath(args.psf_file)

        dec = SimulatedDataset.get_decoder_from_args(
            decoder_file, slen, n_bands, latent_dim
        )
        background = SimulatedDataset.get_background_from_file(
            background_file, slen, n_bands
        )
        psf = SimulatedDataset.get_psf_from_file(psf_file)

        return dec, psf, background

    @classmethod
    def from_args(cls, args, paths: dict):
        args_dict = vars(args)
        parameters = inspect.signature(ImageDecoder).parameters

        decoder_args_names = ["galaxy_decoder", "psf", "background"]
        decoder_args = SimulatedDataset.decoder_args_from_args(args, paths)
        decoder_kwargs = {
            param: value
            for param, value in args_dict.items()
            if param in parameters and param not in decoder_args_names
        }
        return cls(*decoder_args, **decoder_kwargs)
