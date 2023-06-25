import itertools
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from astropy.io import fits
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torch.distributions import Gamma, Poisson, Uniform

from bliss.catalog import SourceType, TileCatalog
from bliss.surveys.sdss import SDSSDownloader, column_to_tensor


class ImagePrior(pl.LightningModule):
    """Prior distribution of objects in an astronomical image.

    After the module is initialized, sampling is done with the sample_prior method.
    The input parameters correspond to the number of sources, the fluxes, whether an
    object is a galaxy or star, and the distributions of galaxy and star shapes.
    """

    def __init__(
        self,
        sdss_fields: DictConfig,
        n_tiles_h: int,
        n_tiles_w: int,
        tile_slen: int,
        batch_size: int,
        min_sources: int,
        max_sources: int,
        mean_sources: float,
        prob_galaxy: float,
        star_flux_min: float,
        star_flux_max: float,
        star_flux_alpha: float,
        galaxy_flux_min: float,
        galaxy_flux_max: float,
        galaxy_alpha: float,
        galaxy_a_concentration: float,
        galaxy_a_loc: float,
        galaxy_a_scale: float,
        galaxy_a_bd_ratio: float,
    ):
        """Initializes ImagePrior.

        Args:
            sdss_fields: sdss frames to sample from,
            n_tiles_h: Image height in tiles,
            n_tiles_w: Image width in tiles,
            tile_slen: Tile side length in pixels,
            batch_size: int,
            min_sources: Minimum number of sources in a tile
            max_sources: Maximum number of sources in a tile
            mean_sources: Mean rate of sources appearing in a tile
            prob_galaxy: Prior probability a source is a galaxy
            star_flux_min: Prior parameter on fluxes
            star_flux_max: Prior parameter on fluxes
            star_flux_alpha: Prior parameter on fluxes (pareto parameter)
            galaxy_flux_min: Minimum flux of a galaxy
            galaxy_flux_max: Maximum flux of a galaxy
            galaxy_alpha: ?
            galaxy_a_concentration: ?
            galaxy_a_loc: ?
            galaxy_a_scale: galaxy scale
            galaxy_a_bd_ratio: galaxy bulge-to-disk ratio
        """
        super().__init__()
        self.n_tiles_h = n_tiles_h
        self.n_tiles_w = n_tiles_w
        self.tile_slen = tile_slen
        # NOTE: bands have to be non-empty
        assert sdss_fields["bands"]
        self.bands = sdss_fields["bands"]
        self.n_bands = len(self.bands)
        self.max_bands = 5  # TODO: fix hard-coding just for SDSS
        self.batch_size = batch_size
        self.sdss = sdss_fields["dir"]

        self.rcf_stars_rest, self.rcf_gals_rest = self._load_color_distribution(
            sdss_fields["field_list"]
        )

        self.min_sources = min_sources
        self.max_sources = max_sources
        self.mean_sources = mean_sources

        self.prob_galaxy = prob_galaxy

        self.star_flux_min = star_flux_min
        self.star_flux_max = star_flux_max
        self.star_flux_alpha = star_flux_alpha

        self.galaxy_flux_min = galaxy_flux_min
        self.galaxy_flux_max = galaxy_flux_max
        self.galaxy_alpha = galaxy_alpha

        self.galaxy_a_concentration = galaxy_a_concentration
        self.galaxy_a_loc = galaxy_a_loc
        self.galaxy_a_scale = galaxy_a_scale
        self.galaxy_a_bd_ratio = galaxy_a_bd_ratio

    def sample_prior(self, batch_rcfs) -> TileCatalog:
        """Samples latent variables from the prior of an astronomical image.

        Args:
            batch_rcfs: run, camcol, field combinations needed for sampling.

        Returns:
            A dictionary of tensors. Each tensor is a particular per-tile quantity; i.e.
            the first three dimensions of each tensor are
            `(batch_size, self.n_tiles_h, self.n_tiles_w)`.
            The remaining dimensions are variable-specific.
        """
        locs = self._sample_locs()
        select_gal_rcfs = []
        select_star_rcfs = []

        for rcf in batch_rcfs:
            rcf = tuple(rcf)
            # get value if key exists, otherwise pick randomly
            select_gal_rcfs.append(
                self.rcf_gals_rest[rcf]
                if rcf in self.rcf_gals_rest
                else random.choice(list(self.rcf_gals_rest.values()))
            )
            select_star_rcfs.append(
                self.rcf_stars_rest[rcf]
                if rcf in self.rcf_stars_rest
                else random.choice(list(self.rcf_stars_rest.values()))
            )

        galaxy_params = self._sample_galaxy_prior(select_gal_rcfs)
        star_fluxes = self._sample_star_fluxes(select_star_rcfs)
        star_log_fluxes = star_fluxes.log()

        n_sources = self._sample_n_sources()
        source_type = self._sample_source_type()

        catalog_params = {
            "n_sources": n_sources,
            "source_type": source_type,
            "locs": locs,
            "galaxy_params": galaxy_params,
            "star_fluxes": star_fluxes,
            "star_log_fluxes": star_log_fluxes,
        }

        return TileCatalog(self.tile_slen, catalog_params)

    def _sample_n_sources(self):
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w)
        n_sources = Poisson(self.mean_sources).sample(latent_dims)
        n_sources = n_sources.clamp(max=self.max_sources, min=self.min_sources)
        # long() is necessary here because the return value is used for indexing subsequently
        return n_sources.long()

    def _sample_locs(self):
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 2)
        return Uniform(0, 1).sample(latent_dims)

    def _sample_source_type(self):
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 1)
        uniform_aux_var = torch.rand(*latent_dims)
        galaxy_bool = uniform_aux_var < self.prob_galaxy
        star_bool = galaxy_bool.bitwise_not()
        return SourceType.STAR * star_bool + SourceType.GALAXY * galaxy_bool

    def _sample_star_fluxes(self, star_ratios):
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources)
        r_flux = self._draw_truncated_pareto(
            self.star_flux_alpha, self.star_flux_min, self.star_flux_max, latent_dims
        )
        total_flux = self.multiflux(star_ratios, r_flux)

        # select specified bands
        bands = np.array(self.bands)
        return total_flux[..., bands]

    def multiflux(self, rcf_ratios, flux_base) -> Tensor:
        """Generate flux values for remaining bands based on draw."""
        total_flux = torch.zeros(
            self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources, 5
        )

        for b, obj_ratios in enumerate(rcf_ratios):
            for h, w, s, bnd in itertools.product(  # noqa: WPS352
                range(self.n_tiles_h),
                range(self.n_tiles_w),
                range(self.max_sources),
                range(self.max_bands),
            ):
                # sample a light source
                flux_sample = obj_ratios[np.random.randint(0, len(obj_ratios) - 1)]
                total_flux[b, h, w, s, bnd] = flux_sample[bnd] * flux_base[b, h, w, s]

        return total_flux

    def _load_color_distribution(self, sdss_fields):
        rcf_stars_rest = {}
        rcf_gals_rest = {}
        # load all star, galaxy fluxes relative to r-band required for sampling
        for field_params in sdss_fields:
            run = field_params["run"]
            camcol = field_params["camcol"]
            fields = field_params["fields"]
            for field in fields:
                sdss_path = Path(self.sdss)
                camcol_dir = sdss_path / str(run) / str(camcol) / str(field)
                po_path = camcol_dir / f"photoObj-{run:06d}-{camcol:d}-{field:04d}.fits"
                if not po_path.exists():
                    SDSSDownloader(run, camcol, field, str(sdss_path)).download_po()
                msg = (
                    f"{po_path} does not exist. "
                    + "Make sure data files are available for fields specified in config."
                )
                assert Path(po_path).exists(), msg
                po_fits = fits.getdata(po_path)

                fluxes = column_to_tensor(po_fits, "psfflux")
                objc_type = column_to_tensor(po_fits, "objc_type").numpy()
                thing_id = column_to_tensor(po_fits, "thing_id").numpy()

                star_fluxes_rest = []
                gal_fluxes_rest = []

                for obj, _ in enumerate(objc_type):
                    if objc_type[obj] == 6 and thing_id[obj] != -1 and torch.all(fluxes[obj] > 0):
                        ratios = (  # noqa: WPS220
                            fluxes[obj] / fluxes[obj][2]
                        )  # noqa: WPS220  # flux relative to r-band
                        star_fluxes_rest.append(ratios)  # noqa: WPS220
                    elif objc_type[obj] == 3 and thing_id[obj] != -1 and torch.all(fluxes[obj] > 0):
                        ratios = (  # noqa: WPS220
                            fluxes[obj] / fluxes[obj][2]
                        )  # noqa: WPS220  # flux relative to r-band
                        gal_fluxes_rest.append(ratios)  # noqa: WPS220

                if star_fluxes_rest:
                    rcf_stars_rest[(run, camcol, field)] = star_fluxes_rest
                if gal_fluxes_rest:
                    rcf_gals_rest[(run, camcol, field)] = gal_fluxes_rest
        return rcf_stars_rest, rcf_gals_rest

    @staticmethod
    def _draw_truncated_pareto(alpha, min_x, max_x, n_samples) -> Tensor:
        # draw pareto conditioned on being less than f_max
        u_max = 1 - (min_x / max_x) ** alpha
        uniform_samples = torch.rand(n_samples) * u_max
        return min_x / (1.0 - uniform_samples) ** (1 / alpha)

    def _sample_galaxy_prior(self, gal_ratios):
        """Sample the latent galaxy params.

        Args:
            gal_ratios: flux ratios for multiband galaxies (why is this an argument?)

        Returns:
            source_params (Tensor): Tensor containing the following parameters:
                - total_flux: the total flux of the galaxy
                - disk_frac: the fraction of flux attributed to the disk (rest goes to bulge)
                - beta_radians: the angle of shear in radians
                - disk_q: the minor-to-major axis ratio of the disk
                - a_d: semi-major axis of disk
                - bulge_q: minor-to-major axis ratio of the bulge
                - a_b: semi-major axis of bulge
        """
        latent_dims = (self.batch_size, self.n_tiles_h, self.n_tiles_w, self.max_sources)

        r_flux = self._draw_truncated_pareto(
            self.galaxy_alpha, self.galaxy_flux_min, self.galaxy_flux_max, latent_dims
        )

        total_flux = self.multiflux(gal_ratios, r_flux)

        # select specified ban
        bands = np.array(self.bands)
        select_flux = total_flux[..., bands]

        disk_frac = Uniform(0, 1).sample(latent_dims)
        beta_radians = Uniform(0, 2 * np.pi).sample(latent_dims)
        disk_q = Uniform(1e-8, 1).sample(latent_dims)
        bulge_q = Uniform(1e-8, 1).sample(latent_dims)

        base_dist = Gamma(self.galaxy_a_concentration, rate=1.0)
        disk_a = base_dist.sample(latent_dims) * self.galaxy_a_scale + self.galaxy_a_loc

        bulge_loc = self.galaxy_a_loc / self.galaxy_a_bd_ratio
        bulge_scale = self.galaxy_a_scale / self.galaxy_a_bd_ratio
        bulge_a = base_dist.sample(latent_dims) * bulge_scale + bulge_loc

        disk_frac = torch.unsqueeze(disk_frac, 4)
        beta_radians = torch.unsqueeze(beta_radians, 4)
        disk_q = torch.unsqueeze(disk_q, 4)
        disk_a = torch.unsqueeze(disk_a, 4)
        bulge_q = torch.unsqueeze(bulge_q, 4)
        bulge_a = torch.unsqueeze(bulge_a, 4)

        param_lst = [select_flux, disk_frac, beta_radians, disk_q, disk_a, bulge_q, bulge_a]

        return torch.cat(param_lst, dim=4)
