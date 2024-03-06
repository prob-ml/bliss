import dataclasses
import math
import warnings

import torch
from torch.utils.data import IterableDataset

from bliss.catalog import FullCatalog

# prevent pytorch_lightning warning for num_workers = 0 in dataloaders with IterableDataset
warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck.*", UserWarning
)


def counts_mask(counts):
    """Identify the first `counts[i]` elements of the ith row of the result.

    Args:
        counts: (m) tensor of counts

    Returns:
        (m,M) tensor where M = counts.max() and result[i, j] = j < counts[i]


    Example:
    counts = torch.tensor([2,1,1])
    counts_mask(counts)
    tensor([[True, True],
            [True, False],
            [True, False]])

    """
    # Determine the max length for the range
    max_count = counts.max()

    # Create a range tensor [0, 1, 2, ..., max_count-1]
    range_tensor = torch.arange(max_count, device=counts.device).unsqueeze(0)

    # Expand counts to match the shape for broadcasting
    expanded_counts = counts.unsqueeze(1)

    # Compute the mask
    return range_tensor < expanded_counts


def draw_trunc_pareto(b, c, loc, scale, size, torch_generator=None):
    """Draw a truncated pareto random variable.

    Args:
        b: float, exponent of the power law
        c: float, truncation of the power law
        loc: float, location of the power law
        scale: float, scale of the power law
        size: tuple of ints, size of the output
        torch_generator: torch.Generator, optional; random number generator

    Returns:
        (size) tensor of draws from the truncated pareto distribution

    Specifically, let the "standardized"
    truncated pareto distribution be the distribution with PDF

    f(x) propto x^{-b-1} I(1<=x<=c)

    Then we return draws from a shifted and scaled version, i.e.

    X ~ f(x)
    Y = X*scale+loc
    """

    # get generator if necessary
    torch_generator = torch.Generator() if torch_generator is None else torch_generator

    # draw from the standardized truncated pareto
    u = torch.rand(size, generator=torch_generator, device=torch_generator.device)

    # get (c^b -1) / c^b)
    ratio = (c ** (-b) - 1) / c ** (-b)

    # get unscaled draws
    x = torch.exp(-torch.log1p(-u * ratio) / b)

    # scale and shift
    return x * scale + loc


@dataclasses.dataclass
class ToySimulator:
    """Fast but simple simulator for testing purposes.

    Args:
        n_bands (int): number of bands
        coadd_depth (int): number of Single-Epoch Observed Images (SEOs) to coadd
        num_workers (int): number of workers for DataLoader
        height (int): height of images
        width (int): width of images
        coadd_wiggle_shift (float): how much coadds actually vary in terms of shift
            this should definitely be less than the alignment_tolerance
        psf_radius_w (float): width of the PSF in pixels (in reference picture)
        psf_radius_h (float): height of the PSF
        star_flux_exponent (float): exponent of the power law for star fluxes
        star_flux_truncation (float): truncation of the power law for star fluxes
        star_flux_loc (float): location of the power law for star fluxes
        star_flux_scale (float): scale of the power law for star fluxes
        source_poisson_rate (float): rate of poisson process for sources
        tile_slen (int): size of tiles
        background (float): background level

    Fluxes drawn using truncated parameters with parameters given by
    star_flux_[foo], i.e.

    flux = draw_trunc_pareto(
        star_flux_exponent,
        star_flux_truncation,
        loc=star_flux_loc,
        scale=star_flux_scale
    )
    """

    n_bands: int = 1
    coadd_depth: int = 5
    num_workers: int = 0
    height: int = 80
    width: int = 80
    coadd_wiggle_shift: float = 0
    psf_radius_w: float = 2
    psf_radius_h: float = 2
    star_flux_exponent: float = 0.43
    star_flux_truncation: float = 640
    star_flux_loc: float = 0.0
    star_flux_scale: float = 5000
    source_poisson_rate: float = 10
    tile_slen: int = 4
    noise_level: float = 10

    def construct_fluxes(self, size, generator=None):
        """Draw fluxes based on our parameters.

        Args:
            size: tuple of ints; output will be size + (n_bands,)
            generator: torch.Generator, optional; random number generator

        Returns:
            (size + (n_bands,)) tensor of fluxes

        """

        # handle generator
        # take cue from generator about device
        generator = torch.Generator() if generator is None else generator
        device = generator.device

        # get total brightness for each star
        base_brightness = draw_trunc_pareto(
            self.star_flux_exponent,
            self.star_flux_truncation,
            loc=self.star_flux_loc,
            scale=self.star_flux_scale,
            size=size,
            torch_generator=generator,
        )  # M x bands

        # modify by some random factor for each band, uniform in [.8,1.2]
        mods = 0.8 + 0.4 * torch.rand(size + (self.n_bands,), generator=generator, device=device)

        # done
        return base_brightness[..., None] * mods

    def sample_catalogs(self, batch_size, generator=None) -> FullCatalog:
        """Sample a FullCatalog for a single fictitious patch of sky."""

        # handle generator
        # take cue from generator about device
        generator = torch.Generator() if generator is None else generator
        device = generator.device

        # get n_sources per batch
        n_sources = torch.poisson(
            torch.ones(batch_size, device=device) * self.source_poisson_rate,
            generator=generator,
        ).long()

        # and total
        max_sources = n_sources.max().item()

        # get fluxes for each source
        # B x max_sources x bands
        source_fluxes = self.construct_fluxes((batch_size, max_sources), generator)

        # use nans to emphasize that some points are not real
        source_fluxes = torch.where(
            counts_mask(n_sources)[:, :, None],
            source_fluxes,
            torch.tensor(math.nan, device=device),
        )

        # positions for eaach sources
        # total_n_sources x 2
        hw = torch.tensor([self.height, self.width], dtype=torch.float32, device=device)
        source_locations = (
            torch.rand((batch_size, max_sources, 2), generator=generator, device=device) * hw
        )

        # max sources per batch
        max_sources = source_fluxes.shape[1]

        # done
        return FullCatalog(
            self.height,
            self.width,
            {
                "plocs": source_locations,
                "source_type": torch.zeros(
                    batch_size, max_sources, 1, dtype=torch.int32, device=device
                ),
                "galaxy_fluxes": torch.zeros(batch_size, max_sources, self.n_bands, device=device),
                "galaxy_params": torch.zeros(batch_size, max_sources, 6, device=device),
                "star_fluxes": source_fluxes,
                "n_sources": n_sources,
            },
        )

    def sample_dithers(self, full_catalog, generator=None) -> torch.tensor:
        """Simulate misalignment of coadds.

        Args:
            full_catalog: FullCatalog, the catalog of sources in each tile
            generator: torch.Generator, optional; random number generator

        Returns:
            (B, n_coadds, n_bands, 2) tensor, offsets for each image
        """

        batch_size = full_catalog["plocs"].shape[0]
        size = (batch_size, self.coadd_depth, self.n_bands, 2)
        eps = torch.rand(size=size, generator=generator, device=generator.device)
        return (eps * 2 - 1) * self.coadd_wiggle_shift

    def sample_images(self, full_catalog, dithers, generator=None) -> torch.tensor:
        """Create images from a catalog and misalignments.

        Args:
            full_catalog: FullCatalog, the catalog of sources in each tile
            dithers: (B, n_coadds, n_bands, 2) tensor, offsets for each image
                representing errors in the alignment
            generator: torch.Generator, optional; random number generator

        Returns:
            (B, n_bands, height, width) tensor, the coadded images

        Raises:
            ValueError: if any source is not a star

        Sample images from full catalog

        If star_fluxes is of size [B,max_sources,n_bands] then we
        expect that star_fluxes[i,j,k] = 0 whenever j>=n_sources[i]

        (this facilitates the batch processing)
        """

        # make sure all of the sources are stars
        if torch.any(full_catalog["source_type"] != 0):
            raise ValueError("all sources must be stars")

        # zero out fluxes based on n_sources
        mask = counts_mask(full_catalog["n_sources"])[:, :, None]  # B x max_sources x 1
        fluxes = torch.where(mask, full_catalog["star_fluxes"], 0)

        # construct meshgrid of size (height,width)
        # pixel value at img[0,0] is assumed to span from coordinate 0,0
        # to coordinate value 1,1
        # so the *center* of img[0,0] pixel is at coordinate (.5,.5)
        x = torch.arange(self.width, device=full_catalog.device) + 0.5
        y = torch.arange(self.height, device=full_catalog.device) + 0.5
        xx, yy = torch.meshgrid(x, y, indexing="ij")

        # uncollated results will have shape [B,height,width,max_sources]
        xx = xx[None, :, :, None]
        yy = yy[None, :, :, None]

        # create B,height,width image for each band
        band_images = []
        for b in range(self.n_bands):
            relevant_fluxes = fluxes[:, :, b]  # B x max_sources

            img = 0
            for coadd in range(self.coadd_depth):
                relevant_dithers = dithers[:, coadd, b]  # B x 2

                # plocs is of size [B,max_sources,2]
                # meshgrid of size [height,width]
                # uncollated result is of size [B,height,width,max_sources]
                plocs_x = (
                    full_catalog["plocs"][:, None, None, :, 0]
                    + relevant_dithers[:, None, None, None, 0]
                )
                plocs_y = (
                    full_catalog["plocs"][:, None, None, :, 1]
                    + relevant_dithers[:, None, None, None, 1]
                )

                # compute brightness [B,height,width]
                # we briefly store [B,height,width,max_sources] result at this
                # stage, reaching apex memory usage
                brightness = (
                    torch.exp(
                        -0.5
                        * (
                            (plocs_x - xx) ** 2 / self.psf_radius_h**2
                            + (plocs_y - yy) ** 2 / self.psf_radius_w**2
                        )  # noqa: C815
                    )
                    * relevant_fluxes[:, None, None]
                ).sum(-1)

                # sample and add it in
                samp = torch.poisson(brightness + self.noise_level, generator=generator)
                img += samp

            band_images.append(img)

        # merge
        return torch.stack(band_images, dim=1)


class ToySimulatedDataset(IterableDataset):
    def __init__(
        self,
        toy_simulator: ToySimulator,
        batch_size: int = 64,
        num_workers: int = 1,
        epoch_size: int = 64,
        generator: torch.Generator = None,
        tile_slen: int = 4,
        max_sources_per_tile: int = 1,
        cache: bool = False,
    ):
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.generator = generator if generator is not None else torch.Generator()
        self.toy_simulator = toy_simulator
        self.tile_slen = tile_slen
        self.max_sources_per_tile = max_sources_per_tile
        self.cache = cache

        if cache:
            self._cached = [self._draw_one() for i in range(self.epoch_size)]

    def __len__(self):
        return self.epoch_size

    def _draw_one(self):
        fc = self.toy_simulator.sample_catalogs(self.batch_size, generator=self.generator)
        diths = self.toy_simulator.sample_dithers(fc, generator=self.generator)
        imgs = self.toy_simulator.sample_images(fc, diths, generator=self.generator)

        return {
            "tile_catalog": fc.to_tile_catalog(
                self.tile_slen,
                self.max_sources_per_tile,
                ignore_extra_sources=True,
            ).to_dict(),
            "images": imgs,
            "background": torch.ones_like(imgs) * self.toy_simulator.noise_level * 0.1,
            "deconvolution": torch.zeros_like(imgs),
            "psf_params": torch.zeros(self.batch_size, 4, 1, device=imgs.device),
        }

    def __iter__(self):
        if self.cache:
            for b in self._cached:
                yield b
        else:
            for _ in range(self.epoch_size):
                yield self._draw_one()
