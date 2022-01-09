import torch
import pytorch_lightning as pl
from einops import rearrange
from torch.distributions import Poisson

from bliss.models import galaxy_net
from bliss.models.location_encoder import get_is_on_from_n_sources
from bliss.models.galaxy_flow import CenteredGalaxyLatentFlow


class ImagePrior(pl.LightningModule):
    """Prior distribution of objects in an astronomical image.

    After the module is initialized, sampling is done with the sample_prior method.
    The input parameters correspond to the number of sources, the fluxes, whether an
    object is a galaxy or star, and the distributions of galaxy and star shapes.

    Attributes:
        n_bands: Number of bands (colors) in the image
        n_tiles_per_image: Number of tiles per image
        min_sources: Minimum number of sources in a tile
        max_sources: Maximum number of sources in a tile
        mean_sources: Mean rate of sources appearing in a tile
        loc_min: Per-tile lower-bound on the location of sources
        loc_max: Per-tile upper-bound on the location of sources
        f_min: Prior parameter on fluxes
        f_max: Prior parameter on fluxes
        alpha: Prior parameter on fluxes
        prob_galaxy: Prior probability a source is a galaxy
        vae: Trained VAE modeling single, centered galaxies
    """

    def __init__(
        self,
        n_bands: int = 1,
        slen: int = 50,
        tile_slen: int = 2,
        min_sources: int = 0,
        max_sources: int = 2,
        mean_sources: int = 0.4,
        loc_min: float = 0.0,
        loc_max: float = 1.0,
        f_min: float = 1e4,
        f_max: float = 1e6,
        alpha: float = 0.5,
        prob_galaxy: float = 0.0,
        autoencoder_ckpt: str = None,
        autoencoder_flow_ckpt: str = None,
    ):
        """Initializes ImagePrior.

        Args:
            n_bands: Number of bands (colors) in the image.
            slen: Side-length of astronomical image (image is assumed to be square).
            tile_slen: Side-length of each tile.
            min_sources: Minimum number of sources in a tile
            max_sources: Maximum number of sources in a tile
            mean_sources: Mean rate of sources appearing in a tile
            loc_min: Per-tile lower-bound on the location of sources
            loc_max: Per-tile upper-bound on the location of sources
            f_min: Prior parameter on fluxes
            f_max: Prior parameter on fluxes
            alpha: Prior parameter on fluxes (pareto parameter)
            prob_galaxy: Prior probability a source is a galaxy
            autoencoder_ckpt: Location of checkpoint of galaxy encoder.
            autoencoder_flow_ckpt: Location of checkpoint of flow for latent distribution
        """
        super().__init__()
        self.n_bands = n_bands
        assert slen % 1 == 0, "slen must be an integer."
        assert slen % tile_slen == 0, "slen must be divisible by tile_slen"
        n_tiles_per_image = (int(slen) / tile_slen) ** 2
        self.n_tiles_per_image = int(n_tiles_per_image)

        assert max_sources > 0, "No sources will be drawn."
        self.min_sources = min_sources
        self.max_sources = max_sources
        self.mean_sources = mean_sources
        self.loc_min = loc_min
        self.loc_max = loc_max
        self.f_min = f_min
        self.f_max = f_max
        self.alpha = alpha

        self.prob_galaxy = float(prob_galaxy)
        if prob_galaxy > 0.0:
            self.vae = galaxy_net.OneCenteredGalaxyAE.load_from_checkpoint(autoencoder_ckpt)
            if autoencoder_flow_ckpt is not None:
                print("INFO: Loading trained normalizing flow for galaxy latents")
                flow = CenteredGalaxyLatentFlow.load_from_checkpoint(autoencoder_flow_ckpt)
                self.vae.dist_main = flow.flow_main
                self.vae.dist_residual = flow.flow_residual
        else:
            self.vae = None

    def sample_prior(self, batch_size: int = 1) -> dict:
        """Samples latent variables from the prior of an astronomical image.

        Args:
            batch_size: The number of samples to draw.

        Returns:
            A dictionary of tensors. Each tensor is a particular per-tile quantity; i.e.
            the first two dimensions of each tensor are batch_size and self.n_tiles_per_image.
            The remaining dimensions are variable-specific.
        """
        n_sources = self._sample_n_sources(batch_size)
        is_on_array = get_is_on_from_n_sources(n_sources, self.max_sources)
        locs = self._sample_locs(is_on_array, batch_size)

        _, _, galaxy_bool, star_bool = self._sample_n_galaxies_and_stars(n_sources, is_on_array)
        galaxy_params = self._sample_galaxy_params(galaxy_bool)
        fluxes = self._sample_fluxes(n_sources, star_bool, batch_size)
        log_fluxes = self._get_log_fluxes(fluxes)

        # per tile quantities.
        return {
            "n_sources": n_sources,
            "locs": locs,
            "galaxy_bool": galaxy_bool,
            "star_bool": star_bool,
            "galaxy_params": galaxy_params,
            "fluxes": fluxes,
            "log_fluxes": log_fluxes,
        }

    @staticmethod
    def _get_log_fluxes(fluxes):
        log_fluxes = torch.where(
            fluxes > 0, fluxes, torch.ones_like(fluxes)
        )  # prevent log(0) errors.
        return torch.log(log_fluxes)

    def _sample_n_sources(self, batch_size):
        # returns number of sources for each batch x tile
        # output dimension is batch_size x n_tiles_per_image

        # always poisson distributed.
        p = torch.full((1,), self.mean_sources, device=self.device, dtype=torch.float)
        m = Poisson(p)
        n_sources = m.sample([batch_size, self.n_tiles_per_image])

        # long() here is necessary because used for indexing and one_hot encoding.
        n_sources = n_sources.clamp(max=self.max_sources, min=self.min_sources)
        return rearrange(n_sources.long(), "b n 1 -> b n")

    def _sample_locs(self, is_on_array, batch_size):
        # output dimension is batch_size x n_tiles_per_image x max_sources x 2

        # 2 = (x,y)
        shape = (
            batch_size,
            self.n_tiles_per_image,
            self.max_sources,
            2,
        )
        locs = torch.rand(*shape, device=is_on_array.device)
        locs *= self.loc_max - self.loc_min
        locs += self.loc_min
        locs *= is_on_array.unsqueeze(-1)

        return locs

    def _sample_n_galaxies_and_stars(self, n_sources, is_on_array):
        # the counts returned (n_galaxies, n_stars) are of shape (batch_size x n_tiles_per_image)
        # the booleans returned (galaxy_bool, star_bool) are of shape
        # (batch_size x n_tiles_per_image x max_sources x 1)
        # this last dimension is so it is parallel to other galaxy/star params.

        batch_size = n_sources.size(0)
        uniform = torch.rand(
            batch_size,
            self.n_tiles_per_image,
            self.max_sources,
            1,
            device=is_on_array.device,
        )
        galaxy_bool = uniform < self.prob_galaxy
        galaxy_bool = (galaxy_bool * is_on_array.unsqueeze(-1)).float()
        star_bool = (1 - galaxy_bool) * is_on_array.unsqueeze(-1)
        n_galaxies = galaxy_bool.sum((-2, -1))
        n_stars = star_bool.sum((-2, -1))
        assert torch.all(n_stars <= n_sources) and torch.all(n_galaxies <= n_sources)

        return n_galaxies, n_stars, galaxy_bool, star_bool

    def _sample_fluxes(self, n_stars, star_bool, batch_size):
        """Samples fluxes.

        Arguments:
            n_stars: Tensor indicating number of stars per tile
            star_bool: Tensor indicating whether each object is a star or not
            batch_size: Size of the batches

        Returns:
            fluxes, tensor shape (batch_size x self.n_tiles_per_image x self.max_sources x n_bands)
        """
        assert n_stars.shape[0] == batch_size

        shape = (batch_size, self.n_tiles_per_image, self.max_sources, 1)
        base_fluxes = draw_pareto_maxed(shape, self.device, self.f_min, self.f_max, self.alpha)

        if self.n_bands > 1:
            shape = (
                batch_size,
                self.n_tiles_per_image,
                self.max_sources,
                self.n_bands - 1,
            )
            colors = torch.randn(*shape, device=base_fluxes.device)
            fluxes = 10 ** (colors / 2.5) * base_fluxes
            fluxes = torch.cat((base_fluxes, fluxes), dim=3)
            fluxes *= star_bool.float()
        else:
            fluxes = base_fluxes * star_bool.float()

        return fluxes

    def _sample_galaxy_params(self, galaxy_bool):
        assert galaxy_bool.shape[1:] == (self.n_tiles_per_image, self.max_sources, 1)
        batch_size = galaxy_bool.size(0)
        n_latent_samples = batch_size * self.n_tiles_per_image * self.max_sources
        if self.vae is not None:
            self.vae = self.vae.to(device=galaxy_bool.device)
            latents = self.vae.sample_latent(n_latent_samples)
        else:
            latents = torch.zeros((n_latent_samples, 1), device=galaxy_bool.device)

        galaxy_params = rearrange(
            latents,
            "(b n s) g -> b n s g",
            b=batch_size,
            n=self.n_tiles_per_image,
            s=self.max_sources,
        )
        return galaxy_params * galaxy_bool


def draw_pareto_maxed(shape, device, f_min, f_max, alpha):
    # draw pareto conditioned on being less than f_max
    u_max = pareto_cdf(f_max, f_min, alpha)
    uniform_samples = torch.rand(*shape, device=device) * u_max
    return f_min / (1.0 - uniform_samples) ** (1 / alpha)


def pareto_cdf(x, f_min, alpha):
    return 1 - (f_min / x) ** alpha
