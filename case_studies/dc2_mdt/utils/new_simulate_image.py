import torch
from torch import nn
from torch.distributions import Poisson, Categorical, Gamma, MixtureSameFamily, Uniform
from einops import repeat


class ConstantLocsPrior:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self, sample_shape):
        mid_point = (self.low + self.high) / 2
        return torch.ones(sample_shape).unsqueeze(-1) * mid_point
    
class UniformLocsPrior:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self, sample_shape):
        return Uniform(self.low, self.high).sample(sample_shape)
    

class CatalogPrior:
    def __init__(
        self,
        max_objects: int,
        img_height: int,
        img_width: int,
        pad: int,
        flux_alpha: float,
        flux_beta: float,
        always_max_count=True,
        constant_locs=True,
    ):
        self.max_objects = max_objects
        self.count_dim = self.max_objects + 1

        self.img_height = img_height
        self.img_width = img_width
        self.pad = pad

        if always_max_count:
            self.count_prior = Categorical(torch.eye(self.count_dim)[-1])
        else:
            self.count_prior = Categorical(torch.ones(self.count_dim))

        self.flux_prior = Gamma(torch.tensor(flux_alpha), torch.tensor(flux_beta))
        if constant_locs:
            self.loc_prior = ConstantLocsPrior(
                torch.zeros(2) + self.pad * torch.ones(2),
                torch.tensor((self.img_height, self.img_width)) - self.pad * torch.ones(2),
            )
        else:
            self.loc_prior = UniformLocsPrior(
                torch.zeros(2) + self.pad * torch.ones(2),
                torch.tensor((self.img_height, self.img_width)) - self.pad * torch.ones(2),
            )

    def sample(
        self,
        num_catalogs,
    ):
        counts = self.count_prior.sample([num_catalogs])  # (n_catalogs, )
        count_indicator = torch.arange(1, self.count_dim).unsqueeze(0) <= counts.unsqueeze(1)  # (n_catalogs, m)
        fluxes = (
            self.flux_prior.sample([num_catalogs, self.max_objects])
            * count_indicator
        )  # (n_catalogs, m)
        locs = self.loc_prior.sample(
            [num_catalogs, self.max_objects]
        ) * count_indicator.unsqueeze(2)  # (n_catalogs, m, 2)
        return {
            "counts": counts, 
            "fluxes": fluxes, 
            "locs": locs
        }



class ImageSimulator(nn.Module):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        max_objects: int,
        psf_stdev: float,
        flux_alpha: float,
        flux_beta: float,
        pad=0,
        always_max_count=True,
        constant_locs=True,
        coadd_images=False,
    ):
        super().__init__()

        self.img_height = img_height
        self.img_width = img_width

        self.max_objects = max_objects

        self.psf_stdev = psf_stdev

        self.flux_alpha = flux_alpha
        self.flux_beta = flux_beta
        self.catalog_prior = CatalogPrior(max_objects=max_objects,
                                          img_height=img_height,
                                          img_width=img_width,
                                          pad=pad,
                                          flux_alpha=flux_alpha,
                                          flux_beta=flux_beta,
                                          always_max_count=always_max_count,
                                          constant_locs=constant_locs)
        self.coadd_images = coadd_images

        self.register_buffer("dummy_param", torch.zeros(0))
        self.register_buffer("psf_marginal_h",
                             (0.5 + torch.arange(self.img_height, dtype=torch.float32)).view(1, self.img_height, 1, 1))
        self.register_buffer("psf_marginal_w",
                             (0.5 + torch.arange(self.img_width, dtype=torch.float32)).view(1, 1, self.img_width, 1))

    @property
    def device(self):
        return self.dummy_param.device

    def psf(self, loc_h, loc_w):
        logpsf = -(
            (self.psf_marginal_h - loc_h.view(-1, 1, 1, self.max_objects)) ** 2
            + (self.psf_marginal_w - loc_w.view(-1, 1, 1, self.max_objects)) ** 2
        ) / (2 * self.psf_stdev ** 2)
        return torch.exp(logpsf - logpsf.logsumexp(dim=(1, 2), keepdim=True))
    
    def _generate_by_catalog(self, tile_cat):
        batch_size = tile_cat["fluxes"].shape[0]
        psf = self.psf(tile_cat["locs"][:, :, 0], tile_cat["locs"][:, :, 1])  # (b, h, w, m)
        source_intensities = (
            tile_cat["fluxes"].view(batch_size, 1, 1, self.max_objects) * psf
        )  # (b, h, w, m)
        if not self.coadd_images:
            images = Poisson(source_intensities).sample()
            random_perm_func = lambda img: img[..., torch.randperm(img.shape[-1])]
            images = torch.vmap(random_perm_func, randomness="different")(images)
        else:
            images = Poisson(source_intensities.sum(dim=-1) + 10).sample()
        return {
            **tile_cat, 
            "source_intensities": source_intensities, 
            "psf": psf,
            "images": images
        }
    
    def _generate(self, batch_size):
        tile_cat = self.catalog_prior.sample(num_catalogs=batch_size)
        tile_cat = {k: v.to(device=self.device) for k, v in tile_cat.items()}
        return self._generate_by_catalog(tile_cat)
    
    def generate(self, batch_size, *, seed=None):
        if seed is not None:
            with torch.random.fork_rng(devices=["cpu", self.device]):
                torch.manual_seed(seed)
                output = self._generate(batch_size)
        else:
            output = self._generate(batch_size)
        return output
    
    def post_dist(self, output_dict):
        assert not self.coadd_images
        gamma_dist = Gamma(output_dict["images"].sum(dim=(-2, -3)) + self.flux_alpha,
                           output_dict["psf"].sum(dim=(-2, -3)) + self.flux_beta)  # (b, m)
        mix = Categorical(torch.ones_like(output_dict["images"][:, 0, 0, :]))
        return MixtureSameFamily(mix, gamma_dist)
