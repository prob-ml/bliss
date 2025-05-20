import torch
from torch import nn
from torch.distributions import Poisson, Categorical, Uniform
from einops import repeat
import copy


class ConstantLocsPrior:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self, sample_shape):
        mid_point = (self.low + self.high) / 2
        sample_shape_dict = {f"s{i}": s 
                             for i, s in enumerate(sample_shape)}
        return repeat(mid_point, 
                      "... ->" + " ".join([f"s{i}" for i in range(len(sample_shape))]) + " ...",
                      **sample_shape_dict)
    

class CatalogPrior:
    def __init__(
        self,
        max_objects: int,
        img_height: int,
        img_width: int,
        min_flux: float,
        pad: int,
    ):
        self.max_objects = max_objects
        self.count_dim = self.max_objects + 1

        self.img_height = img_height
        self.img_width = img_width
        self.pad = pad

        self.min_flux = torch.tensor(min_flux)

        self.count_prior = Categorical(torch.eye(self.count_dim)[-1])
        self.flux_prior = Uniform(self.min_flux, 10 * self.min_flux)
        self.loc_prior = ConstantLocsPrior(
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
        min_flux: float,
        psf_stdev: float,
        background_intensity: float,
    ):
        super().__init__()

        self.img_height = img_height
        self.img_width = img_width

        self.max_objects = max_objects

        self.min_flux = min_flux
        self.psf_stdev = psf_stdev
        self.background_intensity = background_intensity

        self.catalog_prior = CatalogPrior(max_objects=max_objects,
                                          img_height=img_height,
                                          img_width=img_width,
                                          min_flux=min_flux,
                                          pad=0)

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
    
    def _generate(self, batch_size, add_second_source):
        tile_cat = self.catalog_prior.sample(num_catalogs=batch_size)
        tile_cat = {k: v.to(device=self.device) for k, v in tile_cat.items()}
        if add_second_source:
            ori_tile_cat = copy.deepcopy(tile_cat)
            tile_cat["counts"] += 1
            tile_cat["locs"] = torch.cat([tile_cat["locs"], tile_cat["locs"] + 1], dim=-2)
            tile_cat["fluxes"] = torch.cat([tile_cat["fluxes"], tile_cat["fluxes"] * 0.5], dim=-1)
            self.max_objects += 1
        source_intensities = (
            tile_cat["fluxes"].view(batch_size, 1, 1, self.max_objects)
            * self.psf(tile_cat["locs"][:, :, 0], tile_cat["locs"][:, :, 1])
        ).sum(dim=3)  # (b, h, w)
        total_intensities = source_intensities + self.background_intensity
        images = Poisson(total_intensities).sample()
        if add_second_source:
            output_tile_cat = ori_tile_cat
            self.max_objects -= 1
        else:
            output_tile_cat = tile_cat
        return {
            **output_tile_cat, 
            "total_intensities": total_intensities, 
            "images": images
        }
    
    def generate(self, batch_size, *, seed=None, add_second_source=False):
        if seed is not None:
            with torch.random.fork_rng(devices=["cpu", self.device]):
                torch.manual_seed(seed)
                output = self._generate(batch_size, add_second_source)
        else:
            output = self._generate(batch_size, add_second_source)
        return output
