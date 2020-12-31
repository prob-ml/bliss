from abc import abstractmethod, ABC
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import categorical
from .. import device


def get_is_on_from_n_sources(n_sources, max_sources):
    """Return a boolean array of shape=(batch_size, max_sources) whose (k,l)th entry indicates
    whether there are more than l sources on the kth batch.
    """
    assert not torch.any(torch.isnan(n_sources))
    assert torch.all(n_sources >= 0)
    assert torch.all(n_sources.le(max_sources))

    is_on_array = torch.zeros(
        *n_sources.shape, max_sources, device=device, dtype=torch.float
    )

    for i in range(max_sources):
        is_on_array[..., i] = n_sources > i

    return is_on_array


def get_star_bool(n_sources, galaxy_bool):
    assert n_sources.shape[0] == galaxy_bool.shape[0]
    assert galaxy_bool.shape[-1] == 1
    max_sources = galaxy_bool.shape[-2]
    assert n_sources.le(max_sources).all()
    is_on_array = get_is_on_from_n_sources(n_sources, max_sources)
    is_on_array = is_on_array.view(*galaxy_bool.shape)
    star_bool = (1 - galaxy_bool) * is_on_array
    return star_bool


def _argfront(is_on_array, dim):
    # return indices that sort pushing all zeroes of tensor to the back.
    # dim is dimension along which do the ordering.
    assert len(is_on_array.shape) == 2
    indx_sort = (is_on_array != 0).long().argsort(dim=dim, descending=True)
    return indx_sort


def _sample_class_weights(class_weights, n_samples=1):
    """Draw a sample from Categorical variable with probabilities class_weights."""

    assert not torch.any(torch.isnan(class_weights))
    cat_rv = categorical.Categorical(probs=class_weights)
    return cat_rv.sample((n_samples,)).squeeze()


def _get_tile_coords(slen, tile_slen):
    """
    This records (x0, x1) indices each image tile comes from.

    Returns:
        tile_coords (torch.LongTensor):
    """

    nptiles1 = int(slen / tile_slen)
    n_ptiles = nptiles1 ** 2

    def return_coords(i):
        return [(i // nptiles1) * tile_slen, (i % nptiles1) * tile_slen]

    tile_coords = torch.tensor([return_coords(i) for i in range(n_ptiles)])
    tile_coords = tile_coords.long().to(device)

    return tile_coords


def _loc_mean_func(x):
    return torch.sigmoid(x) * (x != 0).float()


def _prob_galaxy_func(x):
    return torch.sigmoid(x).clamp(1e-4, 1 - 1e-4)


def _identity_func(x):
    return x


class BaseEncoder(nn.Module, ABC):
    def __init__(
        self,
        max_detections=1,
        n_bands=1,
        tile_slen=10,
        ptile_slen=16,
        enc_conv_c=20,
        enc_kern=3,
        enc_hidden=256,
        momentum=0.5,
    ):
        """
        This class implements the source encoder, which is supposed to take in a synthetic image of
        size slen * slen and returns a NN latent variable representation of this image.

        Args:
        slen (int): dimension of full image, we assume its square for now
        ptile_slen (int): dimension (in pixels) of the individual
                           image padded tiles (usually 8 for stars, and _ for galaxies).
        n_bands (int): number of bands
        max_detections (int): Number of maximum detections in a single tile.
        n_galaxy_params (int): Number of latent dimensions in the galaxy VAE network.

        """

        super(BaseEncoder, self).__init__()
        self.max_detections = max_detections
        self.n_bands = n_bands
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen

        border_padding = (ptile_slen - tile_slen) / 2
        assert border_padding % 1 == 0, "amount of padding should be an integer"
        self.border_padding = int(border_padding)

        # cache the weights used for the tiling convolution
        self._cache_tiling_conv_weights()

        conv_out_dim = self.enc_conv_c * ptile_slen ** 2
        self.enc_conv = nn.Sequential(
            nn.Conv2d(self.n_bands, enc_conv_c, enc_kern, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(enc_conv_c, enc_conv_c, enc_kern, stride=1, padding=1),
            nn.BatchNorm2d(enc_conv_c, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(enc_conv_c, enc_conv_c, enc_kern, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(enc_conv_c, enc_conv_c, enc_kern, stride=1, padding=1),
            nn.BatchNorm2d(enc_conv_c, momentum=momentum),
            nn.ReLU(),
            nn.Flatten(1, -1),
            nn.Linear(conv_out_dim, enc_hidden),
            nn.BatchNorm1d(enc_hidden, momentum=momentum),
            nn.ReLU(),
            nn.Linear(enc_hidden, enc_hidden),
            nn.BatchNorm1d(enc_hidden, momentum=momentum),
            nn.ReLU(),
            nn.Linear(enc_hidden, enc_hidden),
            nn.BatchNorm1d(enc_hidden, momentum=momentum),
            nn.ReLU(),
        )

        self.enc_final = nn.Linear(enc_hidden, self.dim_out_all)

        # Number of variational parameters used to characterize each source in an image.
        self.n_params_per_source = sum(
            self.variational_params[k]["dim"] for k in self.variational_params
        )

        # There are self.max_detections * (self.max_detections + 1) total possible detections.
        # For each param, for each possible number of detection d, there are d ways of
        # assigning that param.
        # NOTE: Dimensions correspond to the probabilities in ONE tile.
        self.dim_out_all = int(
            0.5
            * self.max_detections
            * (self.max_detections + 1)
            * self.n_params_per_source
        )

        self.indx_mats = {}  # initialize later in child class __init__

    def _cache_tiling_conv_weights(self):
        # this function sets up weights for the "identity" convolution
        # used to divide a full-image into padded tiles.
        # (see get_image_in_tiles).

        # It has a for-loop, but only needs to be set up once.
        # These weights are set up and  cached during the __init__.

        ptile_slen2 = self.ptile_slen ** 2
        self.tile_conv_weights = torch.zeros(
            ptile_slen2 * self.n_bands,
            self.n_bands,
            self.ptile_slen,
            self.ptile_slen,
            device=device,
        )

        for b in range(self.n_bands):
            for i in range(ptile_slen2):
                self.tile_conv_weights[
                    i + b * ptile_slen2, b, i // self.ptile_slen, i % self.ptile_slen
                ] = 1

    def get_images_in_tiles(self, images):
        # divide a full-image into padded tiles using conv2d
        # and weights cached in `_cache_tiling_conv_weights`.

        assert len(images.shape) == 4  # should be batch_size x n_bands x slen x slen
        assert images.shape[1] == self.n_bands

        output = F.conv2d(
            images,
            self.tile_conv_weights,
            stride=self.tile_slen,
            padding=0,
        ).permute([0, 2, 3, 1])

        return output.reshape(-1, self.n_bands, self.ptile_slen, self.ptile_slen)

    # TODO: Idea is there but need to play around with it to see if it is correct.
    def _get_hidden_indices(self):
        """Setup the indices corresponding to entries in h, these are cached since
        same for all h."""

        # initialize matrices containing the indices for each variational param.
        indx_mats = {}
        for k in self.variational_params:
            param_dim = self.variational_params[k]["dim"]
            shape = (self.max_detections + 1, param_dim * self.max_detections)
            indx_mat = torch.full(
                shape,
                self.dim_out_all,
                dtype=torch.long,
                device=device,
            )
            indx_mats[k] = indx_mat

        # add corresponding indices to the index matrices of variational params
        # for a given n_detection.
        curr_indx = 0
        for n_detections in range(1, self.max_detections + 1):
            for k in self.variational_params:
                param_dim = self.variational_params[k]["dim"]
                new_indx = (param_dim * n_detections) + curr_indx
                indx_mats[k][
                    n_detections, 0 : (param_dim * n_detections)
                ] = torch.arange(curr_indx, new_indx)
                curr_indx = new_indx

        return indx_mats

    def _indx_h_for_n_sources(self, h, n_sources, indx_mat, param_dim):
        """
        Index into all possible combinations of variational parameters (h) to obtain actually
        variational parameters for n_sources.
        Args:
            h: shape = (n_ptiles x dim_out_all)
            n_sources: (n_samples x n_tiles)
            param_dim: the dimension of the parameter you are indexing h for. e.g. for locs,
                            dim_per_source = 2, for galaxy params we usually have
                            dim_per_source = 8.
        Returns:
            var_param: shape = (n_samples x n_ptiles x max_detections x dim_per_source)
        """

        # n_samples = (1 x n_ptiles) if this function was called from forward.
        assert len(n_sources.shape) == 2, "Shape: (n_samples x n_ptiles)"
        assert h.size(0) == n_sources.size(1)  # = n_ptiles
        assert h.size(1) == self.dim_out_all

        n_ptiles = h.size(0)
        n_samples = n_sources.size(0)

        # append null column, return zero if indx_mat returns null index (dim_out_all)
        _h = torch.cat((h, torch.zeros(n_ptiles, 1, device=device)), dim=1)

        # select the indices from _h indicated by indx_mat.
        var_param = torch.gather(
            _h,
            1,
            indx_mat[n_sources.transpose(0, 1)].reshape(n_ptiles, -1),
        )

        var_param = var_param.view(
            n_ptiles, n_samples, self.max_detections, param_dim
        ).transpose(0, 1)

        # shape = (n_samples x n_ptiles x max_detections x dim_per_source)
        return var_param

    def _get_var_params_all(self, image_ptiles):
        # get h matrix.
        # image_ptiles shape: (n_ptiles, n_bands, ptile_slen, ptile_slen)
        # Forward to the layer that is shared by all n_sources.
        log_img = torch.log(image_ptiles - image_ptiles.min() + 1.0)
        h = self.enc_conv(log_img)

        # Concatenate all output parameters for all possible n_sources
        return self.enc_final(h)

    def _get_var_params_for_n_sources(self, h, n_sources):
        """
        Returns:
            loc_mean.shape = (n_samples x n_ptiles x max_detections x len(x,y))
            source_param_mean.shape = (n_samples x n_ptiles x max_detections x n_source_params)
        """
        assert not bool(self.indx_mats), "empty indx_mats"

        est_params = {}
        for k in self.variational_params:
            indx_mat = self.indx_mats[k]
            param_dim = self.variational_params[k]["dim"]
            transform = self.variational_params[k]["transform"]
            _param = self._indx_h_for_n_sources(h, n_sources, indx_mat, param_dim)
            param = transform(_param)
            est_params[k] = param

        return est_params

    @staticmethod
    def _get_normal_samples(mean, sd, tile_is_on_array):
        # tile_is_on_array can be either 'tile_is_on_array'/'tile_galaxy_bool'/'tile_star_bool'.
        # return shape = (n_samples x n_ptiles x max_detections x param_dim)
        assert tile_is_on_array.shape[-1] == 1
        return torch.normal(mean, sd) * tile_is_on_array

    @abstractmethod
    def tile_map_estimate(self, *args):
        # conditioned on how many sources are in each of the padded tiles, returned a dict
        # containing the variational parameters on each tile.
        pass

    @abstractmethod
    def forward(self, images, tile_n_sources):
        # conditioned on how many sources are in each of the padded tiles, returned a dict
        # containing the variational parameters on each tile.
        pass

    @abstractmethod
    @property
    def variational_params(self):
        # return a dict with the variational params that should be estimated.
        pass


class ImageEncoder(BaseEncoder):
    def __init__(self, **kwargs):

        super(ImageEncoder, self).__init__(**kwargs)
        self.n_star_params = self.n_bands
        self.log_softmax = nn.LogSoftmax(dim=1)

        # Accounts for categorical probability over # of objects.
        self.dim_out_all += 1 + self.max_detections
        self.indx_mats = self._get_hidden_indices()

        # TODO: Assign it to the last indices that were not used up.
        # get index for prob_n_sources, assigned indices that were not used.
        self.prob_n_source_indx = torch.zeros(
            self.max_detections + 1, dtype=torch.long, device=device
        )
        self.prob_n_source_indx = torch.arange(1, 3)

    def _get_logprob_n_from_var_params(self, h):
        """
        Obtain log probability of number of n_sources.

        * Example: If max_detections = 3, then Tensor will be (n_tiles x 3) since will return
        probability of having 0,1,2 stars.
        """
        free_probs = h[:, self.prob_n_source_indx]
        return self.log_softmax(free_probs)

    def forward(self, images, tile_n_sources):
        # images shape = (batch_size x n_bands x Slen x Slen)
        # tile_n_sources shape = (batch_size x n_tiles)
        # will unsqueeze and squeeze n_sources later, since used for indexing.
        assert len(tile_n_sources.shape) == 2
        batch_size = images.shape[0]
        n_sources = tile_n_sources.clamp(max=self.max_detections)

        # h.shape = (n_ptiles x self.dim_out_all)
        ptiles = self.get_images_in_tiles(images)
        h = self._get_var_params_all(ptiles)

        # get probability of n_sources and other params.
        # shape = (n_ptiles x (max_detections+1))
        # e.g. loc_mean has shape = (1 x n_ptiles x max_detections x len(x,y))
        n_source_log_probs = self._get_logprob_n_from_var_params(h)
        var_params = self._get_var_params_for_n_sources(h, n_sources)

        # squeeze to remove the 1 up front in the shape.
        var_params["n_source_log_probs"] = n_source_log_probs.view(batch_size, -1)
        var_params = {
            key: param.view(batch_size, -1, param.shape[2], param.shape[3])
            for key, param in var_params.items()
        }

        return var_params

    def tile_map_n_sources(self, images):
        # get map estimate for n_sources in each tile, for each batch_size
        batch_size = images.shape[0]
        ptiles = self.get_images_in_tiles(images)
        h = self._get_var_params_all(ptiles)
        log_probs_n_sources_per_tile = self._get_logprob_n_from_var_params(h)
        tile_n_sources = torch.argmax(log_probs_n_sources_per_tile, dim=1)
        tile_n_sources = tile_n_sources.view(batch_size, -1)

        return tile_n_sources

    def tile_map_estimate(self, images, tile_n_sources):

        # tile_is_on_array shape = (batchsize x n_tiles x max_detections)
        tile_is_on_array = get_is_on_from_n_sources(tile_n_sources, self.max_detections)
        tile_is_on_array = tile_is_on_array.unsqueeze(-1).float()

        # get variational parameters: these are on image tiles
        pred = self(images, tile_n_sources)

        # galaxy booleans
        tile_galaxy_bool = (pred["prob_galaxy"] > 0.5).float()
        tile_galaxy_bool *= tile_is_on_array

        # set sd so we return map estimates.
        # first locs
        locs_sd = torch.zeros_like(pred["loc_logvar"])
        tile_locs = self._get_normal_samples(
            pred["loc_mean"], locs_sd, tile_is_on_array
        )
        tile_locs = tile_locs.clamp(0, 1)

        # then log_fluxes
        tile_star_bool = get_star_bool(tile_n_sources, tile_galaxy_bool)
        log_flux_sd = torch.zeros_like(pred["log_flux_logvar"])
        tile_log_fluxes = self._get_normal_samples(
            pred["log_flux_mean"], log_flux_sd, tile_is_on_array
        )
        tile_log_fluxes *= tile_star_bool
        tile_fluxes = tile_log_fluxes.exp() * tile_star_bool

        return {
            "locs": tile_locs,
            "galaxy_bool": tile_galaxy_bool,
            "log_fluxes": tile_log_fluxes,
            "fluxes": tile_fluxes,
        }

    @property
    def variational_params(self):
        # transform is a function applied directly on NN output.
        return {
            "loc_mean": {"dim": 2, "transform": _loc_mean_func},
            "loc_logvar": {"dim": 2, "transform": _identity_func},
            "log_flux_mean": {"dim": self.n_bands, "transform": _identity_func},
            "log_flux_logvar": {"dim": self.n_bands, "transform": _identity_func},
            "prob_galaxy": {
                "dim": 1,
                "transform": _prob_galaxy_func,
            },
        }


class GalaxyEncoder(BaseEncoder):
    def __init__(self, n_galaxy_params=8, **kwargs):
        super(GalaxyEncoder, self).__init__(**kwargs)
        self.n_galaxy_params = n_galaxy_params
        self.indx_mats = self._get_hidden_indices()

    def tile_map_estimate(self, images, tile_n_sources, galaxy_bool):
        pred = self(images, tile_n_sources)
        sd = torch.zeros_like(pred["galaxy_param_logvar"])
        tile_galaxy_param = self._get_normal_samples(pred["loc_mean"], sd, galaxy_bool)
        return {"galaxy_params": tile_galaxy_param}

    def forward(self, images, tile_n_sources):
        assert len(tile_n_sources.shape) == 2
        batch_size = images.shape[0]
        n_sources = tile_n_sources.clamp(max=self.max_detections)

        ptiles = self.get_images_in_tiles(images)
        h = self._get_var_params_all(ptiles)
        var_params = self._get_var_params_for_n_sources(h, n_sources)

        var_params = {
            key: param.view(batch_size, -1, param.shape[2], param.shape[3])
            for key, param in var_params.items()
        }

        return var_params

    @property
    def variational_params(self):
        # transform is a function applied directly on NN output.
        return {
            "galaxy_param_mean": {
                "dim": self.n_galaxy_params,
                "transform": _identity_func,
            },
            "galaxy_param_logvar": {
                "dim": self.n_galaxy_params,
                "transform": _identity_func,
            },
        }
