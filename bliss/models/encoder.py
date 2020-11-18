import math
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


def get_full_params(slen: int, tile_params: dict):
    # NOTE: off sources should have tile_locs == 0.
    # NOTE: assume that each param in each tile is already pushed to the front.
    required = {"n_sources", "locs"}
    optional = {"galaxy_bool", "galaxy_params", "fluxes", "log_fluxes"}
    assert type(slen) is int
    assert type(tile_params) is dict
    assert required.issubset(tile_params.keys())
    # tile_params does not contain extraneous keys
    for param_name in tile_params:
        assert param_name in required or param_name in optional

    tile_n_sources = tile_params["n_sources"]
    tile_locs = tile_params["locs"]

    # tile_locs shape = (n_samples x n_tiles_per_image x max_detections x 2)
    assert len(tile_locs.shape) == 4
    n_samples = tile_locs.shape[0]
    n_tiles_per_image = tile_locs.shape[1]
    max_detections = tile_locs.shape[2]
    tile_slen = slen / math.sqrt(n_tiles_per_image)
    n_ptiles = n_samples * n_tiles_per_image
    assert int(tile_slen) == tile_slen, "Image cannot be subdivided into tiles!"
    tile_slen = int(tile_slen)

    # coordinates on tiles.
    tile_coords = _get_tile_coords(slen, tile_slen)
    assert tile_coords.shape[0] == n_tiles_per_image, "# tiles one image don't match"

    # get is_on_array
    tile_is_on_array_sampled = get_is_on_from_n_sources(tile_n_sources, max_detections)
    n_sources = tile_is_on_array_sampled.sum(dim=(1, 2))  # per sample.
    max_sources = n_sources.max().int().item()

    # recenter and renormalize locations.
    tile_is_on_array = tile_is_on_array_sampled.view(n_ptiles, -1)
    _tile_locs = tile_locs.view(n_ptiles, -1, 2)
    bias = tile_coords.repeat(n_samples, 1).unsqueeze(1).float()
    _locs = (_tile_locs * tile_slen + bias) / slen
    _locs *= tile_is_on_array.unsqueeze(2)

    # sort locs and clip
    locs = _locs.view(n_samples, -1, 2)
    _indx_sort = _argfront(locs[..., 0], dim=1)
    indx_sort = _indx_sort.unsqueeze(2)
    locs = torch.gather(locs, 1, indx_sort.repeat(1, 1, 2))
    locs = locs[:, 0:max_sources, ...]

    params = {"n_sources": n_sources, "locs": locs}

    # now do the same for the rest of the parameters (without scaling or biasing ofc)
    # for same reason no need to multiply times is_on_array
    for param_name in tile_params:
        if param_name in optional:
            # make sure works galaxy bool has same format as well.
            tile_param = tile_params[param_name]
            assert len(tile_param.shape) == 4
            _param = tile_param.view(n_samples, n_tiles_per_image, max_detections, -1)
            param_dim = _param.size(-1)
            param = _param.view(n_samples, -1, param_dim)
            param = torch.gather(param, 1, indx_sort.repeat(1, 1, param_dim))
            param = param[:, 0:max_sources, ...]

            params[param_name] = param

    return params


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


class ImageEncoder(nn.Module):
    def __init__(
        self,
        n_bands=1,
        tile_slen=2,
        ptile_slen=8,
        max_detections=2,
        n_galaxy_params=8,
        background_pad_value=686.0,
        enc_conv_c=20,
        enc_kern=3,
        enc_hidden=256,
        momentum=0.5,
        pad_border_w_constant=True,
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
        super(ImageEncoder, self).__init__()
        # image parameters
        self.n_bands = n_bands
        self.background_pad_value = background_pad_value

        # padding
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        self.edge_padding = (ptile_slen - tile_slen) / 2
        assert self.edge_padding % 1 == 0, "amount of padding should be an integer"
        self.edge_padding = int(self.edge_padding)
        self.pad_border_w_constant = pad_border_w_constant

        # cache the weights used for the tiling convolution
        self._cache_tiling_conv_weights()

        # max number of detections
        self.max_detections = max_detections

        # convolutional NN parameters
        self.enc_conv_c = enc_conv_c
        self.enc_kern = enc_kern
        self.enc_hidden = enc_hidden

        self.momentum = momentum

        # convolutional NN
        conv_out_dim = self.enc_conv_c * ptile_slen ** 2
        self.enc_conv = nn.Sequential(
            nn.Conv2d(
                self.n_bands, self.enc_conv_c, self.enc_kern, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.enc_conv_c, self.enc_conv_c, self.enc_kern, stride=1, padding=1
            ),
            nn.BatchNorm2d(self.enc_conv_c, momentum=self.momentum),
            nn.ReLU(),
            nn.Conv2d(
                self.enc_conv_c, self.enc_conv_c, self.enc_kern, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.enc_conv_c, self.enc_conv_c, self.enc_kern, stride=1, padding=1
            ),
            nn.BatchNorm2d(self.enc_conv_c, momentum=self.momentum),
            nn.ReLU(),
            nn.Flatten(1, -1),
            nn.Linear(conv_out_dim, self.enc_hidden),
            nn.BatchNorm1d(self.enc_hidden, momentum=self.momentum),
            nn.ReLU(),
            nn.Linear(self.enc_hidden, self.enc_hidden),
            nn.BatchNorm1d(self.enc_hidden, momentum=self.momentum),
            nn.ReLU(),
            nn.Linear(self.enc_hidden, self.enc_hidden),
            nn.BatchNorm1d(self.enc_hidden, momentum=self.momentum),
            nn.ReLU(),
        )

        # There are self.max_detections * (self.max_detections + 1)
        #  total possible detections, and each detection has
        #  4 + 2*n parameters (2 means and 2 variances for each loc + mean and variance for
        #  n source_param's (flux per band or galaxy params.) + 1 for the Bernoulli variable
        #  of whether the source is a star or galaxy.
        self.n_star_params = n_bands
        self.n_galaxy_params = n_galaxy_params
        self.n_params_per_source = (
            4 + 2 * (self.n_star_params + self.n_galaxy_params) + 1
        )

        # The first term corresponds to: for each param, for each possible number of detection d,
        # there are d ways of assigning that param.
        # The second and third term accounts for categorical probability over # of objects.
        # These dimensions correspond to the probabilities in ONE tile.
        self.dim_out_all = int(
            0.5
            * self.max_detections
            * (self.max_detections + 1)
            * self.n_params_per_source
            + 1
            + self.max_detections
        )

        self.variational_params = [
            ("loc_mean", 2, _loc_mean_func),
            ("loc_logvar", 2),
            ("galaxy_param_mean", self.n_galaxy_params),
            ("galaxy_param_logvar", self.n_galaxy_params),
            ("log_flux_mean", self.n_star_params),
            ("log_flux_logvar", self.n_star_params),
            ("prob_galaxy", 1, _prob_galaxy_func),
        ]
        self.n_variational_params = len(self.variational_params)

        self.indx_mats, self.prob_n_source_indx = self._get_hidden_indices()

        self.enc_final = nn.Linear(self.enc_hidden, self.dim_out_all)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def _create_indx_mats(self):
        indx_mats = []
        for i in range(self.n_variational_params):
            param_dim = self.variational_params[i][1]
            shape = (self.max_detections + 1, param_dim * self.max_detections)
            indx_mat = torch.full(
                shape,
                self.dim_out_all,
                dtype=torch.long,
                device=device,
            )
            indx_mats.append(indx_mat)
        return indx_mats

    def _update_indx_mat_for_n_detections(self, indx_mats, curr_indx, n_detections):
        # add corresponding indices to index matrices for n_detections.
        for i in range(self.n_variational_params):
            param_dim = self.variational_params[i][1]
            indx_mat = indx_mats[i]
            new_indx = (param_dim * n_detections) + curr_indx
            indx_mat[n_detections, 0 : (param_dim * n_detections)] = torch.arange(
                curr_indx, new_indx
            )
            indx_mats[i] = indx_mat
            curr_indx = new_indx

        return indx_mats, curr_indx

    def _get_hidden_indices(self):
        """Setup the indices corresponding to entries in h, these are cached since
        same for all h."""

        indx_mats = self._create_indx_mats()  # same order as self.variational_params
        prob_n_source_indx = torch.zeros(
            self.max_detections + 1, dtype=torch.long, device=device
        )

        for n_detections in range(1, self.max_detections + 1):
            # index corresponding to where we left off in last iteration.
            curr_indx = (
                int(0.5 * n_detections * (n_detections - 1) * self.n_params_per_source)
                + (n_detections - 1)
                + 1
            )

            # add corresponding indices to the index matrices of variational params.
            indx_mats, curr_indx = self._update_indx_mat_for_n_detections(
                indx_mats, curr_indx, n_detections
            )

            # the categorical prob will go at the end of the rest.
            prob_n_source_indx[n_detections] = curr_indx

        return indx_mats, prob_n_source_indx

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

    def _get_logprob_n_from_var_params(self, h):
        """
        Obtain log probability of number of n_sources.

        * Example: If max_detections = 3, then Tensor will be (n_tiles x 3) since will return
        probability of having 0,1,2 stars.
        """
        free_probs = h[:, self.prob_n_source_indx]
        return self.log_softmax(free_probs)

    def _get_var_params_for_n_sources(self, h, n_sources):
        """
        Returns:
            loc_mean.shape = (n_samples x n_ptiles x max_detections x len(x,y))
            source_param_mean.shape = (n_samples x n_ptiles x max_detections x n_source_params)
        """

        estimated_params = {}
        for i in range(self.n_variational_params):
            indx_mat = self.indx_mats[i]
            param_info = self.variational_params[i]
            param_name = param_info[0]
            param_dim = param_info[1]

            # obtain hidden function to apply if included, otherwise do nothing.
            hidden_function = param_info[2] if len(param_info) > 2 else lambda x: x
            _param = self._indx_h_for_n_sources(h, n_sources, indx_mat, param_dim)
            param = hidden_function(_param)
            estimated_params[param_name] = param

        return estimated_params

    def _get_var_params_all(self, image_ptiles):
        """get h matrix.

        image_ptiles shape: (n_ptiles, n_bands, ptile_slen, ptile_slen)
        """
        # Forward to the layer that is shared by all n_sources.
        log_img = torch.log(image_ptiles - image_ptiles.min() + 1.0)
        h = self.enc_conv(log_img)

        # Concatenate all output parameters for all possible n_sources
        return self.enc_final(h)

    def forward(self, image_ptiles, n_sources):
        # image_ptiles shape = (n_ptiles x n_bands x ptile_slen x ptile_slen)
        # n_sources shape = (n_ptiles)
        # will unsqueeze and squeeze n_sources later, since used for indexing.
        assert len(n_sources.shape) == 1
        n_sources = n_sources.unsqueeze(0)

        # h.shape = (n_ptiles x self.dim_out_all)
        h = self._get_var_params_all(image_ptiles)

        # get probability of n_sources
        # shape = (n_ptiles x (max_detections+1))
        n_source_log_probs = self._get_logprob_n_from_var_params(h)

        # e.g. loc_mean has shape = (1 x n_ptiles x max_detections x len(x,y))
        n_sources = n_sources.clamp(max=self.max_detections)
        var_params = self._get_var_params_for_n_sources(h, n_sources)
        # squeeze if possible to account for non-sampling case.
        var_params = {key: param.squeeze(0) for key, param in var_params.items()}
        var_params["n_source_log_probs"] = n_source_log_probs.squeeze(0)

        # dictionary with names as in self.variational_params
        return var_params

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
        assert images.size(1) == self.n_bands

        if self.pad_border_w_constant:
            pad = [self.edge_padding] * 4
            images = F.pad(images, pad=pad, value=self.background_pad_value)

        output = F.conv2d(
            images,
            self.tile_conv_weights,
            stride=self.tile_slen,
            padding=0,
        ).permute([0, 2, 3, 1])

        return output.reshape(-1, self.n_bands, self.ptile_slen, self.ptile_slen)

    @staticmethod
    def _get_samples(pred, tile_is_on_array, tile_galaxy_bool):
        # shape = (n_samples x n_ptiles x max_detections x param_dim)
        assert tile_is_on_array.shape[-1] == 1
        assert tile_galaxy_bool.shape[-1] == 1
        loc_mean, loc_sd = pred["loc_mean"], pred["loc_sd"]
        galaxy_param_mean = pred["galaxy_param_mean"]
        galaxy_param_sd = pred["galaxy_param_sd"]
        log_flux_mean, log_flux_sd = pred["log_flux_mean"], pred["log_flux_sd"]

        tile_locs = torch.normal(loc_mean, loc_sd).clamp(0, 1)
        tile_locs *= tile_is_on_array

        tile_galaxy_params = torch.normal(galaxy_param_mean, galaxy_param_sd)
        tile_galaxy_params *= tile_is_on_array * tile_galaxy_bool

        tile_log_fluxes = torch.normal(log_flux_mean, log_flux_sd)
        tile_log_fluxes *= tile_is_on_array * (1 - tile_galaxy_bool)

        return tile_locs, tile_galaxy_params, tile_log_fluxes

    def sample_encoder(self, image, n_samples):
        assert image.size(0) == 1, "Sampling only works for a single image."
        image_ptiles = self.get_images_in_tiles(image)
        h = self._get_var_params_all(image_ptiles)
        log_probs_n_sources_per_tile = self._get_logprob_n_from_var_params(h)

        # sample number of sources.
        # tile_n_sources shape = (n_samples x n_ptiles)
        # tile_is_on_array shape = (n_samples x n_ptiles x max_detections x 1)
        probs_n_sources_per_tile = torch.exp(log_probs_n_sources_per_tile)
        tile_n_sources = _sample_class_weights(probs_n_sources_per_tile, n_samples)
        tile_n_sources = tile_n_sources.view(n_samples, -1)
        tile_is_on_array = get_is_on_from_n_sources(tile_n_sources, self.max_detections)
        tile_is_on_array = tile_is_on_array.unsqueeze(-1).float()

        # get var_params conditioned on n_sources
        pred = self._get_var_params_for_n_sources(h, tile_n_sources)

        # other quantities based on var_params
        # tile_galaxy_bool shape = (n_samples x n_ptiles x max_detections x 1)
        tile_galaxy_bool = torch.bernoulli(pred["prob_galaxy"]).float()
        tile_galaxy_bool *= tile_is_on_array
        pred["loc_sd"] = torch.exp(0.5 * pred["loc_logvar"])
        pred["galaxy_param_sd"] = torch.exp(0.5 * pred["galaxy_param_logvar"])
        pred["log_flux_sd"] = torch.exp(0.5 * pred["log_flux_logvar"])
        tile_locs, tile_galaxy_params, tile_log_fluxes = self._get_samples(
            pred, tile_is_on_array, tile_galaxy_bool
        )

        tile_star_bool = get_star_bool(tile_n_sources, tile_galaxy_bool)
        tile_fluxes = tile_log_fluxes.exp() * tile_star_bool
        return {
            "n_sources": tile_n_sources,
            "locs": tile_locs,
            "galaxy_bool": tile_galaxy_bool,
            "galaxy_params": tile_galaxy_params,
            "log_fluxes": tile_log_fluxes,
            "fluxes": tile_fluxes,
        }

    def tiled_map_estimate(self, image):
        # NOTE: make sure to use inside a `with torch.no_grad()` and with .eval() if applicable.
        
        batchsize = image.shape[0]
        n_ptiles = image.shape[1]
        
        image_ptiles = self.get_images_in_tiles(image)
        h = self._get_var_params_all(image_ptiles)
        log_probs_n_sources_per_tile = self._get_logprob_n_from_var_params(h)

        # get map estimate for n_sources in each tile.
        # tile_n_sources shape = (batchsize x n_ptiles)
        # tile_is_on_array shape = (batchsize x n_ptiles x max_detections x 1)
        tile_n_sources = torch.argmax(log_probs_n_sources_per_tile, dim=1)
        tile_n_sources = tile_n_sources.view(batchsize, -1)
        tile_is_on_array = get_is_on_from_n_sources(tile_n_sources, self.max_detections)
        tile_is_on_array = tile_is_on_array.unsqueeze(-1).float()

        # get variational parameters: these are on image tiles
        # shape (all) = (1 x (batchsize x n_ptiles) x max_detections x param_dim)
        pred = self._get_var_params_for_n_sources(h, tile_n_sources.flatten().unsqueeze(0))
    
        # now reshape 
        pred = {key: param.view(batchsize, n_ptiles, param.shape[2], param.shape[3]) for key, param in pred.items()}
    
        # set sd so we return map estimates.
        tile_galaxy_bool = (pred["prob_galaxy"] > 0.5).float()
        tile_galaxy_bool *= tile_is_on_array
        pred["loc_sd"] = torch.zeros_like(pred["loc_logvar"])
        pred["galaxy_param_sd"] = torch.zeros_like(pred["galaxy_param_logvar"])
        pred["log_flux_sd"] = torch.zeros_like(pred["log_flux_logvar"])

        tile_locs, tile_galaxy_params, tile_log_fluxes = self._get_samples(
            pred, tile_is_on_array, tile_galaxy_bool
        )

        tile_star_bool = get_star_bool(tile_n_sources, tile_galaxy_bool)
        tile_fluxes = tile_log_fluxes.exp() * tile_star_bool

        return {
            "n_sources": tile_n_sources,
            "locs": tile_locs,
            "galaxy_bool": tile_galaxy_bool,
            "galaxy_params": tile_galaxy_params,
            "log_fluxes": tile_log_fluxes,
            "fluxes": tile_fluxes,
        }

    def map_estimate(self, image):
        slen = image.shape[-1]

        if not self.pad_border_w_constant:
            slen = slen - 2 * self.edge_padding

        tile_estimate = self.tiled_map_estimate(image)
        estimate = get_full_params(slen, tile_estimate)
        return estimate
