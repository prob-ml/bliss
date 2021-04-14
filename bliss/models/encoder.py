import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import categorical


def get_mgrid(slen):
    offset = (slen - 1) / 2
    x, y = np.mgrid[-offset : (offset + 1), -offset : (offset + 1)]
    mgrid = torch.tensor(np.dstack((y, x))) / offset
    # mgrid is between -1 and 1
    # then scale slightly because of the way f.grid_sample
    # parameterizes the edges: (0, 0) is center of edge pixel
    return mgrid.float() * (slen - 1) / slen


def get_is_on_from_n_sources(n_sources, max_sources):
    """Return a boolean array of shape=(batch_size, max_sources) whose (k,l)th entry indicates
    whether there are more than l sources on the kth batch.
    """
    assert not torch.any(torch.isnan(n_sources))
    assert torch.all(n_sources >= 0)
    assert torch.all(n_sources.le(max_sources))

    is_on_array = torch.zeros(
        *n_sources.shape, max_sources, device=n_sources.device, dtype=torch.float
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


def get_full_params(tile_params: dict, slen: int, wlen: int = None):
    # NOTE: off sources should have tile_locs == 0.
    # NOTE: assume that each param in each tile is already pushed to the front.

    # check slen, wlen
    if wlen is None:
        wlen = slen
    assert isinstance(slen, int) and isinstance(wlen, int)

    # dictionary of tile_params is consistent and no extraneous keys.
    required = {"n_sources", "locs"}
    optional = {"galaxy_bool", "galaxy_params", "fluxes", "log_fluxes"}
    assert required.issubset(tile_params.keys())
    for param_name in tile_params:
        assert param_name in required or param_name in optional

    tile_n_sources = tile_params["n_sources"]
    tile_locs = tile_params["locs"]

    # tile_locs shape = (n_samples x n_tiles_per_image x max_detections x 2)
    assert len(tile_locs.shape) == 4
    n_samples = tile_locs.shape[0]
    n_tiles_per_image = tile_locs.shape[1]
    max_detections = tile_locs.shape[2]
    n_ptiles = n_samples * n_tiles_per_image

    # calculate tile_slen
    tile_slen = np.sqrt(slen * wlen / n_tiles_per_image)
    assert tile_slen % 1 == 0, "Image cannot be subdivided into tiles!"
    assert slen % tile_slen == 0 and wlen % tile_slen == 0, "incompatible side lengths."
    tile_slen = int(tile_slen)

    # coordinates on tiles.
    x_coords = torch.arange(0, slen, tile_slen, device=tile_n_sources.device).long()
    y_coords = torch.arange(0, wlen, tile_slen, device=tile_n_sources.device).long()
    tile_coords = torch.cartesian_prod(x_coords, y_coords)
    assert tile_coords.shape[0] == n_tiles_per_image, "# tiles one image don't match"

    # get is_on_array
    tile_is_on_array_sampled = get_is_on_from_n_sources(tile_n_sources, max_detections)
    n_sources = tile_is_on_array_sampled.sum(dim=(1, 2))  # per sample.
    max_sources = n_sources.max().int().item()

    # recenter and renormalize locations.
    tile_is_on_array = tile_is_on_array_sampled.view(n_ptiles, -1)
    _tile_locs = tile_locs.view(n_ptiles, -1, 2)
    bias = tile_coords.repeat(n_samples, 1).unsqueeze(1).float()
    _locs = _tile_locs * tile_slen + bias
    _locs[..., 0] /= slen
    _locs[..., 1] /= wlen
    _locs *= tile_is_on_array.unsqueeze(2)

    # sort locs and clip
    locs = _locs.view(n_samples, -1, 2)
    _indx_sort = _argfront(locs[..., 0], dim=1)
    indx_sort = _indx_sort.unsqueeze(2)
    locs = torch.gather(locs, 1, indx_sort.repeat(1, 1, 2))
    locs = locs[:, 0:max_sources]

    params = {"n_sources": n_sources, "locs": locs}

    # now do the same for the rest of the parameters (without scaling or biasing)
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
    cat_rv = categorical.Categorical(probs=class_weights)
    return cat_rv.sample((n_samples,)).squeeze()


def _loc_mean_func(x):
    return torch.sigmoid(x) * (x != 0).float()


def _prob_galaxy_func(x):
    return torch.sigmoid(x).clamp(1e-4, 1 - 1e-4)


def _identity_func(x):
    return x


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout, downsample=False):
        super().__init__()
        self.downsample = downsample
        stride = 1
        if downsample:
            stride = 2
            self.sc_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)
            self.sc_bn = nn.BatchNorm2d(out_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.drop1 = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            identity = self.sc_bn(self.sc_conv(identity))

        out = x + identity
        out = F.relu(out)
        return out


class EncoderCNN(nn.Module):
    def __init__(self, n_bands, channel, dropout):
        super().__init__()
        self.layer = self._make_layer(n_bands, channel, dropout)

    def forward(self, x):
        x = self.layer(x)
        return x

    def _make_layer(self, n_bands, channel, dropout):
        layers = [
            nn.Conv2d(n_bands, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
        ]
        in_channel = channel
        for i in range(3):
            downsample = True
            if i == 0:
                downsample = False
            layers += [ConvBlock(in_channel, channel, dropout, downsample)]
            layers += [
                ConvBlock(channel, channel, dropout, False),
                ConvBlock(channel, channel, dropout, False),
            ]
            in_channel = channel
            channel = channel * 2
        return nn.Sequential(*layers)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        max_detections=1,
        n_bands=1,
        tile_slen=2,
        ptile_slen=6,
        channel=8,
        spatial_dropout=0,
        dropout=0,
        hidden=128,
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
        super().__init__()
        self.max_detections = max_detections
        self.n_bands = n_bands
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        border_padding = (ptile_slen - tile_slen) / 2
        assert tile_slen <= ptile_slen
        assert border_padding % 1 == 0, "amount of border padding should be an integer"
        self.border_padding = int(border_padding)

        # cache the weights used for the tiling convolution
        self._cache_tiling_conv_weights()

        self.enc_conv = EncoderCNN(n_bands, channel, spatial_dropout)

        # Number of variational parameters used to characterize each source in an image.
        self.n_params_per_source = sum(
            self.variational_params[k]["dim"] for k in self.variational_params
        )

        # There are self.max_detections * (self.max_detections + 1) total possible detections.
        # For each param, for each possible number of detection d, there are d ways of assignment.
        # NOTE: Dimensions correspond to the probabilities in ONE tile.
        self.dim_out_all = int(
            0.5 * self.max_detections * (self.max_detections + 1) * self.n_params_per_source
            + 1
            + self.max_detections
        )
        dim_enc_conv_out = ((self.ptile_slen + 1) // 2 + 1) // 2
        self.enc_final = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(channel * 4 * dim_enc_conv_out ** 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.dim_out_all),
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

        # get index for prob_n_sources, assigned indices that were not used.
        indx_mats, last_indx = self._get_hidden_indices()
        for k, v in indx_mats.items():
            self.register_buffer(k + "_indx", v, persistent=False)
        self.register_buffer(
            "prob_n_source_indx",
            torch.arange(last_indx, self.dim_out_all),
            persistent=False,
        )
        assert self.prob_n_source_indx.shape[0] == self.max_detections + 1

        # grid for center cropped tiles
        self.register_buffer("cached_grid", get_mgrid(self.ptile_slen), persistent=False)

        # misc
        self.register_buffer("swap", torch.tensor([1, 0]), persistent=False)

    def get_images_in_tiles_new(self, image):
        channels = image.shape[1]
        window = self.ptile_slen
        stride = self.tile_slen
        tiles = F.unfold(image, kernel_size=window, stride=stride)
        # b=batch, c=channel, h=tile_height, w=tiles_width, n=num_of_tiles_for_each_batch
        tiles = rearrange(tiles, "b (c h w) n -> (b n) c h w", c=channels, h=window, w=window)
        return tiles

    def _cache_tiling_conv_weights(self):
        # this function sets up weights for the "identity" convolution
        # used to divide a full-image into padded tiles.
        # (see get_images_in_tiles).

        # It has a for-loop, but only needs to be set up once.
        # These weights are set up and  cached during the __init__.

        ptile_slen2 = self.ptile_slen ** 2
        self.register_buffer(
            "tile_conv_weights",
            torch.zeros(
                ptile_slen2 * self.n_bands,
                self.n_bands,
                self.ptile_slen,
                self.ptile_slen,
            ),
            persistent=False,
        )

        for b in range(self.n_bands):
            for i in range(ptile_slen2):
                self.tile_conv_weights[
                    i + b * ptile_slen2, b, i // self.ptile_slen, i % self.ptile_slen
                ] = 1

    def get_images_in_tiles(self, images):
        # divide a full-image into padded tiles using conv2d
        # and weights cached in `_cache_tiling_conv_weights`.

        assert len(images.shape) == 4  # should be batch_size x n_bands x pslen x pslen
        assert images.shape[1] == self.n_bands
        batch_size = images.shape[0]

        output = F.conv2d(
            images,
            self.tile_conv_weights,
            stride=self.tile_slen,
            padding=0,
        ).permute([0, 2, 3, 1])

        # shape = (n_ptiles x n_bands x ptile_slen, ptile_slen)
        # not borded slen
        pslen, pwlen = images.shape[-2:]
        slen = pslen - self.border_padding * 2
        wlen = pwlen - self.border_padding * 2
        n_ptiles_per_image = slen * wlen / self.tile_slen ** 2
        assert n_ptiles_per_image % 1 == 0, "n_ptiles_per_image must be an int"
        n_ptiles = int(n_ptiles_per_image * batch_size)
        old_implementation = output.reshape(
            n_ptiles, self.n_bands, self.ptile_slen, self.ptile_slen
        )
        new_implementation = self.get_images_in_tiles_new(images)
        assert torch.allclose(old_implementation, new_implementation)
        return new_implementation

    def center_ptiles(self, image_ptiles, tile_locs):
        # assume there is at most one source per tile
        # return a centered version of sources in tiles using their true locations in tiles.
        # also we crop them to avoid sharp borders with no bacgkround/noise.

        # round up necessary variables and paramters
        assert len(image_ptiles.shape) == 4
        assert len(tile_locs.shape) == 3
        assert tile_locs.shape[1] == 1
        assert image_ptiles.shape[-1] == self.ptile_slen
        n_ptiles = image_ptiles.shape[0]
        crop_slen = 2 * self.tile_slen
        ptile_slen = self.ptile_slen
        assert tile_locs.shape[0] == n_ptiles

        # get new locs to do the shift
        ptile_locs = tile_locs * self.tile_slen + self.border_padding
        ptile_locs /= ptile_slen
        locs0 = torch.tensor([ptile_slen - 1, ptile_slen - 1]) / 2
        locs0 /= ptile_slen - 1
        locs0 = locs0.view(1, 1, 2).to(image_ptiles.device)
        locs = 2 * locs0 - ptile_locs

        # center tiles on the corresponding source given by locs.
        locs = (locs - 0.5) * 2
        locs = locs.index_select(2, self.swap)  # trps (x,y) coords
        grid_loc = self.cached_grid.view(1, ptile_slen, ptile_slen, 2) - locs.view(-1, 1, 1, 2)
        shifted_tiles = F.grid_sample(image_ptiles, grid_loc, align_corners=True)

        # now that everything is center we can crop easily
        cropped_tiles = shifted_tiles[
            :, :, crop_slen : ptile_slen - crop_slen, crop_slen : ptile_slen - crop_slen
        ]
        return cropped_tiles

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
            )
            indx_mats[k] = indx_mat

        # add corresponding indices to the index matrices of variational params
        # for a given n_detection.
        curr_indx = 0
        for n_detections in range(1, self.max_detections + 1):
            for k in self.variational_params:
                param_dim = self.variational_params[k]["dim"]
                new_indx = (param_dim * n_detections) + curr_indx
                indx_mats[k][n_detections, 0 : (param_dim * n_detections)] = torch.arange(
                    curr_indx, new_indx
                )
                curr_indx = new_indx

        return indx_mats, curr_indx

    def _indx_h_for_n_sources(self, h, n_sources, indx_mat, param_dim):
        """
        Index into all possible combinations of variational parameters (h) to obtain actually
        variational parameters for n_sources.
        Args:
            h: shape = (n_ptiles x dim_out_all)
            n_sources: (n_samples x n_tiles)
            param_dim: the dimension of the parameter you are indexing h.
        Returns:
            var_param: shape = (n_samples x n_ptiles x max_detections x dim_per_source)
        """
        assert len(n_sources.shape) == 2
        assert h.size(0) == n_sources.size(1)
        assert h.size(1) == self.dim_out_all
        n_ptiles = h.size(0)
        n_samples = n_sources.size(0)

        # append null column, return zero if indx_mat returns null index (dim_out_all)
        _h = torch.cat((h, torch.zeros(n_ptiles, 1, device=h.device)), dim=1)

        # select the indices from _h indicated by indx_mat.
        var_param = torch.gather(
            _h,
            1,
            indx_mat[n_sources.transpose(0, 1)].reshape(n_ptiles, -1),
        )

        var_param = var_param.view(n_ptiles, n_samples, self.max_detections, param_dim)
        return var_param.transpose(0, 1)

    def get_var_params_all(self, image_ptiles):
        # get h matrix.
        # Forward to the layer that is shared by all n_sources.
        log_img = torch.log(image_ptiles - image_ptiles.min() + 1.0)
        h = self.enc_conv(log_img)

        # Concatenate all output parameters for all possible n_sources
        return self.enc_final(h)

    def _get_var_params_for_n_sources(self, h, n_sources):
        """
        Args:
            n_sources.shape = (n_samples x n_ptiles)

        Returns:
            loc_mean.shape = (n_sample x n_ptiles x max_detections x len(x,y))
        """
        assert len(n_sources.shape) == 2

        est_params = {}
        for k in self.variational_params:
            indx_mat = getattr(self, k + "_indx")
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

    def _get_logprob_n_from_var_params(self, h):
        """
        Obtain log probability of number of n_sources.

        * Example: If max_detections = 3, then Tensor will be (n_tiles x 3) since will return
        probability of having 0,1,2 stars.
        """
        free_probs = h[:, self.prob_n_source_indx]
        return self.log_softmax(free_probs)

    def forward_sampled(self, image_ptiles, tile_n_sources_sampled):
        # images shape = (n_ptiles x n_bands x pslen x pslen)
        # tile_n_sources shape = (n_samples x n_ptiles)
        assert len(tile_n_sources_sampled.shape) == 2
        assert image_ptiles.shape[0] == tile_n_sources_sampled.shape[1]
        # h.shape = (n_ptiles x self.dim_out_all)
        h = self.get_var_params_all(image_ptiles)

        # get probability of params except n_sources
        # e.g. loc_mean: shape = (n_samples x n_ptiles x max_detections x len(x,y))
        var_params = self._get_var_params_for_n_sources(h, tile_n_sources_sampled)

        # get probability of n_sources
        # n_source_log_probs: shape = (n_ptiles x (max_detections+1))
        n_source_log_probs = self._get_logprob_n_from_var_params(h)
        var_params["n_source_log_probs"] = n_source_log_probs

        return var_params

    def forward(self, image_ptiles, tile_n_sources):
        # images shape = (n_ptiles x n_bands x pslen x pslen)
        # tile_n_sources shape = (n_ptiles)
        assert len(tile_n_sources.shape) == 1
        assert len(image_ptiles.shape) == 4
        assert image_ptiles.shape[0] == tile_n_sources.shape[0]
        tile_n_sources = tile_n_sources.clamp(max=self.max_detections).unsqueeze(0)
        var_params = self.forward_sampled(image_ptiles, tile_n_sources)
        var_params = {key: value.squeeze(0) for key, value in var_params.items()}
        return var_params

    def sample_encoder(self, images, n_samples):
        assert len(images.shape) == 4
        assert images.shape[0] == 1, "Only works for 1 image"
        image_ptiles = self.get_images_in_tiles(images)
        h = self.get_var_params_all(image_ptiles)
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
        pred = self.forward_sampled(image_ptiles, tile_n_sources)

        # other quantities based on var_params
        # tile_galaxy_bool shape = (n_samples x n_ptiles x max_detections x 1)
        tile_galaxy_bool = torch.bernoulli(pred["prob_galaxy"]).float()
        tile_galaxy_bool *= tile_is_on_array
        tile_star_bool = get_star_bool(tile_n_sources, tile_galaxy_bool)
        pred["loc_sd"] = torch.exp(0.5 * pred["loc_logvar"])
        pred["log_flux_sd"] = torch.exp(0.5 * pred["log_flux_logvar"])
        tile_locs = self._get_normal_samples(pred["loc_mean"], pred["loc_sd"], tile_is_on_array)
        tile_log_fluxes = self._get_normal_samples(
            pred["log_flux_mean"], pred["log_flux_sd"], tile_star_bool
        )
        tile_fluxes = tile_log_fluxes.exp() * tile_star_bool
        return {
            "n_sources": tile_n_sources,
            "locs": tile_locs,
            "galaxy_bool": tile_galaxy_bool,
            "log_fluxes": tile_log_fluxes,
            "fluxes": tile_fluxes,
        }

    def tile_map_estimate_from_var_params(self, pred, n_tiles_per_image, batch_size):
        # batch_size = # of images that will be predicted.
        # n_tiles_per_image = # tiles/padded_tiles each image is subdivided into.
        # pred = prediction of variational parameters on each tile.

        # tile_n_sources based on log_prob per tile.
        # tile_is_on_array shape = (n_ptiles x max_detections)
        tile_n_sources = torch.argmax(pred["n_source_log_probs"], dim=1)
        tile_is_on_array = get_is_on_from_n_sources(tile_n_sources, self.max_detections)
        tile_is_on_array = tile_is_on_array.unsqueeze(-1).float()

        # galaxy booleans
        tile_galaxy_bool = (pred["prob_galaxy"] > 0.5).float()
        tile_galaxy_bool *= tile_is_on_array

        # set sd so we return map estimates.
        # first locs
        locs_sd = torch.zeros_like(pred["loc_logvar"])
        tile_locs = self._get_normal_samples(pred["loc_mean"], locs_sd, tile_is_on_array)
        tile_locs = tile_locs.clamp(0, 1)

        # then log_fluxes
        tile_star_bool = get_star_bool(tile_n_sources, tile_galaxy_bool)
        log_flux_sd = torch.zeros_like(pred["log_flux_logvar"])
        tile_log_fluxes = self._get_normal_samples(
            pred["log_flux_mean"], log_flux_sd, tile_is_on_array
        )
        tile_log_fluxes *= tile_star_bool
        tile_fluxes = tile_log_fluxes.exp() * tile_star_bool

        tile_estimate = {
            "locs": tile_locs,
            "galaxy_bool": tile_galaxy_bool,
            "log_fluxes": tile_log_fluxes,
            "fluxes": tile_fluxes,
        }

        # reshape with images' batch_size.
        tile_estimate = {
            key: value.view(batch_size, n_tiles_per_image, self.max_detections, -1)
            for key, value in tile_estimate.items()
        }
        tile_estimate["n_sources"] = tile_n_sources.reshape(batch_size, -1)
        return tile_estimate

    def tile_map_n_sources(self, image_ptiles):
        h = self.get_var_params_all(image_ptiles)
        log_probs_n_sources_per_tile = self._get_logprob_n_from_var_params(h)
        tile_n_sources = torch.argmax(log_probs_n_sources_per_tile, dim=1)
        return tile_n_sources

    def tile_map_estimate(self, images):

        # extract image_ptiles
        batch_size = images.shape[0]
        image_ptiles = self.get_images_in_tiles(images)
        n_tiles_per_image = int(image_ptiles.shape[0] / batch_size)

        # MAP (for n_sources) prediction on var params on each tile
        tile_n_sources = self.tile_map_n_sources(image_ptiles)
        pred = self.forward(image_ptiles, tile_n_sources)

        return self.tile_map_estimate_from_var_params(pred, n_tiles_per_image, batch_size)

    def map_estimate(self, images, slen: int, wlen: int = None):
        # return full estimate of parameters in full image.
        # NOTE: slen*wlen is size of the image without border padding

        if wlen is None:
            wlen = slen
        assert isinstance(slen, int) and isinstance(wlen, int)
        # check image compatibility
        border1 = (images.shape[-2] - slen) / 2
        border2 = (images.shape[-1] - wlen) / 2
        assert border1 == border2, "border paddings on each dimension differ."
        assert slen % self.tile_slen == 0, "incompatible slen"
        assert wlen % self.tile_slen == 0, "incompatible wlen"
        assert border1 == self.border_padding, "incompatible border"

        # obtained estimates per tile, then on full image.
        tile_estimate = self.tile_map_estimate(images)
        estimate = get_full_params(tile_estimate, slen, wlen)
        return estimate

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
