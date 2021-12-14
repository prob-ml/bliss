import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
from torch.distributions import categorical
from torch.nn import functional as F


def get_images_in_tiles(images, tile_slen, ptile_slen):
    """Divides a batch of full images into padded tiles.

    This is similar to nn.conv2d, with a sliding window=ptile_slen and stride=tile_slen.

    Arguments:
        images: Tensor of images with size (batchsize x n_bands x slen x slen)
        tile_slen: Side length of tile
        ptile_slen: Side length of padded tile

    Returns:
        A (batchsize x tiles_per_batch) x n_bands x tile_weight x tile_width image
    """
    assert len(images.shape) == 4
    n_bands = images.shape[1]
    window = ptile_slen
    tiles = F.unfold(images, kernel_size=window, stride=tile_slen)
    # b: batch, c: channel, h: tile height, w: tile width, n: num of total tiles for each batch
    return rearrange(tiles, "b (c h w) n -> (b n) c h w", c=n_bands, h=window, w=window)


def get_is_on_from_n_sources(n_sources, max_sources):
    """Provides tensor which indicates how many sources are present for each batch.

    Return a boolean array of `shape=(*n_sources.shape, max_sources)` whose `(*,l)th` entry
    indicates whether there are more than l sources on the `*th` index.

    Arguments:
        n_sources: Tensor with number of sources per tile.
        max_sources: Maximum number of sources allowed per tile.

    Returns:
        Tensor indicating how many sources are present for each batch.
    """
    assert not torch.any(torch.isnan(n_sources))
    assert torch.all(n_sources >= 0)
    assert torch.all(n_sources.le(max_sources))

    is_on_array = torch.zeros(
        *n_sources.shape,
        max_sources,
        device=n_sources.device,
        dtype=torch.float,
    )

    for i in range(max_sources):
        is_on_array[..., i] = n_sources > i

    return is_on_array


def get_full_params(tile_params: dict, slen: int, wlen: int = None):
    # NOTE: off sources should have tile_locs == 0.
    # NOTE: assume that each param in each tile is already pushed to the front.

    # check slen, wlen
    wlen = slen if wlen is None else wlen
    assert isinstance(slen, int) and isinstance(wlen, int)

    # check dictionary of tile_params is consistent and has no extraneous keys.
    required = {"n_sources", "locs"}
    optional = {"galaxy_bool", "star_bool", "galaxy_params", "fluxes", "log_fluxes", "prob_galaxy"}
    assert required.issubset(tile_params.keys())

    for pname in tile_params:
        assert pname in required or pname in optional or pname == "prob_n_sources"

    # tile_locs shape = (n_samples x n_tiles_per_image x max_detections x 2)
    tile_n_sources = tile_params["n_sources"]
    tile_locs = tile_params["locs"]
    assert len(tile_locs.shape) == 4
    n_samples = tile_locs.shape[0]
    n_tiles_per_image = tile_locs.shape[1]
    max_detections = tile_locs.shape[2]

    # otherwise prob_n_sources makes no sense globally
    if max_detections == 1:
        optional.add("prob_n_sources")

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
    tile_is_on_array = rearrange(tile_is_on_array_sampled, "b n d -> (b n) d")
    tile_locs = rearrange(tile_locs, "b n d xy -> (b n) d xy", xy=2)
    bias = repeat(tile_coords, "n xy -> (r n) 1 xy", r=n_samples).float()

    locs = tile_locs * tile_slen + bias
    locs[..., 0] /= slen
    locs[..., 1] /= wlen
    locs *= tile_is_on_array.unsqueeze(2)

    # sort locs and clip
    locs = locs.view(n_samples, -1, 2)
    indx_sort = _argfront(locs[..., 0], dim=1)
    locs = torch.gather(locs, 1, repeat(indx_sort, "b n -> b n r", r=2))
    locs = locs[:, 0:max_sources]
    params = {"n_sources": n_sources, "locs": locs}

    # now do the same for the rest of the parameters (without scaling or biasing)
    # for same reason no need to multiply times is_on_array
    for param_name, val in tile_params.items():
        if param_name in optional:
            tile_param = val
            assert len(tile_param.shape) == 4
            param = rearrange(tile_param, "b t d k -> b (t d) k")
            param = torch.gather(
                param, 1, repeat(indx_sort, "b n -> b n r", r=tile_param.shape[-1])
            )
            param = param[:, 0:max_sources]
            params[param_name] = param

    assert len(params["locs"].shape) == 3
    assert params["locs"].shape[1] == params["n_sources"].max().int().item()

    # add plocs = pixel locs.
    params["plocs"] = params["locs"].clone()
    params["plocs"][:, :, 0] = params["locs"][:, :, 0] * slen
    params["plocs"][:, :, 1] = params["locs"][:, :, 1] * wlen

    return params


class ImageEncoder(nn.Module):
    """Encodes the distribution of a latent variable representing an astronomical image.

    This class implements the source encoder, which is supposed to take in
    an astronomical image of size slen * slen and returns a NN latent variable
    representation of this image.
    """

    def __init__(
        self,
        max_detections: int = 1,
        n_bands: int = 1,
        tile_slen: int = 2,
        ptile_slen: int = 6,
        channel: int = 8,
        spatial_dropout=0,
        dropout=0,
        hidden: int = 128,
    ):
        """Initializes ImageEncoder.

        Args:
            max_detections: Number of maximum detections in a single tile.
            n_bands: number of bands
            tile_slen: dimension of full image, we assume its square for now
            ptile_slen: dimension (in pixels) of the individual
                            image padded tiles (usually 8 for stars, and _ for galaxies).
            channel: TODO (document this)
            spatial_dropout: TODO (document this)
            dropout: TODO (document this)
            hidden: TODO (document this)
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

        # Number of variational parameters used to characterize each source in an image.
        self.n_params_per_source = sum(param["dim"] for param in self.variational_params.values())

        # There are self.max_detections * (self.max_detections + 1) total possible detections.
        # For each param, for each possible number of detection d, there are d ways of assignment.
        # NOTE: Dimensions correspond to the probabilities in ONE tile.
        self.dim_out_all = int(
            0.5 * self.max_detections * (self.max_detections + 1) * self.n_params_per_source
            + 1
            + self.max_detections,
        )
        dim_enc_conv_out = ((self.ptile_slen + 1) // 2 + 1) // 2

        # networks to be trained
        self.enc_conv = EncoderCNN(n_bands, channel, spatial_dropout)
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

        # get indices into the triangular array of returned parameters
        indx_mats = self._get_hidden_indices()
        for k, v in indx_mats.items():
            self.register_buffer(k + "_indx", v, persistent=False)
        assert self.prob_n_source_indx.shape[0] == self.max_detections + 1

    def forward(self, image_ptiles, tile_n_sources):
        raise NotImplementedError(
            "The forward method for ImageEncoder has changed to encode_for_n_sources()"
        )

    def encode(self, image_ptiles):
        # get h matrix.
        # Forward to the layer that is shared by all n_sources.
        log_img = torch.log(image_ptiles - image_ptiles.min() + 1.0)
        var_params = self.enc_conv(log_img)

        # Concatenate all output parameters for all possible n_sources
        return self.enc_final(var_params)

    def sample(self, images, n_samples):
        assert len(images.shape) == 4
        assert images.shape[0] == 1, "Only works for 1 image"
        image_ptiles = get_images_in_tiles(images, self.tile_slen, self.ptile_slen)
        var_params = self.encode(image_ptiles)
        return self._sample(var_params, n_samples)

    def _sample(self, var_params, n_samples):
        log_probs_n_sources_per_tile = self._get_logprob_n_from_var_params(var_params)

        # sample number of sources.
        # tile_n_sources shape = (n_samples x n_ptiles)
        # tile_is_on_array shape = (n_samples x n_ptiles x max_detections x 1)
        probs_n_sources_per_tile = torch.exp(log_probs_n_sources_per_tile)
        tile_n_sources = _sample_class_weights(probs_n_sources_per_tile, n_samples)
        tile_n_sources = tile_n_sources.view(n_samples, -1)
        tile_is_on_array = get_is_on_from_n_sources(tile_n_sources, self.max_detections)
        tile_is_on_array = tile_is_on_array.unsqueeze(-1).float()

        # get var_params conditioned on n_sources
        pred = self._encode_for_n_sources(var_params, tile_n_sources)

        pred["loc_sd"] = torch.exp(0.5 * pred["loc_logvar"])
        pred["log_flux_sd"] = torch.exp(0.5 * pred["log_flux_logvar"])
        tile_locs = self._get_normal_samples(pred["loc_mean"], pred["loc_sd"], tile_is_on_array)
        tile_log_fluxes = self._get_normal_samples(
            pred["log_flux_mean"], pred["log_flux_sd"], tile_is_on_array
        )
        tile_fluxes = tile_log_fluxes.exp() * tile_is_on_array
        return {
            "n_sources": tile_n_sources,
            "locs": tile_locs,
            "log_fluxes": tile_log_fluxes,
            "fluxes": tile_fluxes,
        }

    def encode_for_n_sources(self, image_ptiles, tile_n_sources):
        """Runs encoder on image ptiles."""
        # images shape = (n_ptiles x n_bands x pslen x pslen)
        # tile_n_sources shape = (n_ptiles,) or (n_samples, p_ptiles)
        # Returns: Dictionary of tensors, with dimensions (n_ptiles x ...)

        var_params = self.encode(image_ptiles)
        return self._encode_for_n_sources(var_params, tile_n_sources)

    def _encode_for_n_sources(self, var_params, tile_n_sources):

        tile_n_sources = tile_n_sources.clamp(max=self.max_detections)
        if len(tile_n_sources.shape) == 1:
            tile_n_sources = tile_n_sources.unsqueeze(0)
            squeeze = True
        elif len(tile_n_sources.shape) == 2:
            squeeze = False
        else:
            raise ValueError("tile_n_sources must have shape size 1 or 2")

        assert var_params.shape[0] == tile_n_sources.shape[1]
        # get probability of params except n_sources
        # e.g. loc_mean: shape = (n_samples x n_ptiles x max_detections x len(x,y))
        var_params_for_n_sources = self._get_var_params_for_n_sources(var_params, tile_n_sources)

        # get probability of n_sources
        # n_source_log_probs: shape = (n_ptiles x (max_detections+1))
        n_source_log_probs = self._get_logprob_n_from_var_params(var_params)
        var_params_for_n_sources["n_source_log_probs"] = n_source_log_probs
        if squeeze:
            var_params_for_n_sources = {
                key: value.squeeze(0) for key, value in var_params_for_n_sources.items()
            }
        return var_params_for_n_sources

    def tile_map_n_sources(self, image_ptiles):
        var_params = self.encode(image_ptiles)
        return self._tile_map_n_sources(var_params)

    def _tile_map_n_sources(self, var_params):
        log_probs_n_sources_per_tile = self._get_logprob_n_from_var_params(var_params)
        return torch.argmax(log_probs_n_sources_per_tile, dim=1)

    def tile_map_estimate(self, images):
        # extract image_ptiles
        batch_size = images.shape[0]
        image_ptiles = get_images_in_tiles(images, self.tile_slen, self.ptile_slen)
        n_tiles_per_image = int(image_ptiles.shape[0] / batch_size)

        # MAP (for n_sources) prediction on var params on each tile
        var_params = self.encode(image_ptiles)
        tile_n_sources = self._tile_map_n_sources(var_params)
        pred = self._encode_for_n_sources(var_params, tile_n_sources)

        tile_n_sources = torch.argmax(pred["n_source_log_probs"], dim=1)
        tile_is_on_array = get_is_on_from_n_sources(tile_n_sources, self.max_detections)
        tile_is_on_array = tile_is_on_array.unsqueeze(-1).float()

        # set sd so we return map estimates.
        # first locs
        locs_sd = torch.zeros_like(pred["loc_logvar"])
        tile_locs = self._get_normal_samples(pred["loc_mean"], locs_sd, tile_is_on_array)
        tile_locs = tile_locs.clamp(0, 1)

        # then log_fluxes
        log_flux_mean = pred["log_flux_mean"]
        log_flux_sd = torch.zeros_like(pred["log_flux_logvar"])
        tile_log_fluxes = self._get_normal_samples(log_flux_mean, log_flux_sd, tile_is_on_array)
        tile_fluxes = tile_log_fluxes.exp() * tile_is_on_array

        # finally prob_n_sources
        prob_n_sources = torch.exp(pred["n_source_log_probs"]).reshape(
            batch_size, n_tiles_per_image, 1, self.max_detections + 1
        )

        bshape = (batch_size, n_tiles_per_image, self.max_detections, -1)  # -1 = param_dim
        return {
            "locs": tile_locs.reshape(*bshape),
            "log_fluxes": tile_log_fluxes.reshape(*bshape),
            "fluxes": tile_fluxes.reshape(*bshape),
            "prob_n_sources": prob_n_sources,
            "n_sources": tile_n_sources.reshape(batch_size, -1),
        }

    @property
    def variational_params(self):
        # transform is a function applied directly on NN output.
        return {
            "loc_mean": {"dim": 2, "transform": _loc_mean_func},
            "loc_logvar": {"dim": 2, "transform": _identity_func},
            "log_flux_mean": {"dim": self.n_bands, "transform": _identity_func},
            "log_flux_logvar": {"dim": self.n_bands, "transform": _identity_func},
        }

    # These methods are only used in testing or case studies, Do we need them or can
    # they be moved to the code that they test? ------------------------

    # --------------------------------------------------------------
    def _get_var_params_for_n_sources(self, h, n_sources):
        """Gets variational parameters for n_sources.

        Arguments:
            h: shape = (n_ptiles x dim_out_all)
            n_sources: Tensor with shape (n_samples x n_ptiles)

        Returns:
            loc_mean.shape = (n_sample x n_ptiles x max_detections x len(x,y))
        """
        assert len(n_sources.shape) == 2

        est_params = {}
        for k, param in self.variational_params.items():
            indx_mat = getattr(self, k + "_indx")
            param_dim = param["dim"]
            transform = param["transform"]
            param = self._indx_h_for_n_sources(h, n_sources, indx_mat, param_dim)
            param = transform(param)
            est_params[k] = param

        return est_params

    @staticmethod
    def _get_normal_samples(mean, sd, tile_is_on_array):
        # tile_is_on_array can be either 'tile_is_on_array'/'tile_galaxy_bool'/'tile_star_bool'.
        # return shape = (n_samples x n_ptiles x max_detections x param_dim)
        assert tile_is_on_array.shape[-1] == 1
        return torch.normal(mean, sd) * tile_is_on_array

    def _get_logprob_n_from_var_params(self, h):
        """Obtains log probability of number of n_sources.

        For example, if max_detections = 3, then Tensor will be (n_tiles x 3) since will return
        probability of having 0,1,2 stars.

        Arguments:
            h: Variational parameters

        Returns:
            Log-probability of number of sources.
        """
        free_probs = h[:, self.prob_n_source_indx]
        return self.log_softmax(free_probs)

    def _get_hidden_indices(self):
        """Setup the indices corresponding to entries in h, cached since same for all h."""

        # initialize matrices containing the indices for each variational param.
        indx_mats = {}
        for k, param in self.variational_params.items():
            param_dim = param["dim"]
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
            for k, param in self.variational_params.items():
                param_dim = param["dim"]
                new_indx = (param_dim * n_detections) + curr_indx
                indx_mats[k][n_detections, 0 : (param_dim * n_detections)] = torch.arange(
                    curr_indx, new_indx
                )
                curr_indx = new_indx

        # assigned indices that were not used to `prob_n_source`
        indx_mats["prob_n_source"] = torch.arange(curr_indx, self.dim_out_all)

        return indx_mats

    def _indx_h_for_n_sources(self, h, n_sources, indx_mat, param_dim):
        """Obtains variational parameters for n_sources.

        Indexes into all possible combinations of variational parameters (h) to obtain actually
        variational parameters for n_sources.

        Arguments:
            h: shape = (n_ptiles x dim_out_all)
            n_sources: (n_samples x n_tiles)
            indx_mat: TODO (to be documented)
            param_dim: the dimension of the parameter you are indexing h.

        Returns:
            var_param: shape = (n_samples x n_ptiles x max_detections x dim_per_source)
        """
        assert len(n_sources.shape) == 2
        assert h.size(0) == n_sources.size(1)
        assert h.size(1) == self.dim_out_all
        n_ptiles = h.size(0)
        h = torch.cat((h, torch.zeros(n_ptiles, 1, device=h.device)), dim=1)

        # select the indices from _h indicated by indx_mat.
        indices = indx_mat[n_sources.transpose(0, 1)].reshape(n_ptiles, -1)
        var_param = torch.gather(h, 1, indices)

        # np: n_ptiles, ns: n_samples
        return rearrange(
            var_param,
            "np (ns d pd) -> ns np d pd",
            np=n_ptiles,
            ns=n_sources.size(0),
            d=self.max_detections,
            pd=param_dim,
        )


class EncoderCNN(nn.Module):
    def __init__(self, n_bands, channel, dropout):
        super().__init__()
        self.layer = self._make_layer(n_bands, channel, dropout)

    def forward(self, x):
        """Runs encoder CNN on inputs."""
        return self.layer(x)

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


class ConvBlock(nn.Module):
    """A Convolution Layer.

    This module is two stacks of Conv2D -> ReLU -> BatchNorm, with dropout
    in the middle, and an option to downsample with a stride of 2.

    Parameters:
        in_channel: Number of input channels
        out_channel: Number of output channels
        dropout: Dropout proportion between [0, 1]
        downsample (optional): Whether to downsample with stride of 2.
    """

    def __init__(self, in_channel: int, out_channel: int, dropout: float, downsample: bool = False):
        """Initializes the module layers."""
        super().__init__()
        self.downsample = downsample
        stride = 1
        if self.downsample:
            stride = 2
            self.sc_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)
            self.sc_bn = nn.BatchNorm2d(out_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.drop1 = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        """Runs convolutional block on inputs."""
        identity = x

        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            identity = self.sc_bn(self.sc_conv(identity))

        out = x + identity
        return F.relu(out)


def _argfront(is_on_array, dim):
    # return indices that sort pushing all zeroes of tensor to the back.
    # dim is dimension along which do the ordering.
    assert len(is_on_array.shape) == 2
    return (is_on_array != 0).long().argsort(dim=dim, descending=True)


def _sample_class_weights(class_weights, n_samples=1):
    """Draw a sample from Categorical variable with probabilities class_weights."""
    cat_rv = categorical.Categorical(probs=class_weights)
    return cat_rv.sample((n_samples,)).squeeze()


def _loc_mean_func(x):
    return torch.sigmoid(x) * (x != 0).float()


def _identity_func(x):
    return x
