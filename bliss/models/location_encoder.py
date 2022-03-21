from typing import Dict, Union

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.distributions import categorical
from torch.nn import functional as F

from bliss.catalog import TileCatalog, get_images_in_tiles, get_is_on_from_n_sources


class LogBackgroundTransform:
    def __init__(self, z_threshold: float = 4.0) -> None:
        self.z_threshold = z_threshold

    def __call__(self, image: Tensor, background: Tensor) -> Tensor:
        return torch.log1p(
            F.relu(image - background + self.z_threshold * background.sqrt(), inplace=True)
        )

    def output_channels(self, input_channels: int) -> int:
        return input_channels


class ConcatBackgroundTransform:
    def __init__(self):
        pass

    def __call__(self, image: Tensor, background: Tensor) -> Tensor:
        return torch.cat((image, background), dim=1)

    def output_channels(self, input_channels: int) -> int:
        return 2 * input_channels


class LocationEncoder(nn.Module):
    """Encodes the distribution of a latent variable representing an astronomical image.

    This class implements the source encoder, which is supposed to take in
    an astronomical image of size slen * slen and returns a NN latent variable
    representation of this image.
    """

    def __init__(
        self,
        input_transform: Union[LogBackgroundTransform, ConcatBackgroundTransform],
        max_detections: int,
        n_bands: int,
        tile_slen: int,
        ptile_slen: int,
        channel: int,
        dropout,
        hidden: int,
        spatial_dropout: float,
    ):
        """Initializes LocationEncoder.

        Args:
            input_transform: Class which determines how input image and bg are transformed.
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

        self.input_transform = input_transform

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
        n_bands_in = self.input_transform.output_channels(n_bands)
        self.enc_conv = EncoderCNN(n_bands_in, channel, spatial_dropout)
        self.enc_final = make_enc_final(
            channel * 4 * dim_enc_conv_out ** 2,
            hidden,
            self.dim_out_all,
            dropout,
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

        # get indices into the triangular array of returned parameters
        indx_mats = self._get_hidden_indices()
        for k, v in indx_mats.items():
            self.register_buffer(k + "_indx", v, persistent=False)
        assert self.prob_n_source_indx.shape[0] == self.max_detections + 1

    def forward(self, image_ptiles, tile_n_sources):
        raise NotImplementedError("The forward method has changed to encode_for_n_sources()")

    def _get_images_in_ptiles(self, image: Tensor, background: Tensor) -> Tensor:
        return get_images_in_tiles(
            self.input_transform(image, background), self.tile_slen, self.ptile_slen
        )

    def encode(self, image: Tensor, background: Tensor) -> Tensor:
        """Encodes variational parameters from image padded tiles.

        Args:
            image: An astronomical image with shape `b * n_bands * h * w`.
            background: Background for `image` with the same shape as `image`.

        Returns:
            A tensor of variational parameters in matrix form per-tile
            (`n_ptiles * D`), where `D` is the total flattened dimension
            of all variational parameters. This matrix is used as input to
            other methods of this class (typically named `var_params`).
        """
        # get h matrix.
        # Forward to the layer that is shared by all n_sources.
        image_ptiles = self._get_images_in_ptiles(image, background)
        log_image_ptiles_flat = rearrange(image_ptiles, "b nth ntw c h w -> (b nth ntw) c h w")
        var_params_conv = self.enc_conv(log_image_ptiles_flat)
        var_params_flat = self.enc_final(var_params_conv)
        return rearrange(
            var_params_flat,
            "(b nth ntw) d -> b nth ntw d",
            b=image_ptiles.shape[0],
            nth=image_ptiles.shape[1],
            ntw=image_ptiles.shape[2],
        )

    def sample(self, var_params: Tensor, n_samples: int) -> Dict[str, Tensor]:
        """Sample from encoded variational distribution.

        Args:
            var_params: The output of `self.encode(ptiles)` which is the variational parameters
                in matrix form. Has size `n_ptiles * n_bands`.
            n_samples:
                The number of samples to draw

        Returns:
            A dictionary of tensors with shape `n_samples * n_ptiles * max_sources* ...`.
            Consists of `"n_sources", "locs", "log_fluxes", and "fluxes"`.
        """
        var_params_flat = rearrange(var_params, "b nth ntw d -> (b nth ntw) d")
        log_probs_n_sources_per_tile = self._get_logprob_n_from_var_params(var_params_flat)

        # sample number of sources.
        # tile_n_sources shape = (n_samples x n_ptiles)
        # tile_is_on_array shape = (n_samples x n_ptiles x max_detections x 1)
        probs_n_sources_per_tile = torch.exp(log_probs_n_sources_per_tile)
        tile_n_sources = _sample_class_weights(probs_n_sources_per_tile, n_samples)
        tile_n_sources = tile_n_sources.view(n_samples, -1)
        tile_is_on_array = get_is_on_from_n_sources(tile_n_sources, self.max_detections)
        tile_is_on_array = tile_is_on_array.unsqueeze(-1).float()

        # get var_params conditioned on n_sources
        pred = self.encode_for_n_sources(var_params_flat, tile_n_sources)

        pred["loc_sd"] = torch.exp(0.5 * pred["loc_logvar"])
        pred["log_flux_sd"] = torch.exp(0.5 * pred["log_flux_logvar"])
        tile_locs = self._get_normal_samples(pred["loc_mean"], pred["loc_sd"], tile_is_on_array)
        tile_log_fluxes = self._get_normal_samples(
            pred["log_flux_mean"], pred["log_flux_sd"], tile_is_on_array
        )
        tile_fluxes = tile_log_fluxes.exp() * tile_is_on_array
        sample_flat = {
            "n_sources": tile_n_sources,
            "locs": tile_locs,
            "log_fluxes": tile_log_fluxes,
            "fluxes": tile_fluxes,
        }
        sample = {}
        for k, v in sample_flat.items():
            if k == "n_sources":
                pattern = "ns (b nth ntw) -> ns b nth ntw"
            else:
                pattern = "ns (b nth ntw) s k -> ns b nth ntw s k"
            sample[k] = rearrange(
                v,
                pattern,
                b=var_params.shape[0],
                nth=var_params.shape[1],
                ntw=var_params.shape[2],
            )

        return sample

    def max_a_post(self, var_params: Tensor) -> TileCatalog:
        """Derive maximum a posteriori from variational parameters.

        Args:
            var_params: The output of `self.encode(ptiles)` which is the variational parameters
                in matrix form. Has size `n_ptiles * n_bands`.

        Returns:
            The maximum a posteriori for each padded tile.
            Has shape `n_ptiles * max_detections * ...`.
            The dictionary contains
            `"locs", "log_fluxes", "fluxes", and "n_sources".`.
        """
        var_params_flat = rearrange(var_params, "b nth ntw d -> (b nth ntw) d")
        tile_n_sources = self.tile_map_n_sources(var_params_flat)
        pred = self.encode_for_n_sources(var_params_flat, tile_n_sources)

        tile_n_sources = torch.argmax(pred["n_source_log_probs"], dim=1)
        tile_is_on_array = get_is_on_from_n_sources(tile_n_sources, self.max_detections)
        tile_is_on_array = tile_is_on_array.unsqueeze(-1).float()
        tile_n_sources_log_prob, _ = torch.max(pred["n_source_log_probs"], dim=1)

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

        max_a_post_flat = {
            "locs": tile_locs,
            "log_fluxes": tile_log_fluxes,
            "fluxes": tile_fluxes,
            "n_sources": tile_n_sources,
            "n_sources_log_prob": tile_n_sources_log_prob,
        }
        max_a_post = {}
        for k, v in max_a_post_flat.items():
            if k in {"n_sources", "n_sources_log_prob"}:
                pattern = "(b nth ntw) -> b nth ntw"
            else:
                pattern = "(b nth ntw) s k -> b nth ntw s k"
            max_a_post[k] = rearrange(
                v, pattern, b=var_params.shape[0], nth=var_params.shape[1], ntw=var_params.shape[2]
            )
        return TileCatalog(self.tile_slen, max_a_post)

    def encode_for_n_sources(self, var_params_flat, tile_n_sources):
        """Get variational parameters conditioned on number of sources in tile.

        Args:
            var_params_flat: The output of `self.encode(ptiles)`,
                where the first three dimensions have been flattened.
                These are the variational parameters in matrix form.
                Has size `(batch_size x n_tiles_h x n_tiles_w) * d`.
            tile_n_sources:
                A tensor of the number of sources in each tile.

        Raises:
            ValueError: If the shape of tile_n_sources is not 1 or 2.

        Returns:
            A dictionary where each member has either shape
            `n_samples x n_ptiles x max_detections x ...`
            or `n_ptiles x max_detections x ...` depending on the shape of `tile_n_sources`.
        """

        tile_n_sources = tile_n_sources.clamp(max=self.max_detections)
        if len(tile_n_sources.shape) == 1:
            tile_n_sources = tile_n_sources.unsqueeze(0)
            squeeze = True
        elif len(tile_n_sources.shape) == 2:
            squeeze = False
        else:
            raise ValueError("tile_n_sources must have shape size 1 or 2")

        assert var_params_flat.shape[0] == tile_n_sources.shape[1]
        # get probability of params except n_sources
        # e.g. loc_mean: shape = (n_samples x n_ptiles x max_detections x len(x,y))
        var_params_for_n_sources = self._get_var_params_for_n_sources(
            var_params_flat, tile_n_sources
        )

        # get probability of n_sources
        # n_source_log_probs: shape = (n_ptiles x (max_detections+1))
        n_source_log_probs = self._get_logprob_n_from_var_params(var_params_flat)
        var_params_for_n_sources["n_source_log_probs"] = n_source_log_probs
        if squeeze:
            var_params_for_n_sources = {
                key: value.squeeze(0) for key, value in var_params_for_n_sources.items()
            }
        return var_params_for_n_sources

    def tile_map_n_sources(self, var_params):
        """Get the maximum a posteriori of the number of soruces in each tile.

        Args:
            var_params: The output of `self.encode(ptiles)` which is the variational parameters
                in matrix form. Has size `n_ptiles * n_bands`.

        Returns:
            A tensor of shape `n_ptiles` with the maximum number of sources in each tile.
        """
        log_probs_n_sources_per_tile = self._get_logprob_n_from_var_params(var_params)
        return torch.argmax(log_probs_n_sources_per_tile, dim=1)

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
        # tile_is_on_array can be either 'tile_is_on_array'/'tile_galaxy_bools'/'tile_star_bools'.
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


def make_enc_final(in_size, hidden, out_size, dropout):
    return nn.Sequential(
        nn.Flatten(1),
        nn.Linear(in_size, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(True),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_size),
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
                ConvBlock(channel, channel, dropout),
                ConvBlock(channel, channel, dropout),
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
        x = F.relu(self.bn1(x), inplace=True)

        x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            identity = self.sc_bn(self.sc_conv(identity))

        out = x + identity
        return F.relu(out, inplace=True)


def _sample_class_weights(class_weights, n_samples=1):
    """Draw a sample from Categorical variable with probabilities class_weights."""
    cat_rv = categorical.Categorical(probs=class_weights)
    return cat_rv.sample((n_samples,)).squeeze()


def _loc_mean_func(x):
    return torch.sigmoid(x) * (x != 0).float()


def _identity_func(x):
    return x
