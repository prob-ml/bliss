import itertools
import math
from typing import Dict, Optional, Union

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

from bliss.catalog import get_is_on_from_n_sources
from bliss.encoders.layers import (
    ConcatBackgroundTransform,
    EncoderCNN,
    LogBackgroundTransform,
    make_enc_final,
)
from bliss.render_tiles import get_images_in_tiles


class OldDetectionEncoder(nn.Module):
    """Encodes the distribution of a latent variable representing an astronomical image.

    This class implements the source encoder, which is supposed to take in
    an astronomical image of size slen * slen and returns a NN latent variable
    representation of this image.
    """

    def __init__(
        self,
        input_transform: Union[LogBackgroundTransform, ConcatBackgroundTransform],
        max_detections: int = 1,
        n_bands: int = 1,
        tile_slen: int = 4,
        ptile_slen: int = 52,
        channel: int = 8,
        hidden: int = 128,
        dropout: float = 0,
        spatial_dropout: float = 0,
        device=torch.device("cpu"),
    ):
        """Initializes DetectionEncoder.

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

        assert tile_slen <= ptile_slen
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen

        assert (ptile_slen - tile_slen) % 2 == 0
        self.border_padding = (ptile_slen - tile_slen) // 2

        # Number of distributional parameters used to characterize each source in an image.
        self.n_params_per_source = sum(param["dim"] for param in self.dist_param_groups.values())

        # the number of total detections for all source counts: 1 + 2 + ... + self.max_detections
        # NOTE: the numerator here is always even
        self.n_total_detections = self.max_detections * (self.max_detections + 1) // 2

        # most of our parameters describe individual detections
        n_source_params = self.n_total_detections * self.n_params_per_source

        # we also have parameters indicating the distribution of the number of detections
        count_simplex_dim = 1 + self.max_detections

        # the total number of distributional parameters per tile
        self.dim_out_all = n_source_params + count_simplex_dim

        dim_enc_conv_out = ((self.ptile_slen + 1) // 2 + 1) // 2

        # networks to be trained
        n_bands_in = self.input_transform.output_channels(n_bands)
        self.enc_conv = EncoderCNN(n_bands_in, channel, spatial_dropout)
        self.enc_final = make_enc_final(
            channel * 4 * dim_enc_conv_out**2,
            hidden,
            self.dim_out_all,
            dropout,
        )
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.device = device

        # the next block of code constructs `self.n_detections_map`, which is a 2d tensor with
        # size (self.max_detections + 1, self.max_detections).
        # There is one row for each possible number of detections (including zero).
        # Each row contains the indices of the relevant detections, padded by a dummy value.
        md, ntd = self.max_detections, self.n_total_detections
        n_detections_map = torch.full((md + 1, md), ntd, device=self.device)  # type: ignore
        tri = torch.tril_indices(md, md, device=self.device)  # type: ignore
        n_detections_map[tri[0] + 1, tri[1]] = torch.arange(ntd, device=self.device)  # type: ignore
        self.register_buffer("n_detections_map", n_detections_map)

    def _final_encoding(self, enc_final_output):
        dim_out_all = enc_final_output.shape[1]
        dim_per_source_params = dim_out_all - (self.max_detections + 1)
        per_source_params, n_source_free_probs = torch.split(
            enc_final_output, [dim_per_source_params, self.max_detections + 1], dim=1
        )
        per_source_params = rearrange(
            per_source_params,
            "n_ptiles (td pps) -> n_ptiles td pps",
            td=self.n_total_detections,
            pps=self.n_params_per_source,
        )

        n_source_log_probs = self.log_softmax(n_source_free_probs)

        return {"per_source_params": per_source_params, "n_source_log_probs": n_source_log_probs}

    def do_encode_batch(self, images_with_background):
        image_ptiles = get_images_in_tiles(
            images_with_background,
            self.tile_slen,
            self.ptile_slen,
        )
        image_ptiles = rearrange(image_ptiles, "n nth ntw b h w -> (n nth ntw) b h w")
        transformed_ptiles = self.input_transform(image_ptiles)
        enc_conv_output = self.enc_conv(transformed_ptiles)
        return self.enc_final(enc_conv_output)

    def forward(self, images, background):
        images_with_background = torch.cat((images, background), dim=1)
        enc_final_output = self.do_encode_batch(images_with_background)
        return self._final_encoding(enc_final_output)

    def sample(
        self,
        dist_params: Dict[str, Tensor],
        n_samples: Union[int, None],
        n_source_weights: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Sample from the encoded variational distribution.

        Args:
            dist_params:
                The distributional parameters in matrix form.
            n_samples:
                The number of samples to draw. If None, the variational mode is taken instead.
            n_source_weights:
                If specified, adjusts the sampling probabilities of n_sources.

        Returns:
            A dictionary of tensors with shape `n_samples * n_ptiles * max_sources * ...`.
            Consists of `"n_sources", "locs", "star_log_fluxes", and "star_fluxes"`.
        """
        if n_source_weights is None:
            max_n_weights = self.max_detections + 1
            n_source_weights = torch.ones(max_n_weights, device=self.device)  # type: ignore
        n_source_weights = n_source_weights.reshape(1, -1)
        ns_log_probs_adj = dist_params["n_source_log_probs"] + n_source_weights.log()
        ns_log_probs_adj -= ns_log_probs_adj.logsumexp(dim=-1, keepdim=True)

        if n_samples is not None:
            n_source_probs = ns_log_probs_adj.exp()
            tile_n_sources = Categorical(probs=n_source_probs).sample((n_samples,))
        else:
            tile_n_sources = torch.argmax(ns_log_probs_adj, dim=-1).unsqueeze(0)
        # get distributional parameters conditioned on the sampled numbers of light sources
        dist_params_n_src = self.encode_for_n_sources(
            dist_params["per_source_params"], tile_n_sources
        )

        tile_is_on_array = get_is_on_from_n_sources(tile_n_sources, self.max_detections)
        tile_is_on_array = tile_is_on_array.unsqueeze(-1)

        if n_samples is not None:
            tile_locs = Normal(dist_params_n_src["loc_mean"], dist_params_n_src["loc_sd"]).rsample()
            tile_log_fluxes = Normal(
                dist_params_n_src["log_flux_mean"], dist_params_n_src["log_flux_sd"]
            ).rsample()
        else:
            tile_locs = dist_params_n_src["loc_mean"]
            tile_log_fluxes = dist_params_n_src["log_flux_mean"]
        tile_locs *= tile_is_on_array  # Is masking here helpful/necessary?
        tile_fluxes = tile_log_fluxes.exp()
        tile_fluxes *= tile_is_on_array

        return {
            "locs": tile_locs,
            "star_log_fluxes": tile_log_fluxes,
            "star_fluxes": tile_fluxes,
            "n_sources": tile_n_sources,
        }

    def variational_mode(
        self, dist_params: Dict[str, Tensor], n_source_weights: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Compute the variational mode. Special case of sample() where first dim is squeezed."""
        detection_params = self.sample(dist_params, None, n_source_weights=n_source_weights)
        return {k: v.squeeze(0) for k, v in detection_params.items()}

    @staticmethod
    def _loc_mean_func(x):
        # I don't think the special case for `x == 0` should be necessary
        return torch.sigmoid(x) * (x != 0).float()

    def encode_for_n_sources(
        self, params_per_source: Tensor, tile_n_sources: Tensor
    ) -> Dict[str, Tensor]:
        """Get distributional parameters conditioned on number of sources in tile.

        Args:
            params_per_source:
                Has size `(batch_size x n_tiles_h x n_tiles_w) * d`.
            tile_n_sources:
                A tensor of the number of sources in each tile.

        Returns:
            A dictionary where each member has shape
            `n_samples x n_ptiles x max_detections x ...`
        """
        assert tile_n_sources.max() <= self.max_detections

        # first, we transform `tile_n_sources` so that it can be used as an index
        # for looking up detections in `params_per_source`
        sindx1 = self.n_detections_map[tile_n_sources]  # type: ignore
        sindx2 = rearrange(sindx1, "ns np md -> np (ns md) 1")
        sindx3 = sindx2.expand(sindx2.size(0), sindx2.size(1), self.n_params_per_source)

        # next, we pad `params_per_source` with a dummy column of zeros that will be looked up
        # (copied) whenever fewer the `max_detections` sources are present. `gather` does the copy.
        pps_padded = F.pad(params_per_source, (0, 0, 0, 1))
        pps_gathered = torch.gather(pps_padded, 1, sindx3)
        params_n_srcs_combined = rearrange(
            pps_gathered, "np (ns md) pps -> ns np md pps", ns=tile_n_sources.size(0)
        )

        # finally, we slice pps5 by parameter group because these groups are treated differently,
        # subsequently
        split_sizes = [v["dim"] for v in self.dist_param_groups.values()]
        dist_params_split = torch.split(params_n_srcs_combined, split_sizes, 3)
        names = self.dist_param_groups.keys()
        params_n_srcs = dict(zip(names, dist_params_split))

        params_n_srcs["loc_mean"] = self._loc_mean_func(params_n_srcs["loc_mean"])
        params_n_srcs["loc_sd"] = (params_n_srcs["loc_logvar"].exp() + 1e-5).sqrt()

        # delete these so we don't accidentally use them
        del params_n_srcs["loc_logvar"]

        return params_n_srcs

    def get_loss(self, batch: Dict[str, Tensor]):
        images, background = batch.pop("images"), batch.pop("background")
        dist_params = self.forward(images, background)

        nslp_flat = rearrange(dist_params["n_source_log_probs"], "n_ptiles ns -> n_ptiles ns")
        truth_flat = batch["n_sources"].reshape(-1)
        counter_loss = F.nll_loss(nslp_flat, truth_flat, reduction="none")

        pred = self.encode_for_n_sources(
            dist_params["per_source_params"],
            rearrange(batch["n_sources"], "b ht wt -> 1 (b ht wt)"),
        )
        locs_log_probs_all = _get_params_logprob_all_combs(
            rearrange(batch["locs"], "b ht wt xy -> (b ht wt) 1 xy", xy=2),
            pred["loc_mean"].squeeze(0),
            pred["loc_sd"].squeeze(0),
        )

        locs_loss = _get_min_perm_loss(
            locs_log_probs_all,
            rearrange(batch["n_sources"], "b ht wt -> (b ht wt) 1").float(),
        )

        loss_vec = locs_loss * (locs_loss.detach() < 1e6).float() + counter_loss
        loss = loss_vec.mean()

        return {
            "loss": loss,
            "counter_loss": counter_loss.mean().detach().item(),
            "locs_loss": locs_loss.mean().detach().item(),
        }

    @property
    def dist_param_groups(self):
        return {
            "loc_mean": {"dim": 2},
            "loc_logvar": {"dim": 2},
        }


def _get_log_probs_all_perms(
    locs_log_probs_all,
    is_on_array,
):
    # get log-probability under every possible matching of estimated source to true source
    n_ptiles = locs_log_probs_all.size(0)
    max_detections = 1

    n_permutations = math.factorial(max_detections)
    locs_log_probs_all_perm = torch.zeros(
        n_ptiles, n_permutations, device=locs_log_probs_all.device
    )

    for i, perm in enumerate(itertools.permutations(range(max_detections))):
        # note that we multiply is_on_array, we only evaluate the loss if the source is on.
        locs_log_probs_all_perm[:, i] = (
            locs_log_probs_all[:, perm].diagonal(dim1=1, dim2=2) * is_on_array
        ).sum(1)

    return locs_log_probs_all_perm


def _get_min_perm_loss(
    locs_log_probs_all,
    is_on_array,
):
    # get log-probability under every possible matching of estimated star to true star
    locs_log_probs_all_perm = _get_log_probs_all_perms(
        locs_log_probs_all,
        is_on_array,
    )

    # find the permutation that minimizes the location losses
    locs_loss, _ = torch.min(-locs_log_probs_all_perm, dim=1)

    return locs_loss


def _get_params_logprob_all_combs(true_params, param_mean, param_sd):
    # return shape (n_ptiles x max_detections x max_detections)
    assert true_params.shape == param_mean.shape == param_sd.shape

    n_ptiles = true_params.size(0)
    max_detections = true_params.size(1)

    # view to evaluate all combinations of log_prob.
    true_params = true_params.view(n_ptiles, 1, max_detections, -1)
    param_mean = param_mean.view(n_ptiles, max_detections, 1, -1)
    param_sd = param_sd.view(n_ptiles, max_detections, 1, -1)

    return Normal(param_mean, param_sd).log_prob(true_params).sum(dim=3)
