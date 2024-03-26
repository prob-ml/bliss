import torch
from einops import pack, rearrange, reduce, unpack
from torch import Tensor, nn
from torch.distributions import Categorical, Normal
from torch.nn import BCELoss

from bliss.catalog import TileCatalog
from bliss.encoders.layers import ConcatBackgroundTransform, EncoderCNN, make_enc_final
from bliss.render_tiles import get_images_in_tiles, get_n_padded_tiles_hw


class DetectionEncoder(nn.Module):
    """Encodes the distribution of a latent variable representing an astronomical image.

    This class implements the source encoder, which is supposed to take in
    an astronomical image of size slen * slen and returns a NN latent variable
    representation of this image.
    """

    def __init__(
        self,
        input_transform: ConcatBackgroundTransform,
        n_bands: int = 1,
        tile_slen: int = 4,
        ptile_slen: int = 52,
        channel: int = 8,
        hidden: int = 128,
        dropout: float = 0,
        spatial_dropout: float = 0,
    ):
        """Initializes DetectionEncoder.

        Args:
            input_transform: Class which determines how input image and bg are transformed.
            n_bands: number of bands
            tile_slen: dimension of full image, we assume its square for now
            ptile_slen: dimension (in pixels) of the individual
                            image padded tiles (usually 8 for stars, and _ for galaxies).
            channel: TODO (document this)
            spatial_dropout: TODO (document this)
            dropout: TODO (document this)
            hidden: TODO (document this)
        """
        assert n_bands == 1, "Only 1 band is supported"
        super().__init__()

        self.input_transform = input_transform
        self.n_bands = n_bands

        assert tile_slen <= ptile_slen
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen

        assert (ptile_slen - tile_slen) % 2 == 0
        self.bp = (ptile_slen - tile_slen) // 2

        # Number of distributional parameters used to characterize each source in an image.
        # 2 for location mean, 2 for location sigma (xy), 1 for for probability of counts.
        self._dim_out_all = 5

        dim_enc_conv_out = ((self.ptile_slen + 1) // 2 + 1) // 2

        # networks to be trained
        n_bands_in = self.input_transform.output_channels(n_bands)
        self._enc_conv = EncoderCNN(n_bands_in, channel, spatial_dropout)
        self._enc_final = make_enc_final(
            channel * 4 * dim_enc_conv_out**2,
            hidden,
            self._dim_out_all,
            dropout,
        )

    def forward(self, images: Tensor, background: Tensor):

        # prepare padded tiles with background
        images_with_background, _ = pack([images, background], "b * h w")
        image_ptiles = get_images_in_tiles(
            images_with_background,
            self.tile_slen,
            self.ptile_slen,
        )
        flat_image_ptiles = rearrange(image_ptiles, "n nth ntw c h w -> (n nth ntw) c h w")

        # encode
        transformed_ptiles = self.input_transform(flat_image_ptiles)
        enc_conv_output = self._enc_conv(transformed_ptiles)
        enc_final_output = self._enc_final(enc_conv_output)

        # split NN output
        locs_mean_raw, locs_logvar_raw, n_source_free_probs = unpack(
            enc_final_output, [(2,), (2,), ()], "np *"
        )

        # final transformation from NN output
        n_source_probs = torch.sigmoid(n_source_free_probs).clamp(1e-4, 1 - 1e-4)
        locs_mean = _locs_mean_func(locs_mean_raw)
        locs_sd = _locs_sd_func(locs_logvar_raw)
        return n_source_probs, locs_mean, locs_sd

    def variational_mode(self, images: Tensor, background: Tensor) -> TileCatalog:
        """Compute the variational mode."""
        _, _, h, w = images.shape
        nth, ntw = get_n_padded_tiles_hw(h, w, self.ptile_slen, self.tile_slen)
        n_source_probs, locs_mean, _ = self.forward(images, background)
        flat_tile_n_sources = n_source_probs.ge(0.5).long()
        flat_tile_locs = locs_mean * rearrange(flat_tile_n_sources, "np -> np 1")

        return TileCatalog.from_flat_dict(
            self.tile_slen, nth, ntw, {"n_sources": flat_tile_n_sources, "locs": flat_tile_locs}
        )

    def sample(self, images: Tensor, background: Tensor, n_samples: int = 1) -> dict[str, Tensor]:
        """Sample from the encoded variational distribution.

        Args:
            images:
                Tensor of images to encode.

            background:
                Tensor consisting of images background

            n_samples:
                The number of samples to draw.

        Returns:
            A dictionary of tensors with shape `n_samples * n_ptiles * ...`.
            Consists of "n_sources" and "locs".
        """
        n_source_probs, locs_mean, locs_sd = self.forward(images, background)

        # sample counts per tile
        tile_n_sources = Categorical(probs=n_source_probs).sample((n_samples,))
        assert tile_n_sources.ndim == 2

        # sample locations and zero out out empty sources
        raw_tile_locs = Normal(locs_mean, locs_sd).sample((n_samples,))
        assert raw_tile_locs.ndim == 3
        tile_locs = raw_tile_locs * rearrange(tile_n_sources, "ns np -> ns np 1")

        assert tile_n_sources.shape[0] == tile_locs.shape[0] == n_samples

        return {"n_sources": tile_n_sources, "locs": tile_locs}

    def get_loss(self, image: Tensor, background: Tensor, true_catalog: TileCatalog):
        assert image.device == background.device == true_catalog.device

        # encode
        n_source_probs, locs_mean, locs_sd = self.forward(image, background)

        # loss from detection count encoding
        n_true_sources_flat = rearrange(true_catalog.n_sources, "b nth ntw -> (b nth ntw)")
        counter_loss = BCELoss(reduction="none")(n_source_probs, n_true_sources_flat.float())

        # now for locations
        flat_true_locs = rearrange(true_catalog.locs, "b nth ntw xy -> (b nth ntw) xy", xy=2)
        locs_log_prob = reduce(
            Normal(locs_mean, locs_sd).log_prob(flat_true_locs), "np xy -> np", "sum", xy=2
        )
        locs_loss = locs_log_prob * n_true_sources_flat.float()

        loss_vec = locs_loss * (locs_loss.detach() < 1e6).float() + counter_loss  # noqa: WPS459
        loss = loss_vec.mean()

        return {
            "loss": loss,
            "counter_loss": counter_loss.detach().mean().item(),
            "locs_loss": locs_loss.detach().mean().item(),
        }


def _locs_mean_func(x: Tensor) -> Tensor:
    # I don't think the special case for `x == 0` should be necessary
    return torch.sigmoid(x) * (x != 0).float()


def _locs_sd_func(x: Tensor) -> Tensor:
    return (x.exp() + 1e-5).sqrt()
