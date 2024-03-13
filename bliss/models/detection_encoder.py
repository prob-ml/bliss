import pytorch_lightning as pl
import torch
from einops import pack, rearrange, unpack
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor, nn
from torch.distributions import Categorical, Normal
from torch.nn.functional import nll_loss

from bliss.catalog import TileCatalog
from bliss.models.encoder_layers import ConcatBackgroundTransform, EncoderCNN, make_enc_final
from bliss.render_tiles import get_images_in_tiles, get_n_padded_tiles_hw


class DetectionEncoder(pl.LightningModule):
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
        super().__init__()

        self.input_transform = input_transform
        self.n_bands = n_bands

        assert tile_slen <= ptile_slen
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen

        assert (ptile_slen - tile_slen) % 2 == 0
        self.bp = (ptile_slen - tile_slen) // 2

        # Number of distributional parameters used to characterize each source in an image.
        # 2 for location mean and 2 for location sigma (xy)
        self.n_params_per_source = 4

        # the total number of distributional parameters per tile
        # the extra 2 is for probability of coutns, "on" and "off" (need both to softmax)
        self.dim_out_all = self.n_params_per_source + 2

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

    def encode(self, images: Tensor, background: Tensor):

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
        enc_conv_output = self.enc_conv(transformed_ptiles)
        enc_final_output = self.enc_final(enc_conv_output)

        # split NN output
        locs_mean_raw, locs_logvar, n_source_free_probs = unpack(
            enc_final_output, [(2,), (2,), (2,)], "np *"
        )

        # final transformation from NN output
        n_source_log_probs = self.log_softmax(n_source_free_probs)
        locs_mean = _loc_mean_func(locs_mean_raw)
        locs_sd = (locs_logvar.exp() + 1e-5).sqrt()
        return n_source_log_probs, locs_mean, locs_sd

    def _get_loss(self, image: Tensor, background: Tensor, true_catalog: TileCatalog):
        assert true_catalog.n_sources.max() <= 1

        # encode
        n_source_log_probs, loc_mean, loc_sd = self.encode(image, background)

        # loss from detection count encoding
        nlsp = rearrange(n_source_log_probs, "np C -> np C", C=2)
        n_true_sources_flat = rearrange(true_catalog.n_sources, "b nth ntw -> (b nth ntw)")
        counter_loss = nll_loss(nlsp, n_true_sources_flat, reduction="none")

        # now for locations
        flat_true_locs = rearrange(true_catalog.locs, "b nth ntw xy -> (b nth ntw) xy", xy=2)
        locs_loss = Normal(loc_mean, loc_sd).log_prob(flat_true_locs) * n_true_sources_flat

        loss_vec = locs_loss * (locs_loss.detach() < 1e6).float() + counter_loss  # noqa: WPS459
        loss = loss_vec.mean()

        return {
            "loss": loss,
            "counter_loss": counter_loss,
            "locs_loss": locs_loss,
        }

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
            Consists of `"n_sources" and "locs".
        """
        n_source_log_probs, locs_mean, locs_sd = self.encode(images, background)

        # sample counts per tile
        n_source_probs = n_source_log_probs.exp()
        tile_n_sources = Categorical(probs=n_source_probs).sample((n_samples,))
        assert tile_n_sources.ndim == 2

        # sample locations and zero out out empty sources
        raw_tile_locs = Normal(locs_mean, locs_sd).sample((n_samples,))
        assert raw_tile_locs.ndim == 3
        tile_locs = raw_tile_locs * rearrange(tile_n_sources, "ns np -> ns np 1")

        assert tile_n_sources.shape[0] == tile_locs.shape[0] == n_samples

        return {"n_sources": tile_n_sources, "locs": tile_locs}

    def variational_mode(self, images: Tensor, background: Tensor) -> TileCatalog:
        """Compute the variational mode."""
        _, _, h, w = images.shape
        nth, ntw = get_n_padded_tiles_hw(h, w, self.ptile_slen, self.tile_slen)
        n_source_log_probs, locs_mean, _ = self.encode(images, background)
        flat_tile_n_sources = torch.argmax(n_source_log_probs, dim=-1)
        tile_locs = locs_mean * rearrange(flat_tile_n_sources, "np -> np 1")

        return TileCatalog.from_flat_dict(
            self.tile_slen, nth, ntw, {"n_sources": flat_tile_n_sources, "locs": tile_locs}
        )


def _loc_mean_func(x):
    # I don't think the special case for `x == 0` should be necessary
    return torch.sigmoid(x) * (x != 0).float()


def plot_image_detections(images, true_cat, est_cat, bp: int, nrows, img_ids):
    # setup figure and axes.
    fig, axes = plt.subplots(nrows=nrows, ncols=nrows, figsize=(20, 20))
    axes = axes.flatten() if nrows > 1 else [axes]

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(img_ids):
            break

        img_id = img_ids[ax_idx]
        true_n_sources = true_cat.n_sources[img_id].item()
        n_sources = est_cat.n_sources[img_id].item()
        ax.set_xlabel(f"True num: {true_n_sources}; Est num: {n_sources}")

        # add white border showing where centers of stars and galaxies can be
        ax.axvline(bp, color="w")
        ax.axvline(images.shape[-1] - bp, color="w")
        ax.axhline(bp, color="w")
        ax.axhline(images.shape[-2] - bp, color="w")

        # plot image first
        image = images[img_id, 0].cpu().numpy()
        vmin = image.min().item()
        vmax = image.max().item()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.matshow(image, vmin=vmin, vmax=vmax, cmap="viridis")
        fig.colorbar(im, cax=cax, orientation="vertical")

        true_cat.plot_plocs(ax, img_id, "galaxy", bp=bp, color="r", marker="x", s=20)
        true_cat.plot_plocs(ax, img_id, "star", bp=bp, color="m", marker="x", s=20)
        est_cat.plot_plocs(ax, img_id, "all", bp=bp, color="b", marker="+", s=30)

        if ax_idx == 0:
            ax.scatter(None, None, color="r", marker="x", s=20, label="t.gal")
            ax.scatter(None, None, color="m", marker="x", s=20, label="t.star")
            ax.scatter(None, None, color="b", marker="+", s=30, label="p.source")
            ax.legend(
                bbox_to_anchor=(0, 1.2, 1.0, 0.102),
                loc="lower left",
                ncol=2,
                mode="expand",
                borderaxespad=0,
            )

    fig.tight_layout()
    return fig
