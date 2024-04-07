import pytorch_lightning as pl
import torch
from einops import pack, rearrange, reduce, unpack
from torch import Tensor
from torch.distributions import Categorical, Normal
from torch.nn import BCELoss
from torch.optim import Adam

from bliss.catalog import TileCatalog
from bliss.datasets.galsim_blends import parse_dataset
from bliss.encoders.layers import ConcatBackgroundTransform, EncoderCNN, make_enc_final
from bliss.render_tiles import get_images_in_tiles, get_n_padded_tiles_hw
from bliss.reporting import DetectionMetrics


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

        # metrics
        self.val_detection_metrics = DetectionMetrics(slack=1.0)

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

    def get_loss(self, images: Tensor, background: Tensor, true_catalog: TileCatalog):
        assert images.device == background.device == true_catalog.device

        # encode
        n_source_probs, locs_mean, locs_sd = self.forward(images, background)

        # loss from detection count encoding
        n_true_sources_flat = rearrange(true_catalog.n_sources, "b nth ntw -> (b nth ntw)")
        counter_loss = BCELoss(reduction="none")(n_source_probs, n_true_sources_flat.float())

        # now for locations
        flat_true_locs = rearrange(true_catalog.locs, "b nth ntw xy -> (b nth ntw) xy", xy=2)
        locs_log_prob = -reduce(  # negative log-probability is the loss!
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

    # pytorch lightning
    def training_step(self, batch, batch_idx):
        """Training step (pytorch lightning)."""
        images, background, truth_cat = parse_dataset(batch, self.tile_slen)
        out = self.get_loss(images, background, truth_cat)

        # logging
        self.log("train/loss", out["loss"], batch_size=truth_cat.batch_size)
        self.log("train/counter_loss", out["counter_loss"], batch_size=truth_cat.batch_size)
        self.log("train/locs_loss", out["locs_loss"], batch_size=truth_cat.batch_size)

        return out["loss"]

    def validation_step(self, batch, batch_idx):
        """Validation step (pytorch lightning)."""
        images, background, truth_cat = parse_dataset(batch, self.tile_slen)
        batch_size = truth_cat.batch_size
        out = self.get_loss(images, background, truth_cat)
        pred_cat = self.variational_mode(images, background)

        # compute tiled metrics
        tiled_metrics = _compute_tiled_metrics(truth_cat, pred_cat, tile_slen=self.tile_slen)

        # compute full metrics with matching
        self.val_detection_metrics.update(truth_cat.to_full_params(), pred_cat.to_full_params())

        # logging
        self.log("val/loss", out["loss"], batch_size=batch_size)
        self.log("val/counter_loss", out["counter_loss"], batch_size=batch_size)
        self.log("val/locs_loss", out["locs_loss"], batch_size=batch_size)
        self.log_dict(
            tiled_metrics, batch_size=batch_size, on_step=True, on_epoch=True, reduce_fx="mean"
        )

        return out["loss"]

    def on_validation_epoch_end(self) -> None:
        out = self.val_detection_metrics.compute()
        out_log = {f"val/full/{p}": q for p, q in out.items()}
        self.log_dict(out_log, reduce_fx="mean")
        self.val_detection_metrics.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)


def _compute_tiled_metrics(
    truth_cat: TileCatalog, pred_cat: TileCatalog, tile_slen: int = 4, prefix: str = "val/tiled/"
):
    # compute simple 'tiled' metrics that do not use matching or FullCatalog
    # thus they are slightly incorrect, but OK for general diagnostics of model improving or not
    n_sources1 = truth_cat.n_sources.flatten()
    n_sources2 = pred_cat.n_sources.flatten()

    # recall
    mask1 = n_sources1 > 0
    n_match = torch.eq(n_sources1[mask1], n_sources2[mask1]).sum().item()
    recall = n_match / n_sources1.sum()

    # precision
    mask2 = n_sources2 > 0
    n_match = torch.eq(n_sources1[mask2], n_sources2[mask2]).sum().item()
    precision = n_match / n_sources2.sum()

    # f1
    f1 = 2 / (precision**-1 + recall**-1)

    # average residual distance for true matches
    match_mask = torch.logical_and(torch.eq(n_sources1, n_sources2), torch.eq(n_sources1, 1))
    locs1_flat = rearrange(truth_cat.locs, "b nth ntw xy -> (b nth ntw) xy", xy=2)
    locs2_flat = rearrange(pred_cat.locs, "b nth ntw xy -> (b nth ntw) xy", xy=2)
    plocs1 = locs1_flat[match_mask] * tile_slen
    plocs2 = locs2_flat[match_mask] * tile_slen
    avg_dist = reduce((plocs1 - plocs2).pow(2), "np xy -> np", "sum").sqrt().mean()

    # prefix
    out = {"precision": precision, "recall": recall, "f1": f1, "avg_dist": avg_dist}

    return {f"{prefix}{p}": q for p, q in out.items()}


def _locs_mean_func(x: Tensor) -> Tensor:
    # I don't think the special case for `x == 0` should be necessary
    return torch.sigmoid(x) * (x != 0).float()


def _locs_sd_func(x: Tensor) -> Tensor:
    return (x.exp() + 1e-5).sqrt()
