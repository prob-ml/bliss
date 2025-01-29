import pytorch_lightning as pl
import torch
from einops import rearrange, reduce, unpack
from torch import Tensor
from torch.distributions import Bernoulli, Normal
from torch.nn import BCELoss
from torch.optim import Adam

from bliss.catalog import TileCatalog
from bliss.datasets.generate_blends import parse_dataset
from bliss.encoders.layers import EncoderCNN, make_enc_final
from bliss.grid import validate_border_padding
from bliss.render_tiles import get_images_in_tiles, get_n_padded_tiles_hw


class DetectionEncoder(pl.LightningModule):
    """Encodes the distribution of a latent variable representing an astronomical image.

    This class implements the source encoder, which is supposed to take in
    an astronomical image of size slen * slen and returns a NN latent variable
    representation of this image.
    """

    def __init__(
        self,
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
            n_bands: number of bands
            tile_slen: size of tiles (squares).
            ptile_slen: size of padded tiles (squares).
            channel: TODO (document this)
            spatial_dropout: TODO (document this)
            dropout: TODO (document this)
            hidden: TODO (document this)
        """
        assert n_bands == 1, "Only 1 band is supported"
        super().__init__()

        self.n_bands = n_bands

        assert tile_slen <= ptile_slen
        self.tile_slen = tile_slen
        self.ptile_slen = ptile_slen
        self.bp = validate_border_padding(tile_slen, ptile_slen)

        # Number of distributional parameters used to characterize each source in an image.
        # 2 for location mean, 2 for location sigma (xy), 1 for for probability of counts.
        self._dim_out_all = 5

        dim_enc_conv_out = ((self.ptile_slen + 1) // 2 + 1) // 2

        # networks to be trained
        self._enc_conv = EncoderCNN(n_bands, channel, spatial_dropout)
        self._enc_final = make_enc_final(
            channel * 4 * dim_enc_conv_out**2,
            hidden,
            self._dim_out_all,
            dropout,
        )

    def forward(self, images: Tensor):
        image_ptiles = get_images_in_tiles(images, self.tile_slen, self.ptile_slen)
        flat_image_ptiles = rearrange(image_ptiles, "n nth ntw c h w -> (n nth ntw) c h w")
        return self.encode_tiled(flat_image_ptiles)

    def encode_tiled(self, flat_image_ptiles: Tensor):
        # encode
        enc_conv_output = self._enc_conv(flat_image_ptiles)
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

    def variational_mode(self, images: Tensor) -> TileCatalog:
        """Compute the variational mode."""
        _, _, h, w = images.shape
        nth, ntw = get_n_padded_tiles_hw(h, w, self.ptile_slen, self.tile_slen)
        n_source_probs, locs_mean, _ = self.forward(images)
        flat_tile_n_sources = n_source_probs.ge(0.5).long()
        flat_tile_locs = locs_mean * rearrange(flat_tile_n_sources, "np -> np 1")

        return TileCatalog.from_flat_dict(
            self.tile_slen, nth, ntw, {"n_sources": flat_tile_n_sources, "locs": flat_tile_locs}
        )

    def variational_model_tiled(self, flat_image_ptiles: Tensor) -> dict[str, Tensor]:
        n_source_probs, locs_mean, _ = self.encode_tiled(flat_image_ptiles)
        flat_tile_n_sources = n_source_probs.ge(0.5).long()
        flat_tile_locs = locs_mean * rearrange(flat_tile_n_sources, "np -> np 1")
        return {"n_sources": flat_tile_n_sources, "locs": flat_tile_locs}

    def sample(self, images: Tensor, n_samples: int = 1) -> dict[str, Tensor]:
        """Sample from the encoded variational distribution.

        Args:
            images:
                Tensor of images to encode.

            n_samples:
                The number of samples to draw.

        Returns:
            A dictionary of tensors with shape `n_samples * n_ptiles * ...`.
            Consists of "n_sources" and "locs".
        """
        n_source_probs, locs_mean, locs_sd = self.forward(images)

        # sample counts per tile
        tile_n_sources = Bernoulli(n_source_probs).sample((n_samples,))
        assert tile_n_sources.ndim == 2

        # sample locations and zero out out empty sources
        raw_tile_locs = Normal(locs_mean, locs_sd).sample((n_samples,))
        assert raw_tile_locs.ndim == 3
        tile_locs = raw_tile_locs * rearrange(tile_n_sources, "ns np -> ns np 1")

        assert tile_n_sources.shape[0] == tile_locs.shape[0] == n_samples

        return {"n_sources": tile_n_sources, "locs": tile_locs}

    def get_loss(self, images: Tensor, true_catalog: TileCatalog):
        assert images.device == true_catalog.device
        b, nth, ntw, _ = true_catalog.locs.shape

        # encode
        out: tuple[Tensor, Tensor, Tensor] = self(images)
        n_source_probs, locs_mean, locs_sd = out

        # loss from detection count encoding
        n_true_sources_flat = rearrange(true_catalog.n_sources, "b nth ntw -> (b nth ntw)")
        counter_loss = BCELoss(reduction="none")(n_source_probs, n_true_sources_flat.float())

        # now for locations
        flat_true_locs = rearrange(true_catalog.locs, "b nth ntw xy -> (b nth ntw) xy", xy=2)
        locs_log_prob = -reduce(  # negative log-probability is the loss!
            Normal(locs_mean, locs_sd).log_prob(flat_true_locs), "np xy -> np", "sum", xy=2
        )
        locs_loss = locs_log_prob * n_true_sources_flat.float()  # loc loss only on "on" sources.

        loss_vec = locs_loss * (locs_loss.detach() < 1e6).float() + counter_loss  # noqa: WPS459

        # per the paper, we take the mean over batches and sum over tiles
        loss_per_tile = rearrange(loss_vec, "(b nth ntw) -> b nth ntw", b=b, nth=nth, ntw=ntw)
        loss_per_batch = reduce(loss_per_tile, "b nth ntw -> b", "sum")
        loss = reduce(loss_per_batch, "b -> ", "mean")

        return {
            "loss": loss,
            "counter_loss": counter_loss.detach().mean(),
            "locs_loss": locs_loss.detach().mean(),
        }

    # pytorch lightning
    def training_step(self, batch, batch_idx):
        """Training step (pytorch lightning)."""
        images, truth_cat, _ = parse_dataset(batch, self.tile_slen)
        out = self.get_loss(images, truth_cat)

        # logging
        self.log("train/loss", out["loss"], batch_size=truth_cat.batch_size)
        self.log("train/counter_loss", out["counter_loss"], batch_size=truth_cat.batch_size)
        self.log("train/locs_loss", out["locs_loss"], batch_size=truth_cat.batch_size)

        return out["loss"]

    def validation_step(self, batch, batch_idx):
        """Validation step (pytorch lightning)."""
        images, truth_cat, _ = parse_dataset(batch, self.tile_slen)
        batch_size = truth_cat.batch_size
        out = self.get_loss(images, truth_cat)
        pred_cat = self.variational_mode(images)

        # compute tiled metrics
        tiled_metrics = _compute_tiled_metrics(truth_cat, pred_cat, tile_slen=self.tile_slen)

        # logging
        self.log("val/loss", out["loss"], batch_size=batch_size)
        self.log("val/counter_loss", out["counter_loss"], batch_size=batch_size)
        self.log("val/locs_loss", out["locs_loss"], batch_size=batch_size)
        self.log_dict(
            tiled_metrics, batch_size=batch_size, on_step=True, on_epoch=True, reduce_fx="mean"
        )

        return out["loss"]

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
    n_match = torch.eq(n_sources1[mask1], n_sources2[mask1]).sum()
    recall = n_match / n_sources1.sum()

    # precision
    mask2 = n_sources2 > 0
    n_match = torch.eq(n_sources1[mask2], n_sources2[mask2]).sum()
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
