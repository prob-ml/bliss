#!/usr/bin/env python3
import warnings
from pathlib import Path
from typing import Union

import hydra
import matplotlib as mpl
import numpy as np
import seaborn as sns
import torch
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from bliss import reporting
from bliss.catalog import FullCatalog, PhotoFullCatalog
from bliss.encoder import Encoder
from bliss.inference import SDSSFrame, SimulatedFrame, reconstruct_scene_at_coordinates
from bliss.models.decoder import ImageDecoder
from bliss.plotting import BlissFigure, add_loc_legend, make_detection_figure, plot_image, plot_locs
from bliss.reporting import compute_mag_bin_metrics

ALL_FIGS = ("detection_sdss", "recon_sdss")


def _compute_detection_metrics(truth: FullCatalog, pred: FullCatalog):
    """Return dictionary containing all metrics between true and predicted catalog."""

    # prepare magnitude bins
    mag_cuts2 = torch.arange(18, 24.5, 0.25)
    mag_cuts1 = torch.full_like(mag_cuts2, fill_value=0)
    mag_cuts = torch.column_stack((mag_cuts1, mag_cuts2))

    mag_bins2 = torch.arange(18, 25, 1.0)
    mag_bins1 = mag_bins2 - 1
    mag_bins = torch.column_stack((mag_bins1, mag_bins2))

    # compute metrics
    cuts_data = compute_mag_bin_metrics(mag_cuts, truth, pred)
    bins_data = compute_mag_bin_metrics(mag_bins, truth, pred)

    # data for scatter plot of misclassifications (over all magnitudes).
    tplocs = truth.plocs.reshape(-1, 2)
    eplocs = pred.plocs.reshape(-1, 2)
    tindx, eindx, dkeep, _ = reporting.match_by_locs(tplocs, eplocs, slack=1.0)

    # compute egprob separately for PHOTO
    egbool = pred["galaxy_bools"].reshape(-1)[eindx][dkeep]
    egprob = pred.get("galaxy_probs", None)
    egprob = egbool if egprob is None else egprob.reshape(-1)[eindx][dkeep]
    full_metrics = {
        "tgbool": truth["galaxy_bools"].reshape(-1)[tindx][dkeep],
        "egbool": egbool,
        "egprob": egprob,
        "tmag": truth["mags"].reshape(-1)[tindx][dkeep],
        "emag": pred["mags"].reshape(-1)[eindx][dkeep],
    }

    return {
        "mag_cuts": mag_cuts2,
        "mag_bins": mag_bins2,
        "cuts_data": cuts_data,
        "bins_data": bins_data,
        "full_metrics": full_metrics,
    }


def _make_classification_figure(
    mags,
    data,
    cuts_or_bins="cuts",
    xlims=(18, 24),
    ylims=(0.5, 1.05),
    ratio=2,
    where_step="mid",
    n_gap=50,
):
    # classification accuracy
    class_acc = data["class_acc"]
    galaxy_acc = data["galaxy_acc"]
    star_acc = data["star_acc"]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [1, ratio]}, sharex=True
    )
    xlabel = r"\rm magnitude " + cuts_or_bins[:-1]
    ax2.plot(mags, galaxy_acc, "-o", label=r"\rm galaxy")
    ax2.plot(mags, star_acc, "-o", label=r"\rm star")
    ax2.plot(mags, class_acc, "-o", label=r"\rm overall")
    ax2.set_xlim(xlims)
    ax2.set_ylim(ylims)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("classification accuracy")
    ax2.legend(loc="lower left", prop={"size": 18})

    # setup histogram up top.
    gcounts = data["n_matches_coadd_gal"]
    scounts = data["n_matches_coadd_star"]
    ax1.step(mags, gcounts, label=r"\rm matched coadd galaxies", where=where_step)
    ax1.step(mags, scounts, label=r"\rm matched coadd stars", where=where_step)
    ymax = max(max(gcounts), max(scounts))
    ymax = np.ceil(ymax / n_gap) * n_gap
    yticks = np.arange(0, ymax, n_gap)
    ax1.legend(loc="best", prop={"size": 16})
    ax1.set_ylim((0, ymax))
    ax1.set_yticks(yticks)
    ax1.set_ylabel(r"\rm Counts")
    plt.subplots_adjust(hspace=0)

    return fig


class SDSSReconstructionFigures(BlissFigure):
    cache = "recon_sdss.pt"

    def __init__(self, scene_name: str, scene_data: dict, *args, **kwargs) -> None:
        self.scene_name = scene_name
        self.scene_data = scene_data
        super().__init__(*args, **kwargs)

    @property
    def rc_kwargs(self):
        return {"fontsize": 22, "tick_label_size": "small", "legend_fontsize": "small"}

    @property
    def cache_name(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        return f"sdss_recon_{self.scene_name}"

    def compute_data(
        self, frame: Union[SDSSFrame, SimulatedFrame], encoder: Encoder, decoder: ImageDecoder
    ):
        h, w, scene_size = self.scene_data["h"], self.scene_data["w"], self.scene_data["size"]
        assert scene_size <= 300, "Scene too large, change slen."
        h_end = h + scene_size
        w_end = w + scene_size
        true = frame.image[:, :, h:h_end, w:w_end]
        coadd_params = frame.get_catalog((h, h_end), (w, w_end))

        recon, tile_map_recon = reconstruct_scene_at_coordinates(
            encoder,
            decoder,
            frame.image,
            frame.background,
            h_range=(h, h_end),
            w_range=(w, w_end),
        )
        resid = (true - recon) / recon.sqrt()

        tile_map_recon = tile_map_recon.cpu()
        recon_map = tile_map_recon.to_full_params()

        # get BLISS probability of n_sources in coadd locations.
        coplocs = coadd_params.plocs.reshape(-1, 2)
        prob_n_sources = tile_map_recon.get_tile_params_at_coord(coplocs)["n_source_log_probs"]
        prob_n_sources = prob_n_sources.exp()

        true = true.cpu()
        recon = recon.cpu()
        resid = resid.cpu()
        return {
            "true": true[0, 0],
            "recon": recon[0, 0],
            "resid": resid[0, 0],
            "coplocs": coplocs,
            "cogbools": coadd_params["galaxy_bools"].reshape(-1),
            "plocs": recon_map.plocs.reshape(-1, 2),
            "gprobs": recon_map["galaxy_probs"].reshape(-1),
            "prob_n_sources": prob_n_sources,
        }

    def create_figure(self, data) -> Figure:
        """Make figures related to reconstruction in SDSS."""

        pad = 6.0
        slen = self.scene_data["size"]
        true, recon, res, coplocs, cogbools, plocs, gprobs, prob_n_sources = data.values()
        assert slen == true.shape[-1] == recon.shape[-1] == res.shape[-1]
        fig, (ax_true, ax_recon, ax_res) = plt.subplots(nrows=1, ncols=3, figsize=(28, 12))

        ax_true.set_title("Original Image", pad=pad)
        ax_recon.set_title("Reconstruction", pad=pad)
        ax_res.set_title("Residual", pad=pad)

        s = 55 * 300 / slen  # marker size
        sp = s * 1.5
        lw = 2 * np.sqrt(300 / slen)

        vrange1 = (800, 1100)
        vrange2 = (-5, 5)
        labels = ["Coadd Galaxies", "Coadd Stars", "BLISS Galaxies", "BLISS Stars"]
        plot_image(fig, ax_true, true, vrange1)
        plot_locs(ax_true, 0, slen, coplocs, cogbools, "+", sp, lw, cmap="cool")
        plot_locs(ax_true, 0, slen, plocs, gprobs, "x", s, lw, cmap="bwr")

        plot_image(fig, ax_recon, recon, vrange1)
        plot_locs(ax_recon, 0, slen, coplocs, cogbools, "+", sp, lw, cmap="cool")
        plot_locs(ax_recon, 0, slen, plocs, gprobs, "x", s, lw, cmap="bwr")
        add_loc_legend(ax_recon, labels, s=s)

        plot_image(fig, ax_res, res, vrange2)
        plot_locs(ax_res, 0, slen, coplocs, cogbools, "+", sp, lw, cmap="cool", alpha=0.5)
        plot_locs(ax_res, 0, slen, plocs, gprobs, "x", s, lw, cmap="bwr", alpha=0.5)
        plt.subplots_adjust(hspace=-0.4)
        plt.tight_layout()

        # plot probability of detection in each true object for blends
        if "blend" in self.name:
            for ii, ploc in enumerate(coplocs.reshape(-1, 2)):
                prob = prob_n_sources[ii].item()
                x, y = ploc[1] + 0.5, ploc[0] + 0.5
                text = r"$\boldsymbol{" + f"{prob:.2f}" + "}$"
                ax_true.annotate(text, (x, y), color="lime")

        return fig


class BlissDetectionCutsFigure(BlissFigure):
    @property
    def cache_name(self) -> str:
        return "sdss_detection"

    @property
    def name(self) -> str:
        return "sdss_bliss_detection_cuts"

    @property
    def rc_kwargs(self):
        sns.set_theme(style="darkgrid")
        return {"tick_label_size": 22, "label_size": 30}

    def compute_data(
        self,
        frame: Union[SDSSFrame, SimulatedFrame],
        photo_cat: PhotoFullCatalog,
        encoder: Encoder,
        decoder: ImageDecoder,
    ) -> dict:
        bp = encoder.border_padding
        h, w = bp, bp
        h_end = ((frame.image.shape[2] - 2 * bp) // 4) * 4 + bp
        w_end = ((frame.image.shape[3] - 2 * bp) // 4) * 4 + bp
        truth_params: FullCatalog = frame.get_catalog((h, h_end), (w, w_end))
        photo_catalog_at_hw = photo_cat.crop_at_coords(h, h_end, w, w_end)

        # obtain predictions from BLISS.
        _, tile_est_params = reconstruct_scene_at_coordinates(
            encoder, decoder, frame.image, frame.background, h_range=(h, h_end), w_range=(w, w_end)
        )
        tile_est_params.set_all_fluxes_and_mags(decoder)
        est_params = tile_est_params.cpu().to_full_params()

        # compute metrics with bliss vs coadd and photo (frame) vs coadd
        bliss_metrics = _compute_detection_metrics(truth_params, est_params)
        photo_metrics = _compute_detection_metrics(truth_params, photo_catalog_at_hw)

        return {"bliss_metrics": bliss_metrics, "photo_metrics": photo_metrics}

    def create_figure(self, data) -> Figure:
        mag_cuts = data["bliss_metrics"]["mag_cuts"]
        cuts_data = data["bliss_metrics"]["cuts_data"]
        return make_detection_figure(mag_cuts, cuts_data, ylims=(0.5, 1.03))


class BlissDetectionBinsFigure(BlissDetectionCutsFigure):
    @property
    def name(self) -> str:
        return "sdss_bliss_detection_bins"

    def create_figure(self, data) -> Figure:
        mag_bins = data["bliss_metrics"]["mag_bins"] - 0.5
        bins_data = data["bliss_metrics"]["bins_data"]
        return make_detection_figure(
            mag_bins, bins_data, xlims=(17, 24), ylims=(0.0, 1.05), n_gap=25
        )


class BlissClassificationCutsFigure(BlissDetectionCutsFigure):
    @property
    def name(self) -> str:
        return "sdss_bliss_classification_cuts"

    def create_figure(self, data) -> Figure:
        mag_cuts = data["bliss_metrics"]["mag_cuts"]
        cuts_data = data["bliss_metrics"]["cuts_data"]
        return _make_classification_figure(mag_cuts, cuts_data, "cuts", ylims=(0.8, 1.03))


class BlissClassificationBinsFigure(BlissDetectionCutsFigure):
    @property
    def name(self) -> str:
        return "sdss_bliss_classification_bins"

    def create_figure(self, data) -> Figure:
        mag_bins = data["bliss_metrics"]["mag_bins"] - 0.5
        bins_data = data["bliss_metrics"]["bins_data"]
        return _make_classification_figure(
            mag_bins, bins_data, "bins", xlims=(17, 24), ylims=(0.0, 1.05), n_gap=25
        )


class BlissMag2ScatterFigure(BlissClassificationCutsFigure):
    @property
    def name(self) -> str:
        return "bliss_mag2_scatter"

    @staticmethod
    def make_mag_mag_scatter_figure(tgbool: np.ndarray, tmag: np.ndarray, emag: np.ndarray):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
        ax1.scatter(tmag[tgbool], emag[tgbool], marker="o", c="r", alpha=0.5)
        ax1.plot([15, 23], [15, 23], c="r", label="x=y line")
        ax2.scatter(tmag[~tgbool], emag[~tgbool], marker="o", c="b", alpha=0.5)
        ax2.plot([15, 23], [15, 23], c="b", label="x=y line")
        ax1.legend(loc="best", prop={"size": 22})
        ax2.legend(loc="best", prop={"size": 22})

        ax1.set_xlabel("True Magnitude")
        ax2.set_xlabel("True Magnitude")
        ax1.set_ylabel("Estimated Magnitude")
        ax2.set_ylabel("Estimated Magnitude")
        ax1.set_title("Matched Coadd Galaxies")
        ax2.set_title("Matched Coadd Stars")

        return fig

    def create_figure(self, data) -> Figure:
        tgbool = data["bliss_metrics"]["full_metrics"]["tgbool"].astype(bool)
        tmag = data["bliss_metrics"]["full_metrics"]["tmag"]
        emag = data["bliss_metrics"]["full_metrics"]["emag"]
        return self.make_mag_mag_scatter_figure(tgbool, tmag, emag)


class BlissMagProbScatterFigure(BlissClassificationCutsFigure):
    @property
    def name(self) -> str:
        return "bliss_mag_prob_scatter"

    @staticmethod
    def make_magnitude_prob_scatter_figure(
        tgbool: np.ndarray, egbool: np.ndarray, tmag: np.ndarray, egprob: np.ndarray
    ):
        # scatter of matched objects magnitude vs classification probability.
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        correct = np.equal(tgbool, egbool)

        ax.scatter(tmag[correct], egprob[correct], marker="+", c="b", label="correct", alpha=0.5)
        ax.scatter(
            tmag[~correct], egprob[~correct], marker="x", c="r", label="incorrect", alpha=0.5
        )
        ax.axhline(0.5, linestyle="--")
        ax.axhline(0.1, linestyle="--")
        ax.axhline(0.9, linestyle="--")
        ax.set_xlabel("True Magnitude")
        ax.set_ylabel("Estimated Probability of Galaxy")
        ax.legend(loc="best", prop={"size": 22})

        return fig

    def create_figure(self, data) -> Figure:
        tgbool = data["bliss_metrics"]["full_metrics"]["tgbool"].astype(bool)
        egbool = data["bliss_metrics"]["full_metrics"]["egbool"].astype(bool)
        tmag = data["bliss_metrics"]["full_metrics"]["tmag"]
        egprob = data["bliss_metrics"]["full_metrics"]["egprob"]
        return self.make_magnitude_prob_scatter_figure(tgbool, egbool, tmag, egprob)


class PhotoDetectionCutsFigure(BlissDetectionCutsFigure):
    @property
    def name(self) -> str:
        return "sdss_photo_detection_cuts"

    def create_figure(self, data) -> Figure:
        mag_cuts = data["photo_metrics"]["mag_cuts"]
        cuts_data = data["photo_metrics"]["cuts_data"]
        return make_detection_figure(mag_cuts, cuts_data, ylims=(0.5, 1.03))


class PhotoDetectionBinsFigure(BlissDetectionCutsFigure):
    @property
    def name(self) -> str:
        return "sdss_photo_detection_bins"

    def create_figure(self, data) -> Figure:
        mag_bins = data["photo_metrics"]["mag_bins"] - 0.5
        bins_data = data["photo_metrics"]["bins_data"]
        return make_detection_figure(
            mag_bins, bins_data, xlims=(17, 24), ylims=(0.0, 1.05), n_gap=25
        )


class PhotoClassificationCutsFigure(BlissDetectionCutsFigure):
    @property
    def name(self) -> str:
        return "sdss_photo_classification_cuts"

    def create_figure(self, data) -> Figure:
        mag_cuts = data["photo_metrics"]["mag_cuts"]
        cuts_data = data["photo_metrics"]["cuts_data"]
        return _make_classification_figure(mag_cuts, cuts_data, "cuts", ylims=(0.8, 1.03))


class PhotoClassificationBinsFigure(BlissDetectionCutsFigure):
    @property
    def name(self) -> str:
        return "sdss_photo_classification_bins"

    def create_figure(self, data) -> Figure:
        mag_bins = data["photo_metrics"]["mag_bins"] - 0.5
        bins_data = data["photo_metrics"]["bins_data"]
        return _make_classification_figure(
            mag_bins, bins_data, "bins", xlims=(17, 24), ylims=(0.0, 1.05), n_gap=25
        )


class PhotoMag2SatterFigure(BlissMag2ScatterFigure):
    @property
    def name(self) -> str:
        return "photo_mag2_scatter"

    def create_figure(self, data) -> Figure:
        tgbool = data["photo_metrics"]["full_metrics"]["tgbool"].astype(bool)
        tmag = data["photo_metrics"]["full_metrics"]["tmag"]
        emag = data["photo_metrics"]["full_metrics"]["emag"]
        return self.make_mag_mag_scatter_figure(tgbool, tmag, emag)


class PhotoMagProbScatterFigure(BlissMagProbScatterFigure):
    @property
    def name(self) -> str:
        return "photo_mag_prob_scatter"

    def create_figure(self, data) -> Figure:
        tgbool = data["photo_metrics"]["full_metrics"]["tgbool"].astype(bool)
        egbool = data["photo_metrics"]["full_metrics"]["egbool"].astype(bool)
        tmag = data["photo_metrics"]["full_metrics"]["tmag"]
        egprob = data["photo_metrics"]["full_metrics"]["egprob"]
        return self.make_magnitude_prob_scatter_figure(tgbool, egbool, tmag, egprob)


def load_models(cfg, device):
    # load models required for SDSS reconstructions.

    location = instantiate(cfg.models.detection_encoder).to(device).eval()
    location.load_state_dict(
        torch.load(cfg.plots.location_checkpoint, map_location=location.device)
    )

    binary = instantiate(cfg.models.binary).to(device).eval()
    binary.load_state_dict(torch.load(cfg.plots.binary_checkpoint, map_location=binary.device))

    galaxy = instantiate(cfg.models.galaxy_encoder).to(device).eval()
    galaxy.load_state_dict(torch.load(cfg.plots.galaxy_checkpoint, map_location=galaxy.device))

    n_images_per_batch = cfg.plots.encoder.n_images_per_batch
    n_rows_per_batch = cfg.plots.encoder.n_rows_per_batch
    encoder = Encoder(
        location.eval(),
        binary.eval(),
        galaxy.eval(),
        n_images_per_batch=n_images_per_batch,
        n_rows_per_batch=n_rows_per_batch,
    )
    encoder = encoder.to(device)
    decoder: ImageDecoder = instantiate(cfg.models.decoder).to(device).eval()
    return encoder, decoder


def load_sdss_data(cfg):
    frame: Union[SDSSFrame, SimulatedFrame] = instantiate(cfg.plots.frame)
    photo_cat = PhotoFullCatalog.from_file(**cfg.plots.photo_catalog)
    return frame, photo_cat


def setup(cfg):
    pcfg = cfg.plots
    figs = set(pcfg.figs)
    cachedir = pcfg.cachedir
    device = torch.device(pcfg.device)
    bfig_kwargs = {
        "figdir": pcfg.figdir,
        "cachedir": cachedir,
        "img_format": pcfg.image_format,
    }

    if not Path(cachedir).exists():
        warnings.warn("Specified cache directory does not exist, will attempt to create it.")
        Path(cachedir).mkdir(exist_ok=True, parents=True)

    assert set(figs).issubset(set(ALL_FIGS))
    return figs, device, bfig_kwargs


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg):
    figs, device, bfig_kwargs = setup(cfg)
    encoder, decoder = load_models(cfg, device)
    frame, photo_cat = load_sdss_data(cfg)
    overwrite = cfg.plots.overwrite

    # FIGURE 1: Classification and Detection metrics
    if "detection_sdss" in figs:
        print("INFO: Creating classification and detection metrics from SDSS frame figures...")
        args = frame, photo_cat, encoder, decoder
        BlissDetectionCutsFigure(overwrite=overwrite, **bfig_kwargs)(*args)
        BlissDetectionBinsFigure(overwrite=False, **bfig_kwargs)(*args)
        BlissClassificationCutsFigure(overwrite=False, **bfig_kwargs)(*args)
        BlissClassificationBinsFigure(overwrite=False, **bfig_kwargs)(*args)
        BlissMag2ScatterFigure(overwrite=False, **bfig_kwargs)(*args)
        BlissMagProbScatterFigure(overwrite=False, **bfig_kwargs)(*args)

        PhotoDetectionCutsFigure(overwrite=False, **bfig_kwargs)(*args)
        PhotoDetectionBinsFigure(overwrite=False, **bfig_kwargs)(*args)
        PhotoClassificationCutsFigure(overwrite=False, **bfig_kwargs)(*args)
        PhotoClassificationBinsFigure(overwrite=False, **bfig_kwargs)(*args)
        PhotoMag2SatterFigure(overwrite=False, **bfig_kwargs)(*args)
        PhotoMagProbScatterFigure(overwrite=False, **bfig_kwargs)(*args)

        mpl.rc_file_defaults()

    # FIGURE 2: Reconstructions on SDSS
    if "recon_sdss" in figs:
        print("INFO: Creating reconstructions from SDSS figures...")
        for scene_name, scene_data in cfg.plots.scenes.items():
            sdss_rec_fig = SDSSReconstructionFigures(
                scene_name, scene_data, overwrite=overwrite, **bfig_kwargs
            )
            sdss_rec_fig(frame, encoder, decoder)
        mpl.rc_file_defaults()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
