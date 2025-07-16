"""Script to create detection encoder related figures."""

import torch
from einops import rearrange, reduce
from matplotlib import pyplot as plt
from tqdm import tqdm

from bliss.catalog import FullCatalog, TileCatalog, collate
from bliss.datasets.io import load_dataset_npz
from bliss.encoders.detection import DetectionEncoder
from bliss.plotting import BlissFigure
from bliss.reporting import (
    compute_tp_fp_per_bin,
    get_blendedness,
    get_residual_measurements,
    get_sep_catalog,
)


class BlendDetectionFigures(BlissFigure):
    def __init__(
        self, *, figdir, cachedir, suffix, overwrite=False, img_format="png", aperture=5.0
    ):
        super().__init__(
            figdir=figdir,
            cachedir=cachedir,
            suffix=suffix,
            overwrite=overwrite,
            img_format=img_format,
        )

        self.aperture = aperture

    @property
    def all_rcs(self) -> dict:
        return {
            "snr_detection": {"fontsize": 40, "major_tick_size": 12, "minor_tick_size": 7},
            "bld_detection": {
                "fontsize": 22,
                "legend_fontsize": 16,
                "tick_label_size": 16,
            },
        }

    @property
    def cache_name(self) -> str:
        return "detection"

    @property
    def fignames(self) -> tuple[str, ...]:
        return ("snr_detection", "bld_detection")

    def compute_data(self, ds_path: str, detection: DetectionEncoder):
        # metadata
        bp = detection.bp
        tile_slen = detection.tile_slen

        # read dataset
        dataset = load_dataset_npz(ds_path)
        images = dataset["images"]
        paddings = dataset["paddings"]
        uncentered_sources = dataset["uncentered_sources"]
        star_bools = dataset["star_bools"]
        noiseless = dataset["noiseless"]

        # paddings include stars for convenience, but we don't want to remove them in this case
        # we want to include snr of stars
        only_stars = uncentered_sources * rearrange(star_bools, "b n 1 -> b n 1 1 1").float()
        all_stars = reduce(only_stars, "b n c h w -> b c h w", "sum")
        new_paddings = paddings - all_stars

        # more metadata
        slen = images.shape[-1] - 2 * bp
        nth = (images.shape[2] - 2 * bp) // tile_slen
        ntw = (images.shape[3] - 2 * bp) // tile_slen

        # get truth catalog
        exclude = ("images", "uncentered_sources", "centered_sources", "noiseless", "paddings")
        true_cat_dict = {p: q for p, q in dataset.items() if p not in exclude}
        truth = FullCatalog(slen, slen, true_cat_dict)

        # add true snr to truth catalog using sep
        meas_truth = get_residual_measurements(
            truth, images, paddings=new_paddings, sources=uncentered_sources, bp=bp, r=self.aperture
        )
        truth["snr"] = meas_truth["snr"].clip(0)

        # add blendedness
        truth["bld"] = get_blendedness(uncentered_sources, noiseless).unsqueeze(-1)

        # encoder images using the detection encoder
        # images don't fit in memory, so we need to encode them in batches
        batch_size = 100
        n_images = images.shape[0]
        n_batches = n_images // batch_size
        tiled_params_list = []
        for i in tqdm(range(n_batches), desc="Encoding images..."):
            image_batch = images[i * batch_size : (i + 1) * batch_size].to(detection.device)
            n_source_probs, locs_mean, locs_sd_raw = detection.forward(image_batch)
            tiled_params = {
                "n_source_probs": n_source_probs.cpu(),
                "locs_mean": locs_mean.cpu(),
                "locs_sd": locs_sd_raw.cpu(),
            }
            tiled_params_list.append(tiled_params)

        # combine the results into one dictionary
        tiled_params = collate(tiled_params_list)

        # now we get full predicted catalogs fro different thresholds
        thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
        pred_cats = {}

        print("INFO:Getting tile catalogs for different thresholds")
        for thres in thresholds:
            n_source_probs = tiled_params["n_source_probs"]
            n_sources = n_source_probs.ge(thres).long()
            tiled_is_on = rearrange(n_source_probs.ge(thres).float(), "n -> n 1")

            locs = tiled_params["locs_mean"] * tiled_is_on
            locs_sd = tiled_params["locs_sd"] * tiled_is_on

            tile_cat = TileCatalog.from_flat_dict(
                tile_slen,
                nth,
                ntw,
                {
                    "n_sources": n_sources,
                    "locs": locs,
                    "locs_sd": locs_sd,
                    "n_source_probs": n_source_probs.reshape(-1, 1),
                },
            )

            pred_cats[thres] = tile_cat.to_full_params()

        # now we obtain the full catalog using SEP for comparison
        print("INFO: SEP measurements...")
        sep_cat = get_sep_catalog(images, slen=slen, bp=bp)

        print("INFO:Compute recall (blendedness)...")
        # First bins, equal sized
        bld = truth["bld"].flatten()
        bld_mask = (bld > 1e-2) * (bld <= 1)
        _bld = bld[bld_mask]
        qs = torch.linspace(0, 1, 12)
        _bld_bins = torch.quantile(_bld, qs)
        bld_bins = torch.column_stack((_bld_bins[:-1], _bld_bins[1:]))

        # compute recall for blendedness
        thresh_out = {tsh: {} for tsh in pred_cats}
        for tsh, cat1 in pred_cats.items():
            counts_per_bin = compute_tp_fp_per_bin(truth, cat1, "bld", bld_bins, only_recall=True)
            tp_recall = counts_per_bin["tp_recall"].sum(axis=-1)
            n_true = counts_per_bin["ntrue"].sum(axis=-1)
            recall = tp_recall / n_true
            thresh_out[tsh]["recall"] = recall

        # compute precision, recall, f1 for SEP catalog
        counts_per_bin_sep = compute_tp_fp_per_bin(
            truth, sep_cat, "bld", bld_bins, only_recall=True
        )
        tp_recall_sep = counts_per_bin_sep["tp_recall"].sum(axis=-1)
        n_true_sep = counts_per_bin_sep["ntrue"].sum(axis=-1)
        recall_sep = tp_recall_sep / n_true_sep

        bld_dict = {
            "bld_bins": bld_bins,
            "thresh_out": thresh_out,
            "sep": {"recall": recall_sep},
        }

        # obtain snr for the predicted catalogs (to calculate precision)
        print("INFO:Residual measurement for each catalog")
        for _, cat in pred_cats.items():
            _dummy_images = torch.zeros(
                images.shape[0], cat.max_n_sources, 1, images.shape[-2], images.shape[-1]
            )
            _meas = get_residual_measurements(
                cat,
                images,
                paddings=torch.zeros_like(images),
                sources=_dummy_images,
                bp=bp,
                r=self.aperture,
                no_bar=False,
            )
            cat["snr"] = _meas["snr"].clip(0)

        # get snr for SEP catalog
        print("INFO: SEP residual measurements...")
        _meas_sep = get_residual_measurements(
            sep_cat,
            images,
            paddings=torch.zeros_like(images),
            sources=torch.zeros_like(uncentered_sources),
            bp=bp,
            r=self.aperture,
            no_bar=False,
        )
        sep_cat["snr"] = _meas_sep["snr"].clip(0)

        # define snr bins
        snr_bins1 = 10 ** torch.arange(0, 3.0, 0.2)
        snr_bins2 = 10 ** torch.arange(0.2, 3.2, 0.2)
        snr_bins = torch.column_stack((snr_bins1, snr_bins2))

        # compute precision, recall, f1 for each threshold and snr bin
        print("INFO:Compute precision, recall, and F1 (SNR)...")
        thresh_out = {tsh: {} for tsh in pred_cats}
        for tsh, cat1 in pred_cats.items():
            counts_per_bin = compute_tp_fp_per_bin(truth, cat1, "snr", snr_bins)

            tp_precision = counts_per_bin["tp_precision"].sum(axis=-1)
            fp_precision = counts_per_bin["fp_precision"].sum(axis=-1)
            tp_recall = counts_per_bin["tp_recall"].sum(axis=-1)
            n_true = counts_per_bin["ntrue"].sum(axis=-1)
            precision = tp_precision / (tp_precision + fp_precision)
            recall = tp_recall / n_true
            f1 = 2 / (recall**-1 + precision**-1)

            thresh_out[tsh]["recall"] = recall
            thresh_out[tsh]["precision"] = precision
            thresh_out[tsh]["f1"] = f1

        # compute precision, recall, f1 for SEP catalog
        counts_per_bin_sep = compute_tp_fp_per_bin(truth, sep_cat, "snr", snr_bins)
        tp_precision_sep = counts_per_bin_sep["tp_precision"].sum(axis=-1)
        fp_precision_sep = counts_per_bin_sep["fp_precision"].sum(axis=-1)
        tp_recall_sep = counts_per_bin_sep["tp_recall"].sum(axis=-1)
        n_true_sep = counts_per_bin_sep["ntrue"].sum(axis=-1)

        precision_sep = tp_precision_sep / (tp_precision_sep + fp_precision_sep)
        recall_sep = tp_recall_sep / n_true_sep
        f1_sep = 2 / (recall_sep**-1 + precision_sep**-1)

        snr_dict = {
            "snr_bins": snr_bins,
            "thresh_out": thresh_out,
            "sep": {"recall": recall_sep, "precision": precision_sep, "f1": f1_sep},
        }

        # return dictionary with all the data
        return {
            "snr": snr_dict,
            "blendedness": bld_dict,
        }

    def _get_snr_detection_figure(self, data):
        # make a 3 column figure with precision, recall, f1 for all thresholds + sep
        # colors for thresholds hsould go from blue (low) to red (high) threshold

        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        axs = axs.flatten()
        ds = data["snr"]

        snr_middle = ds["snr_bins"].mean(axis=-1)

        # precision
        ax = axs[0]
        for tsh, out in ds["thresh_out"].items():
            color = plt.cm.coolwarm(tsh)
            ax.plot(snr_middle, out["precision"], color=color)
        ax.plot(snr_middle, ds["sep"]["precision"], "--k", lw=3)
        ax.set_xlabel(r"\rm SNR")
        ax.set_ylabel(r"\rm Precision")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)

        # recall
        ax = axs[1]
        for tsh1, out1 in ds["thresh_out"].items():
            color = plt.cm.coolwarm(tsh1)
            ax.plot(snr_middle, out1["recall"], color=color)
        ax.plot(snr_middle, ds["sep"]["recall"], "--k", lw=3)
        ax.set_xlabel(r"\rm SNR")
        ax.set_ylabel(r"\rm Recall")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)

        # f1
        ax = axs[2]
        for tsh2, out2 in ds["thresh_out"].items():
            color = plt.cm.coolwarm(tsh2)
            ax.plot(snr_middle, out2["f1"], color=color, label=f"${tsh2:.2f}$")
        ax.plot(snr_middle, ds["sep"]["f1"], "--k", label=r"\rm SEP", lw=3)
        ax.set_xlabel(r"\rm SNR")
        ax.set_ylabel(r"\rm $F_{1}$ Score")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.legend()

        plt.tight_layout()

        return fig

    def _get_blendedness_detection_figure(self, data):
        fig, ax = plt.subplots(figsize=(6, 6))
        ds = data["blendedness"]
        bld_bins = ds["bld_bins"]
        bld_middle = bld_bins.mean(axis=-1)

        # recall
        for tsh, out in ds["thresh_out"].items():
            color = plt.cm.coolwarm(tsh)
            ax.plot(bld_middle, out["recall"], c=color, label=f"${tsh:.2f}$")
        ax.plot(bld_middle, ds["sep"]["recall"], "--k", lw=2, label=r"\rm SEP")
        ax.set_xlabel(r"\rm Blendedness")
        ax.set_ylabel(r"\rm Recall")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(1e-2, 1)
        ax.set_xticks([1e-2, 1e-1, 1])
        ax.set_xscale("log")
        ax.legend()

        plt.tight_layout()
        return fig

    def create_figure(self, fname, data):
        if fname == "snr_detection":
            return self._get_snr_detection_figure(data)
        if fname == "bld_detection":
            return self._get_blendedness_detection_figure(data)
        raise ValueError(f"Unknown figure name: {fname}")
