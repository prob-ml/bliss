"""Script to create detection encoder related figures."""

import numpy as np
import sep_pjw as sep
import torch
from einops import rearrange, reduce
from matplotlib import pyplot as plt
from tqdm import tqdm

from bliss.catalog import FullCatalog, TileCatalog, collate
from bliss.datasets.io import load_dataset_npz
from bliss.encoders.detection import DetectionEncoder
from bliss.plotting import BlissFigure
from bliss.reporting import compute_tp_fp_per_bin, get_fluxes_sep


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
            "blends_detection": {"fontsize": 32},
        }

    @property
    def cache_name(self) -> str:
        return "detection"

    @property
    def fignames(self) -> tuple[str, ...]:
        return ("blends_detection",)

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
        _, _, snr_sep_truth = get_fluxes_sep(
            truth, images, new_paddings, uncentered_sources, bp=bp, r=self.aperture
        )
        truth["snr"] = snr_sep_truth.clip(0)

        # encoder images using the detection encoder
        # images don't fit in memory, so we need to encode them in batches
        batch_size = 100
        n_images = images.shape[0]
        n_batches = n_images // batch_size
        tiled_params_list = []
        for i in tqdm(range(n_batches)):
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

        # obtain snr for the predicted catalogs (to calculate precision)
        for _, cat in pred_cats.items():
            _dummy_images = torch.zeros(
                images.shape[0], cat.max_n_sources, 1, images.shape[-2], images.shape[-1]
            )
            _, _, _snr = get_fluxes_sep(
                cat,
                images,
                torch.zeros_like(images),
                _dummy_images,
                bp=bp,
                r=self.aperture,
            )
            cat["snr"] = _snr.clip(0)

        # now we obtain the full catalog using SEP for comparison
        max_n_sources = 0
        all_sep_params = []

        for ii in range(images.shape[0]):
            im = images[ii, 0].numpy()
            bkg = sep.Background(im)
            catalog = sep.extract(im, 1.5, err=bkg.globalrms, minarea=5)

            x1 = catalog["x"]
            y1 = catalog["y"]

            # need to ignore detected sources that are in the padding
            in_padding = (x1 < 23.5) | (x1 > 63.5) | (y1 < 23.5) | (y1 > 63.5)

            x = x1[np.logical_not(in_padding)]
            y = y1[np.logical_not(in_padding)]

            n = len(x)
            max_n_sources = max(n, max_n_sources)

            all_sep_params.append((n, x, y))

        n_sources = torch.zeros((images.shape[0],)).long()
        plocs = torch.zeros((images.shape[0], max_n_sources, 2))

        for jj in range(images.shape[0]):
            n, x, y = all_sep_params[jj]
            n_sources[jj] = n

            plocs[jj, :n, 0] = torch.from_numpy(y) - bp + 0.5
            plocs[jj, :n, 1] = torch.from_numpy(x) - bp + 0.5

        sep_cat = FullCatalog(slen, slen, {"n_sources": n_sources, "plocs": plocs})

        # get snr for SEP catalog
        _, _, _snr = get_fluxes_sep(
            sep_cat,
            images,
            torch.zeros_like(images),
            torch.zeros_like(uncentered_sources),
            bp=bp,
            r=self.aperture,
        )
        sep_cat["snr"] = _snr.clip(0)

        # define snr bins
        snr_bins1 = 10 ** torch.arange(0, 3.0, 0.2)
        snr_bins2 = 10 ** torch.arange(0.2, 3.2, 0.2)
        snr_bins = torch.column_stack((snr_bins1, snr_bins2))

        # compute precision, recall, f1 for each threshold
        # and each snr bin
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

        # return dictionary with all the data
        return {
            "snr_bins": snr_bins,
            "thresh_out": thresh_out,
            "sep": {"recall": recall_sep, "precision": precision_sep, "f1": f1_sep},
        }

    def _get_detection_figure(self, data):
        # make a 3 column figure with precision, recall, f1 for all thresholds + sep
        # colors for thresholds hsould go from blue (low) to red (high) threshold

        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        axs = axs.flatten()

        snr_middle = data["snr_bins"].mean(axis=-1)

        # precision
        ax = axs[0]
        for tsh, out in data["thresh_out"].items():
            color = plt.cm.coolwarm(tsh)
            ax.plot(snr_middle, out["precision"], color=color)
        ax.plot(snr_middle, data["sep"]["precision"], "-k")
        ax.set_xlabel(r"\rm SNR")
        ax.set_ylabel(r"\rm Precision")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)

        # recall
        ax = axs[1]
        for tsh1, out1 in data["thresh_out"].items():
            color = plt.cm.coolwarm(tsh1)
            ax.plot(snr_middle, out1["recall"], color=color)
        ax.plot(snr_middle, data["sep"]["recall"], "-k")
        ax.set_xlabel(r"\rm SNR")
        ax.set_ylabel(r"\rm Recall")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)

        # f1
        ax = axs[2]
        for tsh2, out2 in data["thresh_out"].items():
            color = plt.cm.coolwarm(tsh2)
            ax.plot(snr_middle, out2["f1"], color=color, label=f"${tsh2:.2f}$")
        ax.plot(snr_middle, data["sep"]["f1"], "-k", label=r"\rm SEP")
        ax.set_xlabel(r"\rm SNR")
        ax.set_ylabel(r"\rm $F_{1}$ Score")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.legend()

        plt.tight_layout()

        return fig

    def create_figure(self, fname, data):
        if fname == "blends_detection":
            return self._get_detection_figure(data)
        raise ValueError(f"Unknown figure name: {fname}")
