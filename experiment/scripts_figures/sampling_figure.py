from functools import partial

import matplotlib.pyplot as plt
import torch
from einops import rearrange, reduce
from matplotlib import pyplot as plt
from tqdm import tqdm

from bliss.catalog import FullCatalog, TileCatalog
from bliss.datasets.io import load_dataset_npz
from bliss.encoders.deblend import GalaxyEncoder
from bliss.encoders.detection import DetectionEncoder
from bliss.plotting import BlissFigure
from bliss.reporting import (
    get_blendedness,
    get_deblended_reconstructions,
    get_residual_measurements,
    match_by_locs,
    pred_in_batches,
)


class SamplingFigure(BlissFigure):
    """Create figures related to sampling centroids and fluxes on the full dataset."""

    def __init__(
        self,
        *,
        figdir,
        cachedir,
        suffix,
        overwrite=False,
        img_format="png",
        aperture=5.0,
        n_samples=100,
    ):
        super().__init__(
            figdir=figdir,
            cachedir=cachedir,
            suffix=suffix,
            overwrite=overwrite,
            img_format=img_format,
        )

        self.aperture = aperture
        self.n_samples = n_samples

    @property
    def all_rcs(self):
        return {
            "samples_residual": {},
        }

    @property
    def cache_name(self) -> str:
        return "sampling"

    @property
    def fignames(self) -> tuple[str, ...]:
        return ("samples_residual",)

    def compute_data(self, ds_path: str, detection: DetectionEncoder, deblender: GalaxyEncoder):
        device = detection.device

        # metadata
        bp = detection.bp
        tile_slen = detection.tile_slen

        # read dataset
        dataset = load_dataset_npz(ds_path)
        _images = dataset["images"]
        _paddings = dataset["paddings"]
        _noiseless = dataset["noiseless"]

        uncentered_sources = dataset["uncentered_sources"]
        galaxy_bools = dataset["galaxy_bools"]
        _tgbools = rearrange(galaxy_bools, "n ms 1 -> n ms 1 1 1 ")
        galaxy_uncentered = uncentered_sources * _tgbools

        # we want to exclude stars only but keep the padding for starters
        star_bools = dataset["star_bools"]
        only_stars = uncentered_sources * rearrange(star_bools, "b n 1 -> b n 1 1 1").float()
        all_stars = reduce(only_stars, "b n c h w -> b c h w", "sum")
        images = _images - all_stars
        paddings = _paddings - all_stars
        noiseless = _noiseless - all_stars

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
            truth, images, paddings=paddings, sources=galaxy_uncentered, bp=bp, r=self.aperture
        )
        truth["snr"] = meas_truth["snr"].clip(0)
        truth["blendedness"] = get_blendedness(galaxy_uncentered, noiseless).unsqueeze(-1)
        truth["fluxes"] = meas_truth["flux"]
        truth["ellips"] = meas_truth["ellips"]
        truth["sigma"] = meas_truth["sigma"]
        # we don't need to align here because we are matching later with the predictions

        # first get variational parameters that will be useful for diagnostics
        def _pred_fnc1(x):
            n_source_probs, locs_mean, locs_sd_raw = detection.forward(x)
            return {
                "n_source_probs": n_source_probs,
                "locs_mean": locs_mean,
                "locs_sd": locs_sd_raw,
            }

        additional_info = pred_in_batches(
            _pred_fnc1,
            images,
            device=device,
            desc="Encoding detections (MAP)",
            no_bar=False,
        )

        # now we sample location first
        _pred_fnc2 = partial(detection.sample, n_samples=self.n_samples)
        samples = pred_in_batches(
            _pred_fnc2,
            images,
            device=device,
            desc="Sampling images (detection)",
            no_bar=False,
            axis=1,
        )

        # set n_sources to 0 if locs are out of bounds, also set locs to 0
        # otherwise `TileCatalog` below breaks
        mask = samples["locs"][..., 0].lt(0) | samples["locs"][..., 0].gt(1)
        mask |= samples["locs"][..., 1].lt(0) | samples["locs"][..., 1].gt(1)
        samples["n_sources"][mask] = 0
        samples["locs"][mask, :] = 0.0

        # get full catalogs from tile catalogs
        cats = []
        for ii in range(self.n_samples):
            tile_cat = TileCatalog.from_flat_dict(
                tile_slen, nth, ntw, {k: v[ii] for k, v in samples.items()}
            )
            cat = tile_cat.to_full_params()
            cats.append(cat)

        all_fluxes = []
        all_ellips = []
        all_sigmas = []

        def _pred_fnc3(x, y):
            return {"gparams": deblender.variational_mode(x, y)}

        # NOTE: slow!
        for cat in tqdm(cats, total=len(cats), desc="Creating reconstructions for each sample"):

            # stars are excluded from all images for now
            # impose that all detections are galaxies
            tile_cat = cat.to_tile_params(tile_slen)
            tile_cat["galaxy_bools"] = rearrange(tile_cat.n_sources, "b x y -> b x y 1")
            _tile_locs = tile_cat.locs.to(device)

            _d = pred_in_batches(_pred_fnc3, images, _tile_locs, device=device, batch_size=500)
            _tile_gparams = _d["gparams"]
            _tile_gparams *= tile_cat["galaxy_bools"]
            tile_cat["galaxy_params"] = _tile_gparams
            new_cat = tile_cat.to_full_params()

            recon_uncentered = get_deblended_reconstructions(
                new_cat, deblender._dec, slen=slen, device=deblender.device, batch_size=500
            )
            meas = get_residual_measurements(
                new_cat,
                images,
                paddings=torch.zeros_like(images),
                sources=recon_uncentered,
            )

            all_fluxes.append(meas["flux"])
            all_ellips.append(meas["ellips"])
            all_sigmas.append(meas["sigma"])

        # now we match
        fluxes = []
        ellips = []
        sigmas = []

        for ii in tqdm(range(len(cats)), desc="Matching catalogs witht truth"):  # read: samples
            f = torch.zeros((images.shape[0], truth.max_n_sources, 1))
            e = torch.zeros((images.shape[0], truth.max_n_sources, 2))
            s = torch.zeros((images.shape[0], truth.max_n_sources, 1))

            for jj in range(images.shape[0]):

                tns = truth.n_sources[jj].item()
                _tplocs = truth.plocs[jj]
                _eplocs = cats[ii].plocs[jj]

                f[jj, :tns, :] = torch.nan
                e[jj, :tns, :] = torch.nan
                s[jj, :tns, :] = torch.nan

                # example: HSC -> 3 pixels for matching (Yr3 Li et al. ~2021)
                tm, em, dkeep, _ = match_by_locs(_tplocs, _eplocs, slack=3)

                for kk in range(len(tm)):
                    if dkeep[kk].item():
                        f[jj, tm[kk], 0] = all_fluxes[ii][jj][em[kk]].item()
                        e[jj, tm[kk], :] = all_ellips[ii][jj][em[kk]]
                        s[jj, tm[kk], 0] = all_sigmas[ii][jj][em[kk]].item()

            fluxes.append(f)
            ellips.append(e)
            sigmas.append(s)

        fluxes = torch.stack(fluxes)
        ellips = torch.stack(ellips)
        sigmas = torch.stack(sigmas)

        assert fluxes.shape[0] == len(cats)

        fluxes_flat = []
        e1_flat = []
        e2_flat = []
        sigmas_flat = []

        # now let's flatten across the true objects in each image
        for ii in tqdm(range(len(cats)), desc="Flattening arrays"):
            f = []
            e1 = []
            e2 = []
            s = []

            for jj in range(images.shape[0]):
                nt = truth.n_sources[jj].item()
                for kk in range(nt):
                    f.append(fluxes[ii, jj, kk, 0].item())
                    e1.append(ellips[ii, jj, kk, 0].item())
                    e2.append(ellips[ii, jj, kk, 1].item())
                    s.append(sigmas[ii, jj, kk, 0].item())
            fluxes_flat.append(f)
            e1_flat.append(e1)
            e2_flat.append(e2)
            sigmas_flat.append(s)

        flat_fluxes = torch.tensor(fluxes_flat).unsqueeze(-1)
        e1_flat = torch.tensor(e1_flat)
        e2_flat = torch.tensor(e2_flat)
        ellips_flat = torch.stack([e1_flat, e2_flat], axis=-1)
        sigmas_flat = torch.tensor(sigmas_flat).unsqueeze(-1)

        assert flat_fluxes.shape[0] == len(cats)
        assert ellips_flat.shape[-1] == 2

        # now flatten truth for convenience
        snr_flat = []
        bld_flat = []
        tflux_flat = []
        te1_flat = []
        te2_flat = []
        tsigmas_flat = []
        tgbools_flat = []
        for ii in range(images.shape[0]):
            nt = truth.n_sources[ii].item()
            for jj in range(nt):
                snr_flat.append(truth["snr"][ii, jj, 0])
                bld_flat.append(truth["blendedness"][ii, jj, 0])
                tflux_flat.append(truth["fluxes"][ii, jj, 0])
                te1_flat.append(truth["ellips"][ii, jj, 0])
                te2_flat.append(truth["ellips"][ii, jj, 1])
                tsigmas_flat.append(truth["sigma"][ii, jj, 0])
                tgbools_flat.append(truth["galaxy_bools"][ii, jj, 0])

        snr_flat = torch.tensor(snr_flat).unsqueeze(-1)
        bld_flat = torch.tensor(bld_flat).unsqueeze(-1)
        tflux_flat = torch.tensor(tflux_flat).unsqueeze(-1)
        tsigmas_flat = torch.tensor(tsigmas_flat).unsqueeze(-1)
        te1_flat = torch.tensor(te1_flat)
        te2_flat = torch.tensor(te2_flat)
        tellips_flat = torch.stack([te1_flat, te2_flat], dim=-1)
        tgbools_flat = torch.tensor(tgbools_flat).unsqueeze(-1)

        assert snr_flat.shape == bld_flat.shape == flat_fluxes.shape[1:]
        assert tellips_flat.shape[0] == snr_flat.shape[0]
        assert tellips_flat.shape[-1] == 2

        return {
            "samples": {
                "flux": flat_fluxes,
                "ellips": ellips_flat,
                "sigma": sigmas_flat,
            },
            "truth": {
                "flux": tflux_flat,
                "ellips": tellips_flat,
                "sigma": tsigmas_flat,
                "galaxy_bools": tgbools_flat,
            },
            "snr": snr_flat,
            "bld": bld_flat,
            **additional_info,
        }

    def create_figure(self, fname, data):
        if fname == "samples_residual":
            return plt.figure(figsize=(6, 6))
        raise ValueError(f"Unknown figure name: {fname}")
