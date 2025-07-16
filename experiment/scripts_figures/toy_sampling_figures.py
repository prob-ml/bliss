import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from matplotlib.figure import Figure
from tqdm import tqdm

from bliss.catalog import FullCatalog, TileCatalog
from bliss.encoders.deblend import GalaxyEncoder
from bliss.encoders.detection import DetectionEncoder
from bliss.plotting import BlissFigure
from bliss.reporting import (
    get_deblended_reconstructions,
    get_residual_measurements,
    match_by_grade,
)


class ToySamplingFigure(BlissFigure):
    """Create figures related to assessingn probabilistic performance on toy blend."""

    def __init__(
        self,
        *,
        figdir,
        cachedir,
        suffix,
        n_samples: int,
        overwrite=False,
        img_format="png",
        aperture=5.0,
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
            "toy_residual_fluxes": {
                "fontsize": 40,
            }
        }

    @property
    def cache_name(self) -> str:
        return "toy_samples"

    @property
    def fignames(self) -> tuple[str, ...]:
        return ("toy_residual_fluxes",)

    def compute_data(
        self, detection: DetectionEncoder, deblender: GalaxyEncoder, *, toy_cache_fpath: str
    ):
        # first, decide image size
        cache_ds = torch.load(toy_cache_fpath)
        images = cache_ds["images"]
        seps = cache_ds["seps"]
        true_plocs = cache_ds["truth"]["plocs"]
        uncentered_true_sources = cache_ds["uncentered_true_sources"]

        slen = 55
        bp = detection.bp
        tile_slen = detection.tile_slen
        device = detection.device
        assert deblender.device == detection.device
        assert images.shape[-1] == slen + 2 * bp
        batch_size = len(seps)

        # get true catalog
        n_sources = 2
        d = {
            "n_sources": torch.full((batch_size,), n_sources),
            "plocs": true_plocs,
            "galaxy_bools": torch.ones(batch_size, n_sources, 1),
            "star_bools": torch.zeros(batch_size, n_sources, 1),
            "star_fluxes": torch.zeros(batch_size, n_sources, 1),
            "star_log_fluxes": torch.zeros(batch_size, n_sources, 1),
            "galaxy_params": torch.zeros(batch_size, n_sources, 2),
        }
        truth = FullCatalog(slen, slen, d)

        # get true fluxes
        truth_res = get_residual_measurements(
            truth, images, paddings=torch.zeros_like(images), sources=uncentered_true_sources
        )
        true_fluxes = truth_res["flux"]

        # get samples
        n_samples = self.n_samples
        samples = detection.sample(images.to(device), n_samples=n_samples)
        samples = {k: v.to("cpu") for k, v in samples.items()}

        # mask out samples out of boundaries
        mask_xy = samples["locs"].ge(0.0001) * samples["locs"].le(1 - 0.0001)
        mask = mask_xy[..., 0].bitwise_and(mask_xy[..., 1])
        new_locs = samples["locs"] * rearrange(mask, "n nt -> n nt 1")
        new_n_sources = samples["n_sources"] * mask

        new_samples = {**samples}
        new_samples["n_sources"] = new_n_sources
        new_samples["locs"] = new_locs

        # get catalogs
        cats = []
        nth = ntw = slen // tile_slen
        for ii in range(n_samples):
            tile_cat = TileCatalog.from_flat_dict(
                tile_slen, nth, ntw, {k: v[ii] for k, v in new_samples.items()}
            )
            cat = tile_cat.to_full_params()
            cats.append(cat)

        all_fluxes = []
        for cat in tqdm(cats, total=len(cats)):
            tile_cat = cat.to_tile_params(tile_slen)
            tile_cat["galaxy_bools"] = rearrange(tile_cat.n_sources, "b x y -> b x y 1")
            tile_locs = tile_cat.locs.to(device)
            _tile_gparams = deblender.variational_mode(images.to(device), tile_locs).to("cpu")
            _tile_gparams *= tile_cat["galaxy_bools"]
            tile_cat["galaxy_params"] = _tile_gparams
            new_cat = tile_cat.to_full_params()

            recon_uncentered = get_deblended_reconstructions(
                new_cat, deblender._dec, slen=slen, device=deblender.device, batch_size=200
            )
            _res = get_residual_measurements(
                new_cat,
                images,
                paddings=torch.zeros_like(images),
                sources=recon_uncentered,
            )
            all_fluxes.append(_res["flux"])

        f1s = []
        f2s = []

        for ii in tqdm(range(len(cats))):  # read: samples
            f1 = []
            f2 = []
            for jj in range(true_plocs.shape[0]):
                _tplocs = true_plocs[jj]
                _eplocs = cats[ii].plocs[jj]
                _fluxes = all_fluxes[ii][jj][:, 0]
                tm, em, dkeep, _ = match_by_grade(
                    locs1=_tplocs,
                    locs2=_eplocs,
                    fluxes1=true_fluxes[jj, :, 0],
                    fluxes2=_fluxes,
                )
                for kk in range(len(tm)):
                    if dkeep[kk].item():
                        if tm[kk] == 0:
                            f1.append(_fluxes[em[kk]].item())
                        elif tm[kk] == 1:
                            f2.append(_fluxes[em[kk]].item())
                        else:
                            raise ValueError
                    else:
                        if tm[kk] == 0:
                            f1.append(np.nan)
                        elif tm[kk] == 1:
                            f2.append(np.nan)
                        else:
                            raise ValueError
            f1s.append(torch.tensor(f1))
            f2s.append(torch.tensor(f2))

        f1s = torch.stack(f1s)
        f2s = torch.stack(f2s)

        return {
            "f1s": f1s,
            "f2s": f2s,
            "seps": seps,
            "true_plocs": true_plocs,
            "true_fluxes": true_fluxes,
        }

    def _get_residual_fluxes_figure(self, data: dict) -> Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        f1s = data["f1s"]
        f2s = data["f2s"]
        seps = data["seps"]
        true_fluxes = data["true_fluxes"]

        # f1s = f1s[:, 0, 0]
        y = (np.nanquantile(f1s, 0.5, axis=0) - true_fluxes[:, 0, 0]) / true_fluxes[:, 0, 0]
        y1 = (np.nanquantile(f1s, 0.159, axis=0) - true_fluxes[:, 0, 0]) / true_fluxes[:, 0, 0]
        y2 = (np.nanquantile(f1s, 0.841, axis=0) - true_fluxes[:, 0, 0]) / true_fluxes[:, 0, 0]

        ax1.plot(seps, y, "C0")
        ax1.set_title(r"\rm Galaxy 1")
        ax1.fill_between(seps, y1, y2, alpha=0.3, color="C0")
        ax1.set_xlabel(r"\rm Separation (pixels)")
        ax1.set_ylabel(r"\rm Residual fluxes")

        y = (np.nanquantile(f2s, 0.5, axis=0) - true_fluxes[:, 1, 0]) / true_fluxes[:, 0, 0]
        y1 = (np.nanquantile(f2s, 0.159, axis=0) - true_fluxes[:, 1, 0]) / true_fluxes[:, 0, 0]
        y2 = (np.nanquantile(f2s, 0.841, axis=0) - true_fluxes[:, 1, 0]) / true_fluxes[:, 0, 0]

        ax2.plot(seps, y, "C0")
        ax2.set_title(r"\rm Galaxy 2")
        ax2.fill_between(seps, y1, y2, alpha=0.3, color="C0")
        ax2.set_xlabel(r"\rm Separation (pixels)")
        ax2.set_ylabel(r"\rm Residual fluxes")

        ax1.axhline(y=0, color="k", linestyle="--")
        ax2.axhline(y=0, color="k", linestyle="--")
        ax1.set_ylim(-0.5, 1.05)
        ax2.set_ylim(-0.5, 1.05)

        return fig

    def create_figure(self, fname: str, data: dict) -> Figure:
        if fname == "toy_residual_fluxes":
            return self._get_residual_fluxes_figure(data)
        raise NotImplementedError("Figure {fname} not implemented.")
