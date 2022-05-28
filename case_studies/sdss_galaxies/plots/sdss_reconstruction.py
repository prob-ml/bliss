from typing import Union

import numpy as np
from matplotlib import pyplot as plt

from bliss import reporting
from bliss.encoder import Encoder
from bliss.inference import SDSSFrame, SimulatedFrame, reconstruct_scene_at_coordinates
from bliss.models.decoder import ImageDecoder
from case_studies.sdss_galaxies.plots.bliss_figures import BlissFigures, set_rc_params


class SDSSReconstructionFigures(BlissFigures):
    cache = "recon_sdss.pt"

    def __init__(self, scenes, figdir, cachedir, overwrite=False, img_format="png") -> None:
        self.scenes = scenes
        super().__init__(figdir, cachedir, overwrite=overwrite, img_format=img_format)

    def compute_data(
        self, frame: Union[SDSSFrame, SimulatedFrame], encoder: Encoder, decoder: ImageDecoder
    ):

        tile_slen = encoder.detection_encoder.tile_slen
        data = {}

        for figname, scene_coords in self.scenes.items():
            h, w, scene_size = scene_coords["h"], scene_coords["w"], scene_coords["size"]
            assert h % tile_slen == 0 and w % tile_slen == 0
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
            data[figname] = {
                "true": true[0, 0],
                "recon": recon[0, 0],
                "resid": resid[0, 0],
                "coplocs": coplocs,
                "cogbools": coadd_params["galaxy_bools"].reshape(-1),
                "plocs": recon_map.plocs.reshape(-1, 2),
                "gprobs": recon_map["galaxy_probs"].reshape(-1),
                "prob_n_sources": prob_n_sources,
            }

        return data

    def create_figures(self, data):  # pylint: disable=too-many-statements
        """Make figures related to reconstruction in SDSS."""
        out_figures = {}

        pad = 6.0
        set_rc_params(fontsize=22, tick_label_size="small", legend_fontsize="small")
        for figname, scene_coords in self.scenes.items():
            slen = scene_coords["size"]
            dvalues = data[figname].values()
            true, recon, res, coplocs, cogbools, plocs, gprobs, prob_n_sources = dvalues
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
            reporting.plot_image(fig, ax_true, true, vrange1)
            reporting.plot_locs(ax_true, 0, slen, coplocs, cogbools, "+", sp, lw, cmap="cool")
            reporting.plot_locs(ax_true, 0, slen, plocs, gprobs, "x", s, lw, cmap="bwr")

            reporting.plot_image(fig, ax_recon, recon, vrange1)
            reporting.plot_locs(ax_recon, 0, slen, coplocs, cogbools, "+", sp, lw, cmap="cool")
            reporting.plot_locs(ax_recon, 0, slen, plocs, gprobs, "x", s, lw, cmap="bwr")
            reporting.add_legend(ax_recon, labels, s=s)

            reporting.plot_image(fig, ax_res, res, vrange2)
            reporting.plot_locs(
                ax_res, 0, slen, coplocs, cogbools, "+", sp, lw, cmap="cool", alpha=0.5
            )
            reporting.plot_locs(ax_res, 0, slen, plocs, gprobs, "x", s, lw, cmap="bwr", alpha=0.5)
            plt.subplots_adjust(hspace=-0.4)
            plt.tight_layout()

            # plot probability of detection in each true object for blends
            if "blend" in figname:
                for ii, ploc in enumerate(coplocs.reshape(-1, 2)):
                    prob = prob_n_sources[ii].item()
                    x, y = ploc[1] + 0.5, ploc[0] + 0.5
                    text = r"$\boldsymbol{" + f"{prob:.2f}" + "}$"
                    ax_true.annotate(text, (x, y), color="lime")

            out_figures[figname] = fig

        return out_figures
