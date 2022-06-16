#!/usr/bin/env python3
import warnings
from pathlib import Path
from typing import Union

import hydra
import matplotlib as mpl
import torch
from hydra.utils import instantiate

from bliss import generate
from bliss.catalog import PhotoFullCatalog
from bliss.encoder import Encoder
from bliss.inference import SDSSFrame, SimulatedFrame
from bliss.models.decoder import ImageDecoder
from case_studies.sdss_galaxies.plots.autoencoder import AEReconstructionFigures
from case_studies.sdss_galaxies.plots.sdss_detection_metrics import DetectionClassificationFigures
from case_studies.sdss_galaxies.plots.sdss_reconstruction import SDSSReconstructionFigures
from case_studies.sdss_galaxies.plots.sim_blend_metrics import BlendSimFigures


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
        location.eval(), binary.eval(), galaxy.eval(), n_images_per_batch, n_rows_per_batch
    )
    encoder = encoder.to(device)

    decoder: ImageDecoder = instantiate(cfg.models.decoder).to(device).eval()

    return encoder, decoder


def setup(cfg):
    figs = set(cfg.plots.figs)
    cachedir = cfg.plots.cachedir
    device = torch.device(cfg.plots.device)
    bfig_kwargs = {
        "figdir": cfg.plots.figdir,
        "cachedir": cachedir,
        "overwrite": cfg.plots.overwrite,
        "img_format": cfg.plots.image_format,
    }

    if not Path(cachedir).exists():
        warnings.warn("Specified cache directory does not exist, will attempt to create it.")
        Path(cachedir).mkdir(exist_ok=True, parents=True)

    return figs, device, bfig_kwargs


def load_sdss_data(cfg):
    frame: Union[SDSSFrame, SimulatedFrame] = instantiate(cfg.plots.frame)
    photo_cat = PhotoFullCatalog.from_file(**cfg.plots.photo_catalog)
    return frame, photo_cat


def make_autoencoder_figure(cfg, bfig_kwargs, device):
    print("INFO: Creating autoencoder figures...")
    autoencoder = instantiate(cfg.models.galaxy_net)
    autoencoder.load_state_dict(torch.load(cfg.models.prior.galaxy_prior.autoencoder_ckpt))
    autoencoder = autoencoder.to(device).eval()

    # generate galsim simulated galaxies images if file does not exist.
    galaxies_file = Path(cfg.plots.simulated_sdss_individual_galaxies)
    if not galaxies_file.exists() or cfg.plots.overwrite:
        print(f"INFO: Generating individual galaxy images and saving to: {galaxies_file}")
        dataset = instantiate(
            cfg.datasets.sdss_galaxies, batch_size=512, n_batches=20, num_workers=20
        )
        imagepath = galaxies_file.parent / (galaxies_file.stem + "_images.png")
        generate.generate(
            dataset, galaxies_file, imagepath, n_plots=25, global_params=("background", "slen")
        )

    # create figure classes and plot.
    ae_figures = AEReconstructionFigures(n_examples=5, **bfig_kwargs)
    ae_figures.save_figures(
        autoencoder, galaxies_file, cfg.plots.psf_file, cfg.plots.sdss_pixel_scale
    )
    mpl.rc_file_defaults()


def make_blend_sim_figure(cfg, encoder, decoder, bfig_kwargs):
    print("INFO: Creating figures for metrics on simulated blended galaxies.")
    blend_file = Path(cfg.plots.simulated_blended_galaxies)

    # create dataset of blends if not existant.
    if not blend_file.exists() or cfg.plots.overwrite:
        print(f"INFO: Creating dataset of simulated galsim blends and saving to {blend_file}")
        dataset = instantiate(cfg.plots.galsim_blended_galaxies)
        imagepath = blend_file.parent / (blend_file.stem + "_images.png")
        global_params = ("background", "slen")
        generate.generate(dataset, blend_file, imagepath, n_plots=25, global_params=global_params)

    blend_fig = BlendSimFigures(**bfig_kwargs)
    blend_fig.save_figures(blend_file, encoder, decoder)


@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    figs, device, bfig_kwargs = setup(cfg)
    encoder, decoder = load_models(cfg, device)
    frame, photo_cat = load_sdss_data(cfg)

    # FIGURE 1: Autoencoder single galaxy reconstruction
    if 1 in figs:
        make_autoencoder_figure(cfg, bfig_kwargs, device)

    # FIGURE 2: Classification and Detection metrics
    if 2 in figs:
        print("INFO: Creating classification and detection metrics from SDSS frame figures...")
        dc_fig = DetectionClassificationFigures(**bfig_kwargs)
        dc_fig.save_figures(frame, photo_cat, encoder, decoder)
        mpl.rc_file_defaults()

    # FIGURE 3: Reconstructions on SDSS
    if 3 in figs:
        print("INFO: Creating reconstructions from SDSS figures...")
        sdss_rec_fig = SDSSReconstructionFigures(cfg.plots.scenes, **bfig_kwargs)
        sdss_rec_fig.save_figures(frame, encoder, decoder)
        mpl.rc_file_defaults()

    if 4 in figs:
        make_blend_sim_figure(cfg, encoder, decoder, bfig_kwargs)

    if not figs.intersection({1, 2, 3, 4}):
        raise NotImplementedError(
            "No figures were created, `cfg.plots.figs` should be a subset of [1,2,3,4]."
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
