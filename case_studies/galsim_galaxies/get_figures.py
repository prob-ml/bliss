#!/usr/bin/env python3
from copy import deepcopy
from pathlib import Path

import click
import pytorch_lightning as pl
import torch

from bliss.encoders.autoencoder import OneCenteredGalaxyAE
from bliss.encoders.binary import BinaryEncoder
from bliss.encoders.deblend import GalaxyEncoder
from bliss.encoders.detection import DetectionEncoder
from bliss.encoders.encoder import Encoder
from bliss.encoders.layers import ConcatBackgroundTransform
from case_studies.galsim_galaxies.scripts_figures.ae_figures import AutoEncoderFigures
from case_studies.galsim_galaxies.scripts_figures.blend_figures import BlendSimulationFigure
from case_studies.galsim_galaxies.scripts_figures.toy_figures import ToySeparationFigure

ALL_FIGS = ("single_gal", "blend_gal", "toy")

pl.seed_everything(42)


def _load_models(device):
    """Load models required for producing results."""

    # encoders
    input_transform = ConcatBackgroundTransform()

    detection = DetectionEncoder(input_transform).to(device).eval()
    detection.load_state_dict(torch.load("models/detection.pt", map_location=device))
    detection.requires_grad_(False)

    binary = BinaryEncoder(input_transform).to(device).eval()
    binary.load_state_dict(torch.load("models/binary.pt", map_location=device))
    binary.requires_grad_(False)

    deblender = GalaxyEncoder("models/autoencoder.pt")
    deblender.load_state_dict(torch.load("models/deblend.pt", map_location=device))
    deblender.requires_grad_(False)

    encoder = Encoder(
        detection.eval(),
        binary.eval(),
        deblender.eval(),
        n_images_per_batch=20,
        n_rows_per_batch=30,
    )
    encoder = encoder.to(device)

    # decoder
    ae = OneCenteredGalaxyAE().to(device).eval()
    ae.load_state_dict(torch.load("models/autoencoder.pt", map_location=device))
    decoder = deepcopy(ae.dec)
    decoder.requires_grad_(False)
    decoder = decoder.eval()
    del ae

    return encoder, decoder


def _make_autoencoder_figures(device, overwrite: bool):
    print("INFO: Creating autoencoder figures...")
    autoencoder = OneCenteredGalaxyAE()
    autoencoder.load_state_dict(torch.load("models/autoencoder.pt"))
    autoencoder = autoencoder.to(device).eval()
    autoencoder.requires_grad_(False)
    galaxies_file = "data/single_galaxies_test.pt"

    # arguments for figures
    args = (autoencoder, galaxies_file)

    # create figure classes and plot.
    AutoEncoderFigures(n_examples=5, overwrite=overwrite, figdir="figures", cachedir="data")(*args)


def _make_blend_figures(encoder, decoder, overwrite: bool):
    print("INFO: Creating figures for metrics on simulated blended galaxies.")
    blend_file = Path("data/blends_test.pt")
    BlendSimulationFigure(overwrite=overwrite, figdir="figures", cachedir="data")(
        blend_file, encoder, decoder
    )


@click.command()
@click.option("-m", "--mode", required=True, type=str, help="which figure to make")
@click.option("-o", "--overwrite", is_flag=True, default=False)
def main(mode: str, overwrite: bool):
    assert mode in {"single_gal", "toy", "blend_gal"}
    device = torch.device("cuda:0")

    # FIGURE 1: Autoencoder single galaxy reconstruction
    if mode == "single_gal":
        _make_autoencoder_figures(device, overwrite)

    if mode in {"toy", "blend_gal"}:
        encoder, decoder = _load_models(device)

    if mode == "blend_gal":
        _make_blend_figures(encoder, decoder, overwrite)

    if mode == "toy":
        print("INFO: Creating figures for testing BLISS on pair galaxy toy example.")
        ToySeparationFigure(overwrite=overwrite, figdir="figures", cachedir="data")(
            encoder, decoder
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
