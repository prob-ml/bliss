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
from experiment.scripts_figures.ae_figures import AutoEncoderFigures
from experiment.scripts_figures.blend_figures import BlendSimulationFigure
from experiment.scripts_figures.toy_figures import ToySeparationFigure

ALL_FIGS = {"single", "blend", "toy"}

pl.seed_everything(42)


def _load_models(device):
    """Load models required for producing results."""

    # encoders
    detection = DetectionEncoder().to(device).eval()
    detection.load_state_dict(torch.load("models/detection.pt", map_location=device))
    detection.requires_grad_(False)

    binary = BinaryEncoder().to(device).eval()
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


def _make_autoencoder_figures(device, tag: str, overwrite: bool):
    print("INFO: Creating autoencoder figures...")
    autoencoder = OneCenteredGalaxyAE()
    autoencoder.load_state_dict(torch.load("models/autoencoder.pt"))
    autoencoder = autoencoder.to(device).eval()
    autoencoder.requires_grad_(False)
    galaxies_file = f"/nfs/turbo/lsa-regier/scratch/ismael/datasets/test_ae_ds_{tag}.pt"

    # arguments for figures
    args = (autoencoder, galaxies_file)

    # create figure classes and plot.
    cachedir = "/nfs/turbo/lsa-regier/scratch/ismael/cache/"
    AutoEncoderFigures(n_examples=5, overwrite=overwrite, figdir="figures", cachedir=cachedir)(
        *args
    )


def _make_blend_figures(encoder, decoder, tag: str, overwrite: bool):
    print("INFO: Creating figures for metrics on simulated blended galaxies.")
    blend_file = Path(f"/nfs/turbo/lsa-regier/scratch/ismael/datasets/test_ds_{tag}.pt")
    cachedir = "/nfs/turbo/lsa-regier/scratch/ismael/cache/"
    BlendSimulationFigure(overwrite=overwrite, figdir="figures", cachedir=cachedir)(
        blend_file, encoder, decoder
    )


@click.command()
@click.option("-m", "--mode", required=True, type=str, help="which figure to make")
@click.option("-t", "--tag", default=None, type=str, help="dataset tag for testing")
@click.option("-o", "--overwrite", is_flag=True, default=False)
def main(mode: str, tag: str, overwrite: bool):
    assert mode in ALL_FIGS
    device = torch.device("cuda:0")

    # FIGURE 1: Autoencoder single galaxy reconstruction
    if mode == "single":
        assert tag is not None
        _make_autoencoder_figures(device, tag, overwrite)

    if mode in {"toy", "blend"}:
        encoder, decoder = _load_models(device)

    if mode == "blend":
        assert tag is not None
        _make_blend_figures(encoder, decoder, tag, overwrite)

    if mode == "toy":
        print("INFO: Creating figures for testing BLISS on pair galaxy toy example.")
        cachedir = "/nfs/turbo/lsa-regier/scratch/ismael/cache/"
        ToySeparationFigure(overwrite=overwrite, figdir="figures", cachedir=cachedir)(
            encoder, decoder
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
