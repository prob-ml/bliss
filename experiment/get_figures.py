#!/usr/bin/env python3
import warnings
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

warnings.filterwarnings("ignore", category=FutureWarning)

ALL_FIGS = {"single", "blend", "toy"}


def _load_models(seed: int, device):
    """Load models required for producing results."""

    # encoders
    detection = DetectionEncoder().to(device).eval()
    detection.load_state_dict(
        torch.load(f"models/detection_{seed}.pt", map_location=device, weights_only=True)
    )
    detection.requires_grad_(False)

    binary = BinaryEncoder().to(device).eval()
    binary.load_state_dict(
        torch.load(f"models/binary_{seed}.pt", map_location=device, weights_only=True)
    )
    binary.requires_grad_(False)

    deblender = GalaxyEncoder("models/autoencoder.pt")
    deblender.load_state_dict(
        torch.load(f"models/deblend_{seed}.pt", map_location=device, weights_only=True)
    )
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
    ae.load_state_dict(torch.load(f"models/autoencoder_{seed}.pt", map_location=device))
    decoder = deepcopy(ae.dec)
    decoder.requires_grad_(False)
    decoder = decoder.eval()
    del ae

    return encoder, decoder


def _make_autoencoder_figures(seed: int, device, test_file: str, overwrite: bool):
    print("INFO: Creating autoencoder figures...")
    autoencoder = OneCenteredGalaxyAE()
    autoencoder.load_state_dict(torch.load(f"models/autoencoder_{seed}.pt", weights_only=True))
    autoencoder = autoencoder.to(device).eval()
    autoencoder.requires_grad_(False)

    # arguments for figures
    args = (autoencoder, test_file)

    # create figure classes and plot.
    cachedir = "/nfs/turbo/lsa-regier/scratch/ismael/cache/"
    AutoEncoderFigures(n_examples=5, overwrite=overwrite, figdir="figures", cachedir=cachedir)(
        *args
    )


def _make_blend_figures(encoder, decoder, test_file: str, overwrite: bool):
    print("INFO: Creating figures for metrics on simulated blended galaxies.")
    cachedir = "/nfs/turbo/lsa-regier/scratch/ismael/cache/"
    BlendSimulationFigure(overwrite=overwrite, figdir="figures", cachedir=cachedir)(
        test_file, encoder, decoder
    )


@click.command()
@click.option("-m", "--mode", required=True, type=str, help="Which figure to make")
@click.option("-s", "--seed", required=True, type=int, help="Consistent seed used to train models.")
@click.option("--test-file-single", default="", type=str, help="Dataset file for testing AE.")
@click.option("--test-file-blends", default="", type=str, help="Dataset file for testing Encoders.")
@click.option("-o", "--overwrite", is_flag=True, default=False, help="Whether to overwrite cache.")
def main(mode: str, seed: int, test_file_single: str, test_file_blends: str, overwrite: bool):
    assert mode in ALL_FIGS

    device = torch.device("cuda:0")
    pl.seed_everything(seed)

    # FIGURE 1: Autoencoder single galaxy reconstruction
    if mode == "single":
        assert test_file_single != "" and Path(test_file_single).exists()
        _make_autoencoder_figures(seed, device, test_file_single, overwrite)

    if mode in {"toy", "blend"}:
        encoder, decoder = _load_models(seed, device)

    if mode == "blend":
        assert test_file_blends != "" and Path(test_file_blends).exists()
        _make_blend_figures(encoder, decoder, test_file_blends, overwrite)

    if mode == "toy":
        print("INFO: Creating figures for testing BLISS on pair galaxy toy example.")
        cachedir = "/nfs/turbo/lsa-regier/scratch/ismael/cache/"
        ToySeparationFigure(overwrite=overwrite, figdir="figures", cachedir=cachedir)(
            encoder, decoder
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
