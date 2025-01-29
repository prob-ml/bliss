#!/usr/bin/env python3
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
import typer

from bliss.encoders.binary import BinaryEncoder
from bliss.encoders.deblend import GalaxyEncoder
from bliss.encoders.detection import DetectionEncoder
from experiment.scripts_figures.binary_figures import BinaryFigures
from experiment.scripts_figures.deblend_figures import DeblendingFigures
from experiment.scripts_figures.detection_figures import BlendDetectionFigures
from experiment.scripts_figures.toy_figures import ToySeparationFigure

warnings.filterwarnings("ignore", category=FutureWarning)

ALL_FIGS = ("detection", "binary", "deblend", "toy")
CACHEDIR = "/nfs/turbo/lsa-regier/scratch/ismael/cache/"


def _load_models(fpaths: dict[str, str], model: str, device):
    """Load models required for producing results."""

    if model == "detection":
        detection_fpath = fpaths["detection"]
        detection = DetectionEncoder().to(device).eval()
        detection.load_state_dict(
            torch.load(detection_fpath, map_location=device, weights_only=True)
        )
        detection.requires_grad_(False)
        return detection

    elif model == "binary":
        binary_fpath = fpaths["binary"]
        binary = BinaryEncoder().to(device).eval()
        binary.load_state_dict(torch.load(binary_fpath, map_location=device, weights_only=True))
        binary.requires_grad_(False)
        return binary

    elif model == "deblend":
        deblend_fpath = fpaths["deblend"]
        ae_fpath = fpaths["ae"]

        deblender = GalaxyEncoder(ae_fpath).to(device).eval()
        deblender.load_state_dict(torch.load(deblend_fpath, map_location=device, weights_only=True))
        deblender.requires_grad_(False)
        return deblender

    raise ValueError(f"model name {model} is not recognized.")


def _make_detection_figure(
    fpaths: dict[str, str],
    test_file: str,
    *,
    aperture: float,
    suffix: str,
    overwrite: bool,
    device: torch.device,
):
    print("INFO: Creating figures for detection encoder performance simulated blended galaxies.")
    _init_kwargs = {
        "overwrite": overwrite,
        "figdir": "figures",
        "suffix": suffix,
        "cachedir": CACHEDIR,
        "aperture": aperture,
    }
    detection = _load_models(fpaths, "detection", device)
    BlendDetectionFigures(**_init_kwargs)(ds_path=test_file, detection=detection)


def _make_deblend_figures(
    fpaths: dict[str, str],
    test_file: str,
    *,
    aperture: float,
    suffix: str,
    overwrite: bool,
    device: torch.device,
):
    print("INFO: Creating figures for detection encoder performance simulated blended galaxies.")
    _init_kwargs = {
        "overwrite": overwrite,
        "figdir": "figures",
        "suffix": suffix,
        "cachedir": CACHEDIR,
        "aperture": aperture,
    }
    deblend = _load_models(fpaths, "deblend", device)
    DeblendingFigures(**_init_kwargs)(ds_path=test_file, deblend=deblend)


def _make_binary_figures(
    fpaths: dict[str, str],
    test_file: str,
    *,
    aperture: float,
    suffix: str,
    overwrite: bool,
    device: torch.device,
):
    print("INFO: Creating figures for detection encoder performance simulated blended galaxies.")
    _init_kwargs = {
        "overwrite": overwrite,
        "figdir": "figures",
        "suffix": suffix,
        "cachedir": CACHEDIR,
        "aperture": aperture,
    }
    binary = _load_models(fpaths, "binary", device)
    BinaryFigures(**_init_kwargs)(ds_path=test_file, binary=binary)


def _make_toy_figures(
    fpaths: dict[str, str],
    *,
    aperture: float,
    suffix: str,
    overwrite: bool,
    device: torch.device,
):
    print("INFO: Creating figures for detection encoder performance simulated blended galaxies.")
    _init_kwargs = {
        "overwrite": overwrite,
        "figdir": "figures",
        "suffix": suffix,
        "cachedir": CACHEDIR,
        "aperture": aperture,
    }
    detection = _load_models(fpaths, "detection", device)
    deblender = _load_models(fpaths, "deblend", device)
    ToySeparationFigure(**_init_kwargs)(detection=detection, deblender=deblender)


def main(
    mode: str,
    suffix: str,
    seed: int = 42,
    aperture: float = 5.0,
    test_file_single: str = "",
    test_file_blends: str = "",
    detection_fpath: str = "",
    deblend_fpath: str = "",
    binary_fpath: str = "",
    ae_fpath: str = "",
    overwrite: bool = False,
):
    assert mode in ALL_FIGS

    device = torch.device("cuda:0")
    pl.seed_everything(seed)

    fpaths = {
        "detection": detection_fpath,
        "binary": binary_fpath,
        "deblend": deblend_fpath,
        "ae": ae_fpath,
    }

    if mode == "detection":
        assert test_file_blends != "" and Path(test_file_blends).exists()
        assert detection_fpath != "" and Path(detection_fpath).exists()
        _make_detection_figure(
            fpaths,
            test_file_blends,
            suffix=suffix,
            overwrite=overwrite,
            aperture=aperture,
            device=device,
        )

    elif mode == "deblend":
        assert test_file_blends != "" and Path(test_file_blends).exists()
        assert deblend_fpath != "" and Path(deblend_fpath).exists()
        assert ae_fpath != "" and Path(ae_fpath).exists(), "Need to provide AE when deblending."
        _make_deblend_figures(
            fpaths,
            test_file_blends,
            suffix=suffix,
            overwrite=overwrite,
            aperture=aperture,
            device=device,
        )

    elif mode == "binary":
        assert test_file_blends != "" and Path(test_file_blends).exists()
        assert binary_fpath != "" and Path(binary_fpath).exists()
        _make_binary_figures(
            fpaths,
            test_file_blends,
            suffix=suffix,
            overwrite=overwrite,
            aperture=aperture,
            device=device,
        )

    if mode == "toy":
        assert detection_fpath != "" and Path(detection_fpath).exists()
        assert deblend_fpath != "" and Path(deblend_fpath).exists()
        assert ae_fpath != "" and Path(ae_fpath).exists(), "Need to provide AE when deblending."
        _make_toy_figures(
            fpaths, aperture=aperture, suffix=suffix, overwrite=overwrite, device=device
        )

    else:
        raise NotImplementedError("The requred figure has not been implemented.")


if __name__ == "__main__":
    typer.run(main)
