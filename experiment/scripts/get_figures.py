#!/usr/bin/env python3
import subprocess
import warnings

import pytorch_lightning as L
import torch
import typer

from bliss.encoders.binary import BinaryEncoder
from bliss.encoders.deblend import GalaxyEncoder
from bliss.encoders.detection import DetectionEncoder
from experiment import CACHE_DIR, DATASETS_DIR, FIGURE_DIR, MODELS_DIR, SEED
from experiment.scripts.figures.binary_figures import BinaryFigures
from experiment.scripts.figures.deblend_figures import DeblendingFigures
from experiment.scripts.figures.detection_figures import BlendDetectionFigures
from experiment.scripts.figures.toy_figures import ToySeparationFigure
from experiment.scripts.figures.toy_sampling_figures import ToySamplingFigure

warnings.filterwarnings("ignore", category=FutureWarning)

ALL_FIGS = ("detection", "binary", "deblend", "toy", "toy_samples", "samples")


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
    print("INFO: Creating figures for detection encoder performance on simulated blended galaxies.")
    _init_kwargs = {
        "overwrite": overwrite,
        "figdir": FIGURE_DIR,
        "suffix": suffix,
        "cachedir": CACHE_DIR,
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
    print("INFO: Creating figures for deblender performance on simulated blended galaxies.")
    _init_kwargs = {
        "overwrite": overwrite,
        "figdir": FIGURE_DIR,
        "suffix": suffix,
        "cachedir": CACHE_DIR,
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
    print("INFO: Creating figures for binary encoder performance on simulated blended galaxies.")
    _init_kwargs = {
        "overwrite": overwrite,
        "figdir": FIGURE_DIR,
        "suffix": suffix,
        "cachedir": CACHE_DIR,
        "aperture": aperture,
    }
    binary = _load_models(fpaths, "binary", device)
    BinaryFigures(**_init_kwargs)(ds_path=test_file, binary=binary)


def _make_toy_figures(
    fpaths: dict[str, str],
    *,
    suffix: str,
    overwrite: bool,
    device: torch.device,
):
    print("INFO: Creating figures for toy experiment.")
    _init_kwargs = {
        "overwrite": overwrite,
        "figdir": FIGURE_DIR,
        "suffix": suffix,
        "cachedir": CACHE_DIR,
    }
    detection = _load_models(fpaths, "detection", device)
    deblender = _load_models(fpaths, "deblend", device)
    ToySeparationFigure(**_init_kwargs)(detection=detection, deblender=deblender)


def _make_toy_sampling_figure(
    fpaths: dict[str, str],
    *,
    suffix: str,
    overwrite: bool,
    device: torch.device,
    aperture: float,
    toy_cache_fpath: str,
):
    print("INFO: Creating figures for toy sampling experiment.")
    _init_kwargs = {
        "overwrite": overwrite,
        "figdir": "figures",
        "suffix": suffix,
        "cachedir": CACHE_DIR,
        "aperture": aperture,
        "n_samples": 100,
    }
    detection = _load_models(fpaths, "detection", device)
    deblender = _load_models(fpaths, "deblend", device)
    ToySamplingFigure(**_init_kwargs)(
        detection=detection, deblender=deblender, toy_cache_fpath=toy_cache_fpath
    )


def main(
    mode: str = typer.Option(),
    aperture: float = 5.0,
    overwrite: bool = False,
):
    assert mode in ALL_FIGS

    device = torch.device("cuda:0")
    L.seed_everything(SEED)

    fpaths = {
        "ae": MODELS_DIR / f"autoencoder_{SEED}.pt",
        "detection": MODELS_DIR / f"detection_{SEED}.pt",
        "binary": MODELS_DIR / f"binary_{SEED}.pt",
        "deblend": MODELS_DIR / f"deblender_{SEED}.pt",
        "test_ds": DATASETS_DIR / f"test_ds_{SEED}.npz",
    }

    # for _, path in fpaths.items():
    # assert path.exists(), "Path does not exist."

    if mode == "detection":
        assert fpaths["detection"].exists()
        assert fpaths["test_ds"].exists()
        _make_detection_figure(
            fpaths,
            fpaths["test_ds"],
            suffix=str(SEED),
            overwrite=overwrite,
            aperture=aperture,
            device=device,
        )

    elif mode == "deblend":
        assert fpaths["detection"].exists()
        assert fpaths["test_ds"].exists()
        _make_deblend_figures(
            fpaths,
            fpaths["test_ds"],
            suffix=str(SEED),
            overwrite=overwrite,
            aperture=aperture,
            device=device,
        )

    elif mode == "binary":
        assert fpaths["binary"].exists()
        assert fpaths["test_ds"].exists()
        _make_binary_figures(
            fpaths,
            fpaths["test_ds"],
            suffix=str(SEED),
            overwrite=overwrite,
            aperture=aperture,
            device=device,
        )

    elif mode == "toy":
        assert fpaths["detection"].exists()
        assert fpaths["deblend"].exists()
        _make_toy_figures(fpaths, suffix=str(SEED), overwrite=overwrite, device=device)

    elif mode == "toy_samples":
        _make_toy_sampling_figure(
            fpaths,
            suffix=str(SEED),
            overwrite=overwrite,
            device=device,
            aperture=aperture,
            toy_cache_fpath=CACHE_DIR / f"toy_separation_{SEED}.pt",
        )

    elif mode == "samples":
        overwrite_txt = "--overwrite" if overwrite else "--no-overwrite"
        cmd = f"./scripts/figures/sampling_figures.py --seed {SEED} --overwrite {overwrite_txt}"
        subprocess.check_call(cmd, shell=True)

    else:
        raise NotImplementedError("The requred figure has not been implemented.")


if __name__ == "__main__":
    typer.run(main)
