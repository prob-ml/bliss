import base64
import subprocess  # noqa: S404
from pathlib import Path
from typing import Literal

import requests
import torch
from astropy import units as u
from astropy.table import Table
from omegaconf import OmegaConf

from bliss.catalog import FullCatalog
from bliss.conf.igs import base_config
from bliss.generate import generate as _generate
from bliss.predict import predict_sdss as _predict_sdss
from bliss.surveys import sdss_download
from bliss.train import train as _train

SurveyType = Literal["decals", "hst", "lsst", "sdss"]


def generate(
    n_batches: int,
    batch_size: int,
    max_images_per_file: int,
    cached_data_path: str,
    **kwargs,
) -> None:
    """Generate and cache simulated images to disk.

    Args:
        n_batches (int): Number of batches to simulate.
        batch_size (int): Number of images per batch.
        max_images_per_file (int): Number of images per cached file.
        cached_data_path (str): Path to directory where cached data will be stored.
        **kwargs: Keyword arguments to override default configuration values.
    """  # noqa: RST210
    cfg = base_config()
    # apply overrides
    cfg.generate.n_batches = n_batches
    cfg.generate.batch_size = batch_size
    cfg.generate.max_images_per_file = max_images_per_file
    cfg.generate.cached_data_path = cached_data_path
    for k, v in kwargs.items():
        OmegaConf.update(cfg, k, v)

    _generate(cfg)


def _download_git_lfs_file(url) -> bytes:
    """Download a file from git-lfs.

    Args:
        url (str): URL to git-lfs file.

    Returns:
        bytes: File contents.
    """
    ptr_file = requests.get(url, timeout=10)
    ptr = ptr_file.json()
    ptr_sha = ptr["sha"]

    blob_file = requests.get(
        f"https://api.github.com/repos/prob-ml/bliss/git/blobs/{ptr_sha}", timeout=10
    )
    blob = blob_file.json()
    blob_content = blob["content"]
    assert blob["encoding"] == "base64"

    blob_decoded = base64.b64decode(blob_content).decode("utf-8").split("\n")
    sha = blob_decoded[1].split(" ")[1].split(":")[1]
    size = int(blob_decoded[2].split(" ")[1])

    lfs_ptr_file = requests.post(
        "https://github.com/prob-ml/bliss.git/info/lfs/objects/batch",
        headers={
            "Accept": "application/vnd.git-lfs+json",
            # Already added when you pass json=
            # 'Content-type': 'application/json',
        },
        json={
            "operation": "download",
            "transfer": ["basic"],
            "objects": [
                {
                    "oid": sha,
                    "size": size,
                }
            ],
        },
        timeout=10,
    )
    lfs_ptr = lfs_ptr_file.json()
    lfs_ptr_download_url = lfs_ptr["objects"][0]["actions"]["download"]["href"]  # noqa: WPS219

    # Get and write weights to pretrained weights path
    file = requests.get(lfs_ptr_download_url, timeout=10)
    return file.content


def load_pretrained_weights_for_survey(survey: SurveyType, pretrained_weights_path) -> None:
    """Load pretrained weights for a survey.

    Args:
        survey (SurveyType): Survey to load pretrained weights for.
        pretrained_weights_path (str): Path to store pretrained weights.
    """
    weights = _download_git_lfs_file(
        f"https://api.github.com/repos/prob-ml/bliss/contents/data/pretrained_models/{survey}.pt"
    )
    with open(pretrained_weights_path, "wb+") as f:
        f.write(weights)


def train(weight_save_path, **kwargs) -> None:
    """Train model on simulated images.

    Args:
        weight_save_path (str): Path to directory where trained model weights will be stored.
        **kwargs: Keyword arguments to override default configuration values.
    """  # noqa: RST210
    cfg = base_config()
    # apply overrides
    cfg.training.weight_save_path = weight_save_path
    for k, v in kwargs.items():
        OmegaConf.update(cfg, k, v)

    _train(cfg)


def train_on_cached_data(
    weight_save_path,
    cached_data_path,
    train_n_batches,
    batch_size,
    val_split_file_idxs,
    **kwargs,
) -> None:
    """Train on cached data.

    Args:
        weight_save_path (str): Path to directory where trained model weights will be stored.
        cached_data_path (str): Path to directory where cached data is stored.
        train_n_batches (int): Number of batches to train on.
        batch_size (int): Number of images per batch.
        val_split_file_idxs (List[int]): List of file indices to use for validation.
        **kwargs: Keyword arguments to override default configuration values.
    """  # noqa: RST210
    cfg = base_config()
    cfg.training.use_cached_simulator = True
    # apply overrides
    cfg.training.weight_save_path = weight_save_path
    cfg.cached_simulator.cached_data_path = cached_data_path
    cfg.cached_simulator.train_n_batches = train_n_batches
    cfg.cached_simulator.batch_size = batch_size
    cfg.cached_simulator.val_split_file_idxs = val_split_file_idxs
    for k, v in kwargs.items():
        OmegaConf.update(cfg, k, v)

    _train(cfg)


def load_survey_via_makefile(survey: SurveyType, run, camcol, field, download_dir: Path):
    with subprocess.Popen(  # noqa: S603, S607
        [
            "make",
            f"RUN={run}",
            f"CAMCOL={camcol}",
            f"FIELD={field}",
            f"DOWNLOAD_DIR={download_dir}",
        ],
        stdout=subprocess.PIPE,
        cwd=Path(__file__).parent / "surveys",
    ) as process:
        assert process.wait() == 0
        # assert files downloaded at download_dir


def load_survey(survey: SurveyType, run, camcol, field, download_dir: Path):
    sdss_download.download_all(run, camcol, field, str(download_dir))
    # assert files downloaded at download_dir


def to_astropy_table(est_cat: FullCatalog):
    assert list(est_cat.keys()) == [
        "star_log_fluxes",
        "star_fluxes",
        "galaxy_bools",
        "star_bools",
        "galaxy_params",
    ]

    # Convert dictionary of tensors to list of dictionaries
    n = [torch.squeeze(v).shape[0] for v in est_cat.values()][0]  # number of rows
    est_cat_list = [{k: torch.squeeze(v)[i].cpu() for k, v in est_cat.items()} for i in range(n)]

    # Convert to astropy table
    est_cat_table = Table(est_cat_list)
    # Convert all _fluxes columns to u.Quantity
    log_nmgy = u.LogUnit(u.nmgy)
    est_cat_table["star_log_fluxes"] = est_cat_table["star_log_fluxes"] * log_nmgy
    est_cat_table["star_fluxes"] = est_cat_table["star_fluxes"] * u.nmgy

    # Create inner table for galaxy_params
    # Convert list of tensors to list of dictionaries
    galaxy_params_names = [
        "galaxy_flux",
        "galaxy_disk_frac",
        "galaxy_beta_radians",
        "galaxy_disk_q",
        "galaxy_a_d",
        "galaxy_bulge_q",
        "galaxy_a_b",
    ]
    galaxy_params_list = []
    for galaxy_params in est_cat_table["galaxy_params"]:
        galaxy_params_dic = {}
        for i, name in enumerate(galaxy_params_names):
            galaxy_params_dic[name] = galaxy_params[i]
        galaxy_params_dic["galaxy_flux"] = galaxy_params_dic["galaxy_flux"] * u.nmgy
        galaxy_params_dic["galaxy_disk_frac"] = (
            galaxy_params_dic["galaxy_disk_frac"] * u.dimensionless_unscaled
        )
        galaxy_params_dic["galaxy_beta_radians"] = (
            galaxy_params_dic["galaxy_beta_radians"] * u.radian
        )
        galaxy_params_dic["galaxy_disk_q"] = (
            galaxy_params_dic["galaxy_disk_q"] * u.dimensionless_unscaled
        )
        galaxy_params_dic["galaxy_a_d"] = galaxy_params_dic["galaxy_a_d"] * u.arcsec
        galaxy_params_dic["galaxy_bulge_q"] = (
            galaxy_params_dic["galaxy_bulge_q"] * u.dimensionless_unscaled
        )
        galaxy_params_dic["galaxy_a_b"] = galaxy_params_dic["galaxy_a_b"] * u.arcsec
        galaxy_params_list.append(galaxy_params_dic)
    galaxy_params_table = Table(galaxy_params_list)
    est_cat_table["galaxy_params"] = galaxy_params_table

    return est_cat_table, galaxy_params_table


def predict_sdss(data_path: str, weight_save_path: str, **kwargs):
    cfg = base_config()
    # apply overrides
    cfg.predict.dataset.sdss_dir = data_path
    cfg.predict.weight_save_path = weight_save_path
    cfg.paths.sdss = cfg.predict.dataset.sdss_dir
    for k, v in kwargs.items():
        OmegaConf.update(cfg, k, v)

    # download survey images if not already downloaded
    run, camcol, field = (
        cfg.predict.dataset.run,
        cfg.predict.dataset.camcol,
        cfg.predict.dataset.fields[0],
    )
    bands = ["u", "g", "r", "i", "z"]
    for band in bands:
        sdss_data_file_path = (
            Path(cfg.predict.dataset.sdss_dir)
            / f"{run}"
            / f"{camcol}"
            / f"{field}"
            / f"frame-{band}-{'{:06d}'.format(run)}-{camcol}-{'{:04d}'.format(field)}.fits"
        )
        if not sdss_data_file_path.exists():
            download_dir = Path(cfg.predict.dataset.sdss_dir)
            load_survey("sdss", run, camcol, field, download_dir)
            break

    est_cat, _, _, _ = _predict_sdss(cfg)
    est_cat_table, galaxy_params_table = to_astropy_table(est_cat)
    return est_cat, est_cat_table, galaxy_params_table
