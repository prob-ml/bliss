import base64
from typing import Literal

import requests
from astropy import coordinates as coords
from astroquery.sdss import SDSS
from omegaconf import OmegaConf

from bliss.conf.igs import base_config
from bliss.generate import generate as _generate
from bliss.predict import predict_sdss as _predict_sdss
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


def load_survey(survey: SurveyType):
    pos = coords.SkyCoord("0h8m05.63s +14d50m23.3s", frame="icrs")
    xid = SDSS.query_region(pos, radius="5 arcsec", spectro=True)  # pylint: disable=E1101
    print(xid)
    sp = SDSS.get_spectra(matches=xid)
    im = SDSS.get_images(matches=xid, band="g")
    return sp, im


def predict():
    cfg = base_config()

    _predict_sdss(cfg)
