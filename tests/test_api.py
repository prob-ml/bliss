import os
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from bliss.api import BlissClient
from bliss.utils.download_utils import download_git_lfs_file


@pytest.fixture(scope="session")
def cwd(tmpdir_factory):
    return tmpdir_factory.mktemp("cwd")


@pytest.fixture(scope="class")
def bliss_client(cwd, cfg):
    client = BlissClient(str(cwd))
    # Hack to apply select conftest.py overrides, since client.base_cfg should be private
    overrides = {
        "training.trainer.accelerator": cfg.training.trainer.accelerator,
        "predict.device": cfg.predict.device,
    }
    for k, v in overrides.items():
        OmegaConf.update(client.base_cfg, k, v)
    return client


@pytest.fixture(scope="class")
def cached_data_path_api(bliss_client):
    bliss_client.cached_data_path = bliss_client.cwd + "/data/cached_dataset"
    bliss_client.generate(n_batches=3, batch_size=8, max_images_per_file=16)
    return bliss_client.cached_data_path


def download_pretrained_weights(bliss_client, cfg, filename):
    # Only test downloading pretrained weights (from Git LFS) if run via GitHub Actions
    if os.environ.get("GITHUB_TOKEN") is None:
        # Run locally, so use pretrained weights from local BLISS_HOME
        # Copy pretrained weights to {cwd}/data/pretrained_models
        local_pretrained_weights_path = (
            Path(cfg.paths.data) / "pretrained_models/zscore_five_band.pt"
        )
        test_pretrained_weights_path = Path(bliss_client.cwd) / f"data/pretrained_models/{filename}"
        test_pretrained_weights_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(local_pretrained_weights_path), str(test_pretrained_weights_path))
    else:
        bliss_client.load_pretrained_weights_for_survey(
            survey="zscore_five_band",  # NOTE: temporary fix to run API tests - better name?
            filename=filename,
            request_headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"},
        )
    not_found_err_msg = (
        "pretrained weights "
        + f"{bliss_client.cwd}/data/pretrained_models/{filename} "
        + "not found"
    )
    assert Path(
        bliss_client.cwd + f"/data/pretrained_models/{filename}"
    ).exists(), not_found_err_msg


@pytest.fixture(scope="class")
def pretrained_weights_filename(bliss_client, cfg):
    filename = "sdss_pretrained.pt"
    download_pretrained_weights(bliss_client, cfg, filename)
    return filename


@pytest.fixture(scope="class")
def weight_save_path(bliss_client, pretrained_weights_filename):
    weight_save_path = "tutorial_encoder/0_fixture.pt"
    bliss_client.train_on_cached_data(
        weight_save_path=weight_save_path,
        train_n_batches=1,
        batch_size=8,
        val_split_file_idxs=[1],
        test_split_file_idxs=[1],
        pretrained_weights_filename=pretrained_weights_filename,
        training={"n_epochs": 1, "trainer": {"check_val_every_n_epoch": 1, "log_every_n_steps": 1}},
    )
    return weight_save_path


@pytest.mark.usefixtures(
    "bliss_client", "cached_data_path_api", "pretrained_weights_filename", "weight_save_path"
)
class TestApi:
    def test_get_dataset_file(self, bliss_client, cached_data_path_api):
        bliss_client.cached_data_path = cached_data_path_api
        dataset0 = bliss_client.get_dataset_file(filename="dataset_0.pt")
        assert isinstance(dataset0, list), "dataset0 must be a list"

    def test_predict_sdss_default_rcf(self, bliss_client, weight_save_path, cfg):
        # If run via GitHub Actions, download from our Git LFS to avoid having to connect to
        # SDSS remote server; else download from SDSS remote server
        if os.environ.get("GITHUB_TOKEN") is not None:
            download_rcf_from_git(run=94, camcol=1, field=12, cwd=bliss_client.cwd)
        bliss_client.predict_sdss(weight_save_path=weight_save_path)

        # This function call internally requires DECaLS catalog data, so download from Git LFS if
        # authenticated via GitHub Actions, else copy from local BLISS_HOME
        if os.environ.get("GITHUB_TOKEN") is not None:
            download_decals_base_from_git(bliss_client.cwd + "/data/decals")
        else:
            local_decals_base_path = Path(cfg.paths.data) / "decals"
            test_decals_base_path = Path(bliss_client.cwd) / "data/decals"
            shutil.copytree(
                str(local_decals_base_path), str(test_decals_base_path), dirs_exist_ok=True
            )
        bliss_client.plot_predictions_in_notebook()


def download_decals_base_from_git(download_dir: str):
    assert os.environ["GITHUB_TOKEN"] is not None, "GITHUB_TOKEN environment variable not found"

    tractor_filename = "tractor-3366m010.fits"
    cutout_filename = "cutout_336.635_-0.9600.fits"
    cutout = download_git_lfs_file(
        f"https://api.github.com/repos/prob-ml/bliss/contents/data/decals/{cutout_filename}",
        headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"},
    )
    tractor = download_git_lfs_file(
        f"https://api.github.com/repos/prob-ml/bliss/contents/data/decals/{tractor_filename}",
        headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"},
    )
    cutout_path = Path(download_dir) / cutout_filename
    tractor_path = Path(download_dir) / tractor_filename
    cutout_path.write_bytes(cutout)
    tractor_path.write_bytes(tractor)


def download_rcf_from_git(run, camcol, field, cwd):
    run_stripped = str(run).lstrip("0")
    field_stripped = str(field).lstrip("0")
    run6 = f"{int(run_stripped):06d}"
    field4 = f"{int(field_stripped):04d}"

    base_url = f"https://api.github.com/repos/prob-ml/bliss/contents/data/sdss/{run}/{camcol}"
    save_path_rc = Path(cwd) / f"data/sdss/{run}/{camcol}"
    save_path_rcf = save_path_rc / f"{field}"
    save_path_rcf.mkdir(parents=True, exist_ok=True)

    assert os.environ["GITHUB_TOKEN"] is not None, "GITHUB_TOKEN environment variable not found"
    headers = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}
    pf = download_git_lfs_file(f"{base_url}/photoField-{run6}-{camcol}.fits", headers)
    po = download_git_lfs_file(
        f"{base_url}/{field}/photoObj-{run6}-{camcol}-{field4}.fits", headers
    )
    psf = download_git_lfs_file(
        f"{base_url}/{field}/psField-{run6}-{camcol}-{field4}.fits", headers
    )
    save_file(f"{save_path_rc}/photoField-{run6}-{camcol}.fits", pf)
    save_file(f"{save_path_rc}/{field}/photoObj-{run6}-{camcol}-{field4}.fits", po)
    save_file(f"{save_path_rc}/{field}/psField-{run6}-{camcol}-{field4}.fits", psf)

    for band in "ugriz":
        fpm = download_git_lfs_file(
            f"{base_url}/{field}/fpM-{run6}-{band}{camcol}-{field4}.fits", headers
        )
        frame = download_git_lfs_file(
            f"{base_url}/{field}/frame-{band}-{run6}-{camcol}-{field4}.fits", headers
        )
        save_file(f"{save_path_rcf}/fpM-{run6}-{band}{camcol}-{field4}.fits", fpm)
        save_file(f"{save_path_rcf}/frame-{band}-{run6}-{camcol}-{field4}.fits", frame)


def save_file(file_loc, data):
    with open(file_loc, "wb") as f:
        f.write(data)
