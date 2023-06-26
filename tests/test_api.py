import bz2
import gzip
from pathlib import Path

import pytest
import requests
from mock_tests import mock_get, mock_post, mock_train
from omegaconf import OmegaConf

from bliss import api
from bliss.api import BlissClient


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
    bliss_client.generate(n_batches=3, batch_size=5, max_images_per_file=10)
    return bliss_client.cached_data_path


@pytest.fixture(scope="class")
def weight_save_path(bliss_client, cfg):
    """Train model for 1 epoch and return path to saved model."""
    weight_save_path = "tutorial_encoder/0_fixture.pt"
    bliss_client.train_on_cached_data(
        weight_save_path=weight_save_path,
        train_n_batches=1,
        batch_size=5,
        val_split_file_idxs=[1],
        training={
            "n_epochs": 1,
            "trainer": {"check_val_every_n_epoch": 1, "log_every_n_steps": 1},
            "pretrained_weights": cfg.predict.weight_save_path,
        },
    )
    return weight_save_path


@pytest.mark.usefixtures("bliss_client", "cached_data_path_api", "weight_save_path")
class TestApi:
    def test_get_dataset_file(self, bliss_client, cached_data_path_api):
        bliss_client.cached_data_path = cached_data_path_api
        dataset0 = bliss_client.get_dataset_file(filename="dataset_0.pt")
        assert isinstance(dataset0, list), "dataset0 must be a list"

    def test_load_pretrained_weights(self, bliss_client, monkeypatch):
        monkeypatch.setattr(requests, "get", mock_get)
        monkeypatch.setattr(requests, "post", mock_post)

        filename = "sdss_pretrained.pt"
        bliss_client.load_pretrained_weights_for_survey(
            survey="sdss",
            filename=filename,
            request_headers={"Authorization": "test"},
        )
        not_found_err_msg = (
            "pretrained weights "
            + f"{bliss_client.cwd}/data/pretrained_models/{filename} "
            + "not found"
        )
        assert Path(
            bliss_client.cwd + f"/data/pretrained_models/{filename}"
        ).exists(), not_found_err_msg

    def test_train(self, bliss_client, monkeypatch):
        monkeypatch.setattr(api, "_train", mock_train)
        bliss_client.train("test", training={"n_epochs": 1})

    def test_load_survey(self, bliss_client, monkeypatch):
        monkeypatch.setattr(requests, "get", mock_get)
        monkeypatch.setattr(requests, "post", mock_post)
        monkeypatch.setattr(gzip, "decompress", lambda x: x)
        monkeypatch.setattr(bz2, "decompress", lambda x: x)

        download_dir = "data"
        bliss_client.load_survey("sdss", 94, 1, 12, download_dir)
        assert Path(f"{bliss_client.cwd}/{download_dir}/94/1/photoField-000094-1.fits").exists()
        assert Path(
            f"{bliss_client.cwd}/{download_dir}/94/1/12/photoObj-000094-1-0012.fits"
        ).exists()
        assert Path(
            f"{bliss_client.cwd}/{download_dir}/94/1/12/frame-r-000094-1-0012.fits"
        ).exists()
        assert Path(f"{bliss_client.cwd}/{download_dir}/94/1/12/fpM-000094-r1-0012.fits").exists()
        assert Path(
            f"{bliss_client.cwd}/{download_dir}/94/1/12/psField-000094-1-0012.fits"
        ).exists()

    def test_predict_sdss_default_rcf(self, bliss_client, weight_save_path, cfg):
        paths = {"sdss": cfg.paths.sdss, "decals": cfg.paths.decals}
        bliss_client.predict_sdss(weight_save_path=weight_save_path, paths=paths)
        bliss_client.plot_predictions_in_notebook()
