import bz2
import gzip
from pathlib import Path

import pytest
import requests
from mock_tests import mock_generate, mock_get, mock_post, mock_predict_sdss, mock_train
from omegaconf import OmegaConf

from bliss import api
from bliss.api import BlissClient


@pytest.fixture(scope="class")
def bliss_client(cfg, tmpdir_factory):
    client = BlissClient(str(tmpdir_factory.mktemp("cwd")))
    # Hack to apply select conftest.py overrides, since client.base_cfg should be private
    overrides = {
        "training.trainer.accelerator": cfg.training.trainer.accelerator,
        "predict.device": cfg.predict.device,
    }
    for k, v in overrides.items():
        OmegaConf.update(client.base_cfg, k, v)
    return client


@pytest.mark.usefixtures("bliss_client")
class TestApi:
    def test_get_dataset_file(self, cfg, bliss_client):
        bliss_client.cached_data_path = cfg.paths.data + "/tests/multiband_data"
        dataset0 = bliss_client.get_dataset_file(filename="dataset_0.pt")
        assert isinstance(dataset0, list), "dataset0 must be a list"

    def test_generate(self, bliss_client, monkeypatch):
        monkeypatch.setattr(api, "_generate", mock_generate)
        bliss_client.generate(n_batches=2, batch_size=1, max_images_per_file=2)
        assert Path(bliss_client.cached_data_path).exists()

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

    def test_train_on_cached_data(self, cfg, bliss_client, monkeypatch):
        monkeypatch.setattr(api, "_train", mock_train)
        bliss_client.train_on_cached_data("test.pt", 4, 4, None, None, "test_pretrained.pt")

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

    def test_predict_sdss_default_rcf(self, cfg, bliss_client, monkeypatch):
        monkeypatch.setattr(api, "_predict_sdss", mock_predict_sdss)
        # cached predict data stored at cfg.paths.data, copied to temp dir in mock_predict_sdss
        bliss_client.predict_sdss("test_path", paths={"data": cfg.paths.data})
        bliss_client.plot_predictions_in_notebook()
