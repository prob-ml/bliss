from pathlib import Path

import mock_tests
import pytest
from astropy.table import Table
from omegaconf import OmegaConf

from bliss import api, generate, train
from bliss.api import BlissClient
from bliss.catalog import FullCatalog


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
    """Basic tests for API functions like generate, train, and predict_sdss.

    Note that these tests are mostly focused on interface and output directory structure, instead
    of the general behavior of those functions (which is tested in other files).
    """

    def test_generate(self, bliss_client, monkeypatch):
        monkeypatch.setattr(generate, "instantiate", mock_tests.mock_simulator)
        monkeypatch.setattr(generate, "itemize_data", mock_tests.mock_itemize_data)
        bliss_client.generate(n_batches=4, batch_size=1, max_images_per_file=2)

        cached_data_path = bliss_client.cached_data_path
        assert Path(cached_data_path).exists()
        assert Path(f"{cached_data_path}/hparams.yaml").exists()
        assert Path(f"{cached_data_path}/dataset_0.pt").exists()
        assert Path(f"{cached_data_path}/dataset_1.pt").exists()
        assert not Path(f"{cached_data_path}/dataset2.pt").exists()

        # need generated data to test get_dataset_file
        dataset0 = bliss_client.get_dataset_file(filename="dataset_0.pt")
        assert isinstance(dataset0, list)

    def test_load_pretrained_weights(self, bliss_client, monkeypatch):
        monkeypatch.setattr("requests.get", mock_tests.mock_get)
        monkeypatch.setattr("requests.post", mock_tests.mock_post)

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
            f"{bliss_client.cwd}/data/pretrained_models/{filename}"
        ).exists(), not_found_err_msg

    def test_train(self, bliss_client, monkeypatch):
        monkeypatch.setattr(train, "instantiate", mock_tests.mock_trainer)
        monkeypatch.setattr(train, "setup_checkpoint_callback", mock_tests.mock_checkpoint_callback)
        bliss_client.train("test_train", training={"n_epochs": 1})

        model_path = f"{bliss_client.base_cfg.paths.output}/test_train"
        assert Path(model_path).exists()
        assert Path(f"{model_path}.log.json").exists()

    def test_train_on_cached_data(self, bliss_client):
        simulator_cfg = {"cached_data_path": "data/tests/multiband_data"}
        training_cfg = {
            "n_epochs": 1,
            "trainer": {"check_val_every_n_epoch": 1, "log_every_n_steps": 1},
        }
        weight_save_path = "tests/test_model.pt"
        bliss_client.train_on_cached_data(
            weight_save_path=weight_save_path,
            train_n_batches=1,
            batch_size=8,
            val_split_file_idxs=[1],
            test_split_file_idxs=[1],
            cached_simulator=simulator_cfg,
            training=training_cfg,
        )

        assert Path(f"{bliss_client.base_cfg.paths.output}/{weight_save_path}").exists()

    def test_load_survey(self, bliss_client, monkeypatch):
        monkeypatch.setattr("requests.get", mock_tests.mock_get)
        monkeypatch.setattr("requests.post", mock_tests.mock_post)
        monkeypatch.setattr("gzip.decompress", lambda x: x)
        monkeypatch.setattr("bz2.decompress", lambda x: x)

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

    def test_predict_sdss_default_rcf(self, bliss_client, monkeypatch):
        monkeypatch.setattr(api, "_predict_sdss", mock_tests.mock_predict_sdss)
        # cached predict data copied to temp dir in mock_tests.mock_predict_sdss
        cat, cat_table, gal_params_table, pred_table = bliss_client.predict_sdss("test_path")
        assert isinstance(cat, FullCatalog)
        assert isinstance(gal_params_table, Table)
        assert isinstance(cat_table, Table)
        assert isinstance(pred_table, Table)

        bliss_client.plot_predictions_in_notebook()
        assert Path(bliss_client.base_cfg.predict.plot.out_file_name).exists()
