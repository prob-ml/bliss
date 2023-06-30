from pathlib import Path

import mock_tests
import numpy as np
import pytest
import torch
from astropy.table import Table
from omegaconf import OmegaConf

from bliss import api, generate, train
from bliss.api import BlissClient
from bliss.catalog import FullCatalog, SourceType


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
        cat, cat_table, pred_table = bliss_client.predict_sdss("test_path")
        assert isinstance(cat, FullCatalog)
        assert isinstance(cat_table, Table)
        assert isinstance(pred_table, Table)

        bliss_client.plot_predictions_in_notebook()
        assert Path(bliss_client.base_cfg.predict.plot.out_file_name).exists()

        # check that cat_table, gal_params_table contains all expected columns
        expected_table_columns = [
            "star_log_flux_u",
            "star_log_flux_g",
            "star_log_flux_r",
            "star_log_flux_i",
            "star_log_flux_z",
            "star_flux_u",
            "star_flux_g",
            "star_flux_r",
            "star_flux_i",
            "star_flux_z",
            "source_type",
            "galaxy_flux_u",
            "galaxy_flux_g",
            "galaxy_flux_r",
            "galaxy_flux_i",
            "galaxy_flux_z",
            "galaxy_disk_frac",
            "galaxy_beta_radians",
            "galaxy_disk_q",
            "galaxy_a_d",
            "galaxy_bulge_q",
            "galaxy_a_b",
        ]
        assert all(
            col in cat_table.colnames for col in expected_table_columns
        ), "cat_table missing columns"

        # check that cat_table, gal_params_table fluxes and log_fluxes in correct order of
        # magnitude (i.e., O(10^1) / O(10^2) for fluxes, O(10^0) for log_fluxes)
        assert np.all(
            np.log10(cat_table["star_flux_u"].value) <= 2
        ), "star_fluxes_u not O(10^1); ensure units are in nmgy"
        assert np.all(
            np.log10(cat_table["star_log_flux_u"].value) <= 1
        ), "star_log_fluxes_u not O(10^0); ensure units are in log(nmgy)"
        assert np.all(
            np.log10(cat_table["galaxy_flux_u"].value) <= 3
        ), "galaxy_flux_u not O(10^1); ensure units are in nmgy"

        # TODO: check that pred_table contains all expected columns

    def test_fullcat_to_astropy_table(self):
        # make 1 x 1 x 1 tensors in catalog
        d = {
            "plocs": torch.tensor([[[0.0, 0.0]]]),
            "n_sources": torch.tensor((1,)),
            "fluxes": torch.tensor([[[0.0]]]),
            "mags": torch.tensor([[[0.0]]]),
            "ra": torch.tensor([[[0.0]]]),
            "dec": torch.tensor([[[0.0]]]),
            "star_log_fluxes": torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0]]]),
            "star_fluxes": torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0]]]),
            "source_type": torch.tensor([[[SourceType.STAR]]]),
            "galaxy_params": torch.rand(1, 1, 6),
            "galaxy_fluxes": torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0]]]),
        }
        cat = FullCatalog(1, 1, d)
        est_cat_table = api.fullcat_to_astropy_table(cat)
        assert isinstance(est_cat_table, Table)
