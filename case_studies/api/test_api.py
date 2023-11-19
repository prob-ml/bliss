from pathlib import Path

import pytest
import torch
from astropy.table import Table
from omegaconf import OmegaConf

from bliss import generate, train
from bliss.catalog import FullCatalog, SourceType
from case_studies.api import api, mock_tests
from case_studies.api.api import BlissClient


@pytest.fixture(scope="class")
def bliss_client(cfg, tmpdir_factory):
    cwd = str(tmpdir_factory.mktemp("cwd"))
    client = BlissClient(cwd)
    image_ids = []
    for sdss_field in cfg.surveys.sdss.fields:
        run, camcol, fields = sdss_field.values()
        for field in fields:
            image_ids.append((run, camcol, field))
    mock_tests.MockSDSSDownloader(image_ids, client.cwd + "/data/sdss")
    # Hack to apply select conftest.py overrides, since client.base_cfg should be private
    overrides = {
        "train.trainer.accelerator": cfg.train.trainer.accelerator,
        "predict.device": cfg.predict.device,
    }
    for k, v in overrides.items():
        OmegaConf.update(client.base_cfg, k, v)
    return client


@pytest.mark.usefixtures("bliss_client")
class TestApi:
    """Basic tests for API functions like generate, train, and predict.

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
        assert not Path(f"{cached_data_path}/dataset_2.pt").exists()

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
        cached_simulator_cfg = {"cached_data_path": "data/tests/multiband_data"}
        train_cfg = {
            "n_epochs": 1,
            "trainer": {"check_val_every_n_epoch": 1, "log_every_n_steps": 1},
        }
        weight_save_path = "tests/test_model.pt"
        bliss_client.train_on_cached_data(
            weight_save_path=weight_save_path,
            splits="0:50/50:100/50:100",
            batch_size=8,
            cached_simulator=cached_simulator_cfg,
            training=train_cfg,
        )

        assert Path(f"{bliss_client.base_cfg.paths.output}/{weight_save_path}").exists()

    def test_load_survey(self, bliss_client, monkeypatch):
        monkeypatch.setattr("requests.get", mock_tests.mock_get)
        monkeypatch.setattr("requests.post", mock_tests.mock_post)
        monkeypatch.setattr("gzip.decompress", lambda x: x)
        monkeypatch.setattr("bz2.decompress", lambda x: x)

        download_dir = "data/sdss"
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
        monkeypatch.setattr(api, "_predict_and_compare", mock_tests.mock_predict_sdss)
        # cached predict data copied to temp dir in mock_tests.mock_predict_sdss
        cat, cat_table, pred_tables = bliss_client.predict(
            survey="sdss", weight_save_path="test_path"
        )
        assert isinstance(cat, FullCatalog)
        assert isinstance(cat_table, Table)
        assert all(isinstance(pred_table, Table) for pred_table in pred_tables.values())

        bliss_client.plot_predictions_in_notebook()
        assert Path(bliss_client.base_cfg.predict.plot.out_file_name).exists()

        # check that cat_table contains all expected columns
        expected_cat_table_columns = [
            "plocs",
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
            col in cat_table.colnames for col in expected_cat_table_columns
        ), "cat_table missing columns"

        # check that pred_table contains all expected columns
        expected_pred_table_columns = [
            "on_prob_false",
            "on_prob_true",
            "galaxy_prob_false",
            "galaxy_prob_true",
        ]

        dist_colnames = [
            "galsim_disk_frac",
            "galsim_beta_radians",
            "galsim_disk_q",
            "galsim_a_d",
            "galsim_bulge_q",
            "galsim_a_b",
            "star_flux_u",
            "star_flux_g",
            "star_flux_r",
            "star_flux_i",
            "star_flux_z",
            "galaxy_flux_u",
            "galaxy_flux_g",
            "galaxy_flux_r",
            "galaxy_flux_i",
            "galaxy_flux_z",
        ]

        expected_pred_table_columns.extend([f"{col}_mean" for col in dist_colnames])
        expected_pred_table_columns.extend([f"{col}_std" for col in dist_colnames])

        some_pred_table = next(iter(pred_tables.values()))
        assert all(
            col in some_pred_table.colnames for col in expected_pred_table_columns
        ), "pred_table missing columns"

    def test_predict_decals_default_brick(self, bliss_client, monkeypatch):
        monkeypatch.setattr(api, "_predict_and_compare", mock_tests.mock_predict_decals_bulk)
        # cached predict data copied to temp dir in mock_tests.mock_predict_decals_bulk
        cat, cat_table, pred_tables = bliss_client.predict(
            survey="decals",
            weight_save_path="test_path",
            surveys={
                "decals": {
                    "sky_coords": [
                        # brick '3366m010' corresponds to SDSS RCF 94-1-12
                        {"ra": 336.6643042496718, "dec": -0.9316385797930247},
                        # brick '1358p297' corresponds to SDSS RCF 3635-1-169
                        {"ra": 135.95496736941683, "dec": 29.646883837721347},
                    ]
                }
            },
        )
        assert isinstance(cat, FullCatalog)
        assert isinstance(cat_table, Table)
        assert all(isinstance(pred_table, Table) for pred_table in pred_tables.values())

        bliss_client.plot_predictions_in_notebook()
        assert Path(bliss_client.base_cfg.predict.plot.out_file_name).exists()

    def test_fullcat_to_astropy_table(self):
        # make 1 x 1 x 1 tensors in catalog
        d = {
            "plocs": torch.tensor([[[0.0, 0.0]]]),
            "n_sources": torch.tensor((1,)),
            "fluxes": torch.tensor([[[0.0]]]),
            "mags": torch.tensor([[[0.0]]]),
            "ra": torch.tensor([[[0.0]]]),
            "dec": torch.tensor([[[0.0]]]),
            "star_fluxes": torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0]]]),
            "source_type": torch.tensor([[[SourceType.STAR]]]),
            "galaxy_params": torch.rand(1, 1, 6),
            "galaxy_fluxes": torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0]]]),
        }
        cat = FullCatalog(1, 1, d)
        est_cat_table = cat.to_astropy_table(["u", "g", "r", "i", "z"])
        assert isinstance(est_cat_table, Table)
