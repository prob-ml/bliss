import shutil
from pathlib import Path

import pytest
from omegaconf import open_dict

from bliss.cli import predict
from bliss.generate import generate
from bliss.surveys.decals import DarkEnergyCameraLegacySurvey as DECaLS
from bliss.surveys.des import DarkEnergySurvey as DES
from bliss.train import train


@pytest.fixture(autouse=True)
def setup_teardown(cfg, monkeypatch):
    # override `align` for now (kernprof analyzes ~40% runtime); TODO: test alignment
    monkeypatch.setattr("bliss.align.align", lambda x, **_args: x)

    checkpoint_dir = cfg.paths.root + "/checkpoints"
    if Path(checkpoint_dir).exists():
        shutil.rmtree(checkpoint_dir)

    yield

    if Path(checkpoint_dir).exists():
        shutil.rmtree(checkpoint_dir)


@pytest.fixture
def des_cached_data_path(cfg, tmpdir_factory):
    # Also tests simulating DES data
    des_cached_data_path = str(tmpdir_factory.mktemp("des_cached_data_path"))
    gen_des_cfg = cfg.copy()
    gen_des_cfg.generate.cached_data_path = des_cached_data_path

    gen_des_cfg.simulator.survey = cfg.surveys.des
    gen_des_cfg.generate.n_batches = 3
    gen_des_cfg.generate.batch_size = 4
    gen_des_cfg.simulator.prior.batch_size = 4
    gen_des_cfg.simulator.prior.reference_band = DES.BANDS.index("r")
    gen_des_cfg.simulator.prior.survey_bands = DES.BANDS
    gen_des_cfg.generate.max_images_per_file = 8
    generate(gen_des_cfg)

    return des_cached_data_path


@pytest.fixture
def des_weight_save_path(cfg, tmpdir_factory, des_cached_data_path):
    # Also tests training on (cached) disk DES data
    train_des_cfg = cfg.copy()
    train_des_cfg.training.data_source = "${cached_simulator}"
    train_des_cfg.cached_simulator.cached_data_path = des_cached_data_path
    train_des_cfg.cached_simulator.batch_size = cfg.prior.batch_size

    train_des_cfg.encoder.bands = list(range(len(DES.BANDS)))
    train_des_cfg.encoder.survey_bands = DES.BANDS
    train_des_cfg.training.pretrained_weights = None
    train_des_cfg.training.testing = True

    des_weight_save_path = str(tmpdir_factory.mktemp("des_weight_save_path") / "des.pt")
    train_des_cfg.training.weight_save_path = des_weight_save_path
    train(train_des_cfg)
    return des_weight_save_path


@pytest.fixture
def decals_cached_data_path(cfg, tmpdir_factory):
    decals_cached_data_path = str(tmpdir_factory.mktemp("decals_cached_data"))
    gen_decals_cfg = cfg.copy()
    gen_decals_cfg.generate.cached_data_path = decals_cached_data_path

    gen_decals_cfg.simulator.survey = "${surveys.decals}"
    gen_decals_cfg.surveys.decals.sky_coords = [
        # brick '3366m010' corresponds to SDSS RCF 94-1-12
        {"ra": 336.6643042496718, "dec": -0.9316385797930247},
    ]
    gen_decals_cfg.surveys.decals.bands = [DECaLS.BANDS.index("r")]

    gen_decals_cfg.generate.n_batches = 3
    gen_decals_cfg.generate.batch_size = 4
    gen_decals_cfg.simulator.prior.batch_size = 4
    gen_decals_cfg.simulator.prior.reference_band = DECaLS.BANDS.index("r")
    gen_decals_cfg.simulator.prior.survey_bands = DECaLS.BANDS
    gen_decals_cfg.generate.max_images_per_file = 8

    gen_decals_cfg.simulator.use_coaddition = True  # NOTE: coadd configuration
    gen_decals_cfg.simulator.coadd_depth = 2  # NOTE: coadd configuration
    generate(gen_decals_cfg)

    return decals_cached_data_path


@pytest.fixture
def decals_weight_save_path(cfg, tmpdir_factory, decals_cached_data_path):
    # Also tests training on (cached) disk DECaLS data
    train_decals_cfg = cfg.copy()
    train_decals_cfg.training.data_source = "${cached_simulator}"
    train_decals_cfg.cached_simulator.cached_data_path = decals_cached_data_path
    train_decals_cfg.cached_simulator.batch_size = cfg.prior.batch_size

    train_decals_cfg.encoder.bands = list(range(len(DECaLS.BANDS)))
    train_decals_cfg.encoder.survey_bands = DECaLS.BANDS
    train_decals_cfg.encoder.image_normalizer.log_transform_stdevs = []
    train_decals_cfg.training.pretrained_weights = None
    train_decals_cfg.training.testing = True

    decals_weight_save_path = str(tmpdir_factory.mktemp("decals_weight_save_path") / "decals.pt")
    train_decals_cfg.training.weight_save_path = decals_weight_save_path
    train(train_decals_cfg)
    return decals_weight_save_path


class TestPredict:
    def test_predict_sdss_multiple_rcfs(self, cfg, monkeypatch):
        crop = lambda _, img: img[:, 0:64, 0:64]
        method_str = "bliss.surveys.sdss.SloanDigitalSkySurvey._crop_image"
        monkeypatch.setattr(method_str, crop)

        the_cfg = cfg.copy()
        the_cfg.surveys.sdss.fields = [
            {"run": 94, "camcol": 1, "fields": [12]},
            {"run": 3635, "camcol": 1, "fields": [169]},
        ]
        astropy_cats = predict(the_cfg)

        assert len(astropy_cats) == len(the_cfg.surveys.sdss.fields)

    def test_predict_decals_multiple_bricks(self, cfg, decals_weight_save_path):
        the_cfg = cfg.copy()
        the_cfg.simulator.prior.reference_band = DECaLS.BANDS.index("r")

        the_cfg.predict.dataset = "${surveys.decals}"
        the_cfg.surveys.decals.sky_coords = [
            # brick '3366m010' corresponds to SDSS RCF 94-1-12
            {"ra": 336.6643042496718, "dec": -0.9316385797930247},
            # brick '1358p297' corresponds to SDSS RCF 3635-1-169
            {"ra": 135.95496736941683, "dec": 29.646883837721347},
        ]
        the_cfg.encoder.bands = list(range(len(DECaLS.BANDS)))
        the_cfg.encoder.survey_bands = DECaLS.BANDS
        the_cfg.encoder.image_normalizer.log_transform_stdevs = []
        the_cfg.predict.weight_save_path = decals_weight_save_path
        with open_dict(the_cfg):
            the_cfg.predict.dataset["load_image_data"] = True
        astropy_cats = predict(the_cfg)

        assert len(astropy_cats) == len(the_cfg.surveys.decals.sky_coords)

    def test_predict_des(self, cfg, des_weight_save_path):
        the_cfg = cfg.copy()
        the_cfg.simulator.prior.reference_band = DES.BANDS.index("r")

        the_cfg.predict.dataset = "${surveys.des}"
        the_cfg.encoder.bands = list(range(len(DES.BANDS)))
        the_cfg.encoder.survey_bands = DES.BANDS
        the_cfg.predict.weight_save_path = des_weight_save_path
        with open_dict(the_cfg):
            the_cfg.predict.dataset["load_image_data"] = True
        astropy_cats = predict(the_cfg)

        assert len(astropy_cats) == len(the_cfg.surveys.des.image_ids)
