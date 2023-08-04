import pytest

from bliss.generate import generate
from bliss.predict import predict
from bliss.surveys.decals import DarkEnergyCameraLegacySurvey as DECaLS
from bliss.surveys.des import DarkEnergySurvey as DES
from bliss.surveys.sdss import SloanDigitalSkySurvey as SDSS
from bliss.train import train


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


class TestPredict:
    def test_predict_sdss_multiple_rcfs(self, cfg, monkeypatch):
        # override `align` for now (kernprof analyzes ~40% runtime); TODO: test alignment
        monkeypatch.setattr("bliss.predict.align", lambda x, **_args: x)

        the_cfg = cfg.copy()
        the_cfg.surveys.sdss.fields = [
            {"run": 94, "camcol": 1, "fields": [12]},
            {"run": 3635, "camcol": 1, "fields": [169]},
        ]
        the_cfg.predict.plot.show_plot = True
        _, _, _, _, preds = predict(the_cfg)

        # TODO: check rest of the return values from predict
        assert len(preds) == len(
            the_cfg.surveys.sdss.fields
        ), f"Expected {len(the_cfg.surveys.sdss.fields)} predictions, got {len(preds)}"

        # TODO: somehow check plot output

    def test_predict_decals_multiple_bricks(self, cfg, monkeypatch):
        # override `align` for now (kernprof analyzes ~40% runtime); TODO: test alignment
        monkeypatch.setattr("bliss.predict.align", lambda x, **_args: x)

        the_cfg = cfg.copy()
        the_cfg.simulator.prior.reference_band = DECaLS.BANDS.index("r")

        the_cfg.predict.plot.show_plot = True
        the_cfg.predict.dataset = "${surveys.decals}"
        the_cfg.surveys.decals.sky_coords = [
            # brick '3366m010' corresponds to SDSS RCF 94-1-12
            {"ra": 336.6643042496718, "dec": -0.9316385797930247},
            # brick '1358p297' corresponds to SDSS RCF 3635-1-169
            {"ra": 135.95496736941683, "dec": 29.646883837721347},
        ]
        the_cfg.encoder.bands = [SDSS.BANDS.index("r")]
        the_cfg.encoder.input_transform_params.log_transform = False
        the_cfg.predict.weight_save_path = "${paths.pretrained_models}/single_band_base_5d.pt"
        _, _, _, _, preds = predict(the_cfg)

        # TODO: check rest of the return values from predict
        assert len(preds) == len(
            the_cfg.surveys.decals.sky_coords
        ), f"Expected {len(the_cfg.surveys.decals.sky_coords)} predictions, got {len(preds)}"

        # TODO: somehow check plot output

    def test_predict_des(self, cfg, monkeypatch, des_weight_save_path):
        monkeypatch.setattr("bliss.predict.align", lambda x, **_args: x)

        the_cfg = cfg.copy()
        the_cfg.simulator.prior.reference_band = DES.BANDS.index("r")

        the_cfg.predict.plot.show_plot = False
        the_cfg.predict.dataset = "${surveys.des}"
        the_cfg.encoder.bands = list(range(len(DES.BANDS)))
        the_cfg.encoder.survey_bands = DES.BANDS
        the_cfg.predict.weight_save_path = des_weight_save_path
        _, _, _, _, preds = predict(the_cfg)

        # TODO: check rest of the return values from predict
        assert len(preds) == len(
            the_cfg.surveys.des.image_ids
        ), f"Expected {len(the_cfg.surveys.des.image_ids)} predictions, got {len(preds)}"

        # TODO: somehow check plot output
