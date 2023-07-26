import pytest

from bliss.generate import generate
from bliss.surveys.des import DES
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


class TestTrain:
    def test_train_sdss(self, cfg):
        train(cfg)

    def test_train_des(self, cfg):
        train_des_cfg = cfg.copy()
        train_des_cfg.simulator.survey = "${surveys.des}"
        train_des_cfg.simulator.prior.reference_band = DES.BANDS.index("r")
        train_des_cfg.simulator.prior.survey_bands = DES.BANDS

        train_des_cfg.encoder.bands = [0, 1, 2, 3]
        train_des_cfg.encoder.survey_bands = ["g", "r", "i", "z"]
        train_des_cfg.encoder.input_transform_params.z_score = False
        train_des_cfg.training.pretrained_weights = None
        train_des_cfg.training.testing = True
        train(train_des_cfg)

    def test_train_des_cached_data(self, cfg, des_cached_data_path):
        train_des_cfg = cfg.copy()
        train_des_cfg.training.data_source = "${cached_simulator}"
        train_des_cfg.cached_simulator.cached_data_path = des_cached_data_path
        train_des_cfg.cached_simulator.batch_size = cfg.prior.batch_size

        train_des_cfg.encoder.bands = [0, 1, 2, 3]
        train_des_cfg.encoder.survey_bands = ["g", "r", "i", "z"]
        train_des_cfg.encoder.input_transform_params.z_score = False
        train_des_cfg.training.pretrained_weights = None
        train_des_cfg.training.testing = True
        train(train_des_cfg)
