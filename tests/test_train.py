from bliss.surveys.des import DarkEnergySurvey as DES
from bliss.train import train


class TestTrain:
    def test_train_sdss(self, cfg):
        train(cfg)

    def test_train_des(self, cfg):
        train_des_cfg = cfg.copy()
        train_des_cfg.simulator.survey = "${surveys.des}"
        train_des_cfg.simulator.prior.reference_band = DES.BANDS.index("r")
        train_des_cfg.simulator.prior.survey_bands = DES.BANDS

        train_des_cfg.encoder.bands = [
            DES.BANDS.index("g"),
            DES.BANDS.index("r"),
            DES.BANDS.index("z"),
        ]
        train_des_cfg.encoder.survey_bands = DES.BANDS
        train_des_cfg.training.pretrained_weights = None
        train_des_cfg.training.testing = True
        train(train_des_cfg)
