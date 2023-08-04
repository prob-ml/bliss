from bliss.surveys.decals import DarkEnergyCameraLegacySurvey as DECaLS
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

    def test_train_decals(self, cfg):
        train_decals_cfg = cfg.copy()
        train_decals_cfg.simulator.survey = "${surveys.decals}"
        train_decals_cfg.simulator.prior.reference_band = DECaLS.BANDS.index("r")
        train_decals_cfg.simulator.prior.survey_bands = DECaLS.BANDS

        train_decals_cfg.encoder.bands = [
            DECaLS.BANDS.index("g"),
            DECaLS.BANDS.index("r"),
            DECaLS.BANDS.index("z"),
        ]
        train_decals_cfg.encoder.survey_bands = DECaLS.BANDS
        train_decals_cfg.training.pretrained_weights = None
        train_decals_cfg.training.testing = True

        train_decals_cfg.simulator.use_coaddition = True
        train_decals_cfg.simulator.coadd_depth = 2
        train(train_decals_cfg)
