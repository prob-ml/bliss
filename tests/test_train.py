import os

from bliss.main import generate, train
from bliss.surveys.decals import DarkEnergyCameraLegacySurvey as DECaLS
from bliss.surveys.des import DarkEnergySurvey as DES


class TestTrain:
    def test_train_sdss(self, cfg):
        train_sdss_cfg = cfg.copy()
        train(train_sdss_cfg.train)

    def test_train_des(self, cfg):
        train_des_cfg = cfg.copy()
        train_des_cfg.simulator.survey = "${surveys.des}"
        train_des_cfg.simulator.prior.reference_band = DES.BANDS.index("r")
        train_des_cfg.simulator.prior.survey_bands = DES.BANDS

        for f in train_des_cfg.variational_factors:
            if f.name in {"star_fluxes", "galaxy_fluxes"}:
                f.dim = 4

        train_des_cfg.encoder.survey_bands = DES.BANDS
        train_des_cfg.train.pretrained_weights = None
        train_des_cfg.train.testing = True
        train(train_des_cfg.train)

    def test_train_decals(self, cfg):
        train_decals_cfg = cfg.copy()
        train_decals_cfg.simulator.survey = "${surveys.decals}"
        train_decals_cfg.simulator.prior.reference_band = DECaLS.BANDS.index("r")
        train_decals_cfg.simulator.prior.survey_bands = DECaLS.BANDS

        for f in train_decals_cfg.variational_factors:
            if f.name in {"star_fluxes", "galaxy_fluxes"}:
                f.dim = 4

        train_decals_cfg.encoder.image_normalizer.log_transform_stdevs = []
        train_decals_cfg.encoder.survey_bands = DECaLS.BANDS
        train_decals_cfg.train.pretrained_weights = None
        train_decals_cfg.train.testing = True

        train_decals_cfg.simulator.use_coaddition = True
        train_decals_cfg.simulator.coadd_depth = 2
        train(train_decals_cfg.train)

    def test_train_with_cached_data(self, cfg, tmp_path):
        the_cfg = cfg.copy()
        the_cfg.paths.output = tmp_path
        the_cfg.generate.cached_data_path = tmp_path
        generate(the_cfg.generate)

        the_cfg.train.data_source = "${cached_simulator}"
        the_cfg.cached_simulator.cached_data_path = tmp_path
        os.chdir(tmp_path)
        the_cfg.train.weight_save_path = str(tmp_path / "encoder.pt")
        train(the_cfg.train)
