import os

from bliss.main import generate, train
from bliss.surveys.decals import DarkEnergyCameraLegacySurvey as DECaLS
from bliss.surveys.des import DarkEnergySurvey as DES


class TestTrain:
    def test_train_sdss(self, cfg):
        train(cfg.train)

    def test_train_des(self, cfg):
        cfg = cfg.copy()
        cfg.simulator.survey = "${surveys.des}"
        cfg.simulator.prior.reference_band = DES.BANDS.index("r")
        cfg.simulator.prior.survey_bands = DES.BANDS

        for f in cfg.variational_factors:
            if f.name in {"star_fluxes", "galaxy_fluxes"}:
                f.dim = 4

        cfg.encoder.survey_bands = DES.BANDS
        cfg.encoder.image_normalizers.psf.num_psf_params = 10
        cfg.train.pretrained_weights = None
        cfg.train.testing = True
        train(cfg.train)

    def test_train_decals(self, cfg):
        cfg = cfg.copy()
        cfg.simulator.survey = "${surveys.decals}"
        cfg.simulator.prior.reference_band = DECaLS.BANDS.index("r")
        cfg.simulator.prior.survey_bands = DECaLS.BANDS

        for f in cfg.variational_factors:
            if f.name in {"star_fluxes", "galaxy_fluxes"}:
                f.dim = 4

        cfg.encoder.survey_bands = DECaLS.BANDS
        cfg.encoder.image_normalizers.psf.num_psf_params = 14
        cfg.train.pretrained_weights = None
        cfg.train.testing = True

        cfg.simulator.use_coaddition = True
        cfg.simulator.coadd_depth = 2
        train(cfg.train)

    def test_train_with_cached_data(self, cfg, tmp_path):
        cfg = cfg.copy()
        cfg.paths.output = tmp_path
        cfg.generate.cached_data_path = tmp_path
        generate(cfg.generate)

        cfg.train.data_source = "${cached_simulator}"
        cfg.cached_simulator.cached_data_path = tmp_path
        os.chdir(tmp_path)
        cfg.train.weight_save_path = str(tmp_path / "encoder.pt")
        train(cfg.train)
