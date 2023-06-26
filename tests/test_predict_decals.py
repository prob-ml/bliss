from bliss.predict import predict_decals


class TestPredictDecals:
    def test_predict_decals(self, cfg):
        the_cfg = cfg.copy()
        the_cfg.predict.dataset["_target_"] = "bliss.surveys.decals.DarkEnergyCameraLegacySurvey"
        the_cfg.predict.dataset.decals_dir = "${paths.decals}"
        the_cfg.predict.dataset.ra = 336.635
        the_cfg.predict.dataset.dec = -0.96
        the_cfg.predict.dataset.width = 2400
        the_cfg.predict.dataset.height = 1489
        the_cfg.predict.dataset.bands = ["g"]
        predict_decals(the_cfg)
