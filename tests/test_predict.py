from bliss.predict import predict


class TestPredict:
    def test_predict_sdss(self, cfg):
        the_cfg = cfg.copy()
        the_cfg.predict.plot.show_plot = True
        predict(the_cfg)

        # TODO: test output of predictions plot

    def test_predict_sdss_single_band(self, cfg):
        the_cfg = cfg.copy()
        the_cfg.predict.plot.show_plot = True
        the_cfg.encoder.bands = [2]
        the_cfg.predict.weight_save_path = "${paths.pretrained_models}/single_band_base.pt"
        predict(the_cfg)

        # TODO: test output of predictions plot

    def test_predict_sdss_multiple_images(self, cfg):
        the_cfg = cfg.copy()
        the_cfg.surveys.sdss.sdss_fields = [
            {"run": 94, "camcol": 1, "fields": [12]},
            {"run": 3900, "camcol": 6, "fields": [269]},
        ]
        the_cfg.predict.plot.show_plot = True
        _, _, _, _, preds = predict(the_cfg)

        # TODO: check rest of the return values from predict
        assert len(preds) == len(
            the_cfg.surveys.sdss.sdss_fields
        ), f"Expected {len(the_cfg.surveys.sdss.sdss_fields)} predictions, got {len(preds)}"

    def test_predict_decals(self, cfg):
        the_cfg = cfg.copy()
        the_cfg.predict.dataset = cfg.surveys.decals
        the_cfg.encoder.bands = [2]
        the_cfg.predict.weight_save_path = "${paths.pretrained_models}/single_band_base.pt"
        predict(the_cfg)
