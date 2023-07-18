from bliss.predict import predict


class TestPredict:
    def test_predict_sdss_multiple_rcfs(self, cfg):
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

    def test_predict_decals_multiple_bricks(self, cfg):
        the_cfg = cfg.copy()
        the_cfg.predict.plot.show_plot = True
        the_cfg.predict.dataset = "${surveys.decals}"
        the_cfg.surveys.decals.sky_coords = [
            # brick '3366m010' corresponds to SDSS RCF 94-1-12
            {"ra": 336.6643042496718, "dec": -0.9316385797930247},
            # brick '1358p297' corresponds to SDSS RCF 3635-1-169
            {"ra": 135.95496736941683, "dec": 29.646883837721347},
        ]
        the_cfg.encoder.bands = [2]
        the_cfg.predict.weight_save_path = "${paths.pretrained_models}/single_band_base.pt"
        _, _, _, _, preds = predict(the_cfg)

        # TODO: check rest of the return values from predict
        assert len(preds) == len(
            the_cfg.surveys.decals.sky_coords
        ), f"Expected {len(the_cfg.surveys.decals.sky_coords)} predictions, got {len(preds)}"

        # TODO: somehow check plot output
