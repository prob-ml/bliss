import shutil
from pathlib import Path

import pytest

from bliss.main import predict


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
        bliss_cats = predict(the_cfg.predict)
        assert len(bliss_cats) == len(the_cfg.surveys.sdss.fields)

        bands = cfg.encoder.survey_bands
        astropy_cats = [c.to_astropy_table(bands) for c in bliss_cats.values()]
        assert len(astropy_cats) == len(the_cfg.surveys.sdss.fields)
