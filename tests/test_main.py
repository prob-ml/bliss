# this file tests the top-level interface to BLISS, which is defined in main.py

import os
import shutil
from pathlib import Path

import pytest
import torch

from bliss.main import generate, predict, train
from bliss.surveys.decals import DarkEnergyCameraLegacySurvey as DECaLS
from bliss.surveys.des import DarkEnergySurvey as DES


class TestGenerate:
    def test_generate_sdss(self, cfg):
        # check that cached dataset exists
        assert cfg.generate.n_image_files > 0 and cfg.generate.n_batches_per_file > 0

        generate(cfg.generate)

        file_path = cfg.generate.cached_data_path + "/dataset_0_size_2.pt"
        assert os.path.exists(file_path), f"{file_path} not found"

        # cursory check of contents of cached dataset
        with open(file_path, "rb") as f:
            cached_dataset = torch.load(f)
            assert isinstance(cached_dataset, list), "cached_dataset must be a list"
            assert isinstance(
                cached_dataset[0], dict
            ), "cached_dataset must be a list of dictionaries"
            assert isinstance(
                cached_dataset[0]["tile_catalog"], dict
            ), "cached_dataset[0]['tile_catalog'] must be a dictionary"
            assert isinstance(
                cached_dataset[0]["images"], torch.Tensor
            ), "cached_dataset[0]['images'] must be a torch.Tensor"
            correct_len = cfg.generate.n_batches_per_file * cfg.generate.simulator.prior.batch_size
            assert len(cached_dataset) == correct_len, (
                f"cached_dataset has length {len(cached_dataset)}, "
                f"but must be list of length {correct_len}"
            )
            assert (
                len(cached_dataset[0]["images"]) == 5
            ), "cached_dataset[0]['images'] must be a 5-D tensor"
            assert cached_dataset[0]["images"][0].shape == (
                cfg.simulator.prior.n_tiles_h * cfg.simulator.tile_slen,
                cfg.simulator.prior.n_tiles_w * cfg.simulator.tile_slen,
            )


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


@pytest.fixture(autouse=True)
def setup_teardown(cfg, monkeypatch):
    # override `align` for now (kernprof analyzes ~40% runtime); TODO: test alignment
    monkeypatch.setattr("bliss.align.align", lambda x, **_args: x)

    checkpoint_dir = cfg.paths.output + "/checkpoints"
    if Path(checkpoint_dir).exists():
        shutil.rmtree(checkpoint_dir)

    yield

    if Path(checkpoint_dir).exists():
        shutil.rmtree(checkpoint_dir)


class TestPredict:
    def test_predict_sdss_multiple_rcfs(self, cfg, monkeypatch):
        crop = lambda _, img: img[:, 100:164, 100:164]
        method_str = "bliss.surveys.sdss.SloanDigitalSkySurvey._crop_image"
        monkeypatch.setattr(method_str, crop)

        cfg = cfg.copy()
        cfg.surveys.sdss.fields = [
            {"run": 94, "camcol": 1, "fields": [12]},
            {"run": 3635, "camcol": 1, "fields": [169]},
        ]
        bliss_cats = predict(cfg.predict)
        assert len(bliss_cats) == len(cfg.surveys.sdss.fields)

        mode_cats = [c["mode_cat"] for c in bliss_cats.values()]
        full_mode_cats = [c.to_full_catalog(cfg.encoder.tile_slen) for c in mode_cats]
        assert len(full_mode_cats) == len(cfg.surveys.sdss.fields)
