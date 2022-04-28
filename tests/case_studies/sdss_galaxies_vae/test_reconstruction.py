import pytest

from case_studies.sdss_galaxies_vae.reconstruction import reconstruct


@pytest.fixture(scope="module")
def reconstruct_overrides():
    return {
        "datasets.simulated.generate_device": "cpu",
        "mode": "reconstruct",
        "reconstruct.outdir": None,
        "reconstruct.real": False,
        "reconstruct.device": "cpu",
        "reconstruct.slen": 80,
    }


def test_reconstruct_sdss(vae_setup, reconstruct_overrides):
    overrides = {
        **reconstruct_overrides,
        "+reconstruct.frame._target_": "bliss.inference.SDSSFrame",
        "+reconstruct.frame.sdss_dir": "${paths.sdss}",
        "+reconstruct.frame.pixel_scale": 0.396,
        "+reconstruct.frame.coadd_file": "${paths.sdss}/coadd_catalog_94_1_12.fits",
        "+reconstruct.test.h": 200 + 50,
        "+reconstruct.test.w": 1700 + 150,
        "+reconstruct.test.size": 100,
        "+reconstruct.photo_catalog.sdss_path": "${paths.sdss}",
        "+reconstruct.photo_catalog.run": 94,
        "+reconstruct.photo_catalog.camcol": 1,
        "+reconstruct.photo_catalog.field": 12,
        "+reconstruct.photo_catalog.band": 2,
    }

    cfg = vae_setup.get_cfg(overrides)
    reconstruct(cfg)


def test_reconstruct_simulated(vae_setup, reconstruct_overrides):
    overrides = {
        **reconstruct_overrides,
        "+reconstruct.frame._target_": "bliss.inference.SimulatedFrame",
        "+reconstruct.frame.dataset": "${datasets.simulated}",
        "+reconstruct.frame.n_tiles_h": 30,
        "+reconstruct.frame.n_tiles_w": 30,
        "+reconstruct.test.h": 24,
        "+reconstruct.test.w": 24,
        "+reconstruct.test.size": 100,
    }
    cfg = vae_setup.get_cfg(overrides)
    reconstruct(cfg)


def test_reconstruct_semisynthetic(vae_setup, reconstruct_overrides):
    overrides = {
        **reconstruct_overrides,
        "+reconstruct.frame._target_": "bliss.inference.SemiSyntheticFrame",
        "+reconstruct.frame.dataset": "${datasets.simulated}",
        "+reconstruct.frame.coadd": "${paths.sdss}/coadd_catalog_94_1_12.fits",
        "+reconstruct.frame.n_tiles_h": 30,
        "+reconstruct.frame.n_tiles_w": 30,
        "+reconstruct.test.h": 24,
        "+reconstruct.test.w": 24,
        "+reconstruct.test.size": 100,
    }
    cfg = vae_setup.get_cfg(overrides)
    reconstruct(cfg)
