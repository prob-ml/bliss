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
        "+reconstruct.frame.coadd_file": "${paths.data}/coadd_catalog_94_1_12.fits",
        "+reconstruct.scenes.sdss_recon1_test.h": 200 + 50,
        "+reconstruct.scenes.sdss_recon1_test.w": 1700 + 150,
        "+reconstruct.scenes.sdss_recon1_test.size": 100,
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
        "+reconstruct.scenes.sdss_recon1_test.h": 24,
        "+reconstruct.scenes.sdss_recon1_test.w": 24,
        "+reconstruct.scenes.sdss_recon1_test.size": 100,
    }
    cfg = vae_setup.get_cfg(overrides)
    reconstruct(cfg)


def test_reconstruct_semisynthetic(model_setup, devices, reconstruct_overrides):
    overrides = {
        **reconstruct_overrides,
        "+reconstruct.frame._target_": "bliss.inference.SemiSyntheticFrame",
        "+reconstruct.frame.dataset": "${datasets.simulated}",
        "+reconstruct.frame.coadd": "${paths.data}/coadd_catalog_94_1_12.fits",
        "+reconstruct.frame.n_tiles_h": 30,
        "+reconstruct.frame.n_tiles_w": 30,
        "+reconstruct.scenes.sdss_recon1_test.h": 24,
        "+reconstruct.scenes.sdss_recon1_test.w": 24,
        "+reconstruct.scenes.sdss_recon1_test.size": 100,
    }
    cfg = model_setup.get_cfg(overrides)
    reconstruct(cfg)
