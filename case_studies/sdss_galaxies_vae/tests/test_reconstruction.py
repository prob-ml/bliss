from case_studies.sdss_galaxies_vae.reconstruction import reconstruct


def test_reconstruct(model_setup, devices):
    overrides = {
        "mode": "reconstruct",
        "reconstruct.outdir": None,
        "reconstruct.real": False,
        "reconstruct.device": "cpu",
        "+reconstruct.scenes.sdss_recon1_test.h": 200 + 50,
        "+reconstruct.scenes.sdss_recon1_test.w": 1700 + 150,
        "+reconstruct.scenes.sdss_recon1_test.size": 100,
        "+reconstruct.scenes.sdss_recon1_test.slen": 80,
    }

    cfg = model_setup.get_cfg(overrides)
    reconstruct(cfg)
