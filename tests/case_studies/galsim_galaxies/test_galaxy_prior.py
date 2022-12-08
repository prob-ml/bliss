from hydra.utils import instantiate


def test_galaxy_prior(galsim_galaxies_setup):
    """Test to ensure galaxy_dataset is passed in to galaxy_prior and flux dist is correct."""
    cfg_unif = galsim_galaxies_setup.get_cfg(
        {
            "datasets.sdss_galaxies.prior.flux_sample": "uniform",
            "training": "sdss_detection_encoder",
        }
    )
    cfg_par = galsim_galaxies_setup.get_cfg(
        {
            "datasets.sdss_galaxies.prior.flux_sample": "pareto",
            "training": "sdss_detection_encoder",
        }
    )

    galds_unif_cfg = cfg_unif.training.dataset.prior.galaxy_prior.galaxy_dataset  # noqa: WPS219
    galds_unif = instantiate(galds_unif_cfg)

    galds_par_cfg = cfg_par.training.dataset.prior.galaxy_prior.galaxy_dataset  # noqa: WPS219
    galds_par = instantiate(galds_par_cfg)

    assert galds_unif.prior.flux_sample == "uniform"
    assert galds_par.prior.flux_sample == "pareto"
