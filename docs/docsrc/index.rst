Bayesian Light Source Separator (BLISS)
=======================================

BLISS is a Bayesian procedure for deblending light sources. BLISS provides:

* **Accurate estimation** of parameters in blended field.
* **Calibrated uncertainties** through fitting an approximate Bayesian posterior.
* **Scalability** of Bayesian inference to entire astronomical surveys.

BLISS uses state-of-the-art methods in variational inference including:

* **Amortized inference**, in which a neural network maps telescope images to an approximate Bayesian posterior on parameters of interest.
* **Variational auto-encoders** (VAEs) to fit a flexible model for galaxy morphology.
* **Wake-sleep algorithm** to jointly fit the approximate posterior and model parameters such as the PSF and the galaxy VAE.

Latest updates
##############

Galaxies
********
* BLISS now includes a galaxy model based on a Variational AutoEncoder that was trained on CATSIM bulge+disk galaxies.
* We are working on testing galaxy detection functionality and developing galaxy shape measurement.

Stars
*****
* BLISS already includes the StarNet functionality from its predecessor repo: `DeblendingStarFields <https://github.com/Runjing-Liu120/DeblendingStarfields>`_.

Table of Contents
#################

.. toctree::
    :maxdepth: 1
    :titlesonly:
    :name: mastertoc

    Installation
    api/index
