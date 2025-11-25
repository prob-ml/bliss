Bayesian Light Source Separator (BLISS)
=======================================

BLISS is a Bayesian method for deblending and cataloging light sources. BLISS provides:

* **Accurate estimation** of parameters in crowded fields with overlapping sources
* **Calibrated uncertainties** through fully Bayesian inference via neural posterior estimation (NPE)
* **Scalability** to petabyte-scale surveys containing billions of celestial objects

BLISS systematically outperforms standard survey pipelines for light source detection, flux measurement, star/galaxy classification, and galaxy shape measurement.

BLISS uses neural posterior estimation (NPE), a form of simulation-based inference in which a neural network maps telescope images directly to an approximate Bayesian posterior. BLISS uses autoregressive tiling with checkerboard patterns that align the variational distribution's conditional dependencies with the true posterior, improving calibration.

Table of Contents
#################

.. toctree::
    :maxdepth: 1
    :titlesonly:
    :name: mastertoc

    Installation
    api/index
    tutorials/index
