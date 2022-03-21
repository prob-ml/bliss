![](http://portal.nersc.gov/project/dasrepo/celeste/sample_sky.jpg)


Bayesian Light Source Separator (BLISS)
========================================
[![](https://img.shields.io/badge/docs-master-blue.svg)](https://prob-ml.github.io/bliss/)
![tests](https://github.com/applied-bayes/bliss/workflows/tests/badge.svg)
[![codecov.io](https://codecov.io/gh/prob-ml/bliss/branch/master/graphs/badge.svg?branch=master&token=Jgzv0gn3rA)](http://codecov.io/github/prob-ml/bliss?branch=master)
[![wemake-python-styleguide](https://img.shields.io/badge/style-wemake-000000.svg)](https://github.com/wemake-services/wemake-python-styleguide)

# Introduction

BLISS is a Bayesian method for deblending and cataloging light sources. BLISS provides
  - __Accurate estimation__ of parameters in blended field.
  - __Calibrated uncertainties__ through fitting an approximate Bayesian posterior.
  - __Scalability__ of Bayesian inference to entire astronomical surveys.

BLISS uses state-of-the-art variational inference techniques including
  - __Amortized inference__, in which a neural network maps telescope images to an approximate Bayesian posterior on parameters of interest.
  - __Variational auto-encoders__ (VAEs) to fit a flexible model for galaxy morphology and deblend galaxies.
  - __Wake-sleep algorithm__ to jointly fit the approximate posterior and model parameters such as the PSF and the galaxy VAE.

# Installation

1. To use and install `bliss` you first need to install [poetry](https://python-poetry.org/docs/).

2. Then, install the [fftw](http://www.fftw.org) library (which is used by `galsim`). With Ubuntu you can install it by running
```
sudo apt-get install libfftw3-dev
```

3. Now, to create a poetry environment with the `bliss` dependencies satisified, run
```
git clone https://github.com/prob-ml/bliss.git
cd bliss
poetry install
poetry shell
```

# Latest updates
### Galaxies
   - BLISS now includes a galaxy model based on a VAE that was trained on Galsim galaxies.
   - BLISS now includes an algorithm for detecting, measuring, and deblending galaxies.

### Stars
   - BLISS already includes the StarNet functionality from its predecessor repo: [DeblendingStarFields](https://github.com/Runjing-Liu120/DeblendingStarfields).


# References

Runjing Liu, Jon D. McAuliffe, Jeffrey Regier, and The LSST Dark Energy Science Collaboration. *Variational Inference for Deblending Crowded Starfields.* [arXiv:2102.02409](https://arxiv.org/abs/2102.02409), 2021.
