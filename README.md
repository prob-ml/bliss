![](http://portal.nersc.gov/project/dasrepo/celeste/sample_sky.jpg)


Bayesian Light Source Separator (BLISS)
========================================
[![](https://img.shields.io/badge/docs-master-blue.svg)](https://prob-ml.github.io/bliss/)
![tests](https://github.com/prob-ml/bliss/workflows/tests/badge.svg)
[![codecov.io](https://codecov.io/gh/prob-ml/bliss/branch/master/graphs/badge.svg?branch=master&token=Jgzv0gn3rA)](http://codecov.io/github/prob-ml/bliss?branch=master)
![case studies](https://github.com/prob-ml/bliss/actions/workflows/case_studies.yml/badge.svg)

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

BLISS is pip installable with the following command: 

```bash
pip install bliss-deblender
``` 

and the required dependencies are listed in the ``[tool.poetry.dependencies]`` block of the ``pyproject.toml`` file.

# Installation (Developers)

1. To use and install `bliss` you first need to install [poetry](https://python-poetry.org/docs/).

2. Then, install the [fftw](http://www.fftw.org) library (which is used by `galsim`). With Ubuntu you can install it by running

```bash
sudo apt-get install libfftw3-dev
```

3. Install git-lfs if you haven't already installed it for another project:

```bash
git-lfs install
```

4. Now download the bliss repo and fetch some pre-trained models and test data from git-lfs:

```bash
git clone https://github.com/prob-ml/bliss.git
```

5. To create a poetry environment with the `bliss` dependencies satisified, run

```bash
cd bliss
poetry install
poetry shell
```

6. Verify that bliss is installed correctly by running the tests both on your CPU (default) and on your GPU:

```bash
pytest
pytest --gpu
```

7. Finally, if you are planning to contribute code to this repository, consider installing our pre-commit hooks so that your code commits will be checked locally for compliance with our coding conventions:

```bash
pre-commit --install
```

# Latest updates
## Galaxies
   - BLISS now includes a galaxy model based on a VAE that was trained on Galsim galaxies.
   - BLISS now includes an algorithm for detecting, measuring, and deblending galaxies.

## Stars
   - BLISS already includes the StarNet functionality from its predecessor repo: [DeblendingStarFields](https://github.com/Runjing-Liu120/DeblendingStarfields).


# References

Mallory Wang, Ismael Mendoza, Cheng Wang, Camille Avestruz, and Jeffrey Regier. *Statistical Inference for Coadded Astronomical Images.* Machine Learning and the Physical Sciences workshop, NeurIPS 2022. [arXiv:2211.09300](https://arxiv.org/abs/2211.09300)

Derek Hansen, Ismael Mendoza, Runjing Liu, Ziteng Pang, Zhe Zhao, Camille Avestruz, and Jeffrey Regier. *Scalable Bayesian Inference for Detection and Deblending in Astronomical Images*. ICML Workshop on Machine Learning for Astrophysics, 2022. [arXiv:2207.05642](https://arxiv.org/abs/2207.05642)

Runjing Liu, Jon D. McAuliffe, Jeffrey Regier, and The LSST Dark Energy Science Collaboration. *Variational Inference for Deblending Crowded Starfields*, 2021. [arXiv:2102.02409](https://arxiv.org/abs/2102.02409)
