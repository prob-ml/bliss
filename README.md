![](http://portal.nersc.gov/project/dasrepo/celeste/sample_sky.jpg)


Bayesian Light Source Separator (BLISS)
========================================
[![pipeline status](https://gitlab.com/prob-ml/bliss/badges/master/pipeline.svg)](https://gitlab.com/prob-ml/bliss/-/pipelines)
[![codecov.io](https://codecov.io/gh/prob-ml/bliss/branch/master/graphs/badge.svg?branch=master&token=Jgzv0gn3rA)](http://codecov.io/github/prob-ml/bliss?branch=master)
[![PyPI](https://img.shields.io/pypi/v/bliss-toolkit.svg)](https://pypi.org/project/bliss-toolkit)

# Introduction

BLISS is a Bayesian method for deblending and cataloging light sources. BLISS provides:
  - __Accurate estimation__ of parameters in crowded fields with overlapping sources
  - __Calibrated uncertainties__ through fully Bayesian inference via neural posterior estimation (NPE)
  - __Scalability__ to petabyte-scale surveys containing billions of celestial objects

BLISS uses neural posterior estimation (NPE), a form of simulation-based inference in which a neural network maps telescope images directly to an approximate Bayesian posterior. BLISS uses autoregressive tiling with checkerboard patterns that align the variational distribution's conditional dependencies with the true posterior, improving calibration.
BLISS systematically outperforms standard survey pipelines for light source detection, flux measurement, star/galaxy classification, and galaxy shape measurement.

# Installation

**Requirements:** Python 3.12 or higher

BLISS is pip installable with the following command:

```bash
pip install bliss-toolkit
```

The required dependencies are listed in the `[tool.poetry.dependencies]` block of the `pyproject.toml` file.

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
git clone git@github.com:prob-ml/bliss.git
```

5. To create a poetry environment with the `bliss` dependencies satisfied, run

```bash
cd bliss
poetry install
```

6. Verify that bliss is installed correctly by running the tests both on your CPU (default) and on your GPU:

```bash
pytest
pytest --gpu
```

7. Finally, if you are planning to contribute code to this repository, consider installing our pre-commit hooks so that your code commits will be checked locally for compliance with our coding conventions:

```bash
pre-commit install
```

# Usage

BLISS provides a command-line interface powered by [Hydra](https://hydra.cc/) for configuration management. The basic usage pattern is:

```bash
bliss mode={generate,train,predict} [config options]
```

- `mode=generate` - Generate synthetic training data from the forward model
- `mode=train` - Train encoder networks on simulated or real data
- `mode=predict` - Run inference on astronomical images to produce catalogs

Configuration files are located in `bliss/conf/` and can be composed and overridden via command line arguments.

# Case Studies

The `case_studies/` directory contains research applications of BLISS:

- **weak_lensing/** - Shear (γ) and convergence (κ) estimation, validated on DC2 and DECaLS simulations
- **redshift/** - Photo-z estimation (BLISS-PZ) with multiple variational families
- **galaxy_clustering/** - Galaxy cluster detection and membership prediction, validated on DES DR2
- **dc2_cataloging/** - Full cataloging pipeline for DC2 simulation
- **strong_lensing/** - Strong gravitational lens detection
- **spatial_tiling/** - Spatially autoregressive tiling
- **psf_variation/** - Handling spatially varying PSFs

# References

Yicun Duan, Xinyue Li, Camille Avestruz, Jeffrey Regier, and LSST Dark Energy Collaboration. [Neural posterior estimation for cataloging astronomical images from the Legacy Survey of Space and Time](https://arxiv.org/abs/2510.15315). arXiv:2510.15315. 2024. [[code](https://github.com/prob-ml/bliss)]

Jeffrey Regier. [Neural posterior estimation with autoregressive tiling for detecting objects in astronomical images](https://arxiv.org/abs/2510.03074). arXiv:2510.03074. 2024. [[code](https://github.com/prob-ml/bliss)]

Aakash Patel, Tianqing Zhang, Camille Avestruz, Jeffrey Regier, and LSST Dark Energy Science Collaboration. [Neural posterior estimation for cataloging astronomical images with spatially varying backgrounds and point spread functions](https://doi.org/10.3847/1538-3881/ad5e6e). The Astronomical Journal. 2024. [[code](https://github.com/prob-ml/bliss)]

Runjing Liu, Jon D. McAuliffe, Jeffrey Regier, and The LSST Dark Energy Science Collaboration. [Variational inference for deblending crowded starfields](https://www.jmlr.org/papers/volume24/21-0169/21-0169.pdf). Journal of Machine Learning Research. 2023.

Mallory Wang, Ismael Mendoza, Cheng Wang, Camille Avestruz, and Jeffrey Regier. [Statistical inference for coadded astronomical images](https://ml4physicalsciences.github.io/2022/files/NeurIPS_ML4PS_2022_167.pdf). NeurIPS Workshop on Machine Learning and the Physical Sciences. 2022.

Yash Patel and Jeffrey Regier. [Scalable Bayesian inference for detecting strong gravitational lensing systems](https://ml4physicalsciences.github.io/2022/files/NeurIPS_ML4PS_2022_155.pdf). NeurIPS Workshop on Machine Learning and the Physical Sciences. 2022.

Derek Hansen, Ismael Mendoza, Runjing Liu, Ziteng Pang, Zhe Zhao, Camille Avestruz, and Jeffrey Regier. [Scalable Bayesian inference for detection and deblending in astronomical images](https://ml4astro.github.io/icml2022/assets/27.pdf). ICML Workshop on Machine Learning for Astrophysics. 2022.
