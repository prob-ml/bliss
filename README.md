![](http://portal.nersc.gov/project/dasrepo/celeste/sample_sky.jpg)


Bayesian Light Source Separator (BLISS)
========================================
[![](https://img.shields.io/badge/docs-master-blue.svg)](https://prob-ml.github.io/bliss/)
[![tests](https://github.com/prob-ml/bliss/workflows/tests/badge.svg)](https://github.com/prob-ml/bliss/actions/workflows/tests.yml)
[![codecov.io](https://codecov.io/gh/prob-ml/bliss/branch/master/graphs/badge.svg?branch=master&token=Jgzv0gn3rA)](http://codecov.io/github/prob-ml/bliss?branch=master)
[![PyPI](https://img.shields.io/pypi/v/bliss-toolkit.svg)](https://pypi.org/project/bliss-toolkit)

# Introduction

BLISS is a Bayesian method for deblending and cataloging light sources. BLISS provides
  - __Accurate estimation__ of parameters in crowded fields with overlapping sources.
  - __Calibrated uncertainties__ through fitting an approximate Bayesian posterior.
  - __Scalability__ of Bayesian inference to large astronomical surveys (SDSS, DES, DC2/LSST).

# Why BLISS?

Astronomical cataloging is a mathematically ill-posed problem, meaning traditional deterministic pipelines produce statistically incoherent results without calibrated uncertainties. BLISS addresses this fundamental limitation by providing fully Bayesian inference through neural posterior estimation.

**Performance**: BLISS systematically outperforms standard survey pipelines (including LSST's) for light source detection, flux measurement, star/galaxy classification, and galaxy shape measurement.

**Scalability**: BLISS scales to petabyte-scale surveys containing billions of celestial objects, having achieved state-of-the-art results on SDSS, DES, and DC2 simulations.

BLISS uses neural posterior estimation (NPE), a form of simulation-based inference (SBI), with techniques including
  - __Amortized inference__, in which a neural network maps telescope images to an approximate Bayesian posterior on parameters of interest.
  - __Autoregressive tiling__ with K-color checkerboard patterns that align the variational distribution's conditional dependencies with the true posterior, greatly improving calibration.

Beyond basic cataloging, BLISS supports multiple downstream scientific applications:
  - __Weak gravitational lensing__ - shear and convergence estimation.
  - __Photometric redshifts (BLISS-PZ)__ - direct redshift prediction from images with multiple variational families.
  - __Galaxy cluster detection__ - cluster membership prediction validated on DES data.

# Installation

**Requirements:** Python 3.10 or higher

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
export POETRY_VIRTUALENVS_IN_PROJECT=1
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

Configuration files are located in `bliss/conf/` and can be composed and overridden via command line arguments. See the [documentation](https://prob-ml.github.io/bliss/) for details.

# Latest updates

## Weak Gravitational Lensing (2024)
   - Shear (γ) and convergence (κ) parameter estimation
   - Extensive validation on DC2 and DECaLS weak lensing simulation (descwl)
   - Redshift-binned analysis for tomographic weak lensing studies
   - Custom encoder architecture optimized for shear estimation

## Photometric Redshifts (2024)
   - BLISS-PZ: Direct photo-z estimation from multi-band images
   - Multiple variational families: discrete categorical, continuous, B-spline basis, mixture density networks
   - Competitive performance with RAIL pipeline methods
   - Integration with DC2 redshift data

## Galaxy Cluster Detection (2024)
   - Cluster membership prediction from DES DR2 data
   - Validated against redMaPPer cluster catalog
   - Full inference capability over large survey volumes

## Cataloging Improvements
   - Support for multiple surveys: SDSS, DES, DC2/LSST
   - Autoregressive tiling for efficient large-image processing
   - Spatially varying PSF and background handling
   - Rich catalog system with WCS support and coordinate transformations

## Stars and Galaxies
   - BLISS includes StarNet functionality for crowded star fields
   - Parametric galaxy model (bulge + disk) for realistic galaxy rendering
   - Detection, measurement, and deblending of both stars and galaxies

# Case Studies

The `case_studies/` directory contains research applications of BLISS:

- **weak_lensing/** - Shear and convergence estimation
- **redshift/** - Photo-z estimation (BLISS-PZ) with multiple variational families
- **galaxy_clustering/** - Galaxy cluster detection and membership prediction
- **dc2_cataloging/** - Full cataloging pipeline for DC2 simulation
- **dc2_multidetection/** - Multi-detection strategies for DC2
- **strong_lensing/** - Strong gravitational lens detection
- **spatial_tiling/** - Tiling strategies for large images
- **psf_variation/** - Handling spatially varying PSFs
- **prior_elicitation/** - Methods for specifying priors
- **multiband/** - Multi-band image processing

Each case study includes training scripts, configuration files, and evaluation notebooks.

# References

Yicun Duan, Xinyue Li, Camille Avestruz, Jeffrey Regier, and LSST Dark Energy Collaboration. [Neural posterior estimation for cataloging astronomical images from the Legacy Survey of Space and Time](https://arxiv.org/abs/2510.15315). arXiv:2510.15315. 2024. [[code](https://github.com/prob-ml/bliss)]

Jeffrey Regier. [Neural posterior estimation with autoregressive tiling for detecting objects in astronomical images](https://arxiv.org/abs/2510.03074). arXiv:2510.03074. 2024. [[code](https://github.com/prob-ml/bliss)]

Aakash Patel, Tianqing Zhang, Camille Avestruz, Jeffrey Regier, and LSST Dark Energy Science Collaboration. [Neural posterior estimation for cataloging astronomical images with spatially varying backgrounds and point spread functions](https://doi.org/10.3847/1538-3881/ad5e6e). The Astronomical Journal. 2024. [[code](https://github.com/prob-ml/bliss)]

Runjing Liu, Jon D. McAuliffe, Jeffrey Regier, and The LSST Dark Energy Science Collaboration. [Variational inference for deblending crowded starfields](https://www.jmlr.org/papers/volume24/21-0169/21-0169.pdf). Journal of Machine Learning Research. 2023.

Mallory Wang, Ismael Mendoza, Cheng Wang, Camille Avestruz, and Jeffrey Regier. [Statistical inference for coadded astronomical images](https://ml4physicalsciences.github.io/2022/files/NeurIPS_ML4PS_2022_167.pdf). NeurIPS Workshop on Machine Learning and the Physical Sciences. 2022.

Yash Patel and Jeffrey Regier. [Scalable Bayesian inference for detecting strong gravitational lensing systems](https://ml4physicalsciences.github.io/2022/files/NeurIPS_ML4PS_2022_155.pdf). NeurIPS Workshop on Machine Learning and the Physical Sciences. 2022.

Derek Hansen, Ismael Mendoza, Runjing Liu, Ziteng Pang, Zhe Zhao, Camille Avestruz, and Jeffrey Regier. [Scalable Bayesian inference for detection and deblending in astronomical images](https://ml4astro.github.io/icml2022/assets/27.pdf). ICML Workshop on Machine Learning for Astrophysics. 2022.
