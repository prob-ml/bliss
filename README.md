![](http://portal.nersc.gov/project/dasrepo/celeste/sample_sky.jpg)


Bayesian Light Source Separator (BLISS)
========================================
[![](https://img.shields.io/badge/docs-master-blue.svg)](https://prob-ml.github.io/bliss/)
[![tests](https://github.com/prob-ml/bliss/workflows/tests/badge.svg)](https://github.com/prob-ml/bliss/actions/workflows/tests.yml)
[![codecov.io](https://codecov.io/gh/prob-ml/bliss/branch/master/graphs/badge.svg?branch=master&token=Jgzv0gn3rA)](http://codecov.io/github/prob-ml/bliss?branch=master)
[![PyPI](https://img.shields.io/pypi/v/bliss-toolkit.svg)](https://pypi.org/project/bliss-toolkit)

# Introduction

Over the next decade, surveys like the Rubin Observatory's Legacy Survey of Space and Time (LSST) and the Euclid Space Telescope will produce petabytes of imaging data at unprecedented depth. Extracting scientific information from these images, with well-calibrated uncertainties, is a central computational challenge facing modern cosmology. We have entered an era of systematics-limited cosmology, where our understanding of the Universe is constrained more by systematic uncertainties than by data scarcity. In LSST images, over 60% of imaged sources will visually overlap (blending), and traditional pipelines discard faint objects through conservative flux cuts, excluding high-redshift galaxies that provide the greatest constraining power for dark energy.

The BLISS project is an interdisciplinary collaboration between statisticians and physicists. BLISS (Bayesian Light Source Separator) is a Bayesian method for analyzing astronomical images from large cosmological surveys. BLISS provides:
  - __Accurate estimation__ of parameters in crowded fields with overlapping sources
  - __Calibrated uncertainties__ through fully Bayesian inference via neural posterior estimation (NPE)
  - __Scalability__ to petabyte-scale surveys containing billions of celestial objects

BLISS uses neural posterior estimation (NPE), a form of simulation-based inference in which a neural network maps telescope images directly to an approximate Bayesian posterior. Unlike traditional MCMC samplers, which require expensive per-pixel likelihood evaluations and are intractable at survey scale, NPE amortizes the cost of inference: once trained on simulated data, the network performs posterior inference near-instantaneously. Autoregressive tiling with checkerboard patterns aligns the variational distribution's conditional dependencies with the true posterior, improving calibration. By propagating uncertainty through principled posterior distributions, BLISS enables astronomers to extract scientific information from the full depth of survey data, rather than only its brightest fraction. BLISS systematically outperforms standard survey pipelines for light source detection, flux measurement, star/galaxy classification, and galaxy shape measurement.

# Installation

**Requirements:** Python 3.12 or higher

BLISS is pip installable with the following command:

```bash
pip install bliss-toolkit
```

The required dependencies are listed in the `[project]` block of the `pyproject.toml` file.

# Installation (Developers)

1. Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install the [fftw](http://www.fftw.org) library (which is used by `galsim`). With Ubuntu you can install it by running

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

5. To create a virtual environment with the `bliss` dependencies satisfied, run

```bash
cd bliss
uv sync
```

6. Activate the virtual environment:

```bash
source .venv/bin/activate
```

7. Verify that bliss is installed correctly by running the tests both on your CPU (default) and on your GPU:

```bash
pytest
pytest --gpu
```

8. Finally, if you are planning to contribute code to this repository, consider installing our pre-commit hooks so that your code commits will be checked locally for compliance with our coding conventions:

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

- **weak_lensing/** - Tomographic shear (γ) and convergence (κ) mapping
- **redshift/** - Photo-z estimation (BLISS-PZ) with multiple variational families
- **galaxy_clustering/** - Galaxy cluster detection and membership prediction, validated on DES DR2
- **dc2_cataloging/** - Full cataloging pipeline for DC2 simulation
- **strong_lensing/** - Strong gravitational lens detection
- **spatial_tiling/** - Spatially autoregressive tiling for astronomical cataloging
- **psf_variation/** - Spatially varying PSFs for astronomical cataloging

# References

Yicun Duan, Xinyue Li, Camille Avestruz, Jeffrey Regier, and LSST Dark Energy Science Collaboration. [Neural posterior estimation for cataloging astronomical images from the Legacy Survey of Space and Time](https://doi.org/10.3847/1538-3881/ae30df). The Astronomical Journal. 2026.

Jeffrey Regier. [Neural posterior estimation with autoregressive tiling for detecting objects in astronomical images](https://doi.org/10.1214/25-AOAS2125). Annals of Applied Statistics. 2026.

Ismael Mendoza, Derek Hansen, Runjing Liu, Zhe Zhao, Ziteng Pang, Axel Guinot, Camille Avestruz, Jeffrey Regier, and LSST Dark Energy Science Collaboration. [Simulation-based inference for probabilistic galaxy detection and deblending](https://doi.org/10.33232/001c.158908). The Open Journal of Astrophysics. 2026.

Aakash Patel, Tianqing Zhang, Camille Avestruz, Jeffrey Regier, and LSST Dark Energy Science Collaboration. [Neural posterior estimation for cataloging astronomical images with spatially varying backgrounds and point spread functions](https://iopscience.iop.org/article/10.3847/1538-3881/adef32). The Astronomical Journal. 2025.

Runjing Liu, Jon D. McAuliffe, Jeffrey Regier, and LSST Dark Energy Science Collaboration. [Variational inference for deblending crowded starfields](https://www.jmlr.org/papers/volume24/21-0169/21-0169.pdf). Journal of Machine Learning Research. 2023.

Mallory Wang, Ismael Mendoza, Cheng Wang, Camille Avestruz, and Jeffrey Regier. [Statistical inference for coadded astronomical images](https://ml4physicalsciences.github.io/2022/files/NeurIPS_ML4PS_2022_167.pdf). NeurIPS Workshop on Machine Learning and the Physical Sciences. 2022.

Yash Patel and Jeffrey Regier. [Scalable Bayesian inference for detecting strong gravitational lensing systems](https://ml4physicalsciences.github.io/2022/files/NeurIPS_ML4PS_2022_155.pdf). NeurIPS Workshop on Machine Learning and the Physical Sciences. 2022.

Derek Hansen, Ismael Mendoza, Runjing Liu, Ziteng Pang, Zhe Zhao, Camille Avestruz, and Jeffrey Regier. [Scalable Bayesian inference for detection and deblending in astronomical images](https://ml4astro.github.io/icml2022/assets/27.pdf). ICML Workshop on Machine Learning for Astrophysics. 2022.
