Installation Guide
===================

Quick Install
#############

**Requirements:** Python 3.12 or higher, CUDA-capable GPU

BLISS is pip installable::

    pip install bliss-toolkit

Developer Installation
######################

1. Install `poetry <https://python-poetry.org/docs/>`_::

    curl -sSL https://install.python-poetry.org | python3 -
    poetry config virtualenvs.in-project true

2. Install the `fftw <http://www.fftw.org>`_ library (used by GalSim)::

    sudo apt-get install libfftw3-dev

3. Install git-lfs if you haven't already::

    git-lfs install

4. Clone the repo and install dependencies::

    git clone git@github.com:prob-ml/bliss.git
    cd bliss
    poetry install

5. Verify installation by running tests::

    pytest
    pytest --gpu

6. (Optional) Install pre-commit hooks for development::

    pre-commit install

GPU Requirements
################

BLISS requires a CUDA-capable GPU for training. Check your CUDA version with::

    nvidia-smi

BLISS uses PyTorch with CUDA 12.8 support, which works on both older GPUs (RTX 2080 Ti) and newer architectures.

Core Dependencies
#################

BLISS requires the following core packages:

- **PyTorch** (>=2.7.0): Deep learning framework
- **PyTorch Lightning** (>=2.3.3): Training infrastructure
- **Hydra** (>=1.0.4): Configuration management
- **GalSim** (>=2.4.10): Galaxy image simulation
- **Astropy** (>=6.1.1): Astronomical utilities
- **NumPy**, **SciPy**, **Matplotlib**: Scientific computing

Optional Dependencies
#####################

For specific use cases, you may need additional packages:

**Weak Lensing Analysis**::

    pip install healpy

**Galaxy Clustering**::

    pip install h5py pyarrow

**Photo-z Estimation**::

    pip install scikit-learn

**Visualization**::

    pip install plotly seaborn

**TensorBoard Logging**::

    pip install tensorboard

All optional dependencies are included in the development installation (``poetry install``).

Troubleshooting
###############

**GalSim installation fails:**

Ensure FFTW is installed::

    sudo apt-get install libfftw3-dev  # Ubuntu/Debian
    brew install fftw                   # macOS

**CUDA not found:**

Verify your CUDA installation::

    nvcc --version
    nvidia-smi

**Memory errors during training:**

Reduce batch size in configuration::

    bliss mode=train cached_simulator.batch_size=8
