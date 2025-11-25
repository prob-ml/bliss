Installation Guide
===================

Quick Install
#############

**Requirements:** Python 3.12 or higher

BLISS is pip installable::

    pip install bliss-toolkit

Developer Installation
######################

1. Install `poetry <https://python-poetry.org/docs/>`_.

2. Install the `fftw <http://www.fftw.org>`_ library (used by galsim)::

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
