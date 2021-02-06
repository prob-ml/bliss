Installation Guide
===================

1. To use and install `bliss` you first need to install `poetry <https://python-poetry.org/docs/>`_.

2. Then, install the `fftw <http://www.fftw.org>`_ library (which is used by `galsim`). For example, in unix you can simply do::

    sudo apt-get install libfftw3-dev

3. If everything is installed correctly you should be able to run the commands below. This will create a poetry environment that contains only the `bliss` dependencies.::

    git clone https://github.com/prob-ml/bliss.git
    cd bliss
    poetry install
    poetry shell

4. Finally, set an environment variable called :code:`BLISS_HOME` to the absolute path of the bliss directory.
