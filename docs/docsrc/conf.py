# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

sys.path.insert(0, os.path.abspath("../../bliss"))


# -- Project information -----------------------------------------------------

# pylint: disable=redefined-builtin
project = "bliss"
copyright = "2021, Probabilistic Machine Learning Research Group"
author = "Derek Hansen, Bryan Liu, Ismael Mendoza, Ziteng Pang, Jeffrey Regier, Zhe Zhao"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "nbsphinx",
]

intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "pytorch_lightning": ("https://pytorch-lightning.readthedocs.io/en/stable/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []  # type: ignore


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# For nbsphinx
nbsphinx_execute = "never"

# Below code is necessary to ensure pandoc is available for use by nbsphinx.
# See https://stackoverflow.com/questions/62398231/building-docs-fails-due-to-missing-pandoc/71585691#71585691 # noqa: E501 # pylint: disable=line-too-long
from inspect import getsourcefile  # noqa: E402 # pylint: disable=wrong-import-position

# Get path to directory containing this file, conf.py.
PATH_OF_THIS_FILE = getsourcefile(lambda: 0)  # noqa: WPS522
DOCS_DIRECTORY = os.path.dirname(os.path.abspath(PATH_OF_THIS_FILE))  # type: ignore


def ensure_pandoc_installed(_):
    import pypandoc  # pylint: disable=import-outside-toplevel

    # Download pandoc if necessary. If pandoc is already installed and on
    # the PATH, the installed version will be used. Otherwise, we will
    # download a copy of pandoc into docs/bin/ and add that to our PATH.
    pandoc_dir = os.path.join(DOCS_DIRECTORY, "bin")
    # Add dir containing pandoc binary to the PATH environment variable
    if pandoc_dir not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + pandoc_dir
    pypandoc.ensure_pandoc_installed(
        version="2.11.4",
        targetfolder=pandoc_dir,
        delete_installer=True,
    )


def setup(app):
    app.connect("builder-inited", ensure_pandoc_installed)
