bliss package
=============

BLISS is organized into several subpackages:

- **Core modules**: Entry points, catalogs, datasets, and utilities
- **Encoder**: Neural network architecture for posterior estimation
- **Simulator**: Image generation and prior distributions
- **Surveys**: Survey-specific data loaders (SDSS, DES, DC2)

Core Modules
------------

bliss.main module
^^^^^^^^^^^^^^^^^

The main entry point for BLISS workflows: data generation, training, and prediction.

.. automodule:: bliss.main
   :members:
   :no-undoc-members:

bliss.catalog module
^^^^^^^^^^^^^^^^^^^^

Tile and full catalog data structures for representing light source parameters.

.. automodule:: bliss.catalog
   :members:
   :no-undoc-members:

bliss.cached_dataset module
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dataset classes for loading cached simulated or survey data.

.. automodule:: bliss.cached_dataset
   :members:
   :no-undoc-members:

bliss.align module
^^^^^^^^^^^^^^^^^^

Image alignment utilities for multi-band reprojection.

.. automodule:: bliss.align
   :members:
   :no-undoc-members:

bliss.data_augmentation module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data augmentation transforms for training.

.. automodule:: bliss.data_augmentation
   :members:
   :no-undoc-members:

bliss.global_env module
^^^^^^^^^^^^^^^^^^^^^^^

Global environment variables for distributed training.

.. automodule:: bliss.global_env
   :members:
   :no-undoc-members:

Encoder Package
---------------

Neural network components for variational inference.

.. automodule:: bliss.encoder
   :members:
   :no-undoc-members:

bliss.encoder.encoder module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Main Encoder class implementing the neural posterior estimator.

.. automodule:: bliss.encoder.encoder
   :members:
   :no-undoc-members:

bliss.encoder.variational_dist module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Variational distribution factors for different latent variables.

.. automodule:: bliss.encoder.variational_dist
   :members:
   :no-undoc-members:

bliss.encoder.metrics module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluation metrics for detection, classification, and flux estimation.

.. automodule:: bliss.encoder.metrics
   :members:
   :no-undoc-members:

bliss.encoder.convnets module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Feature extraction neural networks.

.. automodule:: bliss.encoder.convnets
   :members:
   :no-undoc-members:

bliss.encoder.convnet_layers module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Building blocks for convolutional networks.

.. automodule:: bliss.encoder.convnet_layers
   :members:
   :no-undoc-members:

bliss.encoder.image_normalizer module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Image normalization strategies (CLAHE, asinh, PSF).

.. automodule:: bliss.encoder.image_normalizer
   :members:
   :no-undoc-members:

bliss.encoder.sample_image_renders module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualization utilities for sample predictions.

.. automodule:: bliss.encoder.sample_image_renders
   :members:
   :no-undoc-members:

Surveys Package
---------------

Survey-specific data loaders and utilities.

.. automodule:: bliss.surveys
   :members:
   :no-undoc-members:

bliss.surveys.survey module
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Base class for survey implementations.

.. automodule:: bliss.surveys.survey
   :members:
   :no-undoc-members:

bliss.surveys.sdss module
^^^^^^^^^^^^^^^^^^^^^^^^^

Sloan Digital Sky Survey data loader.

.. automodule:: bliss.surveys.sdss
   :members:
   :no-undoc-members:

bliss.surveys.des module
^^^^^^^^^^^^^^^^^^^^^^^^

Dark Energy Survey data loader.

.. automodule:: bliss.surveys.des
   :members:
   :no-undoc-members:

bliss.surveys.dc2 module
^^^^^^^^^^^^^^^^^^^^^^^^

DESC DC2 simulated survey data loader.

.. automodule:: bliss.surveys.dc2
   :members:
   :no-undoc-members:

bliss.surveys.download_utils module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Utilities for downloading survey data.

.. automodule:: bliss.surveys.download_utils
   :members:
   :no-undoc-members:

Simulator Package
-----------------

Image simulation and prior distributions.

.. automodule:: bliss.simulator
   :members:
   :no-undoc-members:

bliss.simulator.decoder module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Image decoder for rendering synthetic telescope images.

.. automodule:: bliss.simulator.decoder
   :members:
   :no-undoc-members:

bliss.simulator.prior module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prior distributions over light source catalogs.

.. automodule:: bliss.simulator.prior
   :members:
   :no-undoc-members:

bliss.simulator.psf module
^^^^^^^^^^^^^^^^^^^^^^^^^^

Point spread function utilities.

.. automodule:: bliss.simulator.psf
   :members:
   :no-undoc-members:
