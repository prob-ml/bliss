Experiment
========
This directory the scripts required to reproduce results from the upcoming publication "Simulation-Based Inference for Probabilistic Galaxy Detection and Deblending".

## Downloading data

The `data` folder included in the repository includes some of the data needed to reproduce experiments:

- `stars_med_june2018.fits`

- `stellar_density_lsst.fits.gz`

Both of this come from Erin Sheldon's simulation input data in `https://github.com/LSSTDESC/descwl-shear-sims`. One of the data products is not included in this repository due to size, please download by running:

```bash
wget https://www.cosmo.bnl.gov/www/esheldon/data/catsim.tar.gz
tar xvfz catsim.tar.gz
```

Then take out the `OneSqDeg.fits` file and put into the `data` folder along with the other input data.

Finally, an environment variable has to be set pointing to this folder (absolute path):

```bash
export BLISS_DATA_DIR=/path/to/data/folder/
```

## Steps to reproduce results

0. First you need to set the paths where you want to save various intermediate outputs like datasets
and models. The paths can be set manually in the `__init__.py`.

The random seed used for all the steps below can also be set here. The default is `52`.

1. **Creating datasets**: Simulated datasets are required to train and evaulate the machine learning models.
These simulated datasets can be reproduced by running:

```bash
./make_datasets.py --all
```

Individual datasets can be created:

```bash
./make_datasets.py --single
./make_datasets.py --blends
./make_datasets.py --tiles
./make_datasets.py --central
```

as described in the paper. Datasets are saved in the path specified in the `DATASETS_DIR` from the `__init__.py`.

2. **Training models**: There is a set of pre-trained models already provided in the `models` folder, but the
models can also be reproduced by running:

```bash
export CUDA_VISIBLE_DEVICES="0"
./train_models.py --all
```

A GPU is required for training the models in a reasonable amount of time. The models can also be trained individually like:

```bash
export CUDA_VISIBLE_DEVICES="0" # set your preferred GPU
./train_models.py --autoencoder
./train_models.py --detection
./train_models.py --binary
./train_models.py --deblend
```

Models are saved in the path specified in the `MODELS_DIR` from the `__init__.py`. Intermediate [pytorch lightning](https://lightning.ai/docs/pytorch/stable/) training metadata is saved in `TORCH_DIR` also specified in `__init__.py`.
It might be useful to specify the version number of the folder where the metadata is saved. For instance running:

```bash
./train_models.py --detection --version 1
```

saves the pytorch lightning metadata in the `{TORCH_DIR} / detection / version_1` folder.

3. Finally, results can be obtained now that all the datasets have been produced and models have been trained:

```bash
export CUDA_VISIBLE_DEVICES="0" # set your preferred GPU
./make_figures.py --all
```

and each individual set of figures (roughly corresponding to a section of the paper) can be obtained via:

```bash
./make_figures --detection
./make_figures --binary
./make_figures --deblend
./make_figures --toy
./make_figures --samples
```

A GPU is also particularly helpful in obtaining results quickly, but not as essential as the training.
