Experiment
========
This directory the scripts required to reproduce results from the upcoming publication "Simulation-Based Inference for Probabilistic Galaxy Detection and Deblending".

## Steps to reproduce results

0. First you need to set the paths where you want to save various intermediate outputs like datasets
and models. The paths can be set manually in `config/__init__.py`.

The random seed used for all the steps below can also be set here. The default is `52`.

1. **Creating datasets**: Simulated datasets are required to train and evaulate the machine learning models.
These simulated datasets can be reproduced by running:

```bash
./make_dataset.py --all
```

Individual datasets can be created:

```bash
./make_datasets.py --single
./make_datasets.py --blends
./make_datasets.py --tiles
./make_datasets.py --central
```

as described in the paper. Datasets are saved in the path specified in the `DATASETS_DIR` from the `config/__init__.py`.

2. **Training models**: There is a set of pre-trained models already provided in the `models` folder, but the
models can also be reproduced by running:

```bash
./train_models.py --all
```

A GPU is required for training the models in a reasonable amount of time. The models can also be trained individually like:

```bash
./train_models.py --detection
./train_models.py --binary
./train_models.py --deblend
```

Models are saved in the path specified in the `MODELS_DIR` from the `config/__init__.py`.

3. Get results...
