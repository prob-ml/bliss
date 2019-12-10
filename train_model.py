#!/usr/bin/env python3

import argparse
import numpy as np
import subprocess
import timeit
from pathlib import Path
import torch
import matplotlib.pyplot as plt

import train_galaxy
import train_catalogue

torch.backends.cudnn.benchmark = True
plt.switch_backend("Agg")

all_models = {
    'centered_galaxy': train_galaxy.TrainGalaxy,
    'catalogue': train_catalogue.TrainCatalogue
}


def training(train_module, epochs=None, seed=None, evaluate=None, **kwargs):

    for epoch in range(0, epochs):
        np.random.seed(seed + epoch)
        start_time = timeit.default_timer()
        batch_loss = train_module.train_epoch()
        elapsed = timeit.default_timer() - start_time
        print('[{}] loss: {:.3f}  \t[{:.1f} seconds]'.format(epoch, batch_loss, elapsed))

        if evaluate is not None:
            if epoch % evaluate == 0:
                train_module.evaluate_and_log(epoch)


def run(args):
    if args['model'] not in all_models:
        raise NotImplementedError("Not implemented this model yet.")

    train_module = all_models[args['model']].from_args(args)
    train_module.vae.cuda()
    training(train_module, **args)


if __name__ == "__main__":

    # Setup arguments.
    parser = argparse.ArgumentParser(description='Training model [argument parser]',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=int, default=0, metavar='DEV',
                        help='GPU device ID')
    parser.add_argument('--dir-name', type=str, default="test", metavar='DIR',
                        help='run-specific directory to read from / write to')
    parser.add_argument('--overwrite', action='store_true',
                        help='Whether to overwrite if directory already exists.')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='Random seed for tensor flow cuda.')
    parser.add_argument('--nocuda', action='store_true',
                        help="whether to using a discrete graphics card")
    parser.add_argument('--evaluate', type=int, default=None, help='Whether to evaluate and log the model at some'
                                                                   'specific number of epochs.')

    # specify model and dataset.
    parser.add_argument('--model', type=str, help='What model we are training?', choices=list(all_models.keys()),
                        required=True)
    parser.add_argument('--dataset', type=str, default=None, metavar='DS',
                        choices=['galbasic', 'synthetic', 'galcatsim'],
                        help='Specifies the dataset to be used to train the model.')

    # General training arguments.
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='input batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',
                        help='number of epochs to train.')
    parser.add_argument('--training-examples', type=int, default=100, metavar='TE',
                        help='Number of training examples (batches) used to train in a single epoch.')
    parser.add_argument('--evaluation-examples', type=int, default=10, metavar='EE',
                        help='Number of evaluation examples (batches) used to evaluate in a single epoch.')

    one_centered_galaxy_group = parser.add_argument_group('[One Centered Galaxy Model]',
                                                          'Specify options for the galaxy model to train.')
    train_galaxy.TrainGalaxy.add_args(one_centered_galaxy_group)

    catalogue_group = parser.add_argument_group('[Catalogue Model]',
                                                'Specify options for the catalogue model to train.')
    train_catalogue.TrainCatalogue.add_args(catalogue_group)

    # we are done.
    pargs = parser.parse_args()
    args_dict = vars(pargs)

    # Additional settings.
    args_dict['dir_name'] = "/home/imendoza/deblend/galaxy-net/data/" + args_dict['dir_name']
    project_dir = Path(args_dict['dir_name'])

    torch.cuda.manual_seed(pargs.seed)
    np.random.seed(pargs.seed)

    # check if directory exists or if we should overwrite.
    if project_dir.is_dir() and not args_dict['overwrite']:
        raise IOError("Directory already exists.")

    elif project_dir.is_dir():
        subprocess.run(f"rm -r {project_dir.as_posix()}", shell=True)

    project_dir.mkdir()

    # run.
    with torch.cuda.device(args_dict['device']):
        run(args_dict)
