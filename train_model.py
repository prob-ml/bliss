#!/usr/bin/env python3

import argparse
import numpy as np
import subprocess
import timeit
from pathlib import Path
import torch
import matplotlib.pyplot as plt

import train_color, train_galaxy

torch.backends.cudnn.benchmark = True
plt.switch_backend("Agg")

all_models = {
    'centered_galaxy': train_galaxy.TrainGalaxy,
    'color': train_color.TrainColor
}


def training(train_module, epochs):
    dir_path = Path(args.dir)
    dir_path.mkdir()

    for epoch in range(0, epochs):
        np.random.seed(args.seed + epoch)
        start_time = timeit.default_timer()
        batch_loss = train_module.train_epoch()
        elapsed = timeit.default_timer() - start_time
        print('[{}] loss: {:.3f}  \t[{:.1f} seconds]'.format(epoch, batch_loss, elapsed))

        if epoch % 10 == 0:
            train_module.evaluate_and_log()


def run():
    if args.model not in all_models:
        raise NotImplementedError("Not implemented this model yet.")

    train_module = all_models[args.model].from_args(args)
    train_module.vae.cuda()
    training(train_module, epochs=args.epochs)


if __name__ == "__main__":
    print('hello')
    print()

    # Setup arguments.
    parser = argparse.ArgumentParser(description='Training model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=int, default=0, metavar='DEV',
                        help='GPU device ID')
    parser.add_argument('--dir', type=str, default="test", metavar='DIR',
                        help='run-specific directory to read from / write to')
    parser.add_argument('--overwrite', action='store_true',
                        help='Whether to overwrite if directory already exists.')
    parser.add_argument('--seed', type=int, default=64, metavar='S',
                        help='Random seed for tensor flow cuda.')
    parser.add_argument('--nocuda', action='store_true',
                        help="whether to using a discrete graphics card")

    # specify model and dataset.
    parser.add_argument('--model', type=str, help='What model we are training?')
    parser.add_argument('--dataset', type=str, default="galbasic", metavar='DS',
                        help='Specifies the dataset to be used to train the model.')

    # General training arguments.
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='input batch size for training.')
    parser.add_argument('--num-examples', type=int, default=1000, metavar='NI',
                        help='Number of examples used to train in a single epoch.')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',
                        help='number of epochs to train.')
    params1 = list(vars(parser.parse_known_args()).keys())

    # galaxy training models
    one_centered_galaxy_group = parser.add_argument_group('One Centered Galaxy Model', 'Specify options for the galaxy '
                                                                                       'model to train')
    train_galaxy.TrainGalaxy.add_args(one_centered_galaxy_group, params1)

    args = parser.parse_args()

    # Additional settings.
    args.dir = "/home/imendoza/deblend/galaxy-net/data/" + args.dir
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # check if directory exists or if we should overwrite.
    if Path(args.dir).is_dir() and not args.overwrite:
        raise IOError("Directory already exists.")

    elif Path(args.dir).is_dir():
        subprocess.run(f"rm -r {args.dir}", shell=True)

    with torch.cuda.device(args.device):
        run()
