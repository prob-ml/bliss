# Author: Qiaozhi Huang
# train to predict redshift using network
# python case_studies/redshift/train_rs.py --resume-model=/home/qiaozhih/bliss/case_studies/redshift/training_runs/00017-runreg/000000_model.pt --resume-opt=/home/qiaozhih/bliss/case_studies/redshift/training_runs/00017-runreg/000000_opt.pt --nick=reg
import json
import os
import re

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from network_rs import LitRegressor, Regressor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


@click.command()
# Optinal features
@click.option('--resume-model', help='Resume from given network pt', metavar='[PATH|URL]',  type=str)
@click.option('--resume-opt',   help='Resume from given optimizer pt', metavar='[PATH|URL]',  type=str)
@click.option('--nick',   help='Nickname for remember', type=str)

def main(resume_model, resume_opt, nick):
    ###### Prepare Training set
    print("start reading dataset!")
    dataset_name = 'desc_dc2_run2.2i_dr6_truth_nona_train'
    # dataset_name = 'desc_dc2_run2.2i_dr6_truth_nona_train_small'
    # dataset_name = 'test'
    path = f'/home/qiaozhih/bliss/data/redshift/dc2/{dataset_name}.pkl'
    photo_z = pd.read_pickle(path)
    print("finish reading dataset!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 512
    seed = 42
    num_bins = 1
    group_size = 128

    print("start tensor dataset preparation!")
    x = photo_z.values[:,:-1].astype(float)
    y = photo_z.values[:, -1].astype(float)
    n_samples, n_features_x = x.shape
    n_features_y = 1
    n_samples = n_samples // group_size * group_size

    x_train = np.array(x[:n_samples])
    y_train = np.array(y[:n_samples])
    # y_train = bin_target(y_train, num_bins) # bin
    tensor_x = torch.Tensor(x_train).view(-1, group_size, n_features_x)
    tensor_y = torch.Tensor(y_train).view(-1, group_size, n_features_y)
    # tensor_x = torch.Tensor(x_train)
    # tensor_y = torch.Tensor(y_train)
    custom_dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    print("finish tensor dataset preparation!")


    ###### Construct Network
    options = {
        'hidden_dim': 512,
        'out_dim': num_bins,
        'n_epochs': 50001,
        'outdir': '/home/qiaozhih/bliss/case_studies/redshift/training_runs/',
        'snap': 1,                                        # how many epoches to save one model once
        'loss_fcn': torch.nn.MSELoss(),                     # loss func
        # 'loss_fcn': torch.nn.CrossEntropyLoss(),                     # loss func
        'dropout_rate': 0.5,
        'learning_rate': 1e-3,
        'group_size': group_size,
    }
    in_dim = x.shape[1]

    reg = Regressor(in_dim, options['hidden_dim'], options['out_dim'], options['dropout_rate'])
    # reg = LitRegressor(in_dim, options['hidden_dim'], options['out_dim'], options['dropout_rate'], options['learning_rate'], options['loss_fcn'])
    optimizer = torch.optim.Adam(reg.parameters(), lr=options['learning_rate'])
    reg = reg.to(device)

    if resume_model is not None and resume_opt is not None:
        print(resume_model)
        reg, optimizer = load_model(reg, optimizer, resume_model, resume_opt)
        print('resume successfully!')

    reg = nn.DataParallel(reg)
    reg = reg.to(device)
    ###### Start Train
    # Pick output directory.
    outdir = options['outdir']
    os.makedirs(outdir, exist_ok=True)
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    nickname = ''
    if nick != None:
        nickname = nick
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-run{nickname}')
    os.makedirs(run_dir)
    print("created running directory!")
    writer = SummaryWriter(log_dir=run_dir)
    logger = TensorBoardLogger(save_dir=run_dir, name='tensorboard_logs')

    # train
    print("start training")
    train(reg, optimizer, dataloader, options, run_dir, writer)

    # trainer = pl.Trainer(max_epochs=options['n_epochs'], default_root_dir=run_dir, logger=logger)
    # trainer.fit(model=reg, train_dataloaders=dataloader)
    writer.close()
    print('finish training!')

def train_one_epoch(model, optimizer, dataloader, options, device='cuda'):
    losses = []
    loss_fcn = options['loss_fcn']
    total_batches = len(dataloader)
    for idx, (x, y) in tqdm(enumerate(dataloader), total=total_batches, unit='batch'):
    # for idx, (x, y) in enumerate(dataloader):
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        # print(f'x shape:{x.shape}')
        # print(f'y shape:{y.shape}')
        x = x.to(device)
        if options['out_dim'] != 1:
            y = y.long().to(device)
        else:
            y = y.to(device)
        N, _ = x.shape
        optimizer.zero_grad()
        if options['out_dim'] != 1:
            loss = loss_fcn(model(x).view(N, -1), y.view(N))
        else:
            loss = loss_fcn(model(x).view(-1), y.view(N))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return torch.tensor(losses).mean()

def train(model, optimizer, dataloader, options, run_dir='.', writer=SummaryWriter()):
    n_epochs = options['n_epochs']
    stats_jsonal = None
    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
    fields = []
    for i in range(n_epochs):
        this_epoch_average_loss = train_one_epoch(model, optimizer, dataloader, options)
        writer.add_scalar("Loss/train", this_epoch_average_loss, i)
        writer.flush()

        if i % options['snap'] == 0:
            # print('Epoch {}: Avg. Loss {}'.format(i, this_epoch_average_loss))
            fields = ['Epoch {}: Avg. Loss {}'.format(i, this_epoch_average_loss)]
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
            path = os.path.join(run_dir, f'{i:06d}')
            save_checkpoint(model, optimizer, path)

def save_checkpoint(model, optimizer, path):
    """
    save optimizer and model's state_dict
    """
    torch.save(model.state_dict(), f"{path}_model.pt")
    torch.save(optimizer.state_dict(), f"{path}_opt.pt")

def load_model(model, optimizer, model_path, optimizer_path):
    """
    load pretrained model/optimizer
    """
    model.load_state_dict(torch.load(model_path))
    optimizer.load_state_dict(torch.load(optimizer_path))
    model.eval()
    return model, optimizer

def bin_target(y, num_bins=30):
    """
    Params:
    y: ndarray
    ---
    Returns:
    ndarry representing the respective bin(indices) for each value of y
    """
    bin_edges = np.linspace(y.min(), y.max(), num_bins + 1)
    y_binned = pd.cut(y, bins=bin_edges, labels=False, include_lowest=True)
    return y_binned

if __name__ == "__main__":
    main()
