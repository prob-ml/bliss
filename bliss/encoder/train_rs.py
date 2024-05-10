# Author: Qiaozhi Huang
# train to predict redshift using network
import pandas as pd
import numpy as np
import torch
from network_rs import Regressor
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import re
import matplotlib.pyplot as plt
import json

def main():
    ###### Prepare Training set
    print("start reading dataset!")
    dataset_name = 'desc_dc2_run2.2i_dr6_truth_nona_train'
    # dataset_name = 'test'
    path = f'/home/qiaozhih/bliss/data/redshift/dc2/{dataset_name}.pkl'
    photo_z = pd.read_pickle(path)
    print("finish reading dataset!")
    device = 'cuda'
    # num_data = 10000
    batch_size = 2048
    # photo_z = photo_z[:num_data]
    seed = 42
    
    print("start tensor dataset preparation!")
    x = photo_z.values[:,:-1].astype(float)
    y = photo_z.values[:, -1].astype(float)
    x_train = np.array(x)
    y_train = np.array(y)
    tensor_x = torch.Tensor(x_train)
    tensor_y = torch.Tensor(y_train)
    custom_dataset = TensorDataset(tensor_x, tensor_y)     
    dataloader = DataLoader(custom_dataset, batch_size=batch_size)
    print("finish tensor dataset preparation!")
    
    
    ###### Construct Network
    options = {
        'hidden_dim': 256,
        'out_dim': 1,
        'n_epochs': 100001,
        'n_pred': 1000, 
        'outdir': '/home/qiaozhih/bliss/bliss/encoder/training_runs/',
        'snap': 100,                                        # how many epoches to save one model once
        'loss_fcn': torch.nn.MSELoss(),                     # loss func
    }
    in_dim = x.shape[1]
    
    reg = Regressor(in_dim, options['hidden_dim'], options['out_dim'], device)
    reg = reg.to(device)
    optimizer = torch.optim.Adam(reg.parameters(), lr=1e-3)

    # if snap is not True:
    #     reg = torch.load(model_path)
    #     optimizer = torch.load(opt_path)
    
    
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
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-run')
    os.makedirs(run_dir)
    print("created running directory!")

    # train
    print("start training")
    train(reg, optimizer, dataloader, options, run_dir)
    print('finish training!')

def train_one_epoch(model, optimizer, dataloader, loss_fcn = torch.nn.MSELoss(), device='cuda'):
    losses = []
    for idx, (x, y) in tqdm(enumerate(dataloader), unit='batch'):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = loss_fcn(model(x).view(-1), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
            
        return torch.tensor(losses).mean()
    
def train(model, optimizer, dataloader, options, run_dir='.'):
    n_epochs = options['n_epochs']
    stats_jsonal = None
    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
    fields = []
    for i in range(n_epochs):
        this_epoch_average_loss = train_one_epoch(model, optimizer, dataloader, loss_fcn=options['loss_fcn'])
        print('Epoch {}: Avg. Loss {}'.format(i, this_epoch_average_loss))

        fields = ['Epoch {}: Avg. Loss {}'.format(i, this_epoch_average_loss)]
        stats_jsonl.write(json.dumps(fields) + '\n')
        stats_jsonl.flush()
        if i % options['snap'] == 0:
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

if __name__ == "__main__":
    main()