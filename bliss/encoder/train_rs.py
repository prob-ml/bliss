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

def main():
    ###### Prepare Training set
    print("start reading dataset!")
    # dataset_name = 'desc_dc2_run2.2i_dr6_truth'
    dataset_name = 'test'
    path = f'/home/qiaozhih/bliss/data/redshift/dc2/{dataset_name}.pkl'
    photo_z = pd.read_pickle(path)
    print("finish reading dataset!")
    device = 'cuda'
    num_data = 10
    photo_z = photo_z[:num_data]
    
    print("start tensor dataset preparation!")
    x = photo_z.values[:,:-1].astype(float)
    y = photo_z.values[:, -1].astype(float)
    x_train = np.array(x)
    y_train = np.array(y)
    tensor_x = torch.Tensor(x_train).to(device)
    tensor_y = torch.Tensor(y_train).to(device)
    custom_dataset = TensorDataset(tensor_x, tensor_y)     
    dataloader = DataLoader(custom_dataset, batch_size=2048)
    train_dataloader, val_dataloader = torch.utils.data.random_split(dataloader, [int(len(dataloader) * 0.8), len(dataloader) - int(len(dataloader) * 0.8)]) 
    print("finish tensor dataset preparation!")
    
    
    ###### Construct Network
    options = {
        'hidden_dim': 256,
        'out_dim': 1,
        'n_epochs': 1000,
        'n_pred': 1000, 
        'outdir': '/home/qiaozhih/bliss/bliss/encoder/training_runs/'
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
    print("Created running directory!")

    # train
    print("Start Training")
    train(reg, optimizer, train_dataloader, options['n_epochs'], torch.nn.MSELoss(), run_dir)
    # preds = []
    # trues = []
    # for idx, (x, y) in tqdm(enumerate(val_dataloader), unit='batch'):
    #     pred_this_batch = reg.net(x)
    #     preds.append(pred_this_batch)
    #     trues.append(y)
        
    #     if idx > options['n_pred']:
    #         break
    # preds = torch.cat(preds)
    # trues = torch.cat(trues)
    # plt.scatter(preds.detach().cpu().numpy(), trues.detach().cpu().numpy(), alpha=0.1, s=3)


def train_one_epoch(model, optimizer, dataloader, loss_fcn = torch.nn.MSELoss()):
    losses = []
    for idx, (x, y) in tqdm(enumerate(dataloader), unit='batch'):
        optimizer.zero_grad()
        loss = loss_fcn(model(x).view(-1), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
            
        return torch.tensor(losses).mean()
    
def train(model, optimizer, dataloader, n_epochs=100, loss_fcn = torch.nn.MSELoss(), run_dir='.'):
    for i in range(n_epochs):
        this_epoch_average_loss = train_one_epoch(model, optimizer, dataloader)
        print('Epoch {}: Avg. Loss {}'.format(i, this_epoch_average_loss))
        if i % 5 == 0:
            # TODO modfy path
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
    return model, optimizer

if __name__ == "__main__":
    main()