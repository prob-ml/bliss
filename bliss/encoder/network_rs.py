# Author: Qiaozhi Huang
# Class for network to predict redshift prediction
import numpy as np
import torch
from tqdm import tqdm
    
class Regressor(torch.nn.Module):
    def __init__(self,
        in_dim,                             # Number of input features.
        hidden_dim,                         # Number of input features.
        out_dim,                            # Number of output features.
        device,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device
        # self.loss_fcn = loss_fcn

        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.Softplus(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Softplus(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Softplus(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
    
        