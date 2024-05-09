# Author: Qiaozhi Huang
# Class for network to predict redshift prediction
import numpy as np
import torch
from torch_utils import misc
from torch_utils.ops import bias_act

class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'relu',   # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'
    
class Regressor(torch.nn.Module):
    def __init__(self,
        in_dim,                             # Number of input features.
        hidden_dim,                         # Number of input features.
        out_dim,                            # Number of output features.
        device,
        loss_fcn = torch.nn.MSELoss()       # loss function
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = device
        self.loss_fcn = loss_fcn

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(
                in_dim, 
                self.hidden_dim
            ),
            torch.nn.Softplus(),
            FullyConnectedLayer(
                self.hidden_dim,
                self.hidden_dim,
            ),
            torch.nn.Softplus(),
            FullyConnectedLayer(
                self.hidden_dim,
                self.out_dim,
            )
        )

    def forward(self, x):
        return self.net(x)
        