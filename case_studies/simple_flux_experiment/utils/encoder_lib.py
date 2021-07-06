import torch
from torch import nn

class MLPEncoder(nn.Module):
    def __init__(self, 
                 slen = 51, 
                 latent_dim = 64):

        super(MLPEncoder, self).__init__()

        # image / model parameters
        self.slen = slen
        self.n_pixels = self.slen ** 2

        self.fc1 = nn.Linear(self.n_pixels, latent_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = nn.Linear(latent_dim, 2)
        
        self.softplus = torch.nn.Softplus()
        
    def forward(self, image):
        
        batchsize = image.shape[0]
        
        # pass through neural network
        h = image.view(batchsize, self.n_pixels)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.fc2(h)
        
        mean = h[:, 0]
        sd = self.softplus(h[:, 1]) + 0.001
    
        return mean, sd
