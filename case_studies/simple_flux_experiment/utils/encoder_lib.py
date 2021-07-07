import torch
from torch import nn

from bliss.models.encoder import ImageEncoder

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
        sd = self.softplus(h[:, 1]) + 1e-6
        
        # sd_scale = torch.sigmoid(h[:, 1])
        # sd = mean * sd_scale
    
        return mean, sd

    
class StarNetEncoder(nn.Module):
    def __init__(self, 
                 slen = 51):

        super(StarNetEncoder, self).__init__()

        # image / model parameters
        self.slen = slen
        
        self.encoder = ImageEncoder(tile_slen=self.slen,
                                    ptile_slen=self.slen)
        
        self.softplus = torch.nn.Softplus()
        
    def forward(self, image):
        
        h = self.encoder.get_var_params_all(image)
        
        mean = h[:, 0]
        sd = self.softplus(h[:, 1]) + 0.001
        
        return mean, sd
        