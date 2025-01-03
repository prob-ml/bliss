import torch

from torch import nn
from case_studies.dc2_new_diffusion.utils.convnet_layers import C3, ConvBlock


class CatalogEncoder(nn.Module):
    def __init__(self, cat_dim, hidden_dim):
        super().__init__()

        self.cat_dim = cat_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            ConvBlock(cat_dim, hidden_dim),
            *[ConvBlock(hidden_dim, hidden_dim) for _ in range(2)],
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(hidden_dim, hidden_dim),
            C3(hidden_dim, hidden_dim, n=3, shortcut=False),
            ConvBlock(hidden_dim, hidden_dim // 2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(hidden_dim // 2, hidden_dim // 2),
            C3(hidden_dim // 2, hidden_dim // 2, n=3, shortcut=False),
            ConvBlock(hidden_dim // 2, hidden_dim // 4),
            ConvBlock(hidden_dim // 4, hidden_dim // 4, kernel_size=1, use_relu=True)
        )

    def forward(self, cat_tensor):
        return self.encoder(cat_tensor)
    

class CatalogDecoder(nn.Module):
    def __init__(self, cat_dim, hidden_dim):
        super().__init__()

        self.cat_dim = cat_dim
        self.hidden_dim = hidden_dim

        self.decoder = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim, kernel_size=5),
            ConvBlock(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2),
            C3(hidden_dim * 2, hidden_dim * 2, n=3),
            ConvBlock(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2),
            C3(hidden_dim * 4, hidden_dim * 4, n=3),
            ConvBlock(hidden_dim * 4, hidden_dim * 4, kernel_size=1),
            nn.Conv2d(hidden_dim * 4, cat_dim, kernel_size=1)
        )

    def forward(self, encoded_cat_tensor):
        return self.decoder(encoded_cat_tensor)
