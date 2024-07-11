from torch import nn

from bliss.encoder.convnet_layers import C3, ConvBlock, Detect


class CatalogNet(nn.Module):
    def __init__(self, num_features, out_channels):
        super().__init__()

        n_hidden_ch = 256
        self.detection_net = nn.Sequential(
            ConvBlock(num_features, n_hidden_ch),
            ConvBlock(n_hidden_ch, n_hidden_ch),
            C3(n_hidden_ch, n_hidden_ch, n=4),
            ConvBlock(n_hidden_ch, n_hidden_ch),
            Detect(n_hidden_ch, out_channels),
        )

    def forward(self, x_features):
        return self.detection_net(x_features)
