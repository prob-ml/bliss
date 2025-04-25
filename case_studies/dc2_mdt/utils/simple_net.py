import torch
import torch.nn as nn
import math

from case_studies.dc2_mdt.utils.convnet_layers import UShapeFeaturesNet, ConvBlock, C3

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        half_d_model = d_model // 2
        log_denominator = -math.log(10000) / (half_d_model - 1)
        denominator_ = torch.exp(torch.arange(half_d_model) * log_denominator)
        self.register_buffer("denominator", denominator_)

    def forward(self, time):
        """
        :param time: shape=(B, )
        :return: Positional Encoding shape=(B, d_model)
        """
        argument = time[:, None] * self.denominator[None, :]  # (B, half_d_model)
        return torch.cat([argument.sin(), argument.cos()], dim=-1)  # (B, d_model)
    

class Block(nn.Module):
    def __init__(self, dim, dim_out, spatial=True):
        """
        Input shape=(B, dim, H, W)
        Output shape=(B, dim_out, H, W)

        :param dim: input channel
        :param dim_out: output channel
        :param groups: number of groups for Group normalization.
        """
        super().__init__()
        if spatial:
            self.proj = nn.Conv2d(dim, dim_out, kernel_size=(3, 3), padding=1)
        else:
            self.proj = nn.Conv2d(dim, dim_out, kernel_size=(1, 1), padding=0)
        assert dim_out % 8 == 0
        self.norm = nn.GroupNorm(num_groups=dim_out // 8, num_channels=dim_out)
        self.activation = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (1 + scale) + shift
        return self.activation(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, spatial=True):
        """
        In abstract, it is composed of two Convolutional layer with residual connection,
        with information of time encoding is passed to first Convolutional layer.

        Input shape=(B, dim, H, W)
        Output shape=(B, dim_out, H, W)

        :param dim: input channel
        :param dim_out: output channel
        :param time_emb_dim: Embedding dimension for time.
        :param group: number of groups for Group normalization.
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim is not None else None
        self.block1 = Block(dim, dim_out, spatial=spatial)
        self.block2 = Block(dim_out, dim_out, spatial=spatial)
        self.residual_conv = nn.Conv2d(dim, dim_out, kernel_size=(1, 1)) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        """

        :param x: (B, dim, H, W)
        :param time_emb: (B, time_emb_dim)
        :return: (B, dim_out, H, W)
        """
        scale_shift = None
        if time_emb is not None:
            scale_shift = self.mlp(time_emb)[..., None, None]  # (B, dim_out*2, 1, 1)
            scale_shift = scale_shift.chunk(2, dim=1)  # len 2 with each element shape (B, dim_out, 1, 1)
        hidden = self.block1(x, scale_shift)
        hidden = self.block2(hidden)
        return hidden + self.residual_conv(x)


class SimpleNet(nn.Module):
    def __init__(self, 
                 n_bands: int, 
                 ch_per_band: int, 
                 in_ch: int, 
                 out_ch: int, 
                 dim: int,
                 num_cond_layers: int,
                 spatial_cond_layers: bool,
                 learn_sigma: bool):
        super().__init__()

        self.num_features = 256
        self.n_bands = n_bands
        self.ch_per_band = ch_per_band

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dim = dim
        self.time_emb_dim = 4 * self.dim
        self.num_cond_layers = num_cond_layers
        self.spatial_cond_layers = spatial_cond_layers
        self.learn_sigma = learn_sigma

        self.fast_inference_mode = False
        self.buffer_image = None
        self.buffer_image_feats = None

        self.initialize_networks()

    def initialize_features_net(self):
        self.features_net = UShapeFeaturesNet(self.n_bands, self.ch_per_band, self.num_features)

    def initialize_time_mlp(self):
        self.time_mlp = nn.Sequential(
            PositionalEncoding(self.dim), 
            nn.Linear(self.dim, self.time_emb_dim),
            nn.GELU(), 
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )

    def initialize_diffusion_layers(self):
        self.cond_layers = nn.ModuleList([ResnetBlock(self.in_ch + self.num_features 
                                                      if i == 0 else self.dim, 
                                                      self.dim, 
                                                      time_emb_dim=self.time_emb_dim, 
                                                      spatial=self.spatial_cond_layers)
                                          for i in range(self.num_cond_layers)])
     
        self.final_conv = nn.Sequential(
            ConvBlock(self.dim, self.dim, kernel_size=1),
            C3(self.dim, self.dim, n=3, spatial=False),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            nn.Conv2d(self.dim, 
                      self.out_ch if not self.learn_sigma else 2 * self.out_ch, 
                      kernel_size=(1, 1))
        )

    def initialize_networks(self):
        self.initialize_features_net()
        self.initialize_time_mlp()
        self.initialize_diffusion_layers()

    def get_image_feats(self, image):
        if self.fast_inference_mode:
            if self.buffer_image is None or (not torch.allclose(self.buffer_image, image)):
                if self.buffer_image is not None:
                    print("WARNING: in fast inference mode, we update the buffered image")
                self.buffer_image = image
                self.buffer_image_feats = self.features_net(image)
            image_feats = self.buffer_image_feats
        else:
            image_feats = self.features_net(image)
        return image_feats

    def forward(self, x, t, image):
        image_feats = self.get_image_feats(image)
        x = torch.cat([x, image_feats], dim=1)
        t = self.time_mlp(t)
        for resnet_block in self.cond_layers:
            x = resnet_block(x, t)
        return self.final_conv(x)


class SimpleARNet(SimpleNet):
    def initialize_diffusion_layers(self):
        n_sources_out_ch = 1 if not self.learn_sigma else 2
        n_sources_dim = self.dim // 2
        other_out_ch = self.out_ch - 1 if not self.learn_sigma else 2 * self.out_ch - 2
        self.n_sources_layers = nn.ModuleList([ResnetBlock(self.in_ch + self.num_features 
                                                            if i == 0 else n_sources_dim, 
                                                            n_sources_dim, 
                                                            time_emb_dim=self.time_emb_dim, 
                                                            spatial=self.spatial_cond_layers)
                                                for i in range(self.num_cond_layers)])
        self.other_layers = nn.ModuleList([ResnetBlock(self.in_ch + self.num_features + n_sources_out_ch 
                                                        if i == 0 else self.dim, 
                                                        self.dim, 
                                                        time_emb_dim=self.time_emb_dim, 
                                                        spatial=self.spatial_cond_layers)
                                            for i in range(self.num_cond_layers)])
     
        self.final_n_sources_conv = nn.Sequential(
            ConvBlock(n_sources_dim, n_sources_dim, kernel_size=1),
            C3(n_sources_dim, n_sources_dim, n=3, spatial=False),
            ConvBlock(n_sources_dim, n_sources_dim, kernel_size=1),
            nn.Conv2d(n_sources_dim, 
                      n_sources_out_ch, 
                      kernel_size=(1, 1))
        )
        self.final_other_conv = nn.Sequential(
            ConvBlock(self.dim, self.dim, kernel_size=1),
            C3(self.dim, self.dim, n=3, spatial=False),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            nn.Conv2d(self.dim, 
                      other_out_ch, 
                      kernel_size=(1, 1))
        )

    def forward(self, x, t, image):
        image_feats = self.get_image_feats(image)
        t = self.time_mlp(t)
    
        n_sources_x = torch.cat([x, image_feats], dim=1)
        for resnet_block in self.n_sources_layers:
            n_sources_x = resnet_block(n_sources_x, t)
        pred_n_sources = self.final_n_sources_conv(n_sources_x)

        other_x = torch.cat([x, image_feats, pred_n_sources.detach()], dim=1)
        for resnet_block in self.other_layers:
            other_x = resnet_block(other_x, t)
        pred_other = self.final_other_conv(other_x)

        if not self.learn_sigma:
            out = torch.cat([pred_n_sources, pred_other], dim=1)
        else:
            pred_n_sources1, pred_n_sources2 = pred_n_sources.chunk(2, dim=1)
            pred_other1, pred_other2 = pred_other.chunk(2, dim=1)
            out = torch.cat([pred_n_sources1, pred_other1, 
                             pred_n_sources2, pred_other2], dim=1)
        return out


class SimpleCondTrueNet(SimpleNet):
    def initialize_diffusion_layers(self):
        self.cond_layers = nn.ModuleList([ResnetBlock(self.in_ch + self.num_features + 1 
                                                      if i == 0 else self.dim, 
                                                      self.dim, 
                                                      time_emb_dim=self.time_emb_dim, 
                                                      spatial=self.spatial_cond_layers)
                                          for i in range(self.num_cond_layers)])
     
        self.final_conv = nn.Sequential(
            ConvBlock(self.dim, self.dim, kernel_size=1),
            C3(self.dim, self.dim, n=3, spatial=False),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            nn.Conv2d(self.dim, 
                      self.out_ch if not self.learn_sigma else 2 * self.out_ch, 
                      kernel_size=(1, 1))
        )

    def forward(self, x, t, image, true_n_sources):
        assert true_n_sources.ndim == 3
        assert true_n_sources.max() == 1
        true_n_sources = true_n_sources.unsqueeze(1).to(dtype=x.dtype)

        image_feats = self.get_image_feats(image)
        x = torch.cat([x, true_n_sources, image_feats], dim=1)
        t = self.time_mlp(t)
        for resnet_block in self.cond_layers:
            x = resnet_block(x, t)
        return self.final_conv(x)


class M2SimpleNet(SimpleNet):
    def initialize_features_net(self):
        self.features_net = UShapeFeaturesNet(self.n_bands, self.ch_per_band, self.num_features, double_downsample=False)

    def initialize_diffusion_layers(self):
        self.cond_layers = nn.ModuleList([ResnetBlock(self.in_ch + self.num_features + 6 
                                                      if i == 0 else self.dim, 
                                                      self.dim, 
                                                      time_emb_dim=self.time_emb_dim, 
                                                      spatial=self.spatial_cond_layers)
                                          for i in range(self.num_cond_layers)])
     
        self.final_conv = nn.Sequential(
            ConvBlock(self.dim, self.dim, kernel_size=1),
            C3(self.dim, self.dim, n=3, spatial=False),
            ConvBlock(self.dim, self.dim, kernel_size=1),
            nn.Conv2d(self.dim, 
                      self.out_ch if not self.learn_sigma else 2 * self.out_ch, 
                      kernel_size=(1, 1))
        )

    def forward(self, x, t, image, true_n_sources_and_locs):
        image_feats = self.get_image_feats(image)
        x = torch.cat([x, image_feats, true_n_sources_and_locs], dim=1)
        t = self.time_mlp(t)
        for resnet_block in self.cond_layers:
            x = resnet_block(x, t)
        return self.final_conv(x)
        