from typing import Optional

import pytorch_lightning as pl
import torch
from einops import rearrange

from bliss.global_env import GlobalEnv
from case_studies.weak_lensing.convnet import WeakLensingNet


class TimeEncoder(pl.LightningModule):
    def __init__(self, t_embed_dim: int, expand_dim: int):
        super().__init__()
        self.expand_dim = expand_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, t_embed_dim // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(t_embed_dim // 2, t_embed_dim),
        )

    def forward(self, t):
        return rearrange(self.net(t), "b d -> b d 1 1").expand(
            -1, -1, self.expand_dim, self.expand_dim
        )


class VelocityNet(pl.LightningModule):
    def __init__(self, z_dim: int, x_embed_dim: int, t_embed_dim: int, channels: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(z_dim + x_embed_dim + t_embed_dim, channels, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(channels, z_dim, kernel_size=3, padding=1),
        )

    def forward(self, zt, x_embedding, t_embedding):
        inputs = torch.cat([zt, x_embedding, t_embedding], dim=1)
        return self.net(inputs)


class FlowMatching(pl.LightningModule):
    def __init__(
        self,
        survey_bands: list,
        n_pixels_per_side: int,
        n_tiles_per_side: int,
        ch_init: int,
        ch_max: int,
        ch_final: int,
        initial_downsample: bool,
        more_up_layers: bool,
        num_bottleneck_layers: int,
        image_normalizers: list,
        t_embed_dim: int,
        num_redshift_bins: int,
        velo_net_channels: int,
        optimizer_params: Optional[dict] = None,
    ):
        super().__init__()
        self.survey_bands = survey_bands
        self.n_pixels_per_side = n_pixels_per_side
        self.n_tiles_per_side = n_tiles_per_side
        self.ch_init = ch_init
        self.ch_max = ch_max
        self.ch_final = ch_final
        self.initial_downsample = initial_downsample
        self.more_up_layers = more_up_layers
        self.num_bottleneck_layers = num_bottleneck_layers
        self.image_normalizers = torch.nn.ModuleList(image_normalizers.values())
        self.t_embed_dim = t_embed_dim
        self.z_dim = 3 * num_redshift_bins  # shear1, shear2, convergence for each bin
        self.velo_net_channels = velo_net_channels
        self.optimizer_params = optimizer_params

        self.initialize_networks()

    def initialize_networks(self):
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        self.image_encoder = WeakLensingNet(
            n_bands=len(self.survey_bands),
            n_pixels_per_side=self.n_pixels_per_side,
            n_tiles_per_side=self.n_tiles_per_side,
            ch_per_band=ch_per_band,
            ch_init=self.ch_init,
            ch_max=self.ch_max,
            ch_final=self.ch_final,
            initial_downsample=self.initial_downsample,
            more_up_layers=self.more_up_layers,
            num_bottleneck_layers=self.num_bottleneck_layers,
            map_to_var_params=False,
        )
        self.time_encoder = TimeEncoder(
            t_embed_dim=self.t_embed_dim, expand_dim=self.n_tiles_per_side
        )
        self.velocity_net = VelocityNet(
            z_dim=self.z_dim,
            x_embed_dim=self.ch_final,
            t_embed_dim=self.t_embed_dim,
            channels=self.velo_net_channels,
        )

    def on_fit_start(self):
        GlobalEnv.current_encoder_epoch = self.current_epoch

    def on_train_epoch_start(self):
        GlobalEnv.current_encoder_epoch = self.current_epoch

    def compute_loss(self, batch):
        x_list = [inorm.get_input_tensor(batch) for inorm in self.image_normalizers]
        x = torch.cat(x_list, dim=2)
        x_embedding = self.image_encoder(x)

        t = torch.rand([x.shape[0], 1], device=x.device)
        t_embedding = self.time_encoder(t)

        shear1 = rearrange(batch["tile_catalog"]["shear_1"], "b h w r -> b r h w")
        shear2 = rearrange(batch["tile_catalog"]["shear_2"], "b h w r -> b r h w")
        convergence = rearrange(batch["tile_catalog"]["convergence"], "b h w r -> b r h w")
        z1 = torch.cat([shear1, shear2, convergence], dim=1)

        z0 = torch.randn_like(z1, device=z1.device)

        zt = (1 - t) * z0 + t * z1

        velo_true = z1 - z0

        velo_pred = self.velocity_net(zt, x_embedding, t_embedding)

        return ((velo_true - velo_pred) ** 2).mean()

    def sample_path(self, zt, x_embedding, t, t_next, method="midpoint"):
        t_embedding = self.time_encoder(t)
        velo_t = self.velocity_net(zt, x_embedding, t_embedding)

        t_diff = t_next.unique() - t.unique()

        if method == "midpoint":
            half_step = 0.5 * t_diff
            z_mid = zt + half_step * velo_t
            t_mid_embedding = self.time_encoder(t + half_step)
            u_mid = self.velocity_net(z_mid, x_embedding, t_mid_embedding)
            return zt + t_diff * u_mid
        if method == "euler":
            return zt + t_diff * velo_t
        raise ValueError("method should be euler or midpoint")

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss, batch_size=batch["images"].shape[0], sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("val_loss", loss, batch_size=batch["images"].shape[0], sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("test_loss", loss, batch_size=batch["images"].shape[0], sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_params)
        return optimizer
