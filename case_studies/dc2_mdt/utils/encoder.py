from typing import Optional

import pytorch_lightning as pl
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import MetricCollection

from bliss.catalog import TileCatalog
from bliss.encoder.metrics import CatalogMatcher
from bliss.global_env import GlobalEnv

from case_studies.dc2_mdt.utils.catalog_parser import CatalogParser
from case_studies.dc2_mdt.utils.gaussian_diffusion import (GaussianDiffusion,
                                                           get_named_beta_schedule,
                                                           ModelMeanType,
                                                           ModelVarType,
                                                           LossType)
from case_studies.dc2_mdt.utils.respace import space_timesteps, SpacedDiffusion
from case_studies.dc2_mdt.utils.mdt_models import MDTv2_S_2
from case_studies.dc2_mdt.utils.resample import create_named_schedule_sampler, ScheduleSampler
from case_studies.dc2_mdt.utils.simple_net import SimpleNet


class DiffusionEncoder(pl.LightningModule):
    def __init__(
        self,
        *,
        survey_bands: list,
        tile_slen: int,
        image_size: list,
        image_normalizers: dict,
        catalog_parser: CatalogParser,
        matcher: CatalogMatcher,
        mode_metrics: MetricCollection,
        d_objective: str,
        d_beta_schedule: str,
        d_learn_sigma: bool,
        d_training_timesteps: int,
        d_sampling_method: str,
        d_sampling_timesteps: int,
        ddim_eta: float,
        acc_grad_batches: int,
        max_fluxes: float,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
        reference_band: int = 2,
        **kwargs,  # other args inherited from base config
    ):
        super().__init__()

        self.survey_bands = survey_bands
        self.tile_slen = tile_slen
        self.image_normalizers = torch.nn.ModuleList(image_normalizers.values())
        self.catalog_parser = catalog_parser
        self.image_size = image_size
        self.mode_metrics = mode_metrics
        self.matcher = matcher
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params if scheduler_params else {"milestones": []}
        self.reference_band = reference_band

        self.d_objective = d_objective
        self.d_beta_schedule = d_beta_schedule
        self.d_learn_sigma = d_learn_sigma
        self.d_training_timesteps = d_training_timesteps
        self.d_sampling_method = d_sampling_method
        assert self.d_sampling_method in ["ddim", "ddpm"]
        self.d_sampling_timesteps = d_sampling_timesteps
        self.ddim_eta = ddim_eta
       
        self.acc_grad_batches = acc_grad_batches
        assert self.acc_grad_batches >= 1
        self.max_fluxes = float(max_fluxes) if max_fluxes != "inf" else torch.inf

        self.my_net = None
        self.training_diffusion: GaussianDiffusion = None
        self.sampling_diffusion: SpacedDiffusion = None
        self.schedule_sampler: ScheduleSampler = None

        self.register_buffer("my_dummy_variable", torch.zeros(0))

        # important: this property activates manual optimization.
        self.automatic_optimization = False

        self.initialize_diffusion()
        self.initialize_schedule_sampler()
        self.initialize_networks()

    @property
    def device(self):
        return self.my_dummy_variable.device

    def initialize_diffusion(self):
        model_mean_type = None
        match self.d_objective:
            case "previous_x":
                model_mean_type = ModelMeanType.PREVIOUS_X
            case "start_x":
                model_mean_type = ModelMeanType.START_X
            case "noise":
                model_mean_type = ModelMeanType.EPSILON
            case "velocity":
                model_mean_type = ModelMeanType.VELOCITY
            case _:
                raise NotImplementedError()
        assert self.d_beta_schedule == "linear"
        diffusion_config = {
            "betas": get_named_beta_schedule("linear", 
                                             self.d_training_timesteps),
            "model_mean_type": model_mean_type,
            "model_var_type": ModelVarType.LEARNED_RANGE 
                              if self.d_learn_sigma 
                              else ModelVarType.FIXED_LARGE,
            "loss_type": LossType.MSE
        }
        self.training_diffusion = GaussianDiffusion(**diffusion_config)
        self.sampling_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(self.d_training_timesteps, 
                                                                                "ddim" + str(self.d_sampling_timesteps)
                                                                                if self.d_sampling_method == "ddim"
                                                                                else str(self.d_sampling_timesteps)),
                                                  **diffusion_config)
        assert np.allclose(self.training_diffusion.betas, 
                           self.sampling_diffusion.ori_betas)
        
    def initialize_schedule_sampler(self):
        self.schedule_sampler = create_named_schedule_sampler("uniform", self.training_diffusion)

    def initialize_networks(self):
        raise NotImplementedError()

    def get_features(self, batch):
        assert batch["images"].size(2) % 16 == 0, "image dims must be multiples of 16"
        assert batch["images"].size(3) % 16 == 0, "image dims must be multiples of 16"
        input_lst = [inorm.get_input_tensor(batch) for inorm in self.image_normalizers]
        return torch.cat(input_lst, dim=2)

    @torch.inference_mode()
    def sample(self, batch):
        x_features = self.get_features(batch)
        diffusion_sampling_config = {
            "model": self.my_net,
            "shape": (x_features.shape[0], 
                      self.catalog_parser.n_params_per_source, 
                      20, 20),
            "clip_denoised": True,
            "model_kwargs": {"image": x_features}
        }
        if self.d_sampling_method == "ddim":
            sample = self.sampling_diffusion.ddim_sample_loop(**diffusion_sampling_config, 
                                                              eta=self.ddim_eta)
        elif self.d_sampling_method == "ddpm":
            sample = self.sampling_diffusion.p_sample_loop(**diffusion_sampling_config)
        else:
            raise NotImplementedError()
        return self.catalog_parser.decode(sample.permute([0, 2, 3, 1]))

    def _compute_loss(self, batch, logging_name):
        raise NotImplementedError()

    def on_fit_start(self):
        GlobalEnv.current_encoder_epoch = self.current_epoch

    def on_train_epoch_start(self):
        GlobalEnv.current_encoder_epoch = self.current_epoch

    def training_step(self, batch, batch_idx):
        """Training step (pytorch lightning)."""
        my_optimizer = self.optimizers()
        mean_loss = self._compute_loss(batch, "train")

        mean_loss = mean_loss / self.acc_grad_batches
        self.manual_backward(mean_loss)
        if (batch_idx + 1) % self.acc_grad_batches == 0:
            my_optimizer.step()
            my_optimizer.zero_grad()

        # step every epoch
        if self.trainer.is_last_batch:
            my_scheduler = self.lr_schedulers()
            my_scheduler.step()

    def update_metrics(self, batch, batch_idx):
        target_tile_cat = TileCatalog(batch["tile_catalog"])
        target_cat = target_tile_cat.to_full_catalog(self.tile_slen)
        target_tile_cat["fluxes"] = target_tile_cat["fluxes"].clamp(max=self.max_fluxes)

        mode_tile_cat = self.sample(batch)
        mode_cat = mode_tile_cat.to_full_catalog(self.tile_slen)
        mode_matching = self.matcher.match_catalogs(target_cat, mode_cat)
        self.mode_metrics.update(target_cat, mode_cat, mode_matching)

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        self._compute_loss(batch, "val")
        self.update_metrics(batch, batch_idx)

    def report_metrics(self, metrics, logging_name, show_epoch=False):
        for k, v in metrics.compute().items():
            self.log(f"{logging_name}/{k}", v, sync_dist=True)

        for metric_name, metric in metrics.items():
            if hasattr(metric, "plot"):  # noqa: WPS421
                try:
                    plot_or_none = metric.plot()
                except NotImplementedError:
                    continue
                name = f"Epoch:{self.current_epoch}" if show_epoch else ""
                name += f"/{logging_name} {metric_name}"
                if self.logger and plot_or_none:
                    fig, _axes = plot_or_none
                    self.logger.experiment.add_figure(name, fig)

    def on_validation_epoch_end(self):
        self.report_metrics(self.mode_metrics, "val/mode", show_epoch=True)
        self.mode_metrics.reset()

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        self._compute_loss(batch, "test")
        self.update_metrics(batch, batch_idx)

    def on_test_epoch_end(self):
        # note: metrics are not reset here, to give notebooks access to them
        self.report_metrics(self.mode_metrics, "test/mode", show_epoch=False)

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Pytorch lightning method."""
        return self.sample(batch)

    def configure_optimizers(self):
        """Configure optimizers for training (pytorch lightning)."""
        my_optimizer = Adam(self.my_net.parameters(), **self.optimizer_params)
        my_scheduler = MultiStepLR(my_optimizer, **self.scheduler_params)
        return [my_optimizer], [my_scheduler]


class MDTEncoder(DiffusionEncoder):
    def initialize_networks(self):
        assert self.tile_slen == 4
        assert self.image_size[0] == self.image_size[1]
        assert self.image_size[0] == 80
        target_ch = self.catalog_parser.n_params_per_source
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        self.my_net = MDTv2_S_2(image_n_bands=len(self.survey_bands), 
                                image_ch_per_band=ch_per_band, 
                                image_feats_ch=128, 
                                input_size=20, 
                                in_channels=target_ch, 
                                decode_layers=6,
                                mask_ratio=0.3,
                                mlp_ratio=4.0,
                                learn_sigma=self.d_learn_sigma)

    def _compute_cur_batch_loss(self, batch):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        target_cat1["fluxes"] = target_cat1["fluxes"].clamp(max=self.max_fluxes)
        x_features = self.get_features(batch)  # (b, c, H, W)
        encoded_catalog_tensor = self.catalog_parser.encode(target_cat1).permute(0, 3, 1, 2)  # (b, k, h, w)
        
        t, weights = self.schedule_sampler.sample(x_features.shape[0], device=self.device)
        train_loss_args = {
            "model": self.my_net,
            "x_start": encoded_catalog_tensor,
            "t": t,
        }
        no_mask_loss = self.training_diffusion.training_losses(**train_loss_args, 
                                                               model_kwargs={"image": x_features})
        masked_loss = self.training_diffusion.training_losses(**train_loss_args, 
                                                              model_kwargs={"image": x_features, 
                                                                            "enable_mask": True})
        return no_mask_loss, masked_loss, weights

    def _compute_loss(self, batch, logging_name):
        no_mask_loss, masked_loss, loss_weights = self._compute_cur_batch_loss(batch)

        batch_size = batch["images"].size(0)
        with torch.inference_mode():
            for k, v in no_mask_loss.items():
                self.log(f"{logging_name}/_no_mask_{k}", 
                         (v * loss_weights).mean(), 
                         batch_size=batch_size, 
                         sync_dist=True)
            for k, v in masked_loss.items():
                self.log(f"{logging_name}/_masked_{k}", 
                         (v * loss_weights).mean(), 
                         batch_size=batch_size, 
                         sync_dist=True)
        
        loss = (no_mask_loss["loss"] * loss_weights).mean() + (masked_loss["loss"] * loss_weights).mean()
        self.log(f"{logging_name}/_loss", loss, batch_size=batch_size, sync_dist=True)
        return loss


class SimpleNetEncoder(DiffusionEncoder):
    def initialize_networks(self):
        assert self.tile_slen == 4
        assert self.image_size[0] == self.image_size[1]
        assert self.image_size[0] == 80
        target_ch = self.catalog_parser.n_params_per_source
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        self.my_net = SimpleNet(n_bands=len(self.survey_bands),
                                ch_per_band=ch_per_band,
                                in_ch=target_ch,
                                out_ch=target_ch,
                                dim=64,
                                num_cond_layers=8,
                                spatial_cond_layers=False,
                                learn_sigma=self.d_learn_sigma)
        
    def _compute_cur_batch_loss(self, batch):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        target_cat1["fluxes"] = target_cat1["fluxes"].clamp(max=self.max_fluxes)
        x_features = self.get_features(batch)  # (b, c, H, W)
        encoded_catalog_tensor = self.catalog_parser.encode(target_cat1).permute(0, 3, 1, 2)  # (b, k, h, w)
        
        t, weights = self.schedule_sampler.sample(x_features.shape[0], device=self.device)
        train_loss_args = {
            "model": self.my_net,
            "x_start": encoded_catalog_tensor,
            "t": t,
        }
        loss_dict = self.training_diffusion.training_losses(**train_loss_args, 
                                                            model_kwargs={"image": x_features})
        return loss_dict, weights

    def _compute_loss(self, batch, logging_name):
        loss_dict, loss_weights = self._compute_cur_batch_loss(batch)

        batch_size = batch["images"].size(0)
        with torch.inference_mode():
            for k, v in loss_dict.items():
                self.log(f"{logging_name}/_no_mask_{k}", 
                         (v * loss_weights).mean(), 
                         batch_size=batch_size, 
                         sync_dist=True)
        
        loss = (loss_dict["loss"] * loss_weights).mean()
        self.log(f"{logging_name}/_loss", loss, batch_size=batch_size, sync_dist=True)
        return loss
