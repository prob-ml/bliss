from typing import Optional

import pytorch_lightning as pl
import numpy as np
import torch
from torch import nn
import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import MetricCollection
from einops import rearrange

from bliss.catalog import TileCatalog
from bliss.encoder.metrics import CatalogMatcher
from bliss.global_env import GlobalEnv
from bliss.encoder.encoder import Encoder as BlissEncoder
from bliss.encoder.convnets import FeaturesNet as BlissFeaturesNet
from bliss.encoder.convnet_layers import ConvBlock as BlissConvBlock
from bliss.encoder.convnet_layers import Detect as BlissDetect
from bliss.encoder.convnet_layers import C3 as BlissC3

from case_studies.dc2_mdt.utils.catalog_parser import CatalogParser
from case_studies.dc2_mdt.utils.gaussian_diffusion import (GaussianDiffusion,
                                                           get_named_beta_schedule,
                                                           ModelMeanType,
                                                           ModelVarType,
                                                           LossType)
from case_studies.dc2_mdt.utils.respace import space_timesteps, SpacedDiffusion
from case_studies.dc2_mdt.utils.mdt_models import MDTv2_S_2, M2_MDTv2_S_2
from case_studies.dc2_mdt.utils.resample import create_named_schedule_sampler, ScheduleSampler, SpeedSampler
from case_studies.dc2_mdt.utils.simple_net import SimpleNet, SimpleARNet, SimpleCondTrueNet, M2SimpleNet


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
        d_use_speed_sampler: bool,
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
        self.d_use_speed_sampler = d_use_speed_sampler
        self.d_sampling_method = d_sampling_method
        assert self.d_sampling_method in ["ddim", "ddpm"]
        self.d_sampling_timesteps = d_sampling_timesteps
        self.ddim_eta = ddim_eta
       
        self.acc_grad_batches = acc_grad_batches
        assert self.acc_grad_batches >= 1
        self.max_fluxes = float(max_fluxes) if max_fluxes != "inf" else torch.inf

        self.my_net = None
        self.diffusion_config: dict = None
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
        self.diffusion_config = {
            "betas": get_named_beta_schedule("linear", 
                                             self.d_training_timesteps),
            "model_mean_type": model_mean_type,
            "model_var_type": ModelVarType.LEARNED_RANGE 
                              if self.d_learn_sigma 
                              else ModelVarType.FIXED_LARGE,
            "loss_type": LossType.MSE
        }
        self.training_diffusion = GaussianDiffusion(**self.diffusion_config)
        self.sampling_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(self.d_training_timesteps, 
                                                                                "ddim" + str(self.d_sampling_timesteps)
                                                                                if self.d_sampling_method == "ddim"
                                                                                else str(self.d_sampling_timesteps)),
                                                  **self.diffusion_config)
        assert np.allclose(self.training_diffusion.betas, 
                           self.sampling_diffusion.ori_betas)
    
    # make inference script's life easier
    def reconfig_sampling(self, new_sampling_time_steps, new_ddim_eta):
        self.sampling_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(self.d_training_timesteps, 
                                                                                "ddim" + str(new_sampling_time_steps)
                                                                                if self.d_sampling_method == "ddim"
                                                                                else str(new_sampling_time_steps)),
                                                  **self.diffusion_config)
        self.ddim_eta = new_ddim_eta
        if self.ddim_eta != new_ddim_eta and self.d_sampling_method != "ddim":
            print("WARNING: you set ddim_eta to a new value, but your sampling method is not ddim")

    def initialize_schedule_sampler(self):
        if self.d_use_speed_sampler:
            self.schedule_sampler = SpeedSampler(diffusion=self.training_diffusion,
                                             lam=0.6, 
                                             k=5, 
                                             tau=700)
        else:
            self.schedule_sampler = create_named_schedule_sampler("uniform", self.training_diffusion)

    def initialize_networks(self):
        raise NotImplementedError()

    def get_image(self, batch):
        assert batch["images"].size(2) % 8 == 0, "image dims must be multiples of 8"
        assert batch["images"].size(3) % 8 == 0, "image dims must be multiples of 8"
        input_lst = [inorm.get_input_tensor(batch) for inorm in self.image_normalizers]
        return torch.cat(input_lst, dim=2)

    @torch.inference_mode()
    def sample(self, batch, return_intermediate=False):
        image = self.get_image(batch)
        diffusion_sampling_config = {
            "model": self.my_net,
            "shape": (image.shape[0], 
                      self.catalog_parser.n_params_per_source, 
                      self.image_size[0] // self.tile_slen, self.image_size[1] // self.tile_slen),
            "clip_denoised": True,
            "model_kwargs": {"image": image}
        }
        if self.d_sampling_method == "ddim":
            if not return_intermediate:
                sample = self.sampling_diffusion.ddim_sample_loop(**diffusion_sampling_config, 
                                                                 eta=self.ddim_eta)
            else:
                sample, inter_samples = self.sampling_diffusion.ddim_sample_loop(**diffusion_sampling_config, 
                                                                                 eta=self.ddim_eta,
                                                                                 return_intermediate=True)
        elif self.d_sampling_method == "ddpm":
            if not return_intermediate:
                sample = self.sampling_diffusion.p_sample_loop(**diffusion_sampling_config)
            else:
                sample, inter_samples = self.sampling_diffusion.p_sample_loop(**diffusion_sampling_config)
        else:
            raise NotImplementedError()
        if not return_intermediate:
            return self.catalog_parser.decode(sample.permute([0, 2, 3, 1]))
        else:
            return self.catalog_parser.decode(sample.permute([0, 2, 3, 1])), [self.catalog_parser.decode(s.permute([0, 2, 3, 1])) for s in inter_samples]

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
        image = self.get_image(batch)  # (b, c, H, W)
        encoded_catalog_tensor = self.catalog_parser.encode(target_cat1).permute(0, 3, 1, 2)  # (b, k, h, w)
        
        t, batch_sample_weights, batch_loss_weights = \
            self.schedule_sampler.sample(image.shape[0], device=self.device)
        train_loss_args = {
            "model": self.my_net,
            "x_start": encoded_catalog_tensor,
            "t": t,
            "loss_weights": batch_loss_weights
        }
        no_mask_loss = self.training_diffusion.training_losses(**train_loss_args, 
                                                               model_kwargs={"image": image})
        masked_loss = self.training_diffusion.training_losses(**train_loss_args, 
                                                              model_kwargs={"image": image, 
                                                                            "enable_mask": True})
        return no_mask_loss, masked_loss, batch_sample_weights

    def _compute_loss(self, batch, logging_name):
        no_mask_loss, masked_loss, batch_sample_weights = self._compute_cur_batch_loss(batch)

        batch_size = batch["images"].size(0)
        with torch.inference_mode():
            for k, v in no_mask_loss.items():
                self.log(f"{logging_name}/_no_mask_{k}", 
                         (v * batch_sample_weights).mean(), 
                         batch_size=batch_size, 
                         sync_dist=True)
            for k, v in masked_loss.items():
                self.log(f"{logging_name}/_masked_{k}", 
                         (v * batch_sample_weights).mean(), 
                         batch_size=batch_size, 
                         sync_dist=True)
        
        loss = (no_mask_loss["loss"] * batch_sample_weights).mean() + \
               (masked_loss["loss"] * batch_sample_weights).mean()
        self.log(f"{logging_name}/_loss", loss, batch_size=batch_size, sync_dist=True)
        return loss


class SimpleNetEncoder(DiffusionEncoder):
    def __init__(self,
                 *, 
                 simple_net_type: str, 
                 **kwargs):
        self.simple_net_type = simple_net_type
        assert self.simple_net_type in ["simple_net", "simple_ar_net", "simple_cond_true_net"]

        super().__init__(**kwargs)

    def initialize_networks(self):
        assert self.tile_slen == 4
        assert self.image_size[0] == self.image_size[1]
        assert self.image_size[0] == 80
        target_ch = self.catalog_parser.n_params_per_source
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        simple_net = None
        match self.simple_net_type:
            case "simple_net":
                simple_net = SimpleNet
            case "simple_ar_net":
                simple_net = SimpleARNet
            case "simple_cond_true_net":
                simple_net = SimpleCondTrueNet
            case _:
                raise NotImplementedError()
        self.my_net = simple_net(n_bands=len(self.survey_bands),
                                    ch_per_band=ch_per_band,
                                    in_ch=target_ch,
                                    out_ch=target_ch,
                                    dim=64,
                                    num_cond_layers=8,
                                    spatial_cond_layers=False,
                                    learn_sigma=self.d_learn_sigma)
        
    @torch.inference_mode()
    def sample(self, batch):
        image = self.get_image(batch)
        model_kwargs = {"image": image}
        if self.simple_net_type == "simple_cond_true_net":
            target_cat = TileCatalog(batch["tile_catalog"])
            target_cat1 = target_cat.get_brightest_sources_per_tile(
                band=self.reference_band, exclude_num=0
            )
            model_kwargs["true_n_sources"] = target_cat1["n_sources"]
        diffusion_sampling_config = {
            "model": self.my_net,
            "shape": (image.shape[0], 
                      self.catalog_parser.n_params_per_source, 
                      20, 20),
            "clip_denoised": True,
            "model_kwargs": model_kwargs
        }
        if self.d_sampling_method == "ddim":
            sample = self.sampling_diffusion.ddim_sample_loop(**diffusion_sampling_config, 
                                                              eta=self.ddim_eta)
        elif self.d_sampling_method == "ddpm":
            sample = self.sampling_diffusion.p_sample_loop(**diffusion_sampling_config)
        else:
            raise NotImplementedError()
        return self.catalog_parser.decode(sample.permute([0, 2, 3, 1]))
        
    def _compute_cur_batch_loss(self, batch):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        target_cat1["fluxes"] = target_cat1["fluxes"].clamp(max=self.max_fluxes)
        image = self.get_image(batch)  # (b, c, H, W)
        encoded_catalog_tensor = self.catalog_parser.encode(target_cat1).permute(0, 3, 1, 2)  # (b, k, h, w)
        
        t, batch_sample_weights, batch_loss_weights = \
            self.schedule_sampler.sample(image.shape[0], device=self.device)
        train_loss_args = {
            "model": self.my_net,
            "x_start": encoded_catalog_tensor,
            "t": t,
            "loss_weights": batch_loss_weights,
        }
        model_kwargs = {"image": image}
        if self.simple_net_type == "simple_cond_true_net":
            model_kwargs["true_n_sources"] = target_cat1["n_sources"]
        loss_dict = self.training_diffusion.training_losses(**train_loss_args, model_kwargs=model_kwargs)
        return loss_dict, batch_sample_weights

    def _compute_loss(self, batch, logging_name):
        loss_dict, batch_sample_weights = self._compute_cur_batch_loss(batch)

        batch_size = batch["images"].size(0)
        with torch.inference_mode():
            for k, v in loss_dict.items():
                self.log(f"{logging_name}/_no_mask_{k}", 
                         (v * batch_sample_weights).mean(), 
                         batch_size=batch_size, 
                         sync_dist=True)
        
        loss = (loss_dict["loss"] * batch_sample_weights).mean()
        self.log(f"{logging_name}/_loss", loss, batch_size=batch_size, sync_dist=True)
        return loss


class M2DiffusionEncoder(DiffusionEncoder):
    @torch.inference_mode()
    def sample(self, batch):
        image = self.get_image(batch)
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        target_cat2 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=1
        )

        target_cat1_n_sources = target_cat1["n_sources"].unsqueeze(1)  # (b, 1, h, w)
        target_cat1_locs = target_cat1["locs"].squeeze(-2).permute(0, 3, 1, 2)  # (b, 2, h, w)
        target_cat2_n_sources = target_cat2["n_sources"].unsqueeze(1)
        target_cat2_locs = target_cat2["locs"].squeeze(-2).permute(0, 3, 1, 2)

        true_n_sources_and_locs = torch.cat([target_cat1_n_sources, target_cat1_locs,
                                             target_cat2_n_sources, target_cat2_locs],
                                             dim=1)  # (b, 6, h, w)
        
        diffusion_sampling_config = {
            "model": self.my_net,
            "shape": (image.shape[0], 
                      self.catalog_parser.n_params_per_source * 2,  # x2 for double detect 
                      self.image_size[0] // self.tile_slen, self.image_size[1] // self.tile_slen),
            "clip_denoised": True,
            "model_kwargs": {"image": image,
                             "true_n_sources_and_locs": true_n_sources_and_locs}
        }
        if self.d_sampling_method == "ddim":
            sample = self.sampling_diffusion.ddim_sample_loop(**diffusion_sampling_config, 
                                                              eta=self.ddim_eta)
        elif self.d_sampling_method == "ddpm":
            sample = self.sampling_diffusion.p_sample_loop(**diffusion_sampling_config)
        else:
            raise NotImplementedError()
        sample1, sample2 = sample.permute([0, 2, 3, 1]).chunk(2, dim=-1)  # (b, h, w, k)
        first_cat = self.catalog_parser.decode(sample1)
        first_cat = self._interpolate_n_sources_and_locs(first_cat, target_cat1)
        second_cat = self.catalog_parser.decode(sample2)
        second_cat = self._interpolate_n_sources_and_locs(second_cat, target_cat2)
        return first_cat.union(second_cat, disjoint=False)
    
    def _interpolate_n_sources_and_locs(self, sample_cat_dict, true_tile_cat):
        sample_cat_dict = copy.copy(sample_cat_dict)
        sample_n_sources = sample_cat_dict["fluxes"][..., 0, 0] > 0.1  # (b, h, w)
        t_n_sources = true_tile_cat["n_sources"] > 0  # (b, h, w)
        sample_cat_dict["n_sources"] = (sample_n_sources & t_n_sources).to(dtype=true_tile_cat["n_sources"].dtype)
        sample_cat_dict["locs"] = rearrange(sample_cat_dict["n_sources"] > 0, "b h w -> b h w 1 1") * true_tile_cat["locs"]
        return TileCatalog(sample_cat_dict)
    
    def _compute_diffusion_loss(self, loss_args, model_kwargs):
        raise NotImplementedError()
    
    def _compute_cur_batch_loss(self, batch):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        target_cat2 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=1
        )
        target_cat1["fluxes"] = target_cat1["fluxes"].clamp(max=self.max_fluxes)
        target_cat2["fluxes"] = target_cat2["fluxes"].clamp(max=self.max_fluxes)
        image = self.get_image(batch)  # (b, c, H, W)
        encoded_cat1 = self.catalog_parser.encode(target_cat1).permute(0, 3, 1, 2)  # (b, k, h, w)
        encoded_cat2 = self.catalog_parser.encode(target_cat2).permute(0, 3, 1, 2)  # (b, k, h, w)

        target_cat1_n_sources = target_cat1["n_sources"].unsqueeze(1)  # (b, 1, h, w)
        target_cat1_locs = target_cat1["locs"].squeeze(-2).permute(0, 3, 1, 2)  # (b, 2, h, w)
        target_cat2_n_sources = target_cat2["n_sources"].unsqueeze(1)
        target_cat2_locs = target_cat2["locs"].squeeze(-2).permute(0, 3, 1, 2)
        
        t, batch_sample_weights, batch_loss_weights = \
            self.schedule_sampler.sample(image.shape[0], device=self.device)
        train_loss_args = {
            "model": self.my_net,
            "x_start": torch.cat([encoded_cat1, encoded_cat2], dim=1),
            "t": t,
            "loss_weights": batch_loss_weights
        }
        true_n_sources_and_locs = torch.cat([target_cat1_n_sources, target_cat1_locs,
                                             target_cat2_n_sources, target_cat2_locs],
                                             dim=1)  # (b, 6, h, w)
        loss_dict = self._compute_diffusion_loss(train_loss_args, model_kwargs={"image": image,
                                                                                "true_n_sources_and_locs": true_n_sources_and_locs})
        return loss_dict, batch_sample_weights

    def _compute_loss(self, batch, logging_name):
        loss_dict, batch_sample_weights = self._compute_cur_batch_loss(batch)

        batch_size = batch["images"].size(0)
        assert "no_mask_loss" in loss_dict
        with torch.inference_mode():
            for k, v in loss_dict["no_mask_loss"].items():
                self.log(f"{logging_name}/_no_mask_{k}", 
                         (v * batch_sample_weights).mean(), 
                         batch_size=batch_size, 
                         sync_dist=True)
            if "masked_loss" in loss_dict:
                for k, v in loss_dict["masked_loss"].items():
                    self.log(f"{logging_name}/_masked_{k}", 
                            (v * batch_sample_weights).mean(), 
                            batch_size=batch_size, 
                            sync_dist=True)
        if "masked_loss" in loss_dict:
            loss = (loss_dict["no_mask_loss"]["loss"] * batch_sample_weights).mean() + \
                   (loss_dict["masked_loss"]["loss"] * batch_sample_weights).mean()
        else:
            loss = (loss_dict["no_mask_loss"]["loss"] * batch_sample_weights).mean()
        self.log(f"{logging_name}/_loss", loss, batch_size=batch_size, sync_dist=True)
        return loss


class M2MDTEncoder(M2DiffusionEncoder):
    def initialize_networks(self):
        assert self.tile_slen == 2
        assert self.image_size[0] == self.image_size[1]
        # assert self.image_size[0] == 112
        target_ch = self.catalog_parser.n_params_per_source * 2  # x2 for double detect
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        self.my_net = M2_MDTv2_S_2(image_n_bands=len(self.survey_bands), 
                                    image_ch_per_band=ch_per_band, 
                                    image_feats_ch=128, 
                                    input_size=self.image_size[0] // self.tile_slen, 
                                    in_channels=target_ch, 
                                    decode_layers=6,
                                    mask_ratio=0.3,
                                    mlp_ratio=4.0,
                                    learn_sigma=self.d_learn_sigma)
        
    def _compute_diffusion_loss(self, loss_args, model_kwargs):
        no_mask_loss = self.training_diffusion.training_losses(**loss_args, model_kwargs=model_kwargs)
        masked_loss = self.training_diffusion.training_losses(**loss_args, model_kwargs={**model_kwargs, "enable_mask": True})
        return {
            "no_mask_loss": no_mask_loss,
            "masked_loss": masked_loss,
        }
        

class M2SimpleNetEncoder(M2DiffusionEncoder):
    def initialize_networks(self):
        assert self.tile_slen == 2
        assert self.image_size[0] == self.image_size[1]
        # assert self.image_size[0] == 112
        target_ch = self.catalog_parser.n_params_per_source * 2  # x2 for double detect
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        self.my_net = M2SimpleNet(n_bands=len(self.survey_bands),
                                    ch_per_band=ch_per_band,
                                    in_ch=target_ch,
                                    out_ch=target_ch,
                                    dim=64,
                                    num_cond_layers=8,
                                    spatial_cond_layers=False,
                                    learn_sigma=self.d_learn_sigma)
        
    def _compute_diffusion_loss(self, loss_args, model_kwargs):
        no_mask_loss = self.training_diffusion.training_losses(**loss_args, model_kwargs=model_kwargs)
        return {
            "no_mask_loss": no_mask_loss,
        }


class M2BlissEncoder(BlissEncoder):
    def initialize_networks(self):
        assert self.tile_slen == 2
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        num_features = 256
        self.features_net = BlissFeaturesNet(
            n_bands=len(self.survey_bands),
            ch_per_band=ch_per_band,
            num_features=num_features,
            double_downsample=(self.tile_slen == 4),
        )

        n_hidden_ch = 256
        self.detection_net = nn.Sequential(
            BlissConvBlock(num_features + 6, n_hidden_ch, kernel_size=1, gn=False),  # +6 for true n_sources and locs
            BlissC3(n_hidden_ch, n_hidden_ch, n=3, spatial=False, gn=False),
            BlissDetect(n_hidden_ch, self.var_dist.n_params_per_source * 2),  # x2 for double detect
        )

    def make_local_context(self, history_cat):
        raise NotImplementedError()
    
    def detect_second(self, x_features_color, history_cat):
        raise NotImplementedError()
    
    def make_color_context(self, history_cat, history_mask):
        raise NotImplementedError()
    
    def sample(self, batch, use_mode=True):
        x_features = self.get_features(batch)
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        target_cat2 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=1
        )

        target_cat1_n_sources = target_cat1["n_sources"].unsqueeze(1)  # (b, 1, h, w)
        target_cat1_locs = target_cat1["locs"].squeeze(-2).permute(0, 3, 1, 2)  # (b, 2, h, w)
        target_cat2_n_sources = target_cat2["n_sources"].unsqueeze(1)
        target_cat2_locs = target_cat2["locs"].squeeze(-2).permute(0, 3, 1, 2)

        true_n_sources_and_locs = torch.cat([target_cat1_n_sources, target_cat1_locs,
                                             target_cat2_n_sources, target_cat2_locs],
                                             dim=1)  # (b, 6, h, w)
        x = torch.cat([x_features, true_n_sources_and_locs], dim=1)
        sample1, sample2 = self.detection_net(x).chunk(2, dim=-1)  # (b, h, w, k)
        first_cat = self.var_dist.sample(sample1, use_mode=use_mode, return_base_cat=True).data
        first_cat = self._interpolate_n_sources_and_locs(first_cat, target_cat1)
        second_cat = self.var_dist.sample(sample2, use_mode=use_mode, return_base_cat=True).data
        second_cat = self._interpolate_n_sources_and_locs(second_cat, target_cat2)
        return first_cat.union(second_cat, disjoint=False)

    def _interpolate_n_sources_and_locs(self, sample_cat_dict, true_tile_cat):
        sample_cat_dict = copy.copy(sample_cat_dict)
        sample_n_sources = sample_cat_dict["fluxes"][..., 0, 0] > 0.1  # (b, h, w)
        t_n_sources = true_tile_cat["n_sources"] > 0  # (b, h, w)
        sample_cat_dict["n_sources"] = (sample_n_sources & t_n_sources).to(dtype=true_tile_cat["n_sources"].dtype)
        sample_cat_dict["locs"] = rearrange(sample_cat_dict["n_sources"] > 0, "b h w -> b h w 1 1") * true_tile_cat["locs"]
        return TileCatalog(sample_cat_dict)
    
    def compute_masked_nll(self, batch, history_mask_patterns, loss_mask_patterns):
        raise NotImplementedError()
    
    def compute_sampler_nll(self, batch):
        raise NotImplementedError()
    
    def _compute_loss(self, batch, logging_name):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        target_cat2 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=1
        )
        target_cat1_n_sources = target_cat1["n_sources"].unsqueeze(1)  # (b, 1, h, w)
        target_cat1_locs = target_cat1["locs"].squeeze(-2).permute(0, 3, 1, 2)  # (b, 2, h, w)
        target_cat2_n_sources = target_cat2["n_sources"].unsqueeze(1)
        target_cat2_locs = target_cat2["locs"].squeeze(-2).permute(0, 3, 1, 2)

        true_n_sources_and_locs = torch.cat([target_cat1_n_sources, target_cat1_locs,
                                             target_cat2_n_sources, target_cat2_locs],
                                             dim=1)  # (b, 6, h, w)

        x_features = self.get_features(batch)
        x = torch.cat([x_features, true_n_sources_and_locs], dim=1)
        pred_cat_param1, pred_cat_param2 = self.detection_net(x).chunk(2, dim=-1)
        nll_1 = self.var_dist.compute_nll(pred_cat_param1, target_cat1)
        nll_2 = self.var_dist.compute_nll(pred_cat_param2, target_cat2)

        loss = (nll_1 + nll_2).sum()

        batch_size = batch["images"].size(0)
        self.log(f"{logging_name}/_loss", loss, batch_size=batch_size, sync_dist=True)

        return loss
