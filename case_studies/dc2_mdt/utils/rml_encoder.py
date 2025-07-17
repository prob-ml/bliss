from typing import Optional

import pytorch_lightning as pl
import copy
from einops import rearrange, repeat
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import MetricCollection
import numpy as np

from bliss.catalog import TileCatalog
from bliss.encoder.metrics import CatalogMatcher
from bliss.global_env import GlobalEnv

from case_studies.dc2_mdt.utils.catalog_parser import CatalogParser
from case_studies.dc2_mdt.utils.reverse_markov_learning import RMLDiffusion
from case_studies.dc2_mdt.utils.rml_df import RMLDF
from case_studies.dc2_mdt.utils.resample import create_named_schedule_sampler, ScheduleSampler, SigmoidSampler
from case_studies.dc2_mdt.utils.mdt_models import M2_MDTv2_RML_S_2, M2_MDTv2_full_RML_S_2, DC2_MDTv2_RML_S_2, M2_MDTv2_RML_DF_full_S_2


class M2RMLEncoder(pl.LightningModule):
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
        d_training_timesteps: int,
        d_sampling_timesteps: int,
        d_schedule_sampler: str,
        d_sigmoid_schedule_bias: float,
        d_rml_m: int,
        d_rml_lambda: float,
        d_rml_beta: float,
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

        self.d_training_timesteps = d_training_timesteps
        self.d_sampling_timesteps = d_sampling_timesteps
        self.d_schedule_sampler = d_schedule_sampler
        self.d_sigmoid_schedule_bias = d_sigmoid_schedule_bias
        self.d_rml_m = d_rml_m
        self.d_rml_lambda = d_rml_lambda
        self.d_rml_beta = d_rml_beta
        self.ddim_eta = ddim_eta
       
        self.acc_grad_batches = acc_grad_batches
        assert self.acc_grad_batches >= 1
        self.max_fluxes = float(max_fluxes) if max_fluxes != "inf" else torch.inf

        self.my_net = None
        self.rml_diffusion: RMLDiffusion = None
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
        self.rml_diffusion = RMLDiffusion(num_timesteps=self.d_training_timesteps,
                                          num_sampling_steps=self.d_sampling_timesteps,
                                          m=self.d_rml_m,
                                          lambda_=self.d_rml_lambda,
                                          beta=self.d_rml_beta)
    
    # make inference script's life easier
    def reconfig_sampling(self, new_sampling_time_steps, new_ddim_eta):
        assert isinstance(self.rml_diffusion, RMLDiffusion)
        self.rml_diffusion = RMLDiffusion(num_timesteps=self.d_training_timesteps,
                                          num_sampling_steps=new_sampling_time_steps,
                                          m=self.d_rml_m,
                                          lambda_=self.d_rml_lambda,
                                          beta=self.d_rml_beta)
        self.ddim_eta = new_ddim_eta

    def initialize_schedule_sampler(self):
        match self.d_schedule_sampler:
            case "sigmoid":
                self.schedule_sampler = SigmoidSampler(self.rml_diffusion, b=self.d_sigmoid_schedule_bias)
            case "uniform":
                self.schedule_sampler = create_named_schedule_sampler("uniform", self.rml_diffusion)
            case _:
                raise NotImplementedError()

    def initialize_networks(self):
        raise NotImplementedError()

    def get_image(self, batch):
        assert batch["images"].size(2) % 8 == 0, "image dims must be multiples of 8"
        assert batch["images"].size(3) % 8 == 0, "image dims must be multiples of 8"
        input_lst = [inorm.get_input_tensor(batch) for inorm in self.image_normalizers]
        return torch.cat(input_lst, dim=2)

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
                             "true_n_sources_and_locs": true_n_sources_and_locs},
            "eta": self.ddim_eta,
        }
        sample = self.rml_diffusion.ddim_sample_loop(**diffusion_sampling_config)
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
        target_tile_cat["fluxes"] = target_tile_cat["fluxes"].clamp(max=self.max_fluxes)
        target_cat = target_tile_cat.to_full_catalog(self.tile_slen)

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
    

class M2MDTRMLEncoder(M2RMLEncoder):
    def initialize_networks(self):
        assert self.tile_slen == 2
        assert self.image_size[0] == self.image_size[1]
        # assert self.image_size[0] == 112
        target_ch = self.catalog_parser.n_params_per_source * 2  # x2 for double detect
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        self.my_net = M2_MDTv2_RML_S_2(image_n_bands=len(self.survey_bands), 
                                        image_ch_per_band=ch_per_band, 
                                        image_feats_ch=128, 
                                        input_size=self.image_size[0] // self.tile_slen, 
                                        in_channels=target_ch, 
                                        decode_layers=6,
                                        mask_ratio=0.3,
                                        mlp_ratio=4.0,
                                        learn_sigma=False)
        
    def _compute_diffusion_loss(self, loss_args, model_kwargs):
        no_mask_loss = self.rml_diffusion.training_losses(**loss_args, model_kwargs=model_kwargs)
        masked_loss = self.rml_diffusion.training_losses(**loss_args, model_kwargs={**model_kwargs, "enable_mask": True})
        return {
            "no_mask_loss": no_mask_loss,
            "masked_loss": masked_loss,
        }
    

class M2RMLFullEncoder(M2RMLEncoder):
    def _interpolate_n_sources_and_locs(self, sample_cat_dict, true_tile_cat):
        raise NotImplementedError()
    
    @torch.inference_mode()
    def sample(self, batch, return_intermediate=False):
        image = self.get_image(batch)
        diffusion_sampling_config = {
            "model": self.my_net,
            "shape": (image.shape[0], 
                      self.catalog_parser.n_params_per_source * 2,  # x2 for double detect 
                      self.image_size[0] // self.tile_slen, self.image_size[1] // self.tile_slen),
            "clip_denoised": True,
            "model_kwargs": {"image": image},
            "eta": self.ddim_eta,
        }
        if not return_intermediate:
            sample = self.rml_diffusion.ddim_sample_loop(**diffusion_sampling_config)
        else:
            sample, inter_pred_x0 = self.rml_diffusion.ddim_sample_loop(**diffusion_sampling_config, 
                                                                        return_intermediate=return_intermediate)
        sample1, sample2 = sample.permute([0, 2, 3, 1]).chunk(2, dim=-1)  # (b, h, w, k)
        first_cat = self.catalog_parser.decode(sample1)
        second_cat = self.catalog_parser.decode(sample2)
        if not return_intermediate:
            return first_cat.union(second_cat, disjoint=False)
        else:
            return first_cat.union(second_cat, disjoint=False), inter_pred_x0
    
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
        
        t, batch_sample_weights, batch_loss_weights = \
            self.schedule_sampler.sample(image.shape[0], device=self.device)
        train_loss_args = {
            "model": self.my_net,
            "x_start": torch.cat([encoded_cat1, encoded_cat2], dim=1),
            "t": t,
            "loss_weights": batch_loss_weights
        }
        loss_dict = self._compute_diffusion_loss(train_loss_args, model_kwargs={"image": image})
        return loss_dict, batch_sample_weights
    

class M2MDTFullRMLEncoder(M2RMLFullEncoder):
    def initialize_networks(self):
        assert self.tile_slen == 2
        assert self.image_size[0] == self.image_size[1]
        # assert self.image_size[0] == 112
        target_ch = self.catalog_parser.n_params_per_source * 2  # x2 for double detect
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        self.my_net = M2_MDTv2_full_RML_S_2(image_n_bands=len(self.survey_bands), 
                                            image_ch_per_band=ch_per_band, 
                                            image_feats_ch=128, 
                                            input_size=self.image_size[0] // self.tile_slen, 
                                            in_channels=target_ch, 
                                            decode_layers=6,
                                            mask_ratio=0.3,
                                            mlp_ratio=4.0,
                                            learn_sigma=False)
        
    def _compute_diffusion_loss(self, loss_args, model_kwargs):
        no_mask_loss = self.rml_diffusion.training_losses(**loss_args, model_kwargs=model_kwargs)
        masked_loss = self.rml_diffusion.training_losses(**loss_args, model_kwargs={**model_kwargs, "enable_mask": True})
        return {
            "no_mask_loss": no_mask_loss,
            "masked_loss": masked_loss,
        }
    

class M2RMLDFFullEncoder(M2RMLEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert len(self.catalog_parser.factors) == 3
        assert self.catalog_parser.factors[0].name == "n_sources"
        assert self.catalog_parser.factors[1].name == "locs"
        assert self.catalog_parser.factors[2].name == "fluxes"

        assert self.d_sampling_timesteps % 3 == 0
        each_var_ss = self.d_sampling_timesteps // 3
        ns_k_vec = self.generate_k_vec(each_var_ss, pad_front=0, pad_rear=each_var_ss * 2)
        locs_x_k_vec = self.generate_k_vec(each_var_ss, pad_front=each_var_ss, pad_rear=each_var_ss)
        locs_y_k_vec = locs_x_k_vec
        fluxes_k_vec = self.generate_k_vec(each_var_ss, pad_front=2 * each_var_ss, pad_rear=0)
        self.k_matrix = torch.stack([ns_k_vec, locs_y_k_vec, locs_x_k_vec, fluxes_k_vec], dim=-1)  # (sampling_steps, k)

    def generate_k_vec(self, sampling_steps, pad_front, pad_rear):
        basic_vec = torch.linspace(0, self.d_training_timesteps - 1, sampling_steps).int().flip(dims=(0,))
        pad_front_vec = torch.full((pad_front, ), fill_value=basic_vec[0].item())
        pad_rear_vec = torch.full((pad_rear, ), fill_value=basic_vec[-1].item())
        return torch.cat([pad_front_vec, basic_vec, pad_rear_vec], dim=0)

    @classmethod
    def rml_sigmoid_loss_weight_fn(cls, t: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor, b: float):
        alpha_2 = alpha ** 2 + 1e-3
        sigma_2 = sigma ** 2 + 1e-3
        if b == 0.0:
            loss_weights = 1 / (1 + sigma_2 / alpha_2)
        else:
            loss_weights = 1 / (1 + torch.exp(b - torch.log(alpha_2 / sigma_2)))
        return loss_weights.to(device=t.device)[t.flatten()].view(t.shape)
    
    @classmethod
    def _rml_one_source_mask(cls, x0_population: torch.Tensor, output_population: torch.Tensor):
        ns_mask = x0_population[..., 0:1, :, :] > 0.0
        output_ns = output_population[..., 0:1, :, :]
        output_other = output_population[..., 1:, :, :]
        return torch.cat([output_ns, 
                          torch.where(ns_mask, output_other, torch.full_like(output_other, fill_value=-1.0))],
                         dim=-3)
        
    @classmethod
    def rml_loss_mask_fn(cls, x0_population: torch.Tensor, output_population: torch.Tensor):
        assert x0_population.ndim == 5
        assert x0_population.shape[2] == 8
        assert output_population.shape == x0_population.shape  # (b, m, 8, h, w)
        first_s_x0_p, second_s_x0_p = x0_population.chunk(2, dim=2)
        first_s_output_p, second_s_output_p = output_population.chunk(2, dim=2)
        first_s_output_p = cls._rml_one_source_mask(first_s_x0_p, first_s_output_p)
        second_s_output_p = cls._rml_one_source_mask(second_s_x0_p, second_s_output_p)
        return torch.cat([first_s_output_p, second_s_output_p], dim=2)

    @classmethod
    def rml_pred_x0_rectify_fn(cls, pred_x0: torch.Tensor):
        assert pred_x0.ndim == 4
        assert pred_x0.shape[1] == 8
        first_s_pred_x0, second_s_pred_x0 = pred_x0.chunk(2, dim=1)
        first_s_pred_x0 = cls._rml_one_source_mask(first_s_pred_x0, first_s_pred_x0)
        second_s_pred_x0 = cls._rml_one_source_mask(second_s_pred_x0, second_s_pred_x0)
        return torch.cat([first_s_pred_x0, second_s_pred_x0], dim=1)

    def initialize_diffusion(self):
        self.rml_diffusion = RMLDF(num_timesteps=self.d_training_timesteps,
                                   m=self.d_rml_m,
                                   lambda_=self.d_rml_lambda,
                                   beta=self.d_rml_beta,
                                   matching_fn=None,
                                   loss_mask_fn=self.rml_loss_mask_fn,
                                   pred_x0_rectify_fn=self.rml_pred_x0_rectify_fn,
                                   loss_weight_fn=lambda t, alpha, sigma: self.rml_sigmoid_loss_weight_fn(t, alpha, sigma, b=self.d_sigmoid_schedule_bias))
    
    def reconfig_sampling(self, new_sampling_time_steps, new_ddim_eta):
        raise NotImplementedError()

    def initialize_schedule_sampler(self):
        pass

    def _interpolate_n_sources_and_locs(self, sample_cat_dict, true_tile_cat):
        raise NotImplementedError()
    
    @torch.inference_mode()
    def sample(self, batch, return_intermediate=False):
        image = self.get_image(batch)
        sampling_k_matrix = repeat(self.k_matrix, 
                                   "ss k -> ss b (2 k) h w", 
                                   b=image.shape[0],
                                   h=self.image_size[0] // self.tile_slen,
                                   w=self.image_size[1] // self.tile_slen).to(device=self.device)
        diffusion_sampling_config = {
            "model": self.my_net,
            "shape": (image.shape[0], 
                      self.catalog_parser.n_params_per_source * 2,  # x2 for double detect 
                      self.image_size[0] // self.tile_slen, self.image_size[1] // self.tile_slen),
            "clip_denoised": True,
            "model_kwargs": {"image": image},
            "eta": self.ddim_eta,
            "k_matrix": sampling_k_matrix,
        }
        if not return_intermediate:
            sample = self.rml_diffusion.ddim_sample_loop(**diffusion_sampling_config)
        else:
            sample, inter_pred_x0 = self.rml_diffusion.ddim_sample_loop(**diffusion_sampling_config, 
                                                                        return_intermediate=True)
        sample1, sample2 = sample.permute([0, 2, 3, 1]).chunk(2, dim=-1)  # (b, h, w, k)
        first_cat = self.catalog_parser.decode(sample1)
        second_cat = self.catalog_parser.decode(sample2)
        if not return_intermediate:
            return first_cat.union(second_cat, disjoint=False)
        else:
            return first_cat.union(second_cat, disjoint=False), inter_pred_x0
        
    def generate_training_t_schedule(self, batch_size):
        pred_ns_mode = torch.from_numpy(np.random.choice(self.d_training_timesteps, size=(batch_size, 1)))
        pred_ns_mode = torch.cat([pred_ns_mode,
                                torch.full((batch_size, 3), fill_value=self.d_training_timesteps - 1, dtype=torch.long)],
                                dim=-1)
        pred_locs_mode = torch.from_numpy(np.random.choice(self.d_training_timesteps, size=(batch_size, 1)))
        pred_locs_mode = torch.cat([torch.full((batch_size, 1), fill_value=0, dtype=torch.long),
                                    pred_locs_mode,
                                    pred_locs_mode,
                                    torch.full((batch_size, 1), fill_value=self.d_training_timesteps - 1, dtype=torch.long)],
                                    dim=-1)
        pred_fluxes_mode = torch.from_numpy(np.random.choice(self.d_training_timesteps, size=(batch_size, 1)))
        pred_fluxes_mode = torch.cat([torch.full((batch_size, 3), fill_value=0, dtype=torch.long),
                                    pred_fluxes_mode],
                                    dim=-1)
        t = torch.cat([pred_ns_mode, pred_locs_mode, pred_fluxes_mode], dim=0)
        t = t[torch.randperm(t.shape[0])[:batch_size]]
        return repeat(t, "b k -> b (2 k) h w", 
                      h=self.image_size[0] // self.tile_slen, 
                      w=self.image_size[1] // self.tile_slen)
    
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
        encoded_locs = torch.stack([encoded_cat1[:, 1:3], encoded_cat2[:, 1:3]], dim=1)  # (b, 2, k_locs, h, w)
        dist_to_ori = torch.sqrt(((encoded_locs + 1) ** 2).sum(dim=2))  # note that there is a +1
        sorted_index = dist_to_ori.argsort(dim=1, descending=True)  # (b, 2, h, w)
        encoded_x_start = torch.stack([encoded_cat1, encoded_cat2], dim=1)
        encoded_x_start = torch.take_along_dim(encoded_x_start, 
                                               repeat(sorted_index, "b m h w -> b m k h w", k=encoded_x_start.shape[2]), 
                                               dim=1)
        encoded_x_start = rearrange(encoded_x_start, "b m k h w -> b (m k) h w")
        
        train_loss_args = {
            "model": self.my_net,
            "x_start": encoded_x_start,
            "t": self.generate_training_t_schedule(image.shape[0]).to(device=self.device),
            "loss_weights": None,
        }
        loss_dict = self._compute_diffusion_loss(train_loss_args, model_kwargs={"image": image})
        return loss_dict, torch.ones(image.shape[0], device=self.device)


class M2MDTRMLDFFullEncoder(M2RMLDFFullEncoder):
    def initialize_networks(self):
        assert self.tile_slen == 2
        assert self.image_size[0] == self.image_size[1]
        # assert self.image_size[0] == 112
        target_ch = self.catalog_parser.n_params_per_source * 2  # x2 for double detect
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        self.my_net = M2_MDTv2_RML_DF_full_S_2(image_n_bands=len(self.survey_bands), 
                                                image_ch_per_band=ch_per_band, 
                                                image_feats_ch=128, 
                                                input_size=self.image_size[0] // self.tile_slen, 
                                                in_channels=target_ch, 
                                                decode_layers=6,
                                                mask_ratio=0.3,
                                                mlp_ratio=4.0,
                                                learn_sigma=False)

    def _compute_diffusion_loss(self, loss_args, model_kwargs):
        no_mask_loss = self.rml_diffusion.training_losses(**loss_args, model_kwargs=model_kwargs)
        masked_loss = self.rml_diffusion.training_losses(**loss_args, model_kwargs={**model_kwargs, "enable_mask": True})
        return {
            "no_mask_loss": no_mask_loss,
            "masked_loss": masked_loss,
        }

class DC2RMLEncoder(M2RMLEncoder):
    @torch.inference_mode()
    def sample(self, batch):
        image = self.get_image(batch)
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        
        diffusion_sampling_config = {
            "model": self.my_net,
            "shape": (image.shape[0], 
                      self.catalog_parser.n_params_per_source,
                      self.image_size[0] // self.tile_slen, 
                      self.image_size[1] // self.tile_slen),
            "clip_denoised": True,
            "model_kwargs": {"image": image},
            "eta": self.ddim_eta,
        }
        sample = self.rml_diffusion.ddim_sample_loop(**diffusion_sampling_config)
        sample = sample.permute([0, 2, 3, 1])
        pred_cat = self.catalog_parser.decode(sample)
        pred_cat = self._interpolate_n_sources_and_locs(pred_cat, target_cat1)
        return pred_cat
    
    def _interpolate_n_sources_and_locs(self, sample_cat_dict, true_tile_cat):
        sample_cat_dict = copy.copy(sample_cat_dict)
        sample_cat_dict["n_sources"] = true_tile_cat["n_sources"]
        sample_cat_dict["locs"] = true_tile_cat["locs"]
        return TileCatalog(sample_cat_dict)
    
    def _compute_cur_batch_loss(self, batch):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        target_cat1["fluxes"] = target_cat1["fluxes"].clamp(max=self.max_fluxes)
        image = self.get_image(batch)  # (b, c, H, W)
        encoded_cat1 = self.catalog_parser.encode(target_cat1).permute(0, 3, 1, 2)  # (b, k, h, w)
        
        t, batch_sample_weights, batch_loss_weights = \
            self.schedule_sampler.sample(image.shape[0], device=self.device)
        train_loss_args = {
            "model": self.my_net,
            "x_start": encoded_cat1,
            "t": t,
            "loss_weights": batch_loss_weights
        }
        loss_dict = self._compute_diffusion_loss(train_loss_args, model_kwargs={"image": image})
        return loss_dict, batch_sample_weights
    

class DC2MDTRMLEncoder(DC2RMLEncoder):
    def initialize_networks(self):
        assert self.tile_slen == 4
        assert self.image_size[0] == self.image_size[1]
        # assert self.image_size[0] == 112
        target_ch = self.catalog_parser.n_params_per_source
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        self.my_net = DC2_MDTv2_RML_S_2(image_n_bands=len(self.survey_bands), 
                                        image_ch_per_band=ch_per_band, 
                                        image_feats_ch=128, 
                                        input_size=self.image_size[0] // self.tile_slen, 
                                        in_channels=target_ch, 
                                        decode_layers=6,
                                        mask_ratio=0.3,
                                        mlp_ratio=4.0,
                                        learn_sigma=False)
        
    def _compute_diffusion_loss(self, loss_args, model_kwargs):
        no_mask_loss = self.rml_diffusion.training_losses(**loss_args, model_kwargs=model_kwargs)
        masked_loss = self.rml_diffusion.training_losses(**loss_args, model_kwargs={**model_kwargs, "enable_mask": True})
        return {
            "no_mask_loss": no_mask_loss,
            "masked_loss": masked_loss,
        }
