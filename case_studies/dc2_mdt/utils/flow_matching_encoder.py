from typing import Optional

import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import MetricCollection
import torchdiffeq
import torchsde
from typing import Union

from bliss.catalog import TileCatalog
from bliss.encoder.metrics import CatalogMatcher
from bliss.global_env import GlobalEnv

from case_studies.dc2_mdt.utils.catalog_parser import CatalogParser
from case_studies.dc2_mdt.utils.mdt_models import M2_MDTv2_full_S_2, DC2_MDTv2_full_S_2
from case_studies.dc2_mdt.utils.flow_matching import (ConditionalFlowMatcher, 
                                                      ExactOptimalTransportConditionalFlowMatcher,
                                                      StochasticConditionalFlowMatcher,
                                                      pad_t_like_x)

class SDEWrapper(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, drift_and_score_fn, sigma, data_ori_shape):
        super().__init__()
        self.drift_and_score_fn = drift_and_score_fn
        self.sigma = sigma
        self.data_ori_shape = data_ori_shape

    # drift
    def f(self, t, y):
        y = y.view(self.data_ori_shape)
        ode_drift, score = self.drift_and_score_fn(x=y, t=t).chunk(2, dim=1)
        return ode_drift.flatten(start_dim=1) + score.flatten(start_dim=1)

    # diffusion
    def g(self, t, y):
        return torch.full_like(y, fill_value=self.sigma)


class M2FMEncoder(pl.LightningModule):
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
        d_flow_matching_type: str,
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

        self.d_flow_matching_type = d_flow_matching_type
       
        self.acc_grad_batches = acc_grad_batches
        assert self.acc_grad_batches >= 1
        self.max_fluxes = float(max_fluxes) if max_fluxes != "inf" else torch.inf

        self.my_net = None
        self.flow_matcher: Union[ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher] = None

        self.register_buffer("my_dummy_variable", torch.zeros(0))

        # important: this property activates manual optimization.
        self.automatic_optimization = False

        self.initialize_flow_matcher()
        self.initialize_networks()

    @property
    def device(self):
        return self.my_dummy_variable.device

    def initialize_flow_matcher(self):
        sigma = 0.0
        match self.d_flow_matching_type:
            case "vanilla":
                self.flow_matcher = ConditionalFlowMatcher(sigma=sigma)
            case "ot":
                self.flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
            case "stochastic":
                self.flow_matcher = StochasticConditionalFlowMatcher(sigma=0.1)
            case _:
                raise NotImplementedError()

    def initialize_networks(self):
        assert self.tile_slen == 2
        assert self.image_size[0] == self.image_size[1]
        # assert self.image_size[0] == 112
        target_ch = self.catalog_parser.n_params_per_source * 2  # x2 for double detect
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        self.my_net = M2_MDTv2_full_S_2(image_n_bands=len(self.survey_bands), 
                                        image_ch_per_band=ch_per_band, 
                                        image_feats_ch=128, 
                                        input_size=self.image_size[0] // self.tile_slen, 
                                        in_channels=target_ch, 
                                        decode_layers=6,
                                        mask_ratio=0.3,
                                        mlp_ratio=4.0,
                                        learn_sigma=(self.d_flow_matching_type == "stochastic"))  # it doesn't learn sigma; just to double the output channel

    def get_image(self, batch):
        assert batch["images"].size(2) % 8 == 0, "image dims must be multiples of 8"
        assert batch["images"].size(3) % 8 == 0, "image dims must be multiples of 8"
        input_lst = [inorm.get_input_tensor(batch) for inorm in self.image_normalizers]
        return torch.cat(input_lst, dim=2)

    @torch.inference_mode()
    def sample(self, batch, ode_config_dict=None, sde_config_dict=None):
        image = self.get_image(batch)
        init_noise = torch.randn(image.shape[0], 
                                    self.catalog_parser.n_params_per_source * 2,  # x2 for double detect 
                                    self.image_size[0] // self.tile_slen, self.image_size[1] // self.tile_slen,
                                device=image.device)
        self.my_net.enter_fast_inference()
        if self.d_flow_matching_type != "stochastic":
            assert sde_config_dict is None
            if ode_config_dict is None:
                ode_config_dict = {
                    "atol": 1e-4,
                    "rtol": 1e-4,
                    "method": "dopri5",
                }
            traj = torchdiffeq.odeint(
                lambda t, x: self.my_net(x=x, t=t, image=image),
                init_noise,
                torch.linspace(0, 1, 2, device=self.device),
                **ode_config_dict,
            )
        else:
            assert ode_config_dict is None
            if sde_config_dict is None:
                sde_config_dict = {
                    "dt": 0.01,
                    "method": "euler",
                    "atol": 1e-4,
                    "rtol": 1e-4,
                }
            sde_obj = SDEWrapper(lambda x, t: self.my_net(x=x, t=t, image=image),
                                 sigma=self.flow_matcher.sigma,
                                 data_ori_shape=init_noise.shape)
            traj = torchsde.sdeint(
                sde_obj,
                init_noise.flatten(start_dim=1),
                ts=torch.linspace(0, 1, 2, device=image.device),
                **sde_config_dict,
            )
            traj = traj.view(traj.shape[0], traj.shape[1], *init_noise.shape[1:])
        self.my_net.exit_fast_inference()
        sample = traj[-1]
        sample = sample.clamp(min=-1.0, max=1.0)
        sample1, sample2 = sample.permute([0, 2, 3, 1]).chunk(2, dim=-1)  # (b, h, w, k)
        first_cat = self.catalog_parser.decode(sample1)
        second_cat = self.catalog_parser.decode(sample2)
        return first_cat.union(second_cat, disjoint=False)

    def _compute_flow_matching_loss(self, x1, images):
        if self.d_flow_matching_type == "vanilla":
            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0=torch.randn_like(x1), 
                                                                               x1=x1)
            reordered_images = images
        elif self.d_flow_matching_type == "ot":
            t, xt, ut, _, reordered_images = self.flow_matcher.guided_sample_location_and_conditional_flow(x0=torch.randn_like(x1), 
                                                                                                            x1=x1, 
                                                                                                            y1=images)
        elif self.d_flow_matching_type == "stochastic":
            t, xt, ut, eps = self.flow_matcher.sample_location_and_conditional_flow(x0=torch.randn_like(x1), 
                                                                                    x1=x1, 
                                                                                    return_noise=True)
            reordered_images = images
            lambda_t = self.flow_matcher.compute_lambda(t)
        else:
            raise NotImplementedError()
        
        no_mask_pred = self.my_net(x=xt, t=t, image=reordered_images, enable_mask=False)
        masked_pred = self.my_net(x=xt, t=t, image=reordered_images, enable_mask=True)
        
        if self.d_flow_matching_type != "stochastic":
            return {
                "no_mask_loss": (ut - no_mask_pred) ** 2,
                "masked_loss": (ut - masked_pred) ** 2,
            }
        else:
            no_mask_pred_ut, no_mask_pred_st = no_mask_pred.chunk(2, dim=1)
            masked_pred_ut, masked_pred_st = masked_pred.chunk(2, dim=1)
            lambda_t = pad_t_like_x(lambda_t, no_mask_pred_st)
            no_mask_flow_loss = (no_mask_pred_ut - ut) ** 2
            no_mask_score_loss = (lambda_t * no_mask_pred_st + eps) ** 2
            masked_flow_loss = (masked_pred_ut - ut) ** 2
            masked_score_loss = (lambda_t * masked_pred_st + eps) ** 2
            return {
                "no_mask_flow_loss": no_mask_flow_loss, 
                "no_mask_score_loss": no_mask_score_loss,
                "masked_flow_loss": masked_flow_loss,
                "masked_score_loss": masked_score_loss,
            }

    
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
        loss_dict = self._compute_flow_matching_loss(x1=torch.cat([encoded_cat1, encoded_cat2], dim=1), 
                                                     images=image)
        return loss_dict

    def _compute_loss(self, batch, logging_name):
        loss_dict = self._compute_cur_batch_loss(batch)

        batch_size = batch["images"].size(0)
        with torch.inference_mode():
            for k, v in loss_dict.items():
                self.log(f"{logging_name}/_{k}", 
                            v.mean(), 
                            batch_size=batch_size, 
                            sync_dist=True)
        loss = torch.stack([v.mean() for v in loss_dict.values()]).sum()
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
        target_tile_cat1 = target_tile_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        target_tile_cat2 = target_tile_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=1
        )
        target_tile_cat["fluxes"] = target_tile_cat["fluxes"].clamp(max=self.max_fluxes)
        target_tile_cat1["fluxes"] = target_tile_cat1["fluxes"].clamp(max=self.max_fluxes)
        target_tile_cat2["fluxes"] = target_tile_cat2["fluxes"].clamp(max=self.max_fluxes)
        target_cat = target_tile_cat.to_full_catalog(self.tile_slen)
        target_cat.target_tile_cat1 = target_tile_cat1
        target_cat.target_tile_cat2 = target_tile_cat2

        sample_tile_cat = self.sample(batch)
        sample_cat = sample_tile_cat.to_full_catalog(self.tile_slen)
        sample_cat.sample_tile_cat = sample_tile_cat
        sample_matching = self.matcher.match_catalogs(target_cat, sample_cat)
        self.mode_metrics.update(target_cat, sample_cat, sample_matching)

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        """Pytorch lightning method."""
        self._compute_loss(batch, "val")
        self.update_metrics(batch, batch_idx)

    def report_metrics(self, metrics, logging_name, show_epoch=False):
        for k, v in metrics.compute().items():
            if isinstance(v, torch.Tensor):
                self.log(f"{logging_name}/{k}", v, sync_dist=True)
            elif isinstance(v, Figure):
                name = f"Epoch:{self.current_epoch}" if show_epoch else ""
                name += f"/{logging_name} {k}"
                self.logger.experiment.add_figure(name, v)
                plt.close(v)
            else:
                raise NotImplementedError()

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


class DC2FMEncoder(M2FMEncoder):
    def initialize_networks(self):
        assert self.tile_slen == 4
        assert self.image_size[0] == self.image_size[1]
        target_ch = self.catalog_parser.n_params_per_source * 2  # x2 for double detect
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        self.my_net = DC2_MDTv2_full_S_2(image_n_bands=6,  # we use the ori image  # image_n_bands=len(self.survey_bands),  
                                        image_ch_per_band=ch_per_band, 
                                        image_feats_ch=128, 
                                        input_size=self.image_size[0] // self.tile_slen, 
                                        in_channels=target_ch, 
                                        decode_layers=6,
                                        mask_ratio=0.3,
                                        mlp_ratio=4.0,
                                        learn_sigma=(self.d_flow_matching_type == "stochastic"))  # it doesn't learn sigma; just to double the output channel
