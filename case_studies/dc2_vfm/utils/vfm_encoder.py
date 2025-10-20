from typing import Optional

import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import MetricCollection
import torchdiffeq

from einops import repeat, rearrange

from bliss.catalog import TileCatalog
from bliss.encoder.metrics import CatalogMatcher
from bliss.global_env import GlobalEnv

from case_studies.dc2_vfm.utils.variational_dist import VariationalDist
from case_studies.dc2_vfm.utils.nn_model import UUNet


class VFMEncoder(pl.LightningModule):
    def __init__(
        self,
        *,
        survey_bands: list,
        tile_slen: int,
        image_size: list,
        image_normalizers: dict,
        variational_dist: VariationalDist,
        matcher: CatalogMatcher,
        mode_metrics: MetricCollection,
        acc_grad_batches: int,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
        reference_band: int = 2,
        **kwargs,  # other args inherited from base config
    ):
        super().__init__()

        self.survey_bands = survey_bands
        self.tile_slen = tile_slen
        self.image_normalizers = torch.nn.ModuleList(image_normalizers.values())
        self.variational_dist = variational_dist
        self.image_size = image_size
        self.mode_metrics = mode_metrics
        self.matcher = matcher
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params if scheduler_params else {"milestones": []}
        self.reference_band = reference_band
       
        self.acc_grad_batches = acc_grad_batches
        assert self.acc_grad_batches >= 1

        self.register_buffer("my_dummy_variable", torch.zeros(0))
        self.my_net: UUNet = None

        # important: this property activates manual optimization.
        self.automatic_optimization = False

        self.initialize_networks()

    @property
    def device(self):
        return self.my_dummy_variable.device

    def initialize_networks(self):
        assert self.tile_slen == 4
        assert self.image_size[0] == self.image_size[1]
        # assert self.image_size[0] == 112
        target_ns_params = self.variational_dist.n_ns_params_per_source * 2  # x2 for double detect
        target_other_params = self.variational_dist.n_other_params_per_source * 2
        input_ns_chs = self.variational_dist.n_ns_chs_per_source * 2
        input_other_chs = self.variational_dist.n_other_chs_per_source * 2
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)

        self.my_net = UUNet(img_n_bands=len(self.survey_bands),
                            img_ch_per_band=ch_per_band,
                            img_backbone_base_ch=64,
                            ns_ch=input_ns_chs,
                            ns_params=target_ns_params,
                            ns_unet_base_ch=64,
                            other_ch=input_other_chs,
                            other_params=target_other_params,
                            other_unet_base_ch=64)

    def get_image(self, batch):
        assert batch["images"].size(2) % 8 == 0, "image dims must be multiples of 8"
        assert batch["images"].size(3) % 8 == 0, "image dims must be multiples of 8"
        input_lst = [inorm.get_input_tensor(batch) for inorm in self.image_normalizers]
        return torch.cat(input_lst, dim=2)

    @torch.inference_mode()
    def sample(self, batch, ode_config_dict=None):
        image = self.get_image(batch)
        n_sources_x0 = torch.randn(image.shape[0], 
                                    self.image_size[0] // self.tile_slen, 
                                    self.image_size[1] // self.tile_slen,
                                    2,  # 2 for double detect 
                                    device=image.device)
        other_x0 = torch.randn(image.shape[0],
                               self.image_size[0] // self.tile_slen, 
                               self.image_size[1] // self.tile_slen,
                               self.variational_dist.n_other_chs_per_source * 2,
                               device=image.device)
        
        if ode_config_dict is None:
            # ode_config_dict = {
            #     "method": "dopri5",
            #     "atol": 1e-4,
            #     "rtol": 1e-4,
            #     "t": torch.linspace(0, 0.999, 2, device=self.device),
            # }
            ode_config_dict = {
                "method": "euler",
                "options": {"step_size": 0.01},
                "atol": 1e-4,
                "rtol": 1e-4,
                "t": torch.linspace(0, 0.99, 2, device=self.device),
            }
        
        self.my_net.enter_fast_inference()
        
        def n_sources_velocity(t, x):
            # assert (t < 1.0).all()  # dopri5 will generate t > 1.0 due to adaptive step
            t = repeat(t.unsqueeze(0), "1 -> b", b=x.shape[0])
            ns_cat1, ns_cat2 = self.my_net.calculate_n_sources_cat(n_sources_xt=x, 
                                                                   t=t, 
                                                                   image=image)
            ns1_mean = self.variational_dist.n_sources_mean(ns_cat1)
            ns2_mean = self.variational_dist.n_sources_mean(ns_cat2)
            ns_mean = torch.stack([ns1_mean, ns2_mean], dim=-1)  # (b, h, w, k)
            expanded_t = t.view(-1, 1, 1, 1)
            vt = (ns_mean - x) / (1 - expanded_t)
            return vt
        
        ns_traj = torchdiffeq.odeint(
            n_sources_velocity,
            n_sources_x0,
            **ode_config_dict,
        )
        est_ns = ns_traj[-1]  # (b, h, w, 2)
        est_ns = (est_ns > 0.5).long()
        est_ns1, est_ns2 = torch.chunk(est_ns, 2, dim=-1)
        est_ns1, est_ns2 = est_ns1.squeeze(-1), est_ns2.squeeze(-1)

        def other_velocity(t, x):
            # assert (t < 1.0).all()
            t = repeat(t.unsqueeze(0), "1 -> b", b=x.shape[0])
            other_cat1, other_cat2 = self.my_net.calculate_other_cat(other_xt=x, 
                                                                     t=t, 
                                                                     image=image, 
                                                                     n_sources_x1=est_ns.float())
            other1_mean = self.variational_dist.other_mean(other_cat1, est_ns1.bool())
            other2_mean = self.variational_dist.other_mean(other_cat2, est_ns2.bool())
            other_mean = torch.cat([other1_mean, other2_mean], dim=-1)  # (b, h, w, k)
            expanded_t = t.view(-1, 1, 1, 1)
            vt = (other_mean - x) / (1 - expanded_t)
            return vt
        
        other_traj = torchdiffeq.odeint(
            other_velocity,
            other_x0,
            **ode_config_dict,
        )
        est_other = other_traj[-1]
        est_other1, est_other2 = torch.chunk(est_other, 2, dim=-1)

        self.my_net.exit_fast_inference()

        first_cat = self.variational_dist.tensor_to_tile_cat(est_ns1, est_other1)
        second_cat = self.variational_dist.tensor_to_tile_cat(est_ns2, est_other2)
        return first_cat.union(second_cat, disjoint=False)

    def _compute_vfm_loss(self, 
                          n_sources_x1: torch.Tensor, 
                          other_x1: torch.Tensor,
                          image: torch.Tensor,
                          true_tile_cat1: TileCatalog, 
                          true_tile_cat2: TileCatalog):
        assert n_sources_x1.ndim == 4  # (b, h, w, k)
        assert n_sources_x1.shape[-1] == 2
        assert other_x1.ndim == 4
        assert n_sources_x1.dtype == torch.float
        t = torch.rand(other_x1.shape[0], 
                       device=other_x1.device, 
                       dtype=other_x1.dtype)
        expanded_t = t.view(-1, 1, 1, 1)
        n_sources_x0 = torch.randn_like(n_sources_x1)
        n_sources_xt = expanded_t * n_sources_x1 + (1 - expanded_t) * n_sources_x0
        other_x0 = torch.randn_like(other_x1)
        other_xt = expanded_t * other_x1 + (1 - expanded_t) * other_x0
        x_cat1, x_cat2 = self.my_net(n_sources_xt=n_sources_xt, 
                                     other_xt=other_xt, 
                                     t=t, 
                                     image=image,
                                     n_sources_x1=n_sources_x1)
        nll1 = self.variational_dist.compute_nll(x_cat1, true_tile_cat1)
        nll2 = self.variational_dist.compute_nll(x_cat2, true_tile_cat2)
        return torch.sum(nll1 + nll2) / (n_sources_x1.sum() * other_x1.shape[-1] + n_sources_x1.numel())

    def _compute_cur_batch_loss(self, batch):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        target_cat2 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=1
        )
        image = self.get_image(batch)  # (b, bands, c, H, W)
        # note that we do torch.log(fluxes) in the following steps
        encoded_n_sources1, encoded_other1 = self.variational_dist.tile_cat_to_tensor(target_cat1)  # (b, h, w, k)
        encoded_n_sources2, encoded_other2 = self.variational_dist.tile_cat_to_tensor(target_cat2)  # (b, h, w, k)
        nll = self._compute_vfm_loss(n_sources_x1=torch.cat([encoded_n_sources1, encoded_n_sources2], dim=-1), 
                                    other_x1=torch.cat([encoded_other1, encoded_other2], dim=-1),
                                    image=image,
                                    true_tile_cat1=target_cat1,
                                    true_tile_cat2=target_cat2)
        return nll

    def _compute_loss(self, batch, logging_name):
        nll = self._compute_cur_batch_loss(batch)
        batch_size = batch["images"].size(0)
        self.log(f"{logging_name}/_loss", nll.item(), batch_size=batch_size, sync_dist=True)
        return nll

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
