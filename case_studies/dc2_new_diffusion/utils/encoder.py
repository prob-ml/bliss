import random
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import MetricCollection

from einops import rearrange

from bliss.catalog import TileCatalog
from bliss.encoder.metrics import CatalogMatcher
from bliss.global_env import GlobalEnv

from case_studies.dc2_new_diffusion.utils.convnet_layers import C3, ConvBlock
from case_studies.dc2_new_diffusion.utils.catalog_parser import CatalogParser
from case_studies.dc2_new_diffusion.utils.diffusion import (DiffusionModel, 
                                                            UpsampleDiffusionModel, 
                                                            LatentDiffusionModel, 
                                                            YNetDiffusionModel, 
                                                            YNetDoubleDetectDiffusionModel,
                                                            SimpleNetDiffusionModel)
from case_studies.dc2_new_diffusion.utils.unet import UNet, YNet, YNetV2, SimpleNetV1, SimpleNetV2
from case_studies.dc2_new_diffusion.utils.autoencoder import CatalogEncoder, CatalogDecoder


class DiffusionEncoder(pl.LightningModule):
    def __init__(
        self,
        survey_bands: list,
        tile_slen: int,
        image_normalizers: dict,
        catalog_parser: CatalogParser,
        image_size: list,
        matcher: CatalogMatcher,
        mode_metrics: MetricCollection,
        ddim_steps: int,
        ddim_objective: str,
        ddim_beta_schedule: str,
        ddim_self_cond: bool,
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
        self.ddim_steps = ddim_steps
        self.ddim_objective = ddim_objective
        self.ddim_beta_schedule = ddim_beta_schedule
        self.ddim_self_cond = ddim_self_cond

        self.detection_diffusion: DiffusionModel = None

        # important: this property activates manual optimization.
        self.automatic_optimization = False

        self.initialize_networks()

    def initialize_networks(self):
        raise NotImplementedError()

    def get_features(self, batch):
        assert batch["images"].size(2) % 16 == 0, "image dims must be multiples of 16"
        assert batch["images"].size(3) % 16 == 0, "image dims must be multiples of 16"

        input_lst = [inorm.get_input_tensor(batch) for inorm in self.image_normalizers]
        return torch.cat(input_lst, dim=2)

    @torch.inference_mode()
    def sample(self, batch, return_inter_output):
        x_features = self.get_features(batch)
        sample_dict = self.detection_diffusion.sample(x_features, return_inter_output)
        return self.catalog_parser.decode(sample_dict["final_pred"]), sample_dict["inter_output"]
    
    @torch.inference_mode()
    def debug_forward(self, batch):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        x_features = self.get_features(batch)  # (b, c, H, W)
        encoded_catalog_tensor = self.catalog_parser.encode(target_cat1)  # (b, h, w, k)
        return self.detection_diffusion(
            target=encoded_catalog_tensor, input_image=x_features
        )

    def _compute_cur_batch_loss(self, batch):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        x_features = self.get_features(batch)  # (b, c, H, W)
        encoded_catalog_tensor = self.catalog_parser.encode(target_cat1)  # (b, h, w, k)
        pred_dict = self.detection_diffusion(
            target=encoded_catalog_tensor, input_image=x_features
        )
        
        return (
            pred_dict["inter_loss"], 
            self.catalog_parser.gating_loss(pred_dict["final_pred_loss"], target_cat1),
            self.catalog_parser.get_gating_for_loss(target_cat1),
        )
    
    def _reweight_loss(self, loss, loss_gating):
        factored_loss = self.catalog_parser.factor_tensor(loss)
        factored_loss_gating = self.catalog_parser.factor_tensor(loss_gating)

        sub_loss = []
        for l, lg in zip(factored_loss, factored_loss_gating, strict=True):
            sub_loss.append(l.sum() / lg.sum() if lg.sum() > 0 else l.sum())
        mean_sub_loss = torch.cat([sl.unsqueeze(0) for sl in sub_loss]).mean()

        return sub_loss, mean_sub_loss

    def _compute_loss(self, batch, logging_name):
        inter_loss, final_pred_loss, loss_gating = self._compute_cur_batch_loss(batch)
        mean_inter_loss = inter_loss.mean()

        sub_final_loss, mean_final_loss = self._reweight_loss(final_pred_loss, loss_gating)

        batch_size = batch["images"].size(0)
        self.log(f"{logging_name}/_final_loss", mean_final_loss, batch_size=batch_size, sync_dist=True)
        self.log(f"{logging_name}/_inter_loss", mean_inter_loss, batch_size=batch_size, sync_dist=True)
        for sl, f in zip(sub_final_loss, self.catalog_parser.factors, strict=True):
            self.log(f"{logging_name}/_final_loss_{f.name}", sl, batch_size=batch_size, sync_dist=True)
        total_loss = mean_final_loss + mean_inter_loss
        self.log(f"{logging_name}/_loss", total_loss, batch_size=batch_size, sync_dist=True)

        return mean_inter_loss, mean_final_loss

    def on_fit_start(self):
        GlobalEnv.current_encoder_epoch = self.current_epoch

    def on_train_epoch_start(self):
        GlobalEnv.current_encoder_epoch = self.current_epoch

    def training_step(self, batch, batch_idx):
        """Training step (pytorch lightning)."""
        raise NotImplementedError()

    def update_metrics(self, batch, batch_idx):
        target_tile_cat = TileCatalog(batch["tile_catalog"])
        target_cat = target_tile_cat.to_full_catalog(self.tile_slen)
        target_cat.ori_tile_cat = target_tile_cat

        mode_tile_cat, _ = self.sample(batch, return_inter_output=False)
        mode_cat = mode_tile_cat.to_full_catalog(self.tile_slen)
        # in cases where model predicts locs at the boundary, to_tile_cat can't recover the original tile cat
        # this is not a bug and will not influence the metrics calculated using full cat
        # the following code is an ugly way to bypass this
        mode_cat.ori_tile_cat = mode_tile_cat
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
        return self.sample(batch, return_inter_output=False)[0]

    def configure_optimizers(self):
        """Configure optimizers for training (pytorch lightning)."""
        raise NotImplementedError()


class UpsampleDiffusionEncoder(DiffusionEncoder):
    def initialize_networks(self):
        assert self.tile_slen == 4
        target_ch = self.catalog_parser.n_params_per_source
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)

        self.unet = UNet(n_bands=len(self.survey_bands),
                         ch_per_band=ch_per_band,
                         xt_in_ch=target_ch,
                         out_ch=target_ch,
                         dim=32,
                         attn_heads=4,
                         attn_head_dim=16)
        
        postprocess_net_ch = 16
        self.postprocess_net = nn.Sequential(
            ConvBlock(target_ch, postprocess_net_ch, kernel_size=5),
            ConvBlock(postprocess_net_ch, postprocess_net_ch * 2, kernel_size=3, stride=2),
            C3(postprocess_net_ch * 2, postprocess_net_ch * 2, n=3),
            ConvBlock(postprocess_net_ch * 2, postprocess_net_ch * 4, kernel_size=3, stride=2),
            C3(postprocess_net_ch * 4, postprocess_net_ch * 4, n=3),
            ConvBlock(postprocess_net_ch * 4, postprocess_net_ch * 4, kernel_size=1),
            nn.Conv2d(postprocess_net_ch * 4, target_ch, kernel_size=1)
        )

        self.detection_diffusion = UpsampleDiffusionModel(
            model=self.unet,
            postprocess_net=self.postprocess_net,
            target_size=(
                target_ch,
                self.image_size[0],
                self.image_size[1],
            ),
            catalog_parser=self.catalog_parser,
            ddim_steps=self.ddim_steps,
            objective=self.ddim_objective,
            beta_schedule=self.ddim_beta_schedule,
            self_condition=self.ddim_self_cond,
        )

    def training_step(self, batch, batch_idx):
        unet_optimizer, postprocess_net_optimizer = self.optimizers()
        mean_inter_loss, mean_final_loss = self._compute_loss(batch, "train")

        unet_optimizer.zero_grad()
        self.manual_backward(mean_inter_loss)
        unet_optimizer.step()

        postprocess_net_optimizer.zero_grad()
        self.manual_backward(mean_final_loss)
        postprocess_net_optimizer.step()

        # step every epoch
        if self.trainer.is_last_batch:
            unet_scheduler, postprocess_net_scheduler = self.lr_schedulers()
            unet_scheduler.step()
            postprocess_net_scheduler.step()

    def configure_optimizers(self):
        unet_optimizer = Adam(self.unet.parameters(), **self.optimizer_params)
        postprocess_net_optimizer = Adam(self.postprocess_net.parameters(), **self.optimizer_params)
        unet_scheduler = MultiStepLR(unet_optimizer, **self.scheduler_params)
        postprocess_net_scheduler = MultiStepLR(postprocess_net_optimizer, **self.scheduler_params)
        return [unet_optimizer, postprocess_net_optimizer], [unet_scheduler, postprocess_net_scheduler]
        

class LatentDiffusionEncoder(DiffusionEncoder):
    def __init__(self, 
                 encoder_pretrained_weights, 
                 decoder_pretrained_weights, 
                 encoder_hidden_dim,
                 decoder_hidden_dim,
                 encoder_output_min,
                 encoder_output_max,
                 latent_scale,
                 **kwargs):
        self.encoder_pretrained_weights = encoder_pretrained_weights
        self.decoder_pretrained_weights = decoder_pretrained_weights
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_output_min = encoder_output_min
        self.encoder_output_max = encoder_output_max
        self.latent_scale = latent_scale
        super().__init__(**kwargs)

    def initialize_networks(self):
        assert self.tile_slen == 4
        target_ch = self.encoder_hidden_dim // 4
        assert self.decoder_hidden_dim == target_ch
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)

        self.unet = UNet(n_bands=len(self.survey_bands),
                         ch_per_band=ch_per_band,
                         xt_in_ch=target_ch,
                         out_ch=target_ch,
                         dim=32,
                         attn_heads=4,
                         attn_head_dim=16)

        cat_dim = self.catalog_parser.n_params_per_source
        self.encoder = CatalogEncoder(cat_dim, hidden_dim=self.encoder_hidden_dim)
        self.decoder = CatalogDecoder(cat_dim, hidden_dim=self.decoder_hidden_dim)

        with open(self.encoder_pretrained_weights, "rb") as en_f:
            encoder_state_dict = torch.load(en_f, map_location=self.device)
        self.encoder.load_state_dict(encoder_state_dict)

        with open(self.decoder_pretrained_weights, "rb") as de_f:
            decoder_state_dict = torch.load(de_f, map_location=self.device)
        self.decoder.load_state_dict(decoder_state_dict)

        self.detection_diffusion = LatentDiffusionModel(
            model=self.unet,
            encoder=self.encoder,
            decoder=self.decoder,
            encoder_output_min=self.encoder_output_min,
            encoder_output_max=self.encoder_output_max,
            scale=self.latent_scale,
            target_size=(
                target_ch,
                self.image_size[0],
                self.image_size[1],
            ),
            catalog_parser=self.catalog_parser,
            ddim_steps=self.ddim_steps,
            objective=self.ddim_objective,
            beta_schedule=self.ddim_beta_schedule,
            self_condition=self.ddim_self_cond,
        )

    def training_step(self, batch, batch_idx):
        unet_optimizer = self.optimizers()
        mean_inter_loss, _mean_final_loss = self._compute_loss(batch, "train")

        unet_optimizer.zero_grad()
        self.manual_backward(mean_inter_loss)
        unet_optimizer.step()

        # step every epoch
        if self.trainer.is_last_batch:
            unet_scheduler = self.lr_schedulers()
            unet_scheduler.step()

    def configure_optimizers(self):
        unet_optimizer = Adam(self.unet.parameters(), **self.optimizer_params)
        unet_scheduler = MultiStepLR(unet_optimizer, **self.scheduler_params)
        return [unet_optimizer], [unet_scheduler]


class NoLatentDiffusionEncoder(DiffusionEncoder):
    def __init__(self, acc_grad_batches, **kwargs):
        super().__init__(**kwargs)
        self.acc_grad_batches = acc_grad_batches
        assert self.acc_grad_batches >= 1

    def training_step(self, batch, batch_idx):
        my_optimizer = self.optimizers()
        mean_inter_loss, _mean_final_loss = self._compute_loss(batch, "train")

        mean_inter_loss = mean_inter_loss / self.acc_grad_batches
        self.manual_backward(mean_inter_loss)
        if (batch_idx + 1) % self.acc_grad_batches == 0:
            my_optimizer.step()
            my_optimizer.zero_grad()

        # step every epoch
        if self.trainer.is_last_batch:
            my_scheduler = self.lr_schedulers()
            my_scheduler.step()

    def configure_optimizers(self):
        my_optimizer = Adam(self.my_net.parameters(), **self.optimizer_params)
        my_scheduler = MultiStepLR(my_optimizer, **self.scheduler_params)
        return [my_optimizer], [my_scheduler]
    

class YNetDiffusionEncoder(NoLatentDiffusionEncoder):
    def __init__(self, ynet_version, ynet_dim, **kwargs):
        self.ynet_version = ynet_version
        assert self.ynet_version in ["v1", "v2"]
        self.ynet_dim = ynet_dim
        super().__init__(**kwargs)
        
    def initialize_networks(self):
        assert self.tile_slen == 4
        target_ch = self.catalog_parser.n_params_per_source
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)

        match self.ynet_version:
            case "v1":
                self.my_net = YNet(n_bands=len(self.survey_bands),
                                    ch_per_band=ch_per_band,
                                    in_ch=target_ch,
                                    out_ch=target_ch,
                                    dim=self.ynet_dim,
                                    attn_heads=4,
                                    attn_head_dim=32,
                                    use_self_cond=self.ddim_self_cond)
            case "v2":
                self.my_net = YNetV2(n_bands=len(self.survey_bands),
                                     ch_per_band=ch_per_band,
                                     in_ch=target_ch,
                                     out_ch=target_ch,
                                     dim=self.ynet_dim,
                                     attn_heads=4,
                                     attn_head_dim=32)
            case _:
                raise NotImplementedError()

        self.detection_diffusion = YNetDiffusionModel(
            model=self.my_net,
            target_size=(
                target_ch,
                self.image_size[0] // 4,
                self.image_size[1] // 4,
            ),
            catalog_parser=self.catalog_parser,
            ddim_steps=self.ddim_steps,
            objective=self.ddim_objective,
            beta_schedule=self.ddim_beta_schedule,
            self_condition=self.ddim_self_cond,
        )


class YNetFullDiffusionEncoder(YNetDiffusionEncoder):
    MAX_FLUXES = 22025.0
    # MAX_FLUXES = torch.inf

    def update_metrics(self, batch, batch_idx):
        target_tile_cat = TileCatalog(batch["tile_catalog"])
        target_tile_cat["fluxes"] = target_tile_cat["fluxes"].clamp(max=self.MAX_FLUXES)
        target_cat = target_tile_cat.to_full_catalog(self.tile_slen)
        target_cat.ori_tile_cat = target_tile_cat

        mode_tile_cat, _ = self.sample(batch, return_inter_output=False)
        mode_cat = mode_tile_cat.to_full_catalog(self.tile_slen)
        # in cases where model predicts locs at the boundary, to_tile_cat can't recover the original tile cat
        # this is not a bug and will not influence the metrics calculated using full cat
        # the following code is an ugly way to bypass this
        mode_cat.ori_tile_cat = mode_tile_cat
        mode_matching = self.matcher.match_catalogs(target_cat, mode_cat)
        self.mode_metrics.update(target_cat, mode_cat, mode_matching)

    def _compute_cur_batch_loss(self, batch):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        assert self.catalog_parser.factors[0].name == "n_sources_multi"
        max_n_sources = (2 ** self.catalog_parser.factors[0].n_params) - 1
        target_cat1["n_sources_multi"] = rearrange(target_cat["n_sources"], 
                                                   "b h w -> b h w 1 1").clamp(max=max_n_sources)
        target_cat1["fluxes"] = target_cat1["fluxes"].clamp(max=self.MAX_FLUXES)
        x_features = self.get_features(batch)  # (b, c, H, W)
        encoded_catalog_tensor = self.catalog_parser.encode(target_cat1)  # (b, h, w, k)
        pred_dict = self.detection_diffusion(
            target=encoded_catalog_tensor, input_image=x_features
        )
        return (
            pred_dict["inter_loss"], 
            self.catalog_parser.gating_loss(pred_dict["final_pred_loss"], target_cat1),
            self.catalog_parser.get_gating_for_loss(target_cat1),
        )
    
    def _compute_loss(self, batch, logging_name):
        inter_loss, final_pred_loss, loss_gating = self._compute_cur_batch_loss(batch)

        sub_inter_loss, mean_inter_loss = self._reweight_loss(inter_loss, loss_gating)
        sub_final_loss, mean_final_loss = self._reweight_loss(final_pred_loss, loss_gating)

        batch_size = batch["images"].size(0)
        self.log(f"{logging_name}/_final_loss", mean_final_loss, batch_size=batch_size, sync_dist=True)
        self.log(f"{logging_name}/_inter_loss", mean_inter_loss, batch_size=batch_size, sync_dist=True)
        for sl, f in zip(sub_final_loss, self.catalog_parser.factors, strict=True):
            self.log(f"{logging_name}/_final_loss_{f.name}", sl, batch_size=batch_size, sync_dist=True)
        for sl, f in zip(sub_inter_loss, self.catalog_parser.factors, strict=True):
            self.log(f"{logging_name}/_inter_loss_{f.name}", sl, batch_size=batch_size, sync_dist=True)
        total_loss = mean_final_loss + mean_inter_loss
        self.log(f"{logging_name}/_loss", total_loss, batch_size=batch_size, sync_dist=True)

        return mean_inter_loss, mean_final_loss


class YNetDoubleDetectDiffusionEncoder(YNetDiffusionEncoder):
    def initialize_networks(self):
        assert self.tile_slen == 4
        target_ch = self.catalog_parser.n_params_per_source
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)

        self.ynet = YNet(n_bands=len(self.survey_bands),
                         ch_per_band=ch_per_band,
                         in_ch=target_ch,
                         out_ch=target_ch,
                         dim=64,
                         attn_heads=4,
                         attn_head_dim=32)

        self.detection_diffusion = YNetDoubleDetectDiffusionModel(
            model=self.ynet,
            target_size=(
                target_ch,
                self.image_size[0] // 4,
                self.image_size[1] // 4,
            ),
            catalog_parser=self.catalog_parser,
            ddim_steps=self.ddim_steps,
            objective=self.ddim_objective,
            beta_schedule=self.ddim_beta_schedule,
            self_condition=self.ddim_self_cond,
        )
    
    @classmethod
    def _tile_cat_disjoint_union(cls, left_cat: TileCatalog, right_cat: TileCatalog):
        assert left_cat.batch_size == right_cat.batch_size
        assert left_cat.n_tiles_h == right_cat.n_tiles_h
        assert left_cat.n_tiles_w == right_cat.n_tiles_w
        assert left_cat.max_sources == 1
        assert right_cat.max_sources == 1

        d = {}
        ns11 = rearrange(left_cat["n_sources"], "b ht wt -> b ht wt 1 1")
        for k, v in left_cat.items():
            if k == "n_sources":
                d[k] = (v.bool() | right_cat[k].bool()).to(dtype=v.dtype)
            else:
                d1 = v
                d2 = right_cat[k]
                d[k] = torch.where(ns11 > 0, d1, d2)
        return TileCatalog(d)
    
    def _compute_cur_batch_loss(self, batch):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        target_cat2 = self._tile_cat_disjoint_union(
            left_cat=target_cat.get_brightest_sources_per_tile(band=self.reference_band, exclude_num=1, top_k=1), 
            right_cat=target_cat1)
        x_features = self.get_features(batch)  # (b, c, H, W)
        encoded_target1 = self.catalog_parser.encode(target_cat1)  # (b, h, w, k)
        encoded_target2 = self.catalog_parser.encode(target_cat2)  # (b, h, w, k)
        if random.random() > 0.5:
            pred_dict = self.detection_diffusion(
                target1=encoded_target1, target2=encoded_target2, input_image=x_features
            )
        else:
            pred_dict = self.detection_diffusion(
                target1=encoded_target2, target2=encoded_target1, input_image=x_features
            )
        
        return (
            pred_dict["inter_loss"], 
            self.catalog_parser.gating_loss(pred_dict["final_pred_loss"], target_cat1),
            self.catalog_parser.get_gating_for_loss(target_cat1),
        )
    
    @torch.inference_mode()
    def sample(self, batch, return_inter_output, locs_slack, init_time):
        x_features = self.get_features(batch)
        sample_dict = self.detection_diffusion.sample(x_features, return_inter_output, locs_slack, init_time)
        return sample_dict["double_detect"], sample_dict["inter_output"]


class SimpleNetDiffusionEncoder(NoLatentDiffusionEncoder):
    MAX_FLUXES = 22025.0
    # MAX_FLUXES = torch.inf

    def __init__(self, model_version, model_kwargs, **kwargs):
        self.model_version = model_version
        self.model_kwargs = model_kwargs
        assert self.model_version in ["v1", "v2"]
        super().__init__(**kwargs)

    def initialize_networks(self):
        assert self.tile_slen == 4
        target_ch = self.catalog_parser.n_params_per_source
        ch_per_band = sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)

        match self.model_version:
            case "v1":
                self.my_net = SimpleNetV1(n_bands=len(self.survey_bands),
                                          ch_per_band=ch_per_band,
                                          in_ch=target_ch,
                                          out_ch=target_ch,
                                          dim=32,
                                          **self.model_kwargs)
            case "v2":
                self.my_net = SimpleNetV2(n_bands=len(self.survey_bands),
                                          ch_per_band=ch_per_band,
                                          in_ch=target_ch,
                                          out_ch=target_ch,
                                          dim=64,
                                          **self.model_kwargs)
            case _:
                raise NotImplementedError()

        self.detection_diffusion = SimpleNetDiffusionModel(
            model=self.my_net,
            target_size=(
                target_ch,
                self.image_size[0] // 4,
                self.image_size[1] // 4,
            ),
            catalog_parser=self.catalog_parser,
            ddim_steps=self.ddim_steps,
            objective=self.ddim_objective,
            beta_schedule=self.ddim_beta_schedule,
            self_condition=self.ddim_self_cond,
        )

    @torch.inference_mode()
    def sample(self, batch, return_inter_output):
        x_features = self.get_features(batch)
        sample_dict = self.detection_diffusion.sample(x_features, return_inter_output)
        
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )

        sample_tile_dict = self.catalog_parser.decode(sample_dict["final_pred"])
        assert "n_sources" not in sample_tile_dict
        assert "locs" not in sample_tile_dict
        # sample_tile_dict["n_sources"] = target_cat1["n_sources"]
        sample_n_sources = sample_tile_dict["fluxes"][..., 0, 2] > 100
        target_n_sources = target_cat1["n_sources"] > 0
        sample_tile_dict["n_sources"] = (sample_n_sources & target_n_sources).to(dtype=target_cat1["n_sources"].dtype)
        # sample_tile_dict["locs"] = target_cat1["locs"]
        sample_tile_dict["locs"] = rearrange(sample_tile_dict["n_sources"] > 0, "b h w -> b h w 1 1") * target_cat1["locs"]

        return TileCatalog(sample_tile_dict), sample_dict["inter_output"]

    def update_metrics(self, batch, batch_idx):
        target_tile_cat = TileCatalog(batch["tile_catalog"])
        target_tile_cat["fluxes"] = target_tile_cat["fluxes"].clamp(max=self.MAX_FLUXES)
        target_cat = target_tile_cat.to_full_catalog(self.tile_slen)

        mode_tile_cat, _ = self.sample(batch, return_inter_output=False)
        mode_cat = mode_tile_cat.to_full_catalog(self.tile_slen)
        mode_matching = self.matcher.match_catalogs(target_cat, mode_cat)

        self.mode_metrics.update(target_cat, mode_cat, mode_matching)

    def _compute_cur_batch_loss(self, batch):
        target_cat = TileCatalog(batch["tile_catalog"])
        target_cat1 = target_cat.get_brightest_sources_per_tile(
            band=self.reference_band, exclude_num=0
        )
        target_cat1["fluxes"] = target_cat1["fluxes"].clamp(max=self.MAX_FLUXES)
        x_features = self.get_features(batch)  # (b, c, H, W)
        encoded_catalog_tensor = self.catalog_parser.encode(target_cat1)  # (b, h, w, k)
        pred_dict = self.detection_diffusion(
            target=encoded_catalog_tensor, input_image=x_features
        )
        return (
            pred_dict["inter_loss"], 
            self.catalog_parser.gating_loss(pred_dict["final_pred_loss"], target_cat1),
            self.catalog_parser.get_gating_for_loss(target_cat1),
        )
    
    def _compute_loss(self, batch, logging_name):
        inter_loss, final_pred_loss, loss_gating = self._compute_cur_batch_loss(batch)

        sub_inter_loss, mean_inter_loss = self._reweight_loss(inter_loss, loss_gating)
        sub_final_loss, mean_final_loss = self._reweight_loss(final_pred_loss, loss_gating)

        batch_size = batch["images"].size(0)
        self.log(f"{logging_name}/_final_loss", mean_final_loss, batch_size=batch_size, sync_dist=True)
        self.log(f"{logging_name}/_inter_loss", mean_inter_loss, batch_size=batch_size, sync_dist=True)
        for sl, f in zip(sub_final_loss, self.catalog_parser.factors, strict=True):
            self.log(f"{logging_name}/_final_loss_{f.name}", sl, batch_size=batch_size, sync_dist=True)
        for sl, f in zip(sub_inter_loss, self.catalog_parser.factors, strict=True):
            self.log(f"{logging_name}/_inter_loss_{f.name}", sl, batch_size=batch_size, sync_dist=True)
        total_loss = mean_final_loss + mean_inter_loss
        self.log(f"{logging_name}/_loss", total_loss, batch_size=batch_size, sync_dist=True)

        return mean_inter_loss, mean_final_loss
