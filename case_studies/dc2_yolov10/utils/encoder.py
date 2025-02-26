from typing import Optional

import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import MetricCollection

from bliss.catalog import TileCatalog, FullCatalog
from bliss.encoder.metrics import CatalogMatcher
from bliss.global_env import GlobalEnv

from case_studies.dc2_yolov10.utils.model import v10Model, v10LocsModel
from case_studies.dc2_yolov10.utils.other import box_xyxy_to_cxcywh


class PlainEncoder(pl.LightningModule):
    def __init__(
        self,
        survey_bands: list,
        tile_slen: int,
        image_normalizers: dict,
        image_size: list,
        matcher: CatalogMatcher,
        mode_metrics: MetricCollection,
        optimizer_params: Optional[dict] = None,
        scheduler_params: Optional[dict] = None,
        reference_band: int = 2,
        **kwargs,  # other args inherited from base config
    ):
        super().__init__()

        self.survey_bands = survey_bands
        self.tile_slen = tile_slen
        self.image_normalizers = torch.nn.ModuleList(image_normalizers.values())
        self.image_size = image_size  # [height, width]
        assert self.image_size[0] == self.image_size[1]
        self.mode_metrics = mode_metrics
        self.matcher = matcher
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params if scheduler_params else {"milestones": []}
        self.reference_band = reference_band

        self.detection_model: v10Model = None

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
    def sample(self, batch, score_threshold):
        normalized_images = self.get_features(batch)
        sample_xyxy, sample_scores = self.detection_model.sample(normalized_images)
        sample_cxcywh = box_xyxy_to_cxcywh(sample_xyxy)

        plocs = sample_cxcywh[:, :, :2].flip(dims=[2]).clamp(min=0, max=self.image_size[0])  # (b, m, 2)
        mask = sample_scores > score_threshold  # (b, m, 1)
        plocs = plocs[mask.squeeze(-1)]  # (total_sources, 2)
        plocs_list = plocs.split(mask.sum(dim=(-2, -1)).tolist(), dim=0)

        plocs_tensor = pad_sequence(plocs_list, batch_first=True)  # (b, m, 2)
        n_sources = torch.tensor([p.shape[0] for p in plocs_list], 
                                 dtype=torch.int64, 
                                 device=plocs_tensor.device)
        assert n_sources.max() == plocs_tensor.shape[1]
        return FullCatalog(height=self.image_size[0],
                           width=self.image_size[1],
                           d={
                               "n_sources": n_sources,
                               "plocs": plocs_tensor,
                           })

    def _compute_cur_batch_loss(self, batch):
        target_cat = TileCatalog(batch["tile_catalog"]).to_full_catalog(tile_slen=self.tile_slen)
        sources_mask = target_cat.is_on_mask.unsqueeze(-1)  # (b, m, 1)

        assert "bbox_hw" in target_cat
        target_cat_galaxy_bools = target_cat.galaxy_bools  # (b, m)
        target_cat_valid_bbox_bools = ~(target_cat["bbox_hw"].isnan().any(dim=-1, keepdim=True))  # (b, m, 1)
        target_cat_valid_bbox_bools &= sources_mask
        assert ((target_cat_galaxy_bools | target_cat_valid_bbox_bools) == target_cat_galaxy_bools).all()

        cxcy = target_cat["plocs"].flip(dims=[2])  # (b, m, 2)
        assert not cxcy.isnan().any()
        wh = target_cat["bbox_hw"].flip(dims=[2])  # (b, m, 2)
        gt_cxcywh = torch.cat([cxcy, wh], dim=-1).nan_to_num(0.0)
        assert ((gt_cxcywh[..., :2] >= 0.0) & (gt_cxcywh[..., :2] <= self.image_size[0])).all()
        assert (gt_cxcywh[..., 2:] >= 0.0).all()
        gt_cxcywh[..., 2:].clamp_(min=5.0)
        gt_cxcywh *= sources_mask

        max_sources = sources_mask.sum(dim=(-1, -2)).max()
        gt_cxcywh = gt_cxcywh[:, :max_sources, :]
        sources_mask = sources_mask[:, :max_sources, :]

        normalized_images = self.get_features(batch)
        return self.detection_model(
            images=normalized_images, 
            gt_cxcywh=gt_cxcywh,
            gt_mask=sources_mask,
        )

    def _compute_loss(self, batch, logging_name):
        loss_dict = self._compute_cur_batch_loss(batch)

        batch_size = batch["images"].size(0)
        total_loss = None
        for k, v in loss_dict["one2one"].items():
            self.log(f"{logging_name}/_one2one_{k}_loss", v, batch_size=batch_size, sync_dist=True)
            if total_loss is None:
                total_loss = v
            else:
                total_loss += v

        for k, v in loss_dict["one2many"].items():
            self.log(f"{logging_name}/_one2many_{k}_loss", v, batch_size=batch_size, sync_dist=True)
            total_loss += v
   
        self.log(f"{logging_name}/_loss", total_loss, batch_size=batch_size, sync_dist=True)
        return total_loss

    def on_fit_start(self):
        GlobalEnv.current_encoder_epoch = self.current_epoch

    def on_train_epoch_start(self):
        GlobalEnv.current_encoder_epoch = self.current_epoch

    def training_step(self, batch, batch_idx):
        """Training step (pytorch lightning)."""
        raise NotImplementedError()

    def update_metrics(self, batch, batch_idx):
        target_cat = TileCatalog(batch["tile_catalog"]).to_full_catalog(tile_slen=self.tile_slen)
        mode_cat = self.sample(batch, score_threshold=0.5)
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
        return self.sample(batch, score_threshold=0.5)

    def configure_optimizers(self):
        """Configure optimizers for training (pytorch lightning)."""
        raise NotImplementedError()


class v10Encoder(PlainEncoder):
    def initialize_networks(self):
        self.detection_model = v10Model(
            n_bands=len(self.survey_bands),
            ch_per_band=sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        )

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        loss = self._compute_loss(batch, "train")

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        # step every epoch
        if self.trainer.is_last_batch:
            scheduler = self.lr_schedulers()
            scheduler.step()

    def configure_optimizers(self):
        optimizer = Adam(self.detection_model.parameters(), 
                         **self.optimizer_params)
        scheduler = MultiStepLR(optimizer, **self.scheduler_params)
        return [optimizer], [scheduler]


class v10LocsEncoder(PlainEncoder):
    def initialize_networks(self):
        self.detection_model = v10LocsModel(
            n_bands=len(self.survey_bands),
            ch_per_band=sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        )

    @torch.inference_mode()
    def sample(self, batch, score_threshold):
        normalized_images = self.get_features(batch)
        sample_locs, sample_scores = self.detection_model.sample(normalized_images)

        plocs = sample_locs.clamp(min=0, max=self.image_size[0])  # (b, m, 2)
        mask = sample_scores > score_threshold  # (b, m, 1)
        plocs = plocs[mask.squeeze(-1)]  # (total_sources, 2)
        plocs_list = plocs.split(mask.sum(dim=(-2, -1)).tolist(), dim=0)

        plocs_tensor = pad_sequence(plocs_list, batch_first=True)  # (b, m, 2)
        n_sources = torch.tensor([p.shape[0] for p in plocs_list], 
                                 dtype=torch.int64, 
                                 device=plocs_tensor.device)
        assert n_sources.max() == plocs_tensor.shape[1]
        return FullCatalog(height=self.image_size[0],
                           width=self.image_size[1],
                           d={
                               "n_sources": n_sources,
                               "plocs": plocs_tensor,
                           })

    def _compute_cur_batch_loss(self, batch):
        target_cat = TileCatalog(batch["tile_catalog"]).to_full_catalog(tile_slen=self.tile_slen)
        sources_mask = target_cat.is_on_mask.unsqueeze(-1)  # (b, m, 1)

        gt_locs = target_cat["plocs"]  # (b, m, 2)
        gt_locs *= sources_mask

        max_sources = sources_mask.sum(dim=(-1, -2)).max()
        gt_locs = gt_locs[:, :max_sources, :]
        sources_mask = sources_mask[:, :max_sources, :]

        normalized_images = self.get_features(batch)
        return self.detection_model(
            images=normalized_images, 
            gt_locs=gt_locs,
            gt_mask=sources_mask,
        )

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        loss = self._compute_loss(batch, "train")

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        # step every epoch
        if self.trainer.is_last_batch:
            scheduler = self.lr_schedulers()
            scheduler.step()

    def configure_optimizers(self):
        optimizer = Adam(self.detection_model.parameters(), 
                         **self.optimizer_params)
        scheduler = MultiStepLR(optimizer, **self.scheduler_params)
        return [optimizer], [scheduler]