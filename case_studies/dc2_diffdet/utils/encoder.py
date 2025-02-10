from typing import Optional

import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import MetricCollection
from torchvision.ops import MultiScaleRoIAlign

from bliss.catalog import TileCatalog, FullCatalog
from bliss.encoder.metrics import CatalogMatcher
from bliss.global_env import GlobalEnv

from case_studies.dc2_diffdet.utils.diffusion import SparseRCNNDiffusionModel
from case_studies.dc2_diffdet.utils.backbone import FeaturesBackbone, FPN
from case_studies.dc2_diffdet.utils.head import DynamicHead


class DiffusionEncoder(pl.LightningModule):
    def __init__(
        self,
        survey_bands: list,
        tile_slen: int,
        image_normalizers: dict,
        image_size: list,
        matcher: CatalogMatcher,
        mode_metrics: MetricCollection,
        ddim_steps: int,
        ddim_beta_schedule: str,
        feature_backbone_path: str,
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
        self.ddim_steps = ddim_steps
        self.ddim_beta_schedule = ddim_beta_schedule
        self.features_backbone_path = feature_backbone_path

        self.detection_diffusion: SparseRCNNDiffusionModel = None

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
        sample_result = self.detection_diffusion.sample(normalized_images)

        plocs = []
        for b_result in sample_result:
            pred_xyxy = b_result["pred_xyxy"]  # (num_boxes_after_nms, 4)
            pred_plocs = torch.stack([(pred_xyxy[:, 3] + pred_xyxy[:, 1]) / 2,
                                      (pred_xyxy[:, 2] + pred_xyxy[:, 0]) / 2],
                                      dim=-1).clamp(min=0.0, max=self.image_size[0])  # (num_boxes_after_num, 2)
            pred_scores = b_result["pred_scores"]  # (num_boxes_after_nms, 1)
            mask = pred_scores > score_threshold
            plocs.append(pred_plocs[mask])
        plocs_tensor = pad_sequence(plocs, batch_first=True)  # (b, m, 2)
        n_sources = torch.tensor([p.shape[0] for p in plocs], dtype=torch.int64, device=plocs_tensor.device)
        assert n_sources.max() == plocs_tensor.shape[1]
        return FullCatalog(height=self.image_size[0],
                           width=self.image_size[1],
                           d={
                               "n_sources": n_sources,
                               "plocs": plocs_tensor,
                           })

    def _compute_cur_batch_loss(self, batch):
        target_cat = TileCatalog(batch["tile_catalog"]).to_full_catalog(tile_slen=self.tile_slen)
        assert "bbox_hw" in target_cat
        target_cat_galaxy_bools = target_cat.galaxy_bools  # (b, m)
        target_cat_valid_bbox_bools = ~(target_cat["bbox_hw"].isnan())  # (b, m)
        assert ((target_cat_galaxy_bools | target_cat_valid_bbox_bools) == target_cat_galaxy_bools).all()
        target_cat_mask = target_cat_valid_bbox_bools  # (b, m)

        gt_cxcywh = []
        for i in range(target_cat_mask.shape[0]):
            b_mask = target_cat_mask[i]  # (m, )
            cxcy = target_cat["plocs"][i][b_mask].flip(dims=[1])  # (valid_sources, 2)
            bbox_wh = target_cat["bbox_hw"][i][b_mask].flip(dims=[1])  # (valid_sources, 2)
            assert ((cxcy >= 0.0) & (cxcy <= self.image_size[0])).all()
            assert (bbox_wh > 0.0).all()
            gt_cxcywh.append(torch.cat([cxcy, bbox_wh], dim=1))

        normalized_images = self.get_features(batch)
        return self.detection_diffusion(
            images=normalized_images, gt_cxcywh=gt_cxcywh,
        )

    def _compute_loss(self, batch, logging_name):
        loss_dict = self._compute_cur_batch_loss(batch)

        batch_size = batch["images"].size(0)
        total_loss = None
        for k, v in loss_dict.items():
            self.log(f"{logging_name}/_{k}_loss", v, batch_size=batch_size, sync_dist=True)
            if total_loss is None:
                total_loss = v
            else:
                total_loss += v
        output_loss = total_loss / len(loss_dict)
        self.log(f"{logging_name}/_loss", output_loss, batch_size=batch_size, sync_dist=True)
        return output_loss

    def on_fit_start(self):
        GlobalEnv.current_encoder_epoch = self.current_epoch

    def on_train_epoch_start(self):
        GlobalEnv.current_encoder_epoch = self.current_epoch

    def training_step(self, batch, batch_idx):
        """Training step (pytorch lightning)."""
        raise NotImplementedError()

    def update_metrics(self, batch, batch_idx):
        target_cat = TileCatalog(batch["tile_catalog"]).to_full_catalog(tile_slen=self.tile_slen)
        target_cat_galaxy_bools = target_cat.galaxy_bools  # (b, m)
        target_cat_valid_bbox_bools = ~(target_cat["bbox_hw"].isnan())  # (b, m)
        assert ((target_cat_galaxy_bools | target_cat_valid_bbox_bools) == target_cat_galaxy_bools).all()
        target_cat_mask = target_cat_valid_bbox_bools  # (b, m)

        compact_indices = torch.argsort(target_cat_mask, dim=-1, descending=True, stable=False)  # (b, m)
        d = {
            "n_sources": target_cat_mask.sum(dim=-1).to(target_cat["n_sources"].dtype)  # (b, )
        }
        n_sources_mask = torch.gather(target_cat_mask, dim=-1, index=compact_indices)
        for k, v in target_cat.items():
            if k == "n_sources":
                continue
            d[k] = torch.gather(v, dim=-1, index=compact_indices) * n_sources_mask
        masked_target_cat = FullCatalog(height=target_cat.height,
                                        width=target_cat.width,
                                        d=d)
        mode_cat = self.sample(batch, score_threshold=0.5)
        mode_matching = self.matcher.match_catalogs(masked_target_cat, mode_cat)
        self.mode_metrics.update(masked_target_cat, mode_cat, mode_matching)

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


class SparseRCNNDiffusionEncoder(DiffusionEncoder):
    def initialize_networks(self):
        self.features_net = FeaturesBackbone(
            n_bands=len(self.survey_bands),
            ch_per_band=sum(inorm.num_channels_per_band() for inorm in self.image_normalizers)
        )

        with open(self.features_backbone_path, "rb") as f:
            ck = torch.load(f, map_location=self.device)
        features_backbone_state_dict = {k: v for k, v in ck["state_dict"].items() if k.startswith("features_net.")}
        self.features_net.load_state_dict(features_backbone_state_dict)
        self.features_net.partial_freeze()

        self.fpn = FPN(feature_backbone=self.features_net,
                       out_ch=256)
        
        self.box_pooler = MultiScaleRoIAlign(featmap_names=self.fpn.fpn_features,
                                             output_size=3, sampling_ratio=2,
                                             canonical_scale=3, canonical_level=2)
        self.dynamic_head = DynamicHead(box_pooler=self.box_pooler,
                                        num_rcnn_heads=6,
                                        rcnn_input_ch=256,
                                        rcnn_hidden_ch=2048,
                                        rcnn_attn_nhead=8,
                                        image_size=self.image_size)
        
        self.detection_diffusion = SparseRCNNDiffusionModel(
            image_size=self.image_size,
            backbone_model=self.fpn,
            output_head_model=self.dynamic_head,
            num_box_proposals=300,
            beta_schedule=self.ddim_beta_schedule,
            ddim_steps=self.ddim_steps,
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
        optimizer = Adam(list(self.fpn.parameters()) + list(self.dynamic_head.parameters()), 
                         **self.optimizer_params)
        scheduler = MultiStepLR(optimizer, **self.scheduler_params)
        return [optimizer], [scheduler]
