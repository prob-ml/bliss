import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.upsampling import Upsample

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import List

from case_studies.dc2_yolov10.utils.conv_layers import (Conv, C2f, C2fCIB, SPPF, PSA, Concat)
from case_studies.dc2_yolov10.utils.loss import v10DetectLoss, v10LocsDetectLoss
from case_studies.dc2_yolov10.utils.other import box_cxcywh_to_xyxy, make_anchors, dist2bbox, bbox2dist, locs2dist


class v10Backbone(nn.Module):
    def __init__(self, n_bands, ch_per_band):
        super().__init__()

        self.n_bands = n_bands
        self.ch_per_band = ch_per_band

        self.preprocess3d = nn.Sequential(
            nn.Conv3d(self.n_bands, 64, [self.ch_per_band, 5, 5], padding=[0, 2, 2]),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            Rearrange("b c 1 h w -> b c h w"),
            # *[Conv(64, 64, k=5) for _ in range(4)],
        )

        self.backbone_blocks = nn.ModuleList([
            Conv(64, 64, k=3, s=2),  # 0: (b, 64, 40, 40)
            C2f(64, 64, n=2, shortcut=True),
            Conv(64, 128, k=3, s=2),  # 2: (b, 128, 20, 20)
            C2f(128, 128, n=4, shortcut=True),
            Conv(128, 256, k=3, s=2),  # 4: (b, 256, 10, 10)
            C2fCIB(256, 256, n=2, shortcut=True),
            SPPF(256, 256, k=5),
            PSA(256, 256),  # 7
        ])

    def forward(self, x):
        x = self.preprocess3d(x)
        output_dict = {}
        for i, m in enumerate(self.backbone_blocks):
            x = m(x)
            if i in [1, 3, 7]:
                output_dict[f"feats_{i}"] = x
        return output_dict


class v10Head(nn.Module):
    def __init__(self):
        super().__init__()

        self.head_blocks = nn.ModuleList([
            Upsample(scale_factor=2, mode="nearest"),  # 0: (b, 256, 20, 20)
            Concat(dimension=1),  # 1: (b, 256 + 128, 20, 20)
            C2f(384, 128, n=2),
            Upsample(scale_factor=2, mode="nearest"),
            Concat(dimension=1),  # 4: (b, 128 + 64, 40, 40)
            C2f(192, 64, n=2),
            Conv(64, 64, k=3, s=2),  # 6: (b, 64, 20, 20)
            Concat(dimension=1),  # 7: (b, 64 + 128, 20, 20)
            C2fCIB(192, 128, n=2, shortcut=True),
            Conv(128, 128, k=3, s=2),  
            Concat(dimension=1),  # 10: (b, 128 + 256, 10, 10)
            C2fCIB(384, 256, n=2, shortcut=True)  # 11
        ])

    def forward(self, backbone_feats):
        x = backbone_feats["feats_7"]
        save_feats = None
        output_list = []
        for i, m in enumerate(self.head_blocks):
            match i:
                case 1:
                    assert isinstance(m, Concat)
                    x = [x, backbone_feats["feats_3"]]
                case 4:
                    assert isinstance(m, Concat)
                    x = [x, backbone_feats["feats_1"]]
                case 7:
                    assert isinstance(m, Concat)
                    x = [x, save_feats]
                case 10:
                    assert isinstance(m, Concat)
                    x = [x, backbone_feats["feats_7"]]
                case _:
                    assert not isinstance(m, Concat)

            x = m(x)

            match i:
                case 2:
                    save_feats = x
                case 5 | 8 | 11:
                    output_list.append(x)
        return output_list
        

class v10Detect(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.nc = 1  # number of classes
        self.image_size = 80
        self.nl = len(ch)  # number of detection layers
        c2, c3 = 64, 64  # channels

        self.dfl_init()
        
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), 
                Conv(c2, c2, 3), 
                nn.Conv2d(c2, self.cv2_output_ch, 1)
            ) 
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1)
            ) 
            for x in ch
        )

        self.bias_init()

        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def dfl_init(self):
        self.register_buffer("stride", torch.tensor([2, 4, 8]))
        self.reg_max = 16
        self.register_buffer("proj", torch.arange(self.reg_max, dtype=torch.float))
        self.cv2_output_ch = 4 * self.reg_max

    def get_dfl(self, anchor_points, target, pred_distri, fg_mask):
        target_ltrb = bbox2dist(anchor_points, target, self.reg_max - 1)[fg_mask]
        target_left = target_ltrb.long()
        target_right = target_left + 1
        weight_left = target_right - target_ltrb
        weight_right = 1 - weight_left
        pred_distri = pred_distri[fg_mask].view(-1, self.reg_max)
        return (
            F.cross_entropy(pred_distri, target_left.view(-1), reduction="none").view(target_left.shape) * weight_left
            + F.cross_entropy(pred_distri, target_right.view(-1), reduction="none").view(target_right.shape) * weight_right
        ).mean(-1, keepdim=True)

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (self.image_size / s) ** 2)

    def forward_feat(self, x, cv2, cv3):
        y = []
        for i in range(self.nl):
            y.append(torch.cat((cv2[i](x[i]), cv3[i](x[i])), 
                               dim=1))
        return y
    
    def decode_pred_distri(self, 
                           anchor_points: torch.Tensor, 
                           anchor_strides: torch.Tensor, 
                           pred_distri: torch.Tensor):
        b, a, c = pred_distri.shape
        pred_prob = pred_distri.view(b, a, 4, c // 4).softmax(dim=-1)
        pred_dist = pred_prob.matmul(self.proj.type(pred_distri.dtype))
        pred_xyxy = dist2bbox(pred_dist, anchor_points, xywh=False) * anchor_strides
        return {
            "pred_xyxy": pred_xyxy
        }
    
    def postprocess(self, preds: List[torch.Tensor]):
        pred_tensor = torch.cat([rearrange(xi, "b c h w -> b c (h w)")
                                for xi in preds], 
                                dim=-1)
        pred_distri, pred_logits = pred_tensor.split((preds[0].shape[1] - self.nc, self.nc), dim=1)

        pred_logits = pred_logits.permute(0, 2, 1).contiguous()  # (b, 2100, 1)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # (b, 2100, c - 1)

        anchor_points, anchor_strides = make_anchors(preds, self.stride, 0.5)
        pred_output = self.decode_pred_distri(anchor_points, anchor_strides, pred_distri)
        return {
            "pred_logits": pred_logits,
            "pred_distri": pred_distri,
            "anchor_points": anchor_points,
            "anchor_strides": anchor_strides,
            **pred_output,
        }

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        one2many_feats = self.forward_feat(x, self.cv2, self.cv3)
        one2one_feats = self.forward_feat([xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3)
        return {
            "one2many": self.postprocess(one2many_feats), 
            "one2one": self.postprocess(one2one_feats),
        }


class v10Model(nn.Module):
    def __init__(self, n_bands, ch_per_band):
        super().__init__()

        self.backbone = v10Backbone(n_bands=n_bands,
                                    ch_per_band=ch_per_band)
        self.head = v10Head()
        self.detect = v10Detect(ch=(64, 128, 256))
        self.criterion = v10DetectLoss(dfl_func=self.detect.get_dfl, 
                                       loss_gain_dict={
                                           "iou": 7.5,
                                           "cls": 0.5,
                                           "dfl": 1.5,
                                       })

    def _infer_images(self, images):
        backbone_feats = self.backbone(images)
        head_output_list = self.head(backbone_feats)
        return self.detect(head_output_list)

    def forward(self, images, gt_cxcywh, gt_mask):
        preds = self._infer_images(images)
        gt_xyxy = box_cxcywh_to_xyxy(gt_cxcywh)
        return self.criterion(preds, gt_xyxy, gt_mask)

    @torch.inference_mode()
    def sample(self, images):
        preds = self._infer_images(images)["one2one"]
        return preds["pred_xyxy"], preds["pred_logits"].sigmoid()


class v10LocsDetect(v10Detect):
    def dfl_init(self):
        self.register_buffer("stride", torch.tensor([2, 4, 8]))
        self.reg_max = 4
        self.register_buffer("proj", 
                             torch.arange(self.reg_max * 2, 
                                          dtype=torch.float) - self.reg_max + 0.5)
        self.cv2_output_ch = 4 * self.reg_max

    def get_dfl(self, anchor_points, target, pred_distri, fg_mask):
        target_dist = locs2dist(anchor_points, target, self.reg_max - 0.5)[fg_mask]  # (in_mask_sources, 2)
        target_left = (rearrange(target_dist, "m k -> m k 1") >= rearrange(self.proj[1:], "p -> 1 1 p")).sum(dim=-1).long()  # (in_mask_sources, 2)
        assert ((target_left >= 0) & (target_left < self.reg_max * 2 - 1)).all()
        target_right = target_left + 1
        weight_left = torch.gather(repeat(self.proj, 
                                          "p -> m p", 
                                          m=target_right.shape[0]),
                                   dim=-1,
                                   index=target_right) - target_dist
        assert ((weight_left <= 1.0) & (weight_left > 0.0)).all()
        weight_right = 1 - weight_left
        pred_distri = pred_distri[fg_mask].view(-1, self.reg_max * 2)  # (in_mask_sources * 2, reg_max * 2)
        return (
            F.cross_entropy(pred_distri, target_left.view(-1), reduction="none").view(target_left.shape) * weight_left
            + F.cross_entropy(pred_distri, target_right.view(-1), reduction="none").view(target_right.shape) * weight_right
        ).mean(-1, keepdim=True)  # (in_mask_sources, 1)

    def decode_pred_distri(self, 
                           anchor_points: torch.Tensor, 
                           anchor_strides: torch.Tensor, 
                           pred_distri: torch.Tensor):
        b, a, c = pred_distri.shape
        pred_prob = pred_distri.view(b, a, 2, c // 2).softmax(dim=-1)
        pred_dist = pred_prob.matmul(self.proj.type(pred_distri.dtype))
        pred_locs = (pred_dist + anchor_points) * anchor_strides
        return {
            "pred_locs": pred_locs
        }
    

class v10LocsModel(nn.Module):
    def __init__(self, n_bands, ch_per_band):
        super().__init__()

        self.backbone = v10Backbone(n_bands=n_bands,
                                    ch_per_band=ch_per_band)
        self.head = v10Head()
        self.detect = v10LocsDetect(ch=(64, 128, 256))
        self.criterion = v10LocsDetectLoss(dfl_func=self.detect.get_dfl,
                                           loss_gain_dict={
                                               "dist": 7.5,
                                               "cls": 0.5,
                                               "dfl": 1.5,
                                           })

    def _infer_images(self, images):
        backbone_feats = self.backbone(images)
        head_output_list = self.head(backbone_feats)
        return self.detect(head_output_list)

    def forward(self, images, gt_locs, gt_mask):
        preds = self._infer_images(images)
        return self.criterion(preds, gt_locs, gt_mask)

    @torch.inference_mode()
    def sample(self, images):
        preds = self._infer_images(images)["one2one"]
        return preds["pred_locs"], preds["pred_logits"].sigmoid()