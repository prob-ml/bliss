import torch
from torch import nn

from case_studies.dc2_yolov10.utils.other import bbox_iou
from case_studies.dc2_yolov10.utils.assigner import TaskAlignedAssigner, DistanceAssigner


class v10DetectLoss(nn.Module):
    def __init__(self, dfl_func, loss_gain_dict):
        super().__init__()
        self.init_loss(dfl_func=dfl_func, loss_gain_dict=loss_gain_dict)
        
    def init_loss(self, dfl_func, loss_gain_dict):
        self.one2many = v8DetectionLoss(tal_topk=10, 
                                        dfl_func=dfl_func, 
                                        loss_gain_dict=loss_gain_dict)
        self.one2one = v8DetectionLoss(tal_topk=1, 
                                       dfl_func=dfl_func, 
                                       loss_gain_dict=loss_gain_dict)
    
    def forward(self, preds, gt_obj, mask_gt):
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, gt_obj, mask_gt)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, gt_obj, mask_gt)
        return {
            "one2many": loss_one2many,
            "one2one": loss_one2one,
        }
    

class v8DetectionLoss(nn.Module):
    def __init__(self, tal_topk, dfl_func, loss_gain_dict):  # model must be de-paralleled
        super().__init__()

        self.register_buffer("dummy_param", torch.zeros(0))

        self.nc = 1
        self.tal_topk = tal_topk
        self.bce = nn.BCEWithLogitsLoss(reduction="none")        
        self.dfl_func = dfl_func
        self.loss_gain_dict = loss_gain_dict

        self.init_assigner_and_name()

    def init_assigner_and_name(self):
        self.assigner = TaskAlignedAssigner(topk=self.tal_topk, 
                                            num_classes=self.nc, 
                                            alpha=0.5, beta=6.0)
        self.pred_obj_name = "xyxy"

    @property
    def device(self):
        return self.dummy_param.device

    def get_loss(self, 
                     pred_logits,
                     pred_distri, 
                     pred_obj, 
                     anchor_points, 
                     target_obj, 
                     target_scores,
                     fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        iou = bbox_iou(pred_obj[fg_mask], target_obj[fg_mask], xywh=False, CIoU=True)
        loss_iou = (1.0 - iou) * weight

        loss_dfl = self.dfl_func(anchor_points=anchor_points,
                                 target=target_obj,
                                 pred_distri=pred_distri,
                                 fg_mask=fg_mask) * weight
        
        loss_cls = self.bce(pred_logits, target_scores.to(dtype=pred_logits.dtype))
        
        return {
            "cls": loss_cls.sum(),
            "iou": loss_iou.sum(), 
            "dfl": loss_dfl.sum(),
        }

    def forward(self, 
                 preds, # dict
                 gt_obj, # (b, max_sources, k)
                 mask_gt,  # (b, max_sources, 1)
                 ):
        assert preds[f"pred_{self.pred_obj_name}"].shape[0] == gt_obj.shape[0]
        assert gt_obj.shape[:2] == mask_gt.shape[:2]

        target_obj, target_scores, fg_mask = self.assigner(
            preds["pred_logits"].detach().sigmoid(),
            preds[f"pred_{self.pred_obj_name}"].detach().type(gt_obj.dtype),
            preds["anchor_points"] * preds["anchor_strides"],
            gt_obj,
            mask_gt,
        )

        loss = {}
        if fg_mask.sum() > 0:
            anchor_strides = preds["anchor_strides"]
            loss.update(
                self.get_loss(
                    preds["pred_logits"],
                    preds["pred_distri"], 
                    preds[f"pred_{self.pred_obj_name}"] / anchor_strides, 
                    preds["anchor_points"], 
                    target_obj / anchor_strides, 
                    target_scores,
                    fg_mask,
                )
            )
        else:
            for k in self.loss_gain_dict.keys():
                loss[k] = torch.zeros(1, device=self.device)

        target_scores_sum = max(target_scores.sum(), 1)
        for k in loss.keys():
            loss[k] *= self.loss_gain_dict[k] / target_scores_sum

        return loss
    

class v10LocsDetectLoss(v10DetectLoss):
    def init_loss(self, dfl_func, loss_gain_dict):
        self.one2many = v8LocsDetectionLoss(tal_topk=8, 
                                            dfl_func=dfl_func, 
                                            loss_gain_dict=loss_gain_dict)
        self.one2one = v8LocsDetectionLoss(tal_topk=1, 
                                           dfl_func=dfl_func, 
                                           loss_gain_dict=loss_gain_dict)


class v8LocsDetectionLoss(v8DetectionLoss):
    def init_assigner_and_name(self):
        self.assigner = DistanceAssigner(select_radius=8,
                                         topk=self.tal_topk, 
                                         num_classes=self.nc, 
                                         alpha=0.5, beta=8.0)
        self.pred_obj_name = "locs"

    def get_loss(self, 
                pred_logits,
                pred_distri, 
                pred_obj, 
                anchor_points, 
                target_obj, 
                target_scores,
                fg_mask):
        assert target_scores.shape[-1] == 1
        weight = target_scores.squeeze(-1)[fg_mask].unsqueeze(-1)

        loss_dist = (pred_obj[fg_mask] - target_obj[fg_mask]) ** 2
        loss_dist *= weight

        loss_dfl = self.dfl_func(anchor_points=anchor_points,
                                 target=target_obj,
                                 pred_distri=pred_distri,
                                 fg_mask=fg_mask) * weight
        
        loss_cls = self.bce(pred_logits, target_scores.to(dtype=pred_logits.dtype))
        
        return {
            "cls": loss_cls.sum(),
            "dist": loss_dist.sum(), 
            "dfl": loss_dfl.sum(),
        }
