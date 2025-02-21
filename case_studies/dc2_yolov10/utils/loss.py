import torch
from torch import nn
import torch.nn.functional as F

from case_studies.dc2_yolov10.utils.other import make_anchors, dist2bbox, bbox_iou, bbox2dist
from case_studies.dc2_yolov10.utils.assigner import TaskAlignedAssigner

class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max

    def forward(self, 
                pred_dist, pred_bboxes, anchor_points, 
                target_bboxes, target_scores,
                fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = (1.0 - iou) * weight

        # DFL loss
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
        loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight

        return loss_iou.sum(), loss_dfl.sum()
    
    @staticmethod
    def _df_loss(pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class v10DetectLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.one2many = v8DetectionLoss(tal_topk=10)
        self.one2one = v8DetectionLoss(tal_topk=1)
    
    def forward(self, preds, gt_bboxes, mask_gt):
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, gt_bboxes, mask_gt)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, gt_bboxes, mask_gt)
        return {
            "one2many": loss_one2many,
            "one2one": loss_one2one,
        }
    

class v8DetectionLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        super().__init__()

        self.register_buffer("dummy_param", torch.zeros(0))

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.register_buffer("stride", torch.tensor([2, 4, 8]))  # model strides
        self.nc = 1  # number of classes
        self.reg_max = 16
        self.no = self.reg_max * 4 + self.nc

        self.assigner = TaskAlignedAssigner(topk=tal_topk, 
                                            num_classes=self.nc, 
                                            alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max - 1)
        self.register_buffer("proj", torch.arange(self.reg_max, dtype=torch.float))

        self.box_gain = 7.5
        self.cls_gain = 0.5
        self.dfl_gain = 1.5

    @property
    def device(self):
        return self.dummy_param.device

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        b, a, c = pred_dist.shape  # batch, anchors, channels
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def forward(self, 
                 preds, # [(b, 65, 40, 40), (b, 65, 20, 20), (b, 65, 10, 10)]
                 gt_bboxes, # xyxy (b, max_sources, 4)
                 mask_gt,  # (b, max_sources, 1)
                 ):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        assert preds[0].shape[0] == gt_bboxes.shape[0]
        assert gt_bboxes.shape[:2] == mask_gt.shape[:2]

        pred_distri, pred_scores = torch.cat([xi.view(preds[0].shape[0], self.no, -1) 
                                              for xi in preds], 
                                              dim=2).split((self.reg_max * 4, self.nc), 
                                                           dim=1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (b, 2100, 1)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # (b, 2100, 64)

        anchor_points, stride_tensor = make_anchors(preds, self.stride, 0.5)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss = {}
        # Cls loss
        loss["cls"] = self.bce(pred_scores, target_scores.to(dtype=pred_scores.dtype)).sum()

        # Bbox loss
        if fg_mask.sum() > 0:
            target_bboxes /= stride_tensor
            loss["box"], loss["dfl"] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, 
                target_bboxes, target_scores,
                fg_mask
            )
        else:
            loss["box"] = torch.zeros(1, device=self.device)
            loss["dfl"] = torch.zeros(1, device=self.device)

        loss["box"] *= self.box_gain / target_scores_sum  # box gain
        loss["cls"] *= self.cls_gain / target_scores_sum  # cls gain
        loss["dfl"] *= self.dfl_gain / target_scores_sum  # dfl gain

        return loss  # loss(box, cls, dfl)