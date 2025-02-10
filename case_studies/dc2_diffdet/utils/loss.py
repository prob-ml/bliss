import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import sigmoid_focal_loss, box_iou

from einops import rearrange

from case_studies.dc2_diffdet.utils.box_func import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou


class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, image_whwh):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.image_whwh = image_whwh  # (1, 4)

        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0

    def label_loss(self, output_logits, match_outcome):
        """Classification loss (NLL)
        """
        target_onehot = torch.stack([i["matched_box_mask"] for i in match_outcome], dim=0).float().unsqueeze(-1)  # (b, num_boxes, 1)
        assert output_logits.shape == target_onehot.shape
        cls_loss = sigmoid_focal_loss(inputs=output_logits, 
                                        targets=target_onehot, 
                                        alpha=self.focal_loss_alpha, 
                                        gamma=self.focal_loss_gamma, 
                                        reduction="sum")
        num_boxes = sum([i["matched_gt_indices"].numel() for i in match_outcome])
        return {"loss_ce": cls_loss / num_boxes if num_boxes > 0 else cls_loss}


    def box_loss(self, output_xyxy, gt_cxcywh, match_outcome):
        batch_size = len(gt_cxcywh)
        pred_xyxy_list = []
        pred_norm_xyxy_list = []
        target_norm_xyxy_list = []
        target_xyxy_list = []
        for batch_idx in range(batch_size):
            valid_query = match_outcome[batch_idx]["matched_box_mask"]
            gt_multi_idx = match_outcome[batch_idx]["matched_gt_indices"]
            if len(gt_multi_idx) == 0:
                continue
            b_output_xyxy = output_xyxy[batch_idx]
            b_output_norm_xyxy = b_output_xyxy / self.image_whwh
            b_target_xyxy = box_cxcywh_to_xyxy(gt_cxcywh[batch_idx])
            b_target_norm_xyxy = b_target_xyxy / self.image_whwh

            pred_xyxy_list.append(b_output_xyxy[valid_query])
            pred_norm_xyxy_list.append(b_output_norm_xyxy[valid_query])
            target_xyxy_list.append(b_target_xyxy[gt_multi_idx])
            target_norm_xyxy_list.append(b_target_norm_xyxy[gt_multi_idx])

        if len(pred_xyxy_list) != 0:
            pred_xyxy = torch.cat(pred_xyxy_list)
            pred_norm_xyxy = torch.cat(pred_norm_xyxy_list)
            target_xyxy = torch.cat(target_xyxy_list)
            target_norm_xyxy = torch.cat(target_norm_xyxy_list)
            num_boxes = pred_xyxy.shape[0]

            losses = {}

            loss_xyxy_l1 = F.l1_loss(pred_norm_xyxy, target_norm_xyxy, reduction="sum")
            losses["loss_xyxy_l1"] = loss_xyxy_l1 / num_boxes
            loss_giou = 1 - torch.diag(generalized_box_iou(pred_xyxy, target_xyxy))
            losses["loss_giou"] = loss_giou.sum() / num_boxes
        else:
            losses = {"loss_xyxy_l1": (output_xyxy * 0.0).sum(),
                      "loss_giou": (output_xyxy * 0.0).sum()}

        return losses

    def forward(self, output, gt_cxcywh):
        # Retrieve the matching between the outputs of the last layer and the targets
        match_outcome = self.matcher({k: v 
                                for k, v in output.items() 
                                if k != "aux_outputs"}, 
                                gt_cxcywh)

        # Compute all the requested losses
        losses = {}
        losses.update(self.label_loss(output["pred_logits"], match_outcome))
        losses.update(self.box_loss(output["pred_xyxy"], gt_cxcywh, match_outcome))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        assert "aux_outputs" in output
        for i, aux_outputs in enumerate(output["aux_outputs"]):
            match_outcome = self.matcher(aux_outputs, gt_cxcywh)
            l_dict = {}
            l_dict.update(self.label_loss(aux_outputs["pred_logits"], match_outcome))
            l_dict.update(self.box_loss(aux_outputs["pred_xyxy"], gt_cxcywh, match_outcome))
            l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
            losses.update(l_dict)
        
        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        return losses


class DynamicKMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-k (dynamic) matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, 
                 weight_class_cost, 
                 weight_xyxy_l1_cost, 
                 weight_giou_cost,
                 image_whwh,
                 ):
        super().__init__()
        assert weight_class_cost > 0
        assert weight_xyxy_l1_cost > 0
        assert weight_giou_cost > 0
        self.weight_class_cost = weight_class_cost
        self.weight_xyxy_l1_cost = weight_xyxy_l1_cost
        self.weight_giou_cost = weight_giou_cost
        self.image_whwh = image_whwh  # (1, 4)

        self.ota_k = 5
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0

    @classmethod
    def _recover_cost_matrix_for_each_batch(cls, 
                                            whole_cost_matrix: torch.Tensor, 
                                            gt_split_size: list):
        assert whole_cost_matrix.shape[1] == sum(gt_split_size)
        whole_cost_matrix = rearrange(whole_cost_matrix, 
                                      "(b num_boxes) total_turth_num -> b num_boxes total_truth_num",
                                      b=len(gt_split_size))
        return [c[i] for i, c in enumerate(whole_cost_matrix.split(gt_split_size, dim=-1))]
        
    @torch.inference_mode()
    def forward(self, output, gt_cxcywh):
        out_prob = torch.sigmoid(output["pred_logits"]).view(-1, 1)  # (b * num_boxes, 1)
        out_xyxy = output["pred_xyxy"].view(-1, 4)  # (b * num_noxes, 4)
        gt_split_size = [g.shape[0] for g in gt_cxcywh]
        gt_cxcywh = torch.cat(gt_cxcywh, dim=0)  # (total_truth_num, 4)
        gt_xyxy = box_cxcywh_to_xyxy(gt_cxcywh)

        loose_in_circle_mask, strict_in_box_mask = self.get_in_boxes_mask(
            box_xyxy_to_cxcywh(out_xyxy), 
            gt_cxcywh,
            gt_split_size,
        )  # (b * num_boxes, ), (b * num_boxes, total_truth_num)

        # Compute the classification cost.
        alpha = self.focal_loss_alpha
        gamma = self.focal_loss_gamma
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = (pos_cost_class - neg_cost_class).unsqueeze(-1)  # (b * num_boxes, 1)

        out_norm_xyxy = out_xyxy / self.image_whwh  # (b * num_boxes, 4)
        gt_norm_xyxy = gt_xyxy / self.image_whwh  # (total_truth_num, 4)
        cost_bbox = torch.cdist(out_norm_xyxy, gt_norm_xyxy, p=1)  # (b * num_boxes, total_truth_num)

        cost_giou = -generalized_box_iou(out_xyxy, gt_xyxy)  # (b * num_boxes, total_truth_num)

        # Final cost matrix
        cost = self.weight_xyxy_l1_cost * cost_bbox + \
                self.weight_class_cost * cost_class + \
                self.weight_giou_cost * cost_giou + \
                100.0 * (~strict_in_box_mask)  # (b * num_boxes, total_truth_num)
        cost[~loose_in_circle_mask] += 10000.0
        sub_cost = self._recover_cost_matrix_for_each_batch(cost, gt_split_size=gt_split_size)

        iou_matrix = box_iou(out_xyxy, gt_xyxy)  # (b * num_boxes, total_truth_num)
        sub_iou_matrix = self._recover_cost_matrix_for_each_batch(iou_matrix, gt_split_size=gt_split_size)
        match_outcome = []
        for s_cost, s_iou in zip(sub_cost, sub_iou_matrix, strict=True):
            match_outcome.append(self.dynamic_k_matching(s_cost, s_iou))

        return match_outcome

    def get_in_boxes_mask(self, pred_cxcywh, gt_cxcywh, gt_split_size):
        gt_xyxy = box_cxcywh_to_xyxy(gt_cxcywh)  # (total_truth_num, 4)

        pred_cx = pred_cxcywh[:, 0:1]  # (b * num_boxes, 1)
        pred_cy = pred_cxcywh[:, 1:2]  # (b * num_boxes, 1)

        # whether the center of each anchor is inside a gt box
        b_l = pred_cx > gt_xyxy[:, 0].unsqueeze(0)  # (b * num_boxes, total_truth_num)
        b_r = pred_cx < gt_xyxy[:, 2].unsqueeze(0)
        b_t = pred_cy > gt_xyxy[:, 1].unsqueeze(0)
        b_b = pred_cy < gt_xyxy[:, 3].unsqueeze(0)
        pred_c_in_gt_boxes = torch.stack([b_l, b_r, b_t, b_b], dim=-1).all(dim=-1)  # (b * num_boxes, total_truth_num)

        center_radius = 2.5
        # Modified to self-adapted sampling --- the center size depends on the size of the gt boxes
        # https://github.com/dulucas/UVO_Challenge/blob/main/Track1/detection/mmdet/core/bbox/assigners/rpn_sim_ota_assigner.py#L212
        b_l = pred_cx > (gt_cxcywh[:, 0] - (center_radius * (gt_xyxy[:, 2] - gt_xyxy[:, 0]))).unsqueeze(0)  # (b * num_boxes, total_truth_num)
        b_r = pred_cx < (gt_cxcywh[:, 0] + (center_radius * (gt_xyxy[:, 2] - gt_xyxy[:, 0]))).unsqueeze(0)
        b_t = pred_cy > (gt_cxcywh[:, 1] - (center_radius * (gt_xyxy[:, 3] - gt_xyxy[:, 1]))).unsqueeze(0)
        b_b = pred_cy < (gt_cxcywh[:, 1] + (center_radius * (gt_xyxy[:, 3] - gt_xyxy[:, 1]))).unsqueeze(0)
        pred_c_in_gt_circles = torch.stack([b_l, b_r, b_t, b_b], dim=-1).all(dim=-1)  # (b * num_boxes, total_truth_num)

        loose_circle_mask = torch.cat([i.any(dim=-1) 
                                       for i in self._recover_cost_matrix_for_each_batch(pred_c_in_gt_circles, 
                                                                                         gt_split_size=gt_split_size)],
                                      dim=0)  # (b * num_boxes, )

        return loose_circle_mask, pred_c_in_gt_boxes

    def dynamic_k_matching(self, cost, iou_matrix):
        num_gt = cost.shape[1]
        assert cost.shape[1] == iou_matrix.shape[1]
        matching_matrix = torch.zeros_like(cost, dtype=torch.bool)  # (num_boxes, num_gt)
        n_candidate_k = self.ota_k

        topk_ious, _ = torch.topk(iou_matrix, n_candidate_k, dim=0)  # (k, num_gt)
        dynamic_ks = torch.clamp(topk_ious.sum(dim=0).int(), min=1)  # (num_gt, )

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = True

        pred_match_gt_num = matching_matrix.sum(dim=1)  # (num_boxes, )
        if (pred_match_gt_num > 1).any():
            tmp_cost = torch.where(matching_matrix, cost, torch.inf)[pred_match_gt_num > 1]
            cost_argmin = torch.argmin(tmp_cost, dim=1)  # (num_boxes_with_two_or_more_matches, )
            matching_matrix[pred_match_gt_num > 1] &= False
            matching_matrix[pred_match_gt_num > 1, cost_argmin,] = True

        iter_times = 0
        while (matching_matrix.sum(dim=0) == 0).any():
            matched_query_id = matching_matrix.sum(dim=1) > 0  # (num_boxes, )
            cost[matched_query_id] += torch.inf
            unmatch_id = torch.nonzero(matching_matrix.sum(dim=0) == 0, 
                                       as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:, gt_idx])
                matching_matrix[:, gt_idx][pos_idx] = True
            
            pred_match_gt_num = matching_matrix.sum(dim=1)  # (num_boxes, )
            if (pred_match_gt_num > 1).any():  # if a query matches more than one gt
                tmp_cost = torch.where(matching_matrix, cost, torch.inf)[pred_match_gt_num > 1]
                cost_argmin = torch.argmin(tmp_cost, dim=1)  # (num_boxes_with_two_or_more_matches, )
                matching_matrix[pred_match_gt_num > 1] &= False
                matching_matrix[pred_match_gt_num > 1, cost_argmin,] = True

            assert iter_times < cost.shape[0]
            iter_times += 1

        assert (matching_matrix.sum(dim=1) < 2).all()
        selected_query = matching_matrix.sum(dim=1) > 0  # (num_boxes, )
        selected_query_indices = selected_query.nonzero().squeeze(-1)  # (num_matched_boxes, )
        gt_indices = matching_matrix[selected_query].int().argmax(dim=1)  # (num_matched_boxes, )
        assert selected_query_indices.shape == gt_indices.shape
        assert len(gt_indices.unique(sorted=False)[0]) == num_gt

        return {
            "matched_box_mask": selected_query, 
            "matched_box_indices": selected_query_indices, 
            "matched_gt_indices": gt_indices
        }
