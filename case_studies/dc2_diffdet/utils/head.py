import math
from typing import List
import torch
from torch import nn
from einops import rearrange
from torchvision.ops import MultiScaleRoIAlign


class MultiheadSelfAttentionModule(nn.Module):
    def __init__(self, input_ch, attn_head):
        super().__init__()
        self.input_ch = input_ch
        self.attn_head = attn_head

        self.attn = nn.MultiheadAttention(self.input_ch, self.attn_head, batch_first=True)
        self.norm = nn.LayerNorm(self.input_ch)

    def forward(self, x):
        x += self.attn(q=x, k=x, v=x, need_weights=False)[0]
        return self.norm(x)


class DenseLinearModule(nn.Module):
    def __init__(self, input_ch, hidden_ch):
        super().__init__()
        self.input_ch = input_ch
        self.hidden_ch = hidden_ch

        self.linear_layer = nn.Sequential(
            nn.Linear(self.input_ch, self.hidden_ch),
            nn.ReLU(),
            nn.Linear(self.hidden_ch, self.input_ch)
        )
        self.norm = nn.LayerNorm(self.input_ch)

    def forward(self, x):
        x += self.linear_layer(x)
        return self.norm(x)


class DynamicConv(nn.Module):
    def __init__(self, input_ch, dynamic_dim, dynamic_num, pooler_resolution):
        super().__init__()

        self.input_ch = input_ch
        self.dynamic_dim = dynamic_dim
        self.dynamic_num = dynamic_num
        assert self.dynamic_num == 2
        self.num_params = self.input_ch * self.dynamic_dim
        self.dynamic_layer = nn.Linear(self.input_ch, self.dynamic_num * self.num_params)

        self.norm1 = nn.LayerNorm(self.dynamic_dim)
        self.norm2 = nn.LayerNorm(self.input_ch)

        self.activation = nn.ReLU(inplace=True)

        num_output = self.input_ch * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.input_ch)
        self.norm3 = nn.LayerNorm(self.input_ch)
        self.norm4 = nn.LayerNorm(self.input_ch)

    def forward(self, 
                pro_features,  # (b, num_boxes, input_ch)
                roi_features,  # (b, num_boxes, input_ch, pr_a, pr_b)
                ):
        b, num_boxes = roi_features.shape[:2]
        pro_features = rearrange(pro_features, 
                                 "b num_boxes input_ch -> (b num_boxes) input_ch")
        roi_features = rearrange(roi_features, 
                                 "b num_boxes input_ch pr_a pr_b -> (b num_boxes) (pr_a pr_b) input_ch")

        parameters = self.dynamic_layer(pro_features)  # (b * num_boxes, num_params * 2)
        param1 = rearrange(parameters[:, :self.num_params],
                           "b_num_boxes (input_ch dynamic_dim) -> b_num_boxes input_ch dynamic_dim",
                           input_ch=self.input_ch, dynamic_dim=self.dynamic_dim)
        param2 = rearrange(parameters[:, self.num_params:],
                           "b_num_boxes (dynamic_dim input_ch) -> b_num_boxes dynamic_dim input_ch",
                           input_ch=self.input_ch, dynamic_dim=self.dynamic_dim)

        features = torch.bmm(roi_features, param1)  # (b * num_boxes, pr^2, dynamic_dim)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)  # (b * num_boxes, pr^2, input_ch)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)  # (b * num_boxes, pr^2 * input_ch)
        features = self.out_layer(features)  # (b * num_boxes, input_ch)
        features = self.norm3(features)
        features = self.activation(features)

        features += pro_features
        features = self.norm4(features)  # (b * num_boxes, input_ch)

        return rearrange(features, 
                         "(b num_boxes) input_ch -> b num_boxes input_ch", 
                         b=b, num_boxes=num_boxes)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    

class DynamicHead(nn.Module):
    def __init__(self, 
                 box_pooler: MultiScaleRoIAlign,
                 num_rcnn_heads: int,  # default: 6
                 rcnn_input_ch: int,  # default: 256
                 rcnn_hidden_ch: int,  # default: 2048
                 rcnn_attn_nhead: int,  # default: 8
                 image_size: List[int]
                 ):
        super().__init__()

        self.box_pooler = box_pooler
        assert self.box_pooler.output_size[0] == self.box_pooler.output_size[1]
        
        self.num_rcnn_heads = num_rcnn_heads
        self.head_series = nn.ModuleList([RCNNHead(input_ch=rcnn_input_ch, 
                                                   hidden_ch=rcnn_hidden_ch, 
                                                   attn_nhead=rcnn_attn_nhead, 
                                                   pooler_resolution=self.box_pooler.output_size[0]) 
                                          for _ in range(self.num_rcnn_heads)])

        self.rcnn_input_ch = rcnn_input_ch
        time_dim = self.rcnn_input_ch * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.rcnn_input_ch),
            nn.Linear(self.rcnn_input_ch, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.image_size = image_size  # [height, width]

    def forward(self, fpn_features, noised_xyxy, t):
        time = self.time_mlp(t)
        xyxy = noised_xyxy
        b, num_boxes = xyxy.shape[:2]

        roi_features = None  # (b, num_boxes, input_ch, pr, pr)
        pro_features = None  # (b, num_boxes, input_ch) or None
        inter_logits = []
        inter_pred_xyxy = []
        for rcnn_head in self.head_series:
            roi_features = self.box_pooler(fpn_features, 
                                           xyxy, 
                                           [tuple(self.image_size)] * t.shape[0])  # (b * num_boxes, input_ch, pr, pr)
            roi_features = rearrange(roi_features, 
                                     "(b num_boxes) input_ch pr_a pr_b -> b num_boxes input_ch pr_a pr_b",
                                     b=b, num_boxes=num_boxes)
            pred_logits, pred_xyxy, pro_features = rcnn_head(roi_features=roi_features,
                                                            pro_features=pro_features,
                                                            xyxy=xyxy,
                                                            time_emb=time)
            inter_logits.append(pred_logits)
            inter_pred_xyxy.append(pred_xyxy)
            xyxy = pred_xyxy.detach()

        return inter_logits, inter_pred_xyxy


class RCNNHead(nn.Module):
    def __init__(self, 
                 *,
                 input_ch, 
                 hidden_ch, 
                 attn_nhead,
                 pooler_resolution,
                 scale_clamp: float=math.log(1e5 / 16), 
                 bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.input_ch = input_ch
        self.hidden_ch = hidden_ch
        self.attn_nhead = attn_nhead

        self.self_attn = nn.MultiheadAttention(self.input_ch, 
                                               self.attn_nhead)
        self.dynamic_conv = DynamicConv(self.input_ch, 
                                         dynamic_dim=64, 
                                         dynamic_num=2, 
                                         pooler_resolution=pooler_resolution)
        self.dense_linear_module = DenseLinearModule(self.input_ch, self.hidden_ch)
        self.time_mlp = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(self.input_ch * 4, self.input_ch * 2)
        )

        num_cls_layers = 1
        self.cls_module = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(self.input_ch, self.input_ch, bias=False),
                nn.LayerNorm(self.input_ch),
                nn.ReLU()
            ) for _ in range(num_cls_layers)],
            nn.Linear(self.input_ch, 1)
        )

        num_reg_layers = 3
        self.reg_module = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(self.input_ch, self.input_ch, bias=False),
                nn.LayerNorm(self.input_ch),
                nn.ReLU()
            ) for _ in range(num_reg_layers)],
            nn.Linear(self.input_ch, 4)
        )
        
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self,
                *, 
                roi_features, # (b, num_boxes, input_ch, pr, pr)
                pro_features, # (b, num_boxes, input_ch) or None
                xyxy,
                time_emb,
                ):
        assert roi_features.shape[-1] == roi_features.shape[-2]
        if pro_features is None:
            pro_features = roi_features.mean(dim=(-2, -1))  # (b, num_boxes, input_ch)

        pro_features = self.self_attn(pro_features)  # (b, num_boxes, input_ch)
        pro_features = self.dynamic_conv(pro_features, roi_features) # (b, num_boxes, input_ch)
        obj_features = self.dense_linear_module(pro_features)  # (b, num_boxes, input_ch)

        scale_shift = self.time_mlp(time_emb)  # (b, input_ch * 2)
        scale_shift = rearrange(scale_shift, "b c -> b 1 c")  # (b, 1, input_ch * 2)
        scale, shift = scale_shift.chunk(2, dim=-1)  # (b, 1, input_ch)
        obj_features = obj_features * (scale + 1) + shift  # (b, num_boxes, input_ch)

        logits = self.cls_module(obj_features)  # (b, num_boxes, 1)
        bboxes_deltas = self.reg_module(obj_features)  # (b, num_boxes, 4)

        pred_xyxy = self.apply_deltas(bboxes_deltas, xyxy)
        
        return logits, pred_xyxy, pro_features

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (b, num_boxes, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (b, num_boxes, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x = boxes[..., 0] + 0.5 * widths
        ctr_y = boxes[..., 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[..., 0::4] / wx
        dy = deltas[..., 1::4] / wy
        dw = deltas[..., 2::4] / ww
        dh = deltas[..., 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths.unsqueeze(-1) + ctr_x.unsqueeze(-1)
        pred_ctr_y = dy * heights.unsqueeze(-1) + ctr_y.unsqueeze(-1)
        pred_w = torch.exp(dw) * widths.unsqueeze(-1)
        pred_h = torch.exp(dh) * heights.unsqueeze(-1)

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[..., 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[..., 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[..., 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[..., 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes
