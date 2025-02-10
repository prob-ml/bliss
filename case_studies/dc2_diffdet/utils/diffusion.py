# adapted from https://github.com/ShoufaChen/DiffusionDet
import math
from typing import List

import torch
from torch import nn
from torchvision.ops import batched_nms
from torch.nn.utils.rnn import pad_sequence
from einops import repeat

from case_studies.dc2_diffdet.utils.loss import SetCriterion, DynamicKMatcher
from case_studies.dc2_diffdet.utils.box_func import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from case_studies.dc2_diffdet.utils.backbone import FPN
from case_studies.dc2_diffdet.utils.head import DynamicHead


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class SparseRCNNDiffusionModel(nn.Module):
    def __init__(self, 
                 *,
                 image_size: List[int],
                 backbone_model: FPN,
                 output_head_model: DynamicHead,
                 num_box_proposals: int,
                 beta_schedule: str,
                 ddim_steps: int,
                 schedule_fn_kwargs = dict(),
                 timesteps = 1000,
                 ddim_sampling_eta = 0.0,
    ):
        super().__init__()

        self.image_size = image_size  # [height, width]
        self.backbone = backbone_model
        self.output_head_model = output_head_model
        self.num_box_proposals = num_box_proposals
        self.dummy_param = nn.Parameter(torch.zero(0))

        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = ddim_steps
        assert self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        self.scale = 2.0

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Loss parameters:
        class_weight = 2.0
        giou_weight = 2.0
        xyxy_l1_weight = 5.0

        # Build Criterion.
        weight_dict = {
            "loss_ce": class_weight, 
            "loss_xyxy_l1": xyxy_l1_weight, 
            "loss_giou": giou_weight
        }
        # for deep_supervision
        aux_weight_dict = {}
        for i in range(self.output_head_model.num_rcnn_heads - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        
        self.dynamick_matcher = DynamicKMatcher(
            weight_class_cost=class_weight, 
            weight_xyxy_l1_cost=xyxy_l1_weight,
            weight_giou_cost=giou_weight,
            image_whwh=self.image_whwh,
        )
        self.criterion = SetCriterion(
            matcher=self.dynamick_matcher, 
            weight_dict=weight_dict, 
            image_whwh=self.image_whwh,
        )

    @property
    def image_whwh(self):
        return torch.tensor(self.image_size[::-1], device=self.device).view(1, -1).repeat(1, 2)  # (1, 4)
    
    @property
    def device(self):
        return self.dummy_param.device

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats, xt_norm_cxcywh, t):
        xt_norm_cxcywh = torch.clamp(xt_norm_cxcywh, min=-self.scale, max=self.scale)
        xt_norm_cxcywh = ((xt_norm_cxcywh / self.scale) + 1) / 2
        xt_norm_xyxy = box_cxcywh_to_xyxy(xt_norm_cxcywh)
        xt_xyxy = xt_norm_xyxy * self.image_whwh[:, None, :]

        output_logits, output_xyxy = self.output_head_model(backbone_feats, xt_xyxy, t)
        x0_xyxy = output_xyxy[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)

        x0_norm_xyxy = x0_xyxy / self.image_whwh[:, None, :]
        x0_norm_cxcywh = box_xyxy_to_cxcywh(x0_norm_xyxy)
        x0_norm_cxcywh = (x0_norm_cxcywh * 2 - 1) * self.scale
        x0_norm_cxcywh = torch.clamp(x0_norm_cxcywh, min=-self.scale, max=self.scale)

        pred_noise = self.predict_noise_from_start(xt_norm_cxcywh, t, x0_norm_cxcywh)

        return pred_noise, x0_norm_cxcywh, output_logits, output_xyxy

    @torch.inference_mode()
    def ddim_sample(self, backbone_feats, batch_size):
        times = torch.linspace(-1, self.num_timesteps - 1, 
                               steps=self.sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        assert time_pairs[-1][1] == -1

        k = 4
        xt_norm_cxcywh = torch.randn([batch_size, self.num_box_proposals, k], 
                                    device=self.device)
        x0_norm_cxcywh = None

        ensemble_scores, ensemble_xyxy = [], []
        for time, time_next in time_pairs:
            time_cond = torch.full((xt_norm_cxcywh.shape[0],), time, device=self.device, dtype=torch.long)

            pred_noise, x0_norm_cxcywh, output_logits, output_xyxy = self.model_predictions(backbone_feats, 
                                                                                            xt_norm_cxcywh, 
                                                                                            time_cond)

            # for box renewal
            output_scores = torch.sigmoid(output_logits[-1]).squeeze(-1)  # (b, num_boxes)
            # for ensemble
            ensemble_scores.append(output_scores)
            ensemble_xyxy.append(output_xyxy[-1])
            if time_next < 0:
                assert time_pairs[-1][1] == time_next
                break

            keep_mask = output_scores > 0.5  # (b, num_boxes)
            compact_indices = keep_mask.argsort(dim=-1, descending=True, stable=False)  # (b, num_boxes)
            compact_indices = repeat(compact_indices, "b num_boxes -> b num_boxes k", k=k)  # (b, num_boxes, 4)
            pred_noise = torch.gather(pred_noise, dim=1, index=compact_indices)
            x0_norm_cxcywh = torch.gather(x0_norm_cxcywh, dim=1, index=compact_indices)
            compact_mask = torch.gather(keep_mask, dim=1, index=compact_indices)  # (b, num_boxes)
            pred_noise *= compact_mask.unsqueeze(-1)
            x0_norm_cxcywh *= compact_mask.unsqueeze(-1)

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(xt_norm_cxcywh)
            xt_norm_cxcywh = x0_norm_cxcywh * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            # for box renewal: replenish with randn boxes
            xt_norm_cxcywh = torch.where(compact_mask, xt_norm_cxcywh, torch.randn_like(xt_norm_cxcywh))
                

        ensemble_xyxy = torch.cat(ensemble_xyxy, dim=1)  # (b, num_boxes * sample_steps, 4)
        ensemble_scores = torch.cat(ensemble_scores, dim=1)  # (b, num_boxes * sample_steps, 1)
        
        # for nms
        results = []
        for output_scores, xyxy in zip(ensemble_scores, ensemble_xyxy):
            keep_indices = batched_nms(xyxy, output_scores.squeeze(-1), 
                                       torch.zeros_like(output_scores.squeeze(-1), 
                                                        device=self.device,
                                                        dtype=torch.int32), 
                                       iou_threshold=0.5)
            results.append({
                "pred_xyxy": xyxy[keep_indices],
                "pred_scores": output_scores[keep_indices],
            })

        return results

    def q_sample(self, x_start, t, noise):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.inference_mode()
    def sample(self, images):
        return self.ddim_sample(self.backbone(images), batch_size=images.shape[0])

    def forward(self, images, gt_cxcywh):
        # gt_cxcywh is in (cx, cy, w, h) format and has the original image scale
        backbone_feats = self.backbone(images)
        noised_xyxy, _noises, t = self.add_noise_to_targets(gt_cxcywh)
        output_logits, output_xyxy = self.output_head_model(backbone_feats, noised_xyxy, t)
        output = {
            "pred_logits": output_logits[-1], 
            "pred_xyxy": output_xyxy[-1]
        }
        # for deep supervise
        output["aux_outputs"] = [{"pred_logits": a, 
                                  "pred_xyxy": b} 
                                  for a, b in zip(output_logits[:-1], output_xyxy[:-1])]

        return self.criterion(output, gt_cxcywh)

    def add_noise_to_targets(self, gt_cxcywh: List[torch.Tensor]):
        gt_norm_cxcywh = [box_xyxy_to_cxcywh(box_cxcywh_to_xyxy(g) / self.image_whwh) for g in gt_cxcywh]
        final_shape = (len(gt_norm_cxcywh), self.num_box_proposals, 4)
        padded_targets = pad_sequence(gt_norm_cxcywh + [torch.zeros(*final_shape[1:], device=self.device)], batch_first=True)[:-1]  # (b, num_boxes, 4)
        assert padded_targets.shape[1] == self.num_box_proposals
        t = torch.randint(0, self.num_timesteps, (final_shape[0],), device=self.device).long()
        noise = torch.randn(*final_shape, device=self.device)

        box_placeholder = torch.randn_like(padded_targets) / 6 + 0.5  # 3sigma = 1/2 --> sigma: 1/6
        box_placeholder[..., 2:] = torch.clip(box_placeholder[..., 2:], min=1e-4)
        targets_len = torch.tensor([target.shape[0] for target in gt_norm_cxcywh], device=self.device).unsqueeze(-1)  # (b, 1)
        targets_mask = torch.arange(self.num_box_proposals, device=self.device).expand(final_shape[0], -1) < targets_len  # (b, num_boxes)
        padded_targets = torch.where(targets_mask.unsqueeze(-1), padded_targets, box_placeholder)

        padded_targets = (padded_targets * 2 - 1) * self.scale
        # noise sample
        noised_norm_cxcywh = self.q_sample(x_start=padded_targets, t=t, noise=noise)
        noised_norm_cxcywh = torch.clamp(noised_norm_cxcywh, min=-self.scale, max=self.scale)
        noised_norm_cxcywh = ((noised_norm_cxcywh / self.scale) + 1) / 2
        noised_norm_xyxy = box_cxcywh_to_xyxy(noised_norm_cxcywh)
        noised_xyxy = noised_norm_xyxy * self.images_whwh[:, None, :]

        return noised_xyxy, noise, t
