# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch

import math
from random import random
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F

from bliss.catalog import TileCatalog
from case_studies.dc2_diffusion.utils.catalog_parser import CatalogParser


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

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
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
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

class DiffusionModel(nn.Module):
    def __init__(
        self,
        *,
        model: nn.Module,
        target_size: list,
        catalog_parser: CatalogParser,
        objective,
        self_condition,
        beta_schedule,
        ddim_steps,
        schedule_fn_kwargs = dict(),
        timesteps = 1000,
        ddim_sampling_eta = 0.0,
    ):
        super().__init__()

        self.model = model
        self.target_size = target_size  # (k, h, w)
        self.dummy_param = nn.Parameter(torch.zeros(0))
        self.catalog_parser = catalog_parser
        self.self_condition = self_condition
        self.objective = objective
        assert objective in {"pred_noise", "pred_x0"}, \
            "objective must be either pred_noise (predict noise) or " \
            "pred_x0 (predict image start)"
        
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

        # sampling related parameters
        self.sampling_timesteps = ddim_steps
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

    @property
    def device(self):
        return self.dummy_param.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def _clip_func(self, x):
        raise NotImplementedError()

    def model_predictions(self, x, t, input_image,
                          x_self_cond=None, 
                          clip_x_start=False, 
                          rederive_pred_noise=False):
        model_output = self.model(x, t, input_image, x_self_cond)
        maybe_clip = self._clip_func if clip_x_start else identity

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start, model_output

    @torch.inference_mode()
    def ddim_sample(self, input_image, return_inter_output):
        times = torch.linspace(-1, self.num_timesteps - 1, 
                               steps=self.sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        assert time_pairs[-1][1] == -1

        xt = torch.randn([input_image.shape[0], *self.target_size], 
                         device=self.device)
        x_start = None
        model_output = None
        inter_output = []
        for time, time_next in time_pairs:
            time_cond = torch.full((xt.shape[0],), time, 
                                   device=self.device, 
                                   dtype=torch.long)
            self_cond = model_output if self.self_condition else None
            pred_noise, x_start, model_output = self.model_predictions(xt, time_cond, input_image, self_cond, 
                                                                        clip_x_start=True, rederive_pred_noise=True)

            if time_next < 0:
                assert time_pairs[-1][1] == time_next
                if return_inter_output:
                    inter_output.append({
                        "pred_noise": pred_noise.cpu(),
                        "x_start": x_start.cpu(),
                        "model_output": model_output.cpu(),
                        "alpha": None,
                        "alpha_next": None,
                        "sigma": None,
                        "c": None,
                        "xt": x_start.cpu(),
                    })
                break

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(xt)
            xt = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            if return_inter_output:
                inter_output.append({
                    "pred_noise": pred_noise.cpu(),
                    "x_start": x_start.cpu(),
                    "model_output": model_output.cpu(),
                    "alpha": alpha,
                    "alpha_next": alpha_next,
                    "sigma": sigma,
                    "c": c,
                    "xt": xt.cpu(),
                })
        return {
            "inter_x_start": x_start,
            "inter_output": inter_output,
            "final_pred": None,
        }

    @torch.inference_mode()
    def sample(self, input_image, return_inter_output):
        return self.ddim_sample(input_image, return_inter_output)

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, target, input_image):
        b = target.shape[0]
        assert target.shape[1:] == self.target_size
        t = torch.randint(0, self.num_timesteps, (b,), device=target.device).long()
        
        noise = torch.randn_like(target)

        # noise sample
        noised_target = self.q_sample(x_start=target, t=t, noise=noise)

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                _, _, x_self_cond = self.model_predictions(noised_target, t, input_image)
                x_self_cond.detach_()

        # predict
        pred_noise, x_start, model_output = self.model_predictions(
            noised_target, t, input_image, x_self_cond,
            clip_x_start=True, rederive_pred_noise=True)
        
        if self.objective == "pred_noise":
            cur_target = noise
        elif self.objective == "pred_x0":
            cur_target = target
        else:
            raise ValueError(f"unknown objective {self.objective}")

        assert model_output.shape == cur_target.shape
        loss = (model_output - cur_target) ** 2
        return {
            "inter_loss": loss.permute([0, 2, 3, 1]),
            "inter_target": target,
            "inter_objective": cur_target,
            "inter_pred_noise": pred_noise,
            "inter_x_start": x_start,
            "inter_model_output": model_output,
            "final_pred": None,
            "final_target": None,
            "final_pred_loss": None,
        }
    
class UpsampleDiffusionModel(DiffusionModel):
    def __init__(self,
                 postprocess_net: nn.Module,
                 **kwargs):
        super().__init__(**kwargs)

        self.postprocess_net = postprocess_net

    def _clip_func(self, x):
        return self.catalog_parser.clip_tensor(x.permute([0, 2, 3, 1])).permute([0, 3, 1, 2])
    
    def forward(self, target, input_image):
        rearranged_target = target.permute([0, 3, 1, 2])  # (b, k, h, w)
        upsampled_target = F.interpolate(rearranged_target, 
                                         scale_factor=4, 
                                         mode="bilinear")  # (b, k, H, W)
        pred_dict = super().forward(upsampled_target, input_image)
        final_pred = self.postprocess_net(pred_dict["inter_x_start"].detach())
        assert final_pred.shape == rearranged_target.shape
        final_pred_loss = (final_pred - rearranged_target) ** 2
        pred_dict["final_pred"] = final_pred
        pred_dict["final_target"] = rearranged_target
        pred_dict["final_pred_loss"] = final_pred_loss.permute([0, 2, 3, 1])
        return pred_dict

    def sample(self, input_image, return_inter_output):
        sample_dict = super().sample(input_image, return_inter_output)
        final_pred = self.postprocess_net(sample_dict["inter_x_start"])
        sample_dict["final_pred"] = self.catalog_parser.clip_tensor(final_pred.permute([0, 2, 3, 1]))
        return sample_dict
    

class LatentDiffusionModel(DiffusionModel):
    def __init__(self, 
                 encoder: nn.Module, 
                 decoder: nn.Module, 
                 encoder_output_min: float,
                 encoder_output_max: float,
                 scale: float,
                 **kwargs):
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder

        self.encoder_output_min = encoder_output_min
        self.encoder_output_max = encoder_output_max

        self.scale = scale

        self.encoder.eval()
        self.decoder.eval()

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def _clip_func(self, x):
        return x.clamp(min=-self.scale, max=self.scale)
    
    @torch.inference_mode()
    def _encode_target(self, rearranged_target):
        new_target = self.encoder(rearranged_target)
        assert new_target.max() <= self.encoder_output_max
        assert new_target.min() >= self.encoder_output_min
        new_target_minus_1_to_1 = (new_target - self.encoder_output_min) / (self.encoder_output_max - self.encoder_output_min) * 2 - 1
        return new_target_minus_1_to_1 * self.scale
    
    @torch.inference_mode()
    def _decode_target(self, sample_result):
        sample_result = sample_result / self.scale
        sample_result = (sample_result + 1) / 2 * (self.encoder_output_max - self.encoder_output_min) + self.encoder_output_min
        decode_result = self.decoder(sample_result)
        return self.catalog_parser.clip_tensor(decode_result.permute([0, 2, 3, 1]))

    def forward(self, target, input_image):
        if self.encoder.training or self.decoder.training:
            self.encoder.eval()
            self.decoder.eval()
        rearranged_target = target.permute([0, 3, 1, 2])  # (b, k, h, w)
        new_target = self._encode_target(rearranged_target)
        pred_dict = super().forward(new_target, input_image)
        final_pred = self._decode_target(pred_dict["inter_x_start"])
        assert final_pred.shape == target.shape
        final_pred_loss = (final_pred - target) ** 2
        pred_dict["final_pred"] = final_pred
        pred_dict["final_target"] = target
        pred_dict["final_pred_loss"] = final_pred_loss
        return pred_dict

    def sample(self, input_image, return_inter_output):
        if self.encoder.training or self.decoder.training:
            self.encoder.eval()
            self.decoder.eval()
        sample_dict = super().sample(input_image, return_inter_output)
        sample_dict["final_pred"] = self._decode_target(sample_dict["inter_x_start"])
        return sample_dict
    

class NoLatentDiffusionModel(DiffusionModel):
    def _clip_func(self, x):
        return self.catalog_parser.clip_tensor(x.permute([0, 2, 3, 1])).permute([0, 3, 1, 2])
    
    def forward(self, target, input_image):
        rearranged_target = target.permute([0, 3, 1, 2])  # (b, k, h, w)
        pred_dict = super().forward(rearranged_target, input_image)
        x_start = pred_dict["inter_x_start"]
        assert x_start.shape == rearranged_target.shape
        final_pred_loss = (x_start - rearranged_target) ** 2
        pred_dict["final_pred"] = x_start
        pred_dict["final_target"] = rearranged_target
        pred_dict["final_pred_loss"] = final_pred_loss.permute([0, 2, 3, 1])
        return pred_dict

    def sample(self, input_image, return_inter_output):
        sample_dict = super().sample(input_image, return_inter_output)
        sample_dict["final_pred"] = self.catalog_parser.clip_tensor(sample_dict["inter_x_start"].permute([0, 2, 3, 1]))
        return sample_dict

class YNetDiffusionModel(NoLatentDiffusionModel):
    pass

class SimpleNetDiffusionModel(NoLatentDiffusionModel):
    pass


class DoubleDetectDiffusionModel(DiffusionModel):
    def forward(self, target1, target2, input_image):
        b = target1.shape[0]
        assert target1.shape[1:] == self.target_size
        t = torch.randint(0, self.num_timesteps, (b,), device=target1.device).long()
        
        noise = torch.randn_like(target1)

        # noise sample
        noised_target1 = self.q_sample(x_start=target1, t=t, noise=noise)

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                _, x_self_cond = self.model_predictions(noised_target1, t, input_image)
                x_self_cond.detach_()

        # predict
        pred_noise, x_start, model_output = self.model_predictions(
            noised_target1, t, input_image, x_self_cond,
            clip_x_start=True, rederive_pred_noise=True)
        
        if self.objective == "pred_x0":
            cur_target = target2
        else:
            raise ValueError(f"unknown objective {self.objective}")

        assert model_output.shape == cur_target.shape
        loss = (model_output - cur_target) ** 2
        return {
            "inter_loss": loss.permute([0, 2, 3, 1]),
            "inter_target": target2,
            "inter_objective": cur_target,
            "inter_pred_noise": pred_noise,
            "inter_x_start": x_start,
            "inter_model_output": model_output,
            "final_pred": None,
            "final_target": None,
            "final_pred_loss": None,
        }


class YNetDoubleDetectDiffusionModel(DoubleDetectDiffusionModel):
    def _clip_func(self, x):
        return self.catalog_parser.clip_tensor(x.permute([0, 2, 3, 1])).permute([0, 3, 1, 2])
    
    def forward(self, target1, target2, input_image):
        rearranged_target1 = target1.permute([0, 3, 1, 2])  # (b, k, h, w)
        rearranged_target2 = target2.permute([0, 3, 1, 2])  # (b, k, h, w)
        pred_dict = super().forward(rearranged_target1, rearranged_target2, input_image)
        x_start = pred_dict["inter_x_start"]
        assert x_start.shape == rearranged_target2.shape
        final_pred_loss = (x_start - rearranged_target2) ** 2
        pred_dict["final_pred"] = x_start
        pred_dict["final_target"] = rearranged_target2
        pred_dict["final_pred_loss"] = final_pred_loss.permute([0, 2, 3, 1])
        return pred_dict

    def _merge_tile_cat(self, 
                        tile_cat1: TileCatalog, 
                        tile_cat2: TileCatalog, 
                        locs_slack: float):
        assert tile_cat1.max_sources == 1
        assert tile_cat2.max_sources == 1

        locs1 = tile_cat1["locs"].squeeze(-2)  # (b, h, w, 2)
        locs2 = tile_cat2["locs"].squeeze(-2)

        locs_dist = ((locs1 - locs2) ** 2).sum(dim=-1).sqrt()  # (b, h, w)
        locs_dist_mask = locs_dist > locs_slack
        two_sources_mask = tile_cat1["n_sources"].bool() & locs_dist_mask  # (b, h, w)
        two_sources_mask11 = rearrange(two_sources_mask, "b h w -> b h w 1 1")

        d = {}
        for k, v in tile_cat1.items():
            if k == "n_sources":
                d[k] = v + two_sources_mask.to(dtype=v.dtype)
            else:
                d1 = torch.cat((v, torch.zeros_like(v)), dim=-2)
                d2 = torch.cat((v, tile_cat2[k]), dim=-2)
                d[k] = torch.where(two_sources_mask11, d2, d1)
        return TileCatalog(d)

    def sample(self, input_image, return_inter_output, locs_slack, init_time):
        sample_dict = super().sample(input_image, return_inter_output)
        sample_dict["final_pred"] = sample_dict["inter_x_start"].permute([0, 2, 3, 1])
        sample1 = self.catalog_parser.decode(sample_dict["final_pred"])
        sample1_tensor = self.catalog_parser.encode(sample1).permute([0, 3, 1, 2])  # (b, k, h, w)
        time_cond = torch.ones((sample1_tensor.shape[0],), 
                                device=self.device, 
                                dtype=torch.long) * init_time
        _, sample2_tensor, _ = self.model_predictions(sample1_tensor, time_cond, input_image,
                                                     x_self_cond=None, 
                                                     clip_x_start=True, 
                                                     rederive_pred_noise=True)
        sample2 = self.catalog_parser.decode(sample2_tensor.permute([0, 2, 3, 1]))
        sample_dict["double_detect"] = self._merge_tile_cat(sample1, sample2,
                                                            locs_slack=locs_slack)
        return sample_dict
