# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch

import math
from random import random

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from scipy.optimize import linear_sum_assignment

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
        model,
        target_size,
        *,
        catalog_parser: CatalogParser,
        objective,
        self_condition,
        beta_schedule,
        ddim_steps,
        correct_bits,
        empty_tile_random_noise,
        add_fake_tiles,
        schedule_fn_kwargs = dict(),
        timesteps = 1000,
        ddim_sampling_eta = 0.0,
    ):
        super().__init__()
        assert not hasattr(model, "random_or_learned_sinusoidal_cond") or not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.dummy_param = nn.Parameter(torch.zeros(0))
        self.inter_target_size = (target_size[2], target_size[0], target_size[1])  # (k, h, w)
        self.catalog_parser = catalog_parser
        self.self_condition = self_condition
        self.objective = objective
        assert objective in {"pred_noise", "pred_x0", "pred_v"}, \
            "objective must be either pred_noise (predict noise) or " \
            "pred_x0 (predict image start) or "\
            "pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

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
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.0)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters
        self.sampling_timesteps = ddim_steps
        assert self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        # ensure that the bits in predicted x_start only have value -bit_value/bit_value
        self.correct_bits = correct_bits

        self.empty_tile_random_noise = empty_tile_random_noise
        self.add_fake_tiles = add_fake_tiles
        assert not (self.empty_tile_random_noise and self.add_fake_tiles)
        if self.empty_tile_random_noise:
            assert self.objective == "pred_x0"
        if self.add_fake_tiles:
            assert self.objective == "pred_x0"
            assert not self.self_condition

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

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def model_predictions(self, x, t, extracted_feats,
                          x_self_cond=None, 
                          clip_x_start=False, 
                          rederive_pred_noise=False,
                          pre_fake_on_mask=None,
                          pre_fake_data=None):
        model_output = self.model(x, t, extracted_feats, x_self_cond)
        clip_func = lambda x: self.catalog_parser.clip_tensor(x.permute([0, 2, 3, 1])).permute([0, 3, 1, 2])
        maybe_clip = clip_func if clip_x_start else identity

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            if self.correct_bits:
                x_start_tilecat = self.catalog_parser.decode(x_start.permute([0, 2, 3, 1]))
                x_start = self.catalog_parser.encode(x_start_tilecat).permute([0, 3, 1, 2])
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            if self.correct_bits:
                x_start_tilecat = self.catalog_parser.decode(x_start.permute([0, 2, 3, 1]))
                x_start = self.catalog_parser.encode(x_start_tilecat).permute([0, 3, 1, 2])
            if not self.add_fake_tiles or pre_fake_on_mask is None:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
            else:
                pred_noise = self.predict_noise_from_start(x, t,
                                                           self._join_x_and_fake_data(
                                                               x_start, pre_fake_data, pre_fake_on_mask,
                                                               ensure_no_overlap=False
                                                           ))
        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            if self.correct_bits:
                x_start_tilecat = self.catalog_parser.decode(x_start.permute([0, 2, 3, 1]))
                x_start = self.catalog_parser.encode(x_start_tilecat).permute([0, 3, 1, 2])
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    @torch.inference_mode()
    def ddim_sample(self, extracted_feats):
        times = torch.linspace(-1, self.num_timesteps - 1, 
                               steps=self.sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        assert time_pairs[-1][1] == -1

        xt = torch.randn([extracted_feats.shape[0], *self.inter_target_size], 
                         device=self.device)
        x_start = None
        pre_fake_on_mask = None
        pre_fake_data = None
        for time, time_next in time_pairs:
            time_cond = torch.full((xt.shape[0],), time, 
                                   device=self.device, 
                                   dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start = self.model_predictions(xt, time_cond, extracted_feats, self_cond, 
                                                         clip_x_start=True, 
                                                         rederive_pred_noise=True,
                                                         pre_fake_on_mask=pre_fake_on_mask,
                                                         pre_fake_data=pre_fake_data)

            if time_next < 0:
                xt = x_start
                assert time_pairs[-1][1] == time_next
                break

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(xt)
            if self.add_fake_tiles:
                fake_on_mask, fake_data = self.catalog_parser.craft_fake_data(
                    x_start.permute([0, 2, 3, 1])
                )
                fake_on_mask = fake_on_mask.permute([0, 3, 1, 2])  # (b, 1, k, h)
                fake_data = fake_data.permute([0, 3, 1, 2])  # (b, k, h, w)
                assert fake_data.shape == x_start.shape
                assert fake_on_mask.shape[1] == 1
                assert fake_data.shape[0] == fake_on_mask.shape[0]
                assert fake_data.shape[2:] == fake_on_mask.shape[2:]
                x_start = self._join_x_and_fake_data(x_start, 
                                                     fake_data, 
                                                     fake_on_mask,
                                                     ensure_no_overlap=True)
                pre_fake_on_mask = fake_on_mask
                pre_fake_data = fake_data
            xt = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            if self.empty_tile_random_noise:
                # we assume the first channel corresponds to n_sources
                # this is ensured by the assertion in CatalogParser's init func
                empty_tile_mask = x_start[:, 0:1, :, :] < 0.0
                xt = torch.where(empty_tile_mask, 
                                 torch.randn_like(xt),
                                 xt)
        return xt

    @torch.inference_mode()
    def sample(self, extracted_feats):
        return self.ddim_sample(extracted_feats).permute([0, 2, 3, 1])

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, target, extracted_feats):
        rearranged_target = target.permute([0, 3, 1, 2])
        assert rearranged_target.shape[1:] == self.inter_target_size
        assert rearranged_target.shape[2:] == extracted_feats.shape[2:]
        assert rearranged_target.shape[0] == extracted_feats.shape[0]
        b = rearranged_target.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=extracted_feats.device).long()
        
        empty_tile_mask = None
        padded_target = None
        if self.empty_tile_random_noise:
            # we assume the first channel corresponds to n_sources
            # this is ensured by the assertion in CatalogParser's init func
            empty_tile_mask = rearranged_target[:, 0:1, :, :] < 0.0
            padded_target = torch.where(empty_tile_mask, 
                                        torch.randn_like(rearranged_target),
                                        rearranged_target)
        elif self.add_fake_tiles:
            fake_on_mask, fake_data = self.catalog_parser.craft_fake_data(target)
            fake_on_mask = fake_on_mask.permute([0, 3, 1, 2])
            fake_data = fake_data.permute([0, 3, 1, 2])
            padded_target = self._join_x_and_fake_data(rearranged_target, 
                                                       fake_data,
                                                       fake_on_mask,
                                                       ensure_no_overlap=True)
        else:
            padded_target = rearranged_target
        
        noise = torch.randn_like(padded_target)

        # noise sample
        noised_target = self.q_sample(x_start=padded_target, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                _, x_self_cond = self.model_predictions(noised_target, t, extracted_feats)
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(noised_target, t, extracted_feats, x_self_cond)
        if self.objective == "pred_noise":
            cur_target = noise
        elif self.objective == "pred_x0":
            cur_target = rearranged_target
        elif self.objective == "pred_v":
            cur_target = self.predict_v(rearranged_target, t, noise)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = (model_out - cur_target) ** 2
        return loss.permute([0, 2, 3, 1])

    def _join_x_and_fake_data(self,
                              x: torch.Tensor, 
                              fake_data: torch.Tensor, 
                              fake_on_mask: torch.Tensor,
                              ensure_no_overlap: bool):
        assert fake_data.shape == x.shape  # (b, k, h, w)
        assert fake_on_mask.shape[1] == 1
        if ensure_no_overlap:
            assert not ((x[:, 0:1, :, :] > 0.0) & fake_on_mask).any()
        return torch.where(fake_on_mask, fake_data, x)