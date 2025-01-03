# adapted from https://github.com/lucidrains/denoising-diffusion-pytorch

import math

import torch
from torch import nn
import torch.nn.functional as F

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
        beta_schedule,
        ddim_steps,
        schedule_fn_kwargs = dict(),
        timesteps = 1000,
        ddim_sampling_eta = 0.0,
    ):
        super().__init__()

        self.model = model
        self.dummy_param = nn.Parameter(torch.zeros(0))
        self.inter_target_size = (target_size[2], target_size[0], target_size[1])  # (k, h, w)
        self.catalog_parser = catalog_parser
        assert catalog_parser.factors[0].fake_data_on_prop == 1.0
        self.objective = objective
        assert objective == "pred_x0"

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

    @property
    def device(self):
        return self.dummy_param.device

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, t, extracted_feats,
                          pre_fake_on_mask=None,
                          pre_fake_data=None):
        model_output = self.model(x, t, extracted_feats, None)
        clip_func = lambda x: self.catalog_parser.clip_tensor(x.permute([0, 2, 3, 1])).permute([0, 3, 1, 2])

        x_start = model_output
        x_start = clip_func(x_start)
        x_start_tilecat = self.catalog_parser.decode(x_start.permute([0, 2, 3, 1]))
        x_start = self.catalog_parser.encode(x_start_tilecat).permute([0, 3, 1, 2])
        if pre_fake_on_mask is None:
            pred_noise = self.predict_noise_from_start(x, t, x_start[:, 1:, :, :])
        else:
            pred_noise = self.predict_noise_from_start(x, t,
                                                        self._join_x_and_fake_data(
                                                            x_start[:, 1:, :, :], 
                                                            pre_fake_data, pre_fake_on_mask,
                                                            ensure_no_overlap=False
                                                        ))

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
        
        use_consistent_x_start = False
        ref_x_start = None
        ref_pred_noise = None
        for i, (time, time_next) in enumerate(time_pairs):
            time_cond = torch.full((xt.shape[0],), time, 
                                   device=self.device, 
                                   dtype=torch.long)
            pred_noise, x_start = self.model_predictions(xt, time_cond, extracted_feats)            
            n_sources_mask = x_start[:, 0:1, :, :] > 0.0  # (b, 1, h, w)
            pred_noise = pred_noise * n_sources_mask
            x_start[:, 1:, :, :] = x_start[:, 1:, :, :] * n_sources_mask
            if use_consistent_x_start:
                if i == 0:
                    ref_x_start = x_start
                    ref_pred_noise = pred_noise
                else:
                    ref_x_start, ref_pred_noise = self._update_ref(
                        ref_x_start=ref_x_start,
                        cur_x_start=x_start,
                        ref_pred_noise=ref_pred_noise,
                        cur_pred_noise=pred_noise,
                    )
                    x_start = ref_x_start
                    pred_noise = ref_pred_noise
                    n_sources_mask = x_start[:, 0:1, :, :] > 0.0

            if time_next < 0:
                assert time_pairs[-1][1] == time_next
                break

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
            pred_noise = torch.where(n_sources_mask, pred_noise, torch.randn_like(pred_noise))

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(xt)

            xt = x_start[:, 1:, :, :] * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
        return x_start

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
        assert rearranged_target.shape[1] - 1 == self.inter_target_size[0] 
        assert rearranged_target.shape[2:] == extracted_feats.shape[2:]
        assert rearranged_target.shape[0] == extracted_feats.shape[0]
        b = rearranged_target.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=extracted_feats.device).long()
        
        fake_on_mask, fake_data = self.catalog_parser.craft_fake_data(target)
        fake_on_mask = fake_on_mask.permute([0, 3, 1, 2])
        fake_data = fake_data.permute([0, 3, 1, 2])
        padded_target = self._join_x_and_fake_data(rearranged_target, 
                                                    fake_data,
                                                    fake_on_mask,
                                                    ensure_no_overlap=True)
        padded_target = padded_target[:, 1:, :, :]  # remove the n_sources because they have same values
        
        noise = torch.randn_like(padded_target)

        # noise sample
        noised_target = self.q_sample(x_start=padded_target, t=t, noise=noise)

        # predict and take gradient step
        model_out = self.model(noised_target, t, extracted_feats, None)

        loss = (model_out - rearranged_target) ** 2
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
    
    def _update_ref(self, 
                    ref_x_start: torch.Tensor,
                    cur_x_start: torch.Tensor,
                    ref_pred_noise: torch.Tensor, 
                    cur_pred_noise: torch.Tensor):
        assert ref_x_start.shape == cur_x_start.shape
        ref_n_sources_mask = ref_x_start[:, 0:1, :, :] > 0.0
        cur_n_sources_mask = cur_x_start[:, 0:1, :, :] > 0.0
        update_mask = ref_n_sources_mask & cur_n_sources_mask
        return (torch.where(update_mask, cur_x_start, ref_x_start),
                torch.where(update_mask, cur_pred_noise, ref_pred_noise))
