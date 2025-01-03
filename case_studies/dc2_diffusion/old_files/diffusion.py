# # adapted from https://github.com/taehoon-yoon/Diffusion-Probabilistic-Models/tree/master

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class DiffusionModel(nn.Module):
#     def __init__(self, model, target_size, time_step=1000, loss_type="l2", ddim_steps=50, eta=0):
#         """Diffusion model. Can do DDPM or DDIM sample

#         Args:
#             model: model for de-noising network
#             target_size: target size (h, w, k)
#             time_step: Gaussian diffusion length T. In paper they used T=1000
#             loss_type: either l1, l2. Default, l2
#         """

#         super().__init__()
#         self.model = model
#         self.dummy_param = nn.Parameter(torch.zeros(0))
#         self.inter_target_size = (target_size[2], target_size[0], target_size[1])  # (k, h, w)
#         self.time_step = time_step
#         self.loss_type = loss_type

#         self.ddim_steps = ddim_steps
#         self.eta = eta

#         ddpm_steps = self.time_step
#         assert (
#             self.ddim_steps <= ddpm_steps
#         ), "DDIM sampling step must be smaller or equal to DDPM sampling step"

#         beta = self.linear_beta_schedule()  # (t, )  t=time_step, in DDPM paper t=1000
#         alpha = 1.0 - beta  # (a1, a2, a3, ... at)
#         alpha_bar = torch.cumprod(alpha, dim=0)  # (a1, a1*a2, a1*a2*a3, ..., a1*a2*~*at)
#         alpha_bar_prev = F.pad(
#             alpha_bar[:-1], pad=(1, 0), value=1.0
#         )  # (1, a1, a1*a2, ..., a1*a2*~*a(t-1))

#         self.register_buffer("beta", beta)
#         self.register_buffer("alpha", alpha)
#         self.register_buffer("alpha_bar", alpha_bar)
#         self.register_buffer("alpha_bar_prev", alpha_bar_prev)

#         # calculation for q(x_t | x_0) consult (4) in DDPM paper.
#         self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
#         self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1 - alpha_bar))

#         # calculation for q(x_{t-1} | x_t, x_0) consult (7) in DDPM paper.
#         self.register_buffer("beta_tilde", beta * ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar)))
#         self.register_buffer(
#             "mean_tilde_x0_coeff", beta * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
#         )
#         self.register_buffer(
#             "mean_tilde_xt_coeff", torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
#         )

#         # calculation for x0 consult (9) in DDPM paper.
#         self.register_buffer("sqrt_recip_alpha_bar", torch.sqrt(1.0 / alpha_bar))
#         self.register_buffer("sqrt_recip_alpha_bar_min_1", torch.sqrt(1.0 / alpha_bar - 1))

#         # calculation for (11) in DDPM paper.
#         self.register_buffer("sqrt_recip_alpha", torch.sqrt(1.0 / alpha))
#         self.register_buffer(
#             "beta_over_sqrt_one_minus_alpha_bar", beta / torch.sqrt(1.0 - alpha_bar)
#         )

#         alpha_bar = self.alpha_bar
#         # One thing you mush notice is that although sampling time is indexed as [1,...T] in paper,
#         # since in computer program we index from [0,...T-1] rather than [1,...T],
#         # value of tau ranges from [-1, ...T-1] where t=-1 indicate initial state (Data distribution)

#         # [tau_1, tau_2, ... tau_S] sec 4.2
#         self.register_buffer(
#             "tau",
#             torch.linspace(-1, ddpm_steps - 1, steps=self.ddim_steps + 1, dtype=torch.long)[1:],
#         )

#         alpha_tau_i = alpha_bar[self.tau]
#         alpha_tau_i_min_1 = F.pad(alpha_bar[self.tau[:-1]], pad=(1, 0), value=1.0)  # alpha_0 = 1

#         # (16) in DDIM
#         self.register_buffer(
#             "sigma",
#             eta
#             * (
#                 (
#                     (1 - alpha_tau_i_min_1)
#                     / (1 - alpha_tau_i)
#                     * (1 - alpha_tau_i / alpha_tau_i_min_1)
#                 ).sqrt()
#             ),
#         )
#         # (12) in DDIM
#         self.register_buffer("coeff", (1 - alpha_tau_i_min_1 - self.sigma**2).sqrt())
#         self.register_buffer("sqrt_alpha_i_min_1", alpha_tau_i_min_1.sqrt())

#         assert self.coeff[0] == 0.0 and self.sqrt_alpha_i_min_1[0] == 1.0, "DDIM parameter error"

#     @property
#     def device(self):
#         return self.dummy_param.device

#     def linear_beta_schedule(self):
#         """linear schedule, proposed in original ddpm paper"""

#         scale = 1000 / self.time_step
#         beta_start = scale * 0.0001
#         beta_end = scale * 0.02
#         return torch.linspace(beta_start, beta_end, self.time_step, dtype=torch.float32)

#     def q_sample(self, x0, t, noise):
#         """Sampling x_t, according to q(x_t | x_0). Consult (4) in DDPM paper.

#         Args:
#             x0: (b, c, h, w), original image
#             t: timestep t
#             noise: (b, c, h, w), We calculate q(x_t | x_0) using re-parameterization trick.
#         """
#         # Get x_t ~ q(x_t | x_0) using re-parameterization trick
#         return (
#             self.sqrt_alpha_bar[t][:, None, None, None] * x0
#             + self.sqrt_one_minus_alpha_bar[t][:, None, None, None] * noise
#         )

#     def forward(self, target, extracted_feats):
#         """Calculate L_simple according to (14) in DDPM paper

#         Args:
#             target: (b, h, w, k), target encoded catalog
#             extracted_features: (b, c, h, w), the extracted features from ori image
#         """
#         rearranged_target = target.permute([0, 3, 1, 2])
#         assert rearranged_target.shape[1:] == self.inter_target_size
#         assert rearranged_target.shape[2:] == extracted_feats.shape[2:]
#         assert rearranged_target.shape[0] == extracted_feats.shape[0]
#         b = rearranged_target.shape[0]
#         t = torch.randint(0, self.time_step, (b,), device=extracted_feats.device).long()  # (b, )
#         noise = torch.randn_like(rearranged_target)  # corresponds to epsilon in (14)
#         noised_target = self.q_sample(rearranged_target, t, noise)  # argument inside epsilon_theta
#         predicted_noise = self.model(noised_target, t, extracted_feats)  # epsilon_theta in (14)

#         if self.loss_type == "l1":
#             loss = torch.abs(predicted_noise - noise)
#         elif self.loss_type == "l2":
#             loss = (predicted_noise - noise) ** 2
#         else:
#             raise NotImplementedError()
#         return loss.permute([0, 2, 3, 1])

#     # @torch.inference_mode()
#     # def ddpm_p_sample(self, xt, t, extracted_feats, clip=True):
#     #     """Sample x_{t-1} from p_{theta}(x_{t-1} | x_t).

#     #     Args:
#     #         xt: (b, c, h, w), noised image at time step t
#     #         t: time step
#     #         extracted_feats: (b, c, h, w), the extracted features from ori image
#     #         clip: [True, False] Whether to clip predicted x_0 to our desired range -1 ~ 1.
#     #     """

#     #     batched_time = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
#     #     pred_noise = self.model(xt, batched_time, extracted_feats)  # corresponds to epsilon_{theta}
#     #     if clip:
#     #         x0 = self.sqrt_recip_alpha_bar[t] * xt - self.sqrt_recip_alpha_bar_min_1[t] * pred_noise
#     #         x0.clamp_(-1.0, 1.0)
#     #         mean = self.mean_tilde_x0_coeff[t] * x0 + self.mean_tilde_xt_coeff[t] * xt
#     #     else:
#     #         mean = self.sqrt_recip_alpha[t] * (
#     #             xt - self.beta_over_sqrt_one_minus_alpha_bar[t] * pred_noise
#     #         )
#     #     variance = self.beta_tilde[t]
#     #     noise = (
#     #         torch.randn_like(xt) if t > 0 else 0.0
#     #     )  # corresponds to z, consult 4: in Algorithm 2.
#     #     x_t_minus_1 = mean + torch.sqrt(variance) * noise
#     #     return x_t_minus_1

#     @torch.inference_mode()
#     def ddim_p_sample(self, xt, i, extracted_feats, clip=True):
#         """Sample x_{tau_(i-1)} from p(x_{tau_(i-1)} | x_{tau_i}), consult (56) in DDIM paper.
#         Calculation is done using (12) in DDIM paper where t-1 has to be changed to tau_(i-1) and t has to be
#         changed to tau_i in (12), for accelerated generation process where total # of de-noising step is S.

#         Args:
#             xt: noisy image at time step tau_i
#             i: i is the index of array tau which is an sub-sequence of [1, ..., T] of length S. See sec. 4.2
#             extracted_feats: (b, c, h, w), the extracted features from ori image
#             clip: Like in GaussianDiffusion p_sample, we can clip(or clamp) the predicted x_0 to -1 ~ 1
#                   for better sampling result. If you see (12) in DDIM paper, sampling x_(t-1) depends on epsilon_theta which is
#                   U-net network predicted noise at time step t. If we want to clip the "predicted x0", we have to
#                   re-calculate the epsilon_theta to make "predicted x0" lie in -1 ~ 1. This is exactly what is going on
#                   if you set clip==True.
#         """
#         t = self.tau[i]
#         batched_time = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
#         pred_noise = self.model(xt, batched_time, extracted_feats)  # corresponds to epsilon_{theta}
#         x0 = self.sqrt_recip_alpha_bar[t] * xt - self.sqrt_recip_alpha_bar_min_1[t] * pred_noise
#         if clip:
#             x0.clamp_(-1.0, 1.0)
#             pred_noise = (self.sqrt_recip_alpha_bar[t] * xt - x0) / self.sqrt_recip_alpha_bar_min_1[
#                 t
#             ]

#         # x0 corresponds to "predicted x0" and pred_noise corresponds to epsilon_theta(xt) in (12) DDIM
#         # Thus self.coeff[i] * pred_noise corresponds to "direction pointing to xt" in (12)
#         mean = self.sqrt_alpha_i_min_1[i] * x0 + self.coeff[i] * pred_noise
#         noise = torch.randn_like(xt) if i > 0 else 0.0
#         # self.sigma[i] * noise corresponds to "random noise" in (12)
#         x_t_minus_1 = mean + self.sigma[i] * noise
#         return x_t_minus_1

#     @torch.inference_mode()
#     def sample(self, extracted_feats, clip=True):
#         """Generate samples.

#         Args:
#             extracted_feats: (b, c, h, w), the extracted features from ori image
#             clip: [True, False]. Explanation in p_sample function.
#             use_ddim: [True, False]. Wheter to use DDIM
#         """

#         xT = torch.randn([extracted_feats.shape[0], *self.inter_target_size], device=self.device)
#         xt = xT
#         sample_time_step = reversed(range(0, self.ddim_steps))
#         for t in sample_time_step:
#             x_t_minus_1 = self.ddim_p_sample(xt, t, extracted_feats, clip)
#             xt = x_t_minus_1

#         x0 = xt
#         x0.clamp_(min=-1.0, max=1.0)
#         return x0.permute([0, 2, 3, 1])
