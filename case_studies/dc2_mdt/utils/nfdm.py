from typing import Callable

import numpy as np
import torch
from torch import nn, Tensor


@torch.inference_mode()
def solve_sde(
    sde: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    z: Tensor,
    ts: float,
    tf: float,
    n_steps: int,
):
    bs = z.shape[0]
    t_steps = torch.linspace(ts, tf, n_steps + 1)
    dt = (tf - ts) / n_steps
    dt_2 = abs(dt) ** 0.5
    path = [z]
    for t in t_steps[:-1]:
        t = t.expand(bs, 1)
        f, g = sde(z, t)
        w = torch.randn_like(z)
        z = z + f * dt + g * w * dt_2
        path.append(z)
    return {
        "samples": z, 
        "t_steps": t_steps, 
        "path": torch.stack(path),
    }


@torch.inference_mode()
def solve_ode(
    ode: Callable[[Tensor, Tensor], Tensor],
    z: Tensor,
    ts: float,
    tf: float,
    n_steps: int,
):
    return solve_sde(lambda z_in, t_in: (ode(z_in, t_in), torch.zeros_like(z_in)), 
                     z, ts, tf, n_steps)
    

class NeuralFlowDiffusion:
    def __init__(self, sde_sampling_steps, ode_sampling_steps):
        self.sde_sampling_steps = sde_sampling_steps
        self.ode_sampling_steps = ode_sampling_steps

    def _get_t_dir(self, affine_transform_model, x, t):
        return torch.autograd.functional.jvp(
            lambda t: affine_transform_model(x, t),
            t,
            torch.ones_like(t)
        )

    def forward_transform(self, affine_transform_model, noise, t, x):
        (m, s), (dm, ds) = self._get_t_dir(affine_transform_model, x, t)
        z = m + s * noise
        dz = dm + ds * noise
        score = - noise / s
        return z, dz, score

    def inverse_transform(self, affine_transform_model, z, t, x):
        (m, s), (dm, ds) = self._get_t_dir(affine_transform_model, x, t)
        noise = (z - m) / s
        dz = dm + ds / s * (z - m)
        score = (m - z) / s ** 2
        return noise, dz, score

    def score_based_sde_drift(self, dz, score, g2):
        return dz - 0.5 * g2 * score

    def training_losses(self,
                        affine_transform_model,
                        volatility_model,
                        x0_model,
                        x_start,
                        t,
                        x0_model_kwargs):
        noise = torch.randn_like(x_start)
        z, f_dz, f_score = self.forward_transform(affine_transform_model, noise, t, x_start)

        pred_x0 = x0_model(xt=z, t=t, **x0_model_kwargs)
        _, r_dz, r_score = self.inverse_transform(affine_transform_model, z, t, pred_x0)

        g2 = volatility_model(t) ** 2
        f_drift = self.score_based_sde_drift(f_dz, f_score, g2)
        r_drift = self.score_based_sde_drift(r_dz, r_score, g2)
        loss = 0.5 * (f_drift - r_drift) ** 2 / g2
        return {
            "loss": loss.sum(dim=1),
        }
    
    def sde_sample(self, 
                   affine_transform_model,
                   volatility_model,
                   x0_model,
                   shape,
                   device,
                   x0_model_kwargs):
        def sde(z_in, t_in):
            t_in = t_in.to(device=device)
            pred_x0 = x0_model(xt=z_in, t=t_in, **x0_model_kwargs)
            _, dz, score = self.inverse_transform(affine_transform_model, z_in, t_in, pred_x0)
            g = volatility_model(t_in)
            g2 = g ** 2
            drift = self.score_based_sde_drift(dz, score, g2)
            return drift, g
        return solve_sde(sde=sde, 
                         z=torch.randn(shape, device=device), 
                         ts=1.0, 
                         tf=0.0, 
                         n_steps=self.sde_sampling_steps)
    
    def ode_sample(self, 
                   affine_transform_model,
                   x0_model,
                   shape,
                   device,
                   x0_model_kwargs):
        def ode(z_in, t_in):
            t_in = t_in.to(device=device)
            pred_x0 = x0_model(xt=z_in, t=t_in, **x0_model_kwargs)
            _, dz, _ = self.inverse_transform(affine_transform_model, z_in, t_in, pred_x0)
            return dz
        return solve_ode(ode=ode, 
                         z=torch.randn(shape, device=device), 
                         ts=1.0, 
                         tf=0.0, 
                         n_steps=self.ode_sampling_steps)
    