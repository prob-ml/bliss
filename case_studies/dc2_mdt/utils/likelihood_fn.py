import torch
import numpy as np
from scipy import integrate
from einops import reduce


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))

def _extract_into_tensor(arr: torch.Tensor, 
                         timesteps: torch.Tensor, 
                         broadcast_shape: tuple):
    """
    Extract values from a 1-D torch tensor for a batch of indices.
    :param arr: the 1-D torch tensor.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    arr = arr.to(device=timesteps.device)
    res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


# def flow_matching_sde_fn(z, t):
#     assert (t < 1.0).all()
#     while t.ndim < z.ndim:
#        t = t.unsqueeze(-1)
#     return -1 * (z / (1 - t)), torch.sqrt(2 * t / (1 - t))


def flow_matching_sde_fn(x0, t):
    while t.ndim < x0.ndim:
       t = t.unsqueeze(-1)
    return -1 * x0, torch.sqrt(2 * t)

def ode_drift_fn(x0_model_fn, sample, t, x0, alpha, sigma):
    t_index = (t * (alpha.shape[0] - 1)).long()
    # assert ((t_index > 0) & (t_index < (alpha.shape[0] - 1))).all()
    assert (t_index > 0).all()
    pred_x0 = x0_model_fn(xt=sample, t=t_index)
    alpha_t = _extract_into_tensor(alpha, t_index, broadcast_shape=pred_x0.shape)
    sigma_t = _extract_into_tensor(sigma, t_index, broadcast_shape=pred_x0.shape)
    score = -1 * ((sample - alpha_t * pred_x0) / (sigma_t ** 2))
    drift, diffusion = flow_matching_sde_fn(x0, t)  # RML uses the flow matching schedule
    return drift - 0.5 * diffusion ** 2 * score

def vmap_ode_drift_fn(x0_model_fn, sample, t, x0, alpha, sigma, n):
    return torch.vmap(lambda dummy: ode_drift_fn(x0_model_fn, sample, t, x0, alpha, sigma),
                      randomness="different")(torch.randn((n, )))

def vmap_div_fn(x0_model_fn, sample, t, x0, alpha, sigma, n):
    with torch.enable_grad():
        sample.requires_grad_(True)
        lambda_nd_fn = lambda cur_sample, noise: torch.sum(
            ode_drift_fn(x0_model_fn, cur_sample, t, x0, alpha, sigma) * noise
        )
        n_noise = torch.randint(low=0, high=2, 
                                size=(n, ) + sample.shape, 
                                device=sample.device).float() * 2 - 1.0
        # n_noise = torch.randn((n, ) + sample.shape, device=sample.device)
        n_div_nd = torch.vmap(torch.func.grad(lambda_nd_fn), 
                            in_dims=(None, 0), 
                            randomness="different")(sample, n_noise)
        n_div = reduce(n_div_nd * n_noise, "n b ... -> n b", reduction="sum")
    sample.requires_grad_(False)
    return n_div

def normal_logp(sample):
    shape = sample.shape
    N = np.prod(shape[1:])
    logps = -N / 2.0 * np.log(2 * np.pi) - reduce(sample ** 2, "b ... -> b", reduction="sum") / 2.0
    return logps


# likelihood for RML
def rml_likelihood(x0_model_fn, x0, num_timesteps, n):
    sigma = torch.linspace(0, 1, num_timesteps, device=x0.device)
    alpha = 1.0 - sigma  # flow matching schedule

    with torch.no_grad():
        shape = x0.shape
        def ode_func(t, x):
            sample = from_flattened_numpy(x[:-shape[0]], shape).to(x0.device).type(torch.float32)
            vec_t = torch.ones(sample.shape[0], device=sample.device) * t
            drift = to_flattened_numpy(
                vmap_ode_drift_fn(x0_model_fn, sample, vec_t, x0, alpha, sigma, n=n).mean(dim=0)
            )
            logp_grad = to_flattened_numpy(
                vmap_div_fn(x0_model_fn, sample, vec_t, x0, alpha, sigma, n=n).mean(dim=0)
            )
            return np.concatenate([drift, logp_grad], axis=0)

        init = np.concatenate([to_flattened_numpy(x0), np.zeros((shape[0],))], axis=0)
        solution = integrate.solve_ivp(ode_func, (sigma[1].item(), 1.0), 
                                       init, rtol=1e-5, atol=1e-5, method="RK45")
        assert solution.success, f"error happens during integrating ode: {solution.message}"
        nfe = solution.nfev
        zp = solution.y[:, -1]
        z = from_flattened_numpy(zp[:-shape[0]], shape).to(x0.device).type(torch.float32)
        delta_logp = from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(x0.device).type(torch.float32)
        prior_logp = normal_logp(z)
        # pleaase note that because you scale the data to be in range [-1, 1]; 
        # this nll is calculated for [-1, 1], not for the data in original scale
        # you need to add an offset to this nll to get the true nll
        nll = -(prior_logp + delta_logp)  
    return nll, z, nfe
    