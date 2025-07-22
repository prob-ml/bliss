import torch
from einops import repeat, reduce


class RMLDiffusion:
    def __init__(self, 
                 num_timesteps, 
                 num_sampling_steps, 
                 m, 
                 lambda_, 
                 beta,
                 matching_fn=None,
                 loss_mask_fn=None,
                 pred_x0_rectify_fn=None):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.sigma = torch.linspace(0, 1, steps=self.num_timesteps, dtype=torch.float64)
        self.alpha = 1.0 - self.sigma
        self.m = m
        self.lambda_ = lambda_
        self.beta = beta
        self.matching_fn = matching_fn
        self.loss_mask_fn = loss_mask_fn
        self.pred_x0_rectify_fn = pred_x0_rectify_fn

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.alpha, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sigma, t, x_start.shape) * noise
        ).float()
    
    @classmethod
    def replicate_fn(cls, n: int, m: int, x: torch.Tensor):
        return repeat(x, "n ... -> n m ...", n=n, m=m)
    
    def compute_rho_diagonal_fn(self, x: torch.Tensor, y: torch.Tensor):
        assert x.shape == y.shape
        # +1e-5 to avoid gradient explosion when (x - y) ** 2 == 0
        l2_norm_beta = (reduce((x - y) ** 2, "n m ... -> n m", reduction="sum") + 1e-5).sqrt() ** self.beta
        return torch.mean(l2_norm_beta, dim=-1)  # (n, )
    
    def compute_rho_fn(self, x: torch.Tensor, y: torch.Tensor):
        assert x.shape == y.shape
        x = x.unsqueeze(2)  # (n, m, 1, ...)
        y = y.unsqueeze(1)  # (n, 1, m, ...)
        # +1e-5 to avoid gradient explosion when (x - y) ** 2 == 0
        l2_norm_beta = (reduce((x - y) ** 2, "n m1 m2 ... -> n m1 m2", reduction="sum") + 1e-5).sqrt() ** self.beta
        off_diag_mask = (1.0 - torch.eye(l2_norm_beta.shape[-1], 
                                        dtype=l2_norm_beta.dtype, 
                                        device=l2_norm_beta.device)).unsqueeze(0)
        l2_norm_beta = l2_norm_beta * off_diag_mask
        return reduce(l2_norm_beta, "n m1 m2 -> n", reduction="mean") * (self.m / (self.m - 1))
    
    def loss_fn(self, t: torch.Tensor, x0: torch.Tensor, xt: torch.Tensor, model_fn):
        n = x0.shape[0]
        x0_population = self.replicate_fn(n=n, m=self.m, x=x0)
        t_population = self.replicate_fn(n=n, m=self.m, x=t)
        xt_population = self.replicate_fn(n=n, m=self.m, x=xt)
        epsilon_population = torch.randn_like(xt_population)

        output_population = model_fn(t=t_population, xt=xt_population, epsilon=epsilon_population)

        if self.matching_fn is not None:
            output_population = self.matching_fn(x0_population, output_population)

        if self.loss_mask_fn is not None:
            output_population = self.loss_mask_fn(x0_population, output_population)

        confinement = self.compute_rho_diagonal_fn(x=x0_population, y=output_population)
        interaction_prediction = self.compute_rho_fn(x=output_population, y=output_population)

        score = 0.5  * self.lambda_ * interaction_prediction - confinement
        return -1 * score
    
    def training_losses(self, 
                        model, 
                        x_start, 
                        t, 
                        model_kwargs=None, 
                        noise=None, 
                        loss_weights=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        xt = self.q_sample(x_start, t, noise=noise)
        if loss_weights is None:
            loss_weights = torch.ones_like(t).float()
        assert loss_weights.ndim == 1 and loss_weights.shape[0] == x_start.shape[0]

        model_fn = lambda t, xt, epsilon: model(xt, t, epsilon=epsilon, is_training=True, **model_kwargs)
        loss = self.loss_fn(t=t, x0=x_start, xt=xt, model_fn=model_fn)
        assert not torch.isnan(loss).any()
        assert not torch.isnan(loss_weights).any()
        assert loss.shape == loss_weights.shape
        return {
            "loss": loss * loss_weights,
        }
    
    def r_ij(self, i, j, s, t, broadcast_shape=None):
        if broadcast_shape is None:
            broadcast_shape = t.shape
        alpha_t = self._extract_into_tensor(self.alpha, t, broadcast_shape)
        alpha_s = self._extract_into_tensor(self.alpha, s, broadcast_shape)
        sigma_t_2 = self._extract_into_tensor(self.sigma, t, broadcast_shape) ** 2
        sigma_s_2 = self._extract_into_tensor(self.sigma, s, broadcast_shape) ** 2
        return ((alpha_t / alpha_s) ** i) * ((sigma_s_2 / sigma_t_2) ** j)
    
    def ddim_mean_variance(self, x0, xt, s, t, churn_factor):
        assert x0.shape == xt.shape
        r01 = self.r_ij(0, 1, s, t, broadcast_shape=xt.shape)
        r11 = self.r_ij(1, 1, s, t, broadcast_shape=xt.shape)
        r12 = self.r_ij(1, 2, s, t, broadcast_shape=xt.shape)
        r22 = self.r_ij(2, 2, s, t, broadcast_shape=xt.shape)
        c2 = churn_factor ** 2
        alpha_s = self._extract_into_tensor(self.alpha, s, broadcast_shape=xt.shape)
        sigma_s_2 = self._extract_into_tensor(self.sigma, s, broadcast_shape=xt.shape) ** 2
        posterior_mean = (c2 * r12 + (1 - c2) * r01) * xt + \
                          alpha_s * (1 - c2 * r22 - (1 - c2) * r11) * x0
        posterior_variance = sigma_s_2 * (1 - (c2 * r11 + (1 - c2)) ** 2)
        return posterior_mean, posterior_variance
    
    def ddim_sample(
        self,
        model,
        xt,
        t,
        s,
        clip_denoised=True,
        model_kwargs=None,
        eta=0.0,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        epsilon = torch.randn_like(xt)
        noise = torch.randn_like(xt)
        pred_x0 = model(xt, t, epsilon=epsilon, is_training=False, **model_kwargs)
        if clip_denoised:
            pred_x0 = pred_x0.clamp(min=-1, max=1)
        if self.pred_x0_rectify_fn is not None:
            pred_x0 = self.pred_x0_rectify_fn(pred_x0)
        mu, variance = self.ddim_mean_variance(x0=pred_x0, xt=xt, s=s, t=t, churn_factor=eta)
        return {
            "sample": (mu + variance.sqrt() * noise).float(),
            "pred_x0": pred_x0.float(),
        }

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        return_intermediate=False,
    ):
        final = None
        intermediate = []
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample["sample"]
            intermediate.append(sample["pred_x0"].cpu())
        if not return_intermediate:
            return final
        else:
            return final, intermediate

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            xt = noise
        else:
            xt = torch.randn(*shape, device=device)
        
        times = torch.linspace(0, self.num_timesteps - 1, steps=self.num_sampling_steps).int().tolist()[::-1]
        time_pairs = list(zip(times[:-1], times[1:]))  # (t, s)
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            time_pairs = tqdm(time_pairs)

        for t, s in time_pairs:
            t_tensor = torch.tensor([t] * shape[0], device=device)
            s_tensor = torch.tensor([s] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    xt,
                    t=t_tensor,
                    s=s_tensor,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                xt = out["sample"]

    @classmethod
    def _extract_into_tensor(cls, 
                             arr: torch.Tensor, 
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
        res = arr[timesteps]
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res + torch.zeros(broadcast_shape, device=timesteps.device, dtype=torch.float64)
