import torch
from einops import repeat, reduce, rearrange


class RMLDF:
    def __init__(self, 
                 num_timesteps,
                 m, 
                 lambda_, 
                 beta,
                 matching_fn=None,
                 loss_mask_fn=None,
                 pred_x0_rectify_fn=None,
                 loss_weight_fn=None):
        self.num_timesteps = num_timesteps
        self.sigma = torch.linspace(0, 1, steps=self.num_timesteps, dtype=torch.float64)
        self.alpha = 1.0 - self.sigma
        self.m = m
        self.lambda_ = lambda_
        self.beta = beta
        self.matching_fn = matching_fn
        self.loss_mask_fn = loss_mask_fn
        self.pred_x0_rectify_fn = pred_x0_rectify_fn
        self.loss_weight_fn = loss_weight_fn

    @classmethod
    def _extract_arr_by_t(cls, arr: torch.Tensor, t: torch.Tensor):
        return arr.to(device=t.device)[t.flatten()].view(t.shape)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        assert t.shape == x_start.shape
        return (
            self._extract_arr_by_t(self.alpha, t) * x_start
            + self._extract_arr_by_t(self.sigma, t) * noise
        ).float()
    
    @classmethod
    def replicate_fn(cls, n: int, m: int, x: torch.Tensor):
        return repeat(x, "n ... -> n m ...", n=n, m=m)
    
    def compute_rho_diagonal_fn(self, x: torch.Tensor, y: torch.Tensor, loss_weight=None):
        assert x.shape == y.shape
        if loss_weight is None:
            loss_weight = torch.ones_like(x)
        else:
            assert loss_weight.ndim == x.ndim - 1
            loss_weight = loss_weight.unsqueeze(1)
        # +1e-5 to avoid gradient explosion when (x - y) ** 2 == 0
        l2_norm_beta = (reduce((x - y) ** 2 * loss_weight, 
                               "n m ... -> n m", 
                               reduction="sum") + 1e-5).sqrt() ** self.beta
        return torch.mean(l2_norm_beta, dim=-1)  # (n, )
    
    def compute_rho_fn(self, x: torch.Tensor, y: torch.Tensor, loss_weight=None):
        assert x.shape == y.shape
        x = x.unsqueeze(2)  # (n, m, 1, ...)
        y = y.unsqueeze(1)  # (n, 1, m, ...)
        if loss_weight is None:
            loss_weight = torch.ones_like(x)
        else:
            assert loss_weight.ndim == x.ndim - 2
            loss_weight = rearrange(loss_weight, "n ... -> n 1 1 ...")
        # +1e-5 to avoid gradient explosion when (x - y) ** 2 == 0
        l2_norm_beta = (reduce((x - y) ** 2 * loss_weight, 
                               "n m1 m2 ... -> n m1 m2", 
                               reduction="sum") + 1e-5).sqrt() ** self.beta
        off_diag_mask = (1.0 - torch.eye(l2_norm_beta.shape[-1], 
                                        dtype=l2_norm_beta.dtype, 
                                        device=l2_norm_beta.device)).unsqueeze(0)
        l2_norm_beta = l2_norm_beta * off_diag_mask
        return reduce(l2_norm_beta, "n m1 m2 -> n", reduction="mean") * (self.m / (self.m - 1))
    
    def loss_fn(self, t: torch.Tensor, x0: torch.Tensor, xt: torch.Tensor, model_fn):
        n = x0.shape[0]
        assert t.shape == x0.shape
        x0_population = self.replicate_fn(n=n, m=self.m, x=x0)
        t_population = self.replicate_fn(n=n, m=self.m, x=t)
        xt_population = self.replicate_fn(n=n, m=self.m, x=xt)
        epsilon_population = torch.randn_like(xt_population)

        output_population = model_fn(t=t_population, xt=xt_population, epsilon=epsilon_population)

        if self.matching_fn is not None:
            output_population = self.matching_fn(x0_population, output_population)

        if self.loss_mask_fn is not None:
            output_population = self.loss_mask_fn(x0_population, output_population)

        if self.loss_weight_fn is not None:
            loss_weight = self.loss_weight_fn(t, alpha=self.alpha, sigma=self.sigma)
        else:
            loss_weight = None

        confinement = self.compute_rho_diagonal_fn(x=x0_population, y=output_population, loss_weight=loss_weight)
        interaction_prediction = self.compute_rho_fn(x=output_population, y=output_population, loss_weight=loss_weight)

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

        assert loss_weights is None, "please use loss_weight_fn to set the loss weight"

        model_fn = lambda t, xt, epsilon: model(xt, t, epsilon=epsilon, is_training=True, **model_kwargs)
        loss = self.loss_fn(t=t, x0=x_start, xt=xt, model_fn=model_fn)
        assert not torch.isnan(loss).any()
        return {
            "loss": loss,
        }
    
    def r_ij(self, i, j, s, t):
        alpha_t = self._extract_arr_by_t(self.alpha, t)
        alpha_s = self._extract_arr_by_t(self.alpha, s)
        sigma_t_2 = self._extract_arr_by_t(self.sigma, t) ** 2
        sigma_s_2 = self._extract_arr_by_t(self.sigma, s) ** 2
        return torch.where(s != t, 
                           ((alpha_t / alpha_s) ** i) * ((sigma_s_2 / sigma_t_2) ** j),
                           1.0)
    
    def ddim_mean_variance(self, x0, xt, s, t, churn_factor):
        assert x0.shape == xt.shape
        assert s.shape == x0.shape
        assert t.shape == s.shape
        r01 = self.r_ij(0, 1, s, t)
        r11 = self.r_ij(1, 1, s, t)
        r12 = self.r_ij(1, 2, s, t)
        r22 = self.r_ij(2, 2, s, t)
        c2 = churn_factor ** 2
        alpha_s = self._extract_arr_by_t(self.alpha, s)
        sigma_s_2 = self._extract_arr_by_t(self.sigma, s) ** 2
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
        noise = torch.rand_like(xt)
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
        k_matrix,
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
            k_matrix,
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
        k_matrix,
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
        assert k_matrix.shape[1:] == shape
        if noise is not None:
            xt = noise
        else:
            xt = torch.randn(*shape, device=device)

        m_list = list(range(k_matrix.shape[0]))
        time_pair_index = list(zip(m_list[:-1], m_list[1:]))
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            time_pair_index = tqdm(time_pair_index)
        for i, (t, s) in enumerate(time_pair_index):
            t_tensor = k_matrix[t]
            s_tensor = k_matrix[s]
            assert (t_tensor >= s_tensor).all()
            if i == len(time_pair_index) - 1:
                assert (s_tensor == 0).all()
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
