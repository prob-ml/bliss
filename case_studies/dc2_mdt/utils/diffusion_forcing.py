import torch
from einops import reduce


class DiffusionForcing:
    def __init__(self, *, betas):
        betas = torch.tensor(betas, dtype=torch.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

    @classmethod
    def _extract_arr_by_t(cls, arr: torch.Tensor, t: torch.Tensor):
        return arr.to(device=t.device)[t.flatten()].view(t.shape)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        assert t.shape == x_start.shape
        return (self._extract_arr_by_t(self.sqrt_alphas_cumprod, t) * x_start + \
                self._extract_arr_by_t(self.sqrt_one_minus_alphas_cumprod, t) * noise).float()

    def training_losses(self, 
                        model, 
                        x_start, 
                        t, 
                        model_kwargs=None, 
                        noise=None, 
                        loss_weights=None):
        assert t.shape == x_start.shape
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        xt = self.q_sample(x_start, t, noise=noise)
        if loss_weights is None:
            loss_weights = torch.ones(t.shape[0], device=t.device)
        assert loss_weights.ndim == 1 and loss_weights.shape[0] == x_start.shape[0]
        pred_noise = model(xt, t, **model_kwargs)
        loss = reduce((noise - pred_noise) ** 2, "b ... -> b", reduction="mean")
        assert not torch.isnan(loss).any()
        assert not torch.isnan(loss_weights).any()
        assert loss.shape == loss_weights.shape
        return {
            "loss": loss * loss_weights,
        }
    
    def get_x0_from_noise(self, xt, t, noise):
        return (self._extract_arr_by_t(self.sqrt_recip_alphas_cumprod, t) * xt - \
                self._extract_arr_by_t(self.sqrt_recipm1_alphas_cumprod, t) * noise).float()
    
    def get_noise_from_x0(self, xt, t, x0):
        return ((self._extract_arr_by_t(self.sqrt_recip_alphas_cumprod, t) * xt - x0) / \
                 self._extract_arr_by_t(self.sqrt_recipm1_alphas_cumprod, t)).float()
    
    def ddim_sample(
        self,
        model,
        xt,
        t,
        s,
        clip_denoised=True,
        model_kwargs=None,
        eta=0.0,
        memory_dict=None,
    ):
        assert (t >= s).all()
        if model_kwargs is None:
            model_kwargs = {}
        pred_eps = model(xt, t, **model_kwargs)
        pred_x0 = self.get_x0_from_noise(xt, t, pred_eps)
        if clip_denoised:
            pred_x0 = pred_x0.clamp(min=-1, max=1)
            pred_eps = self.get_noise_from_x0(xt, t, pred_x0)
        alphas_bar_t = self._extract_arr_by_t(self.alphas_cumprod, t)
        alphas_bar_s = self._extract_arr_by_t(self.alphas_cumprod, s)
        sigma = eta * ((1 - alphas_bar_t / alphas_bar_s) * (1 - alphas_bar_s) / (1 - alphas_bar_t)).sqrt()
        c = (1 - alphas_bar_s - sigma ** 2).sqrt()
        noise = torch.randn_like(xt)
        sample = (pred_x0 * alphas_bar_s.sqrt() + \
                    c * pred_eps + \
                    sigma * noise).float()
        sample = torch.where(s == 0, pred_x0, sample)
        
        if memory_dict is not None:
            time0_mask = memory_dict["time0_mask"]
            preserved_x0 = memory_dict["preserved_x0"]
        else:
            time0_mask = torch.zeros_like(pred_x0, dtype=torch.bool)
            preserved_x0 = torch.zeros_like(pred_x0)
        sample = torch.where(time0_mask, preserved_x0, sample)
        pred_x0 = torch.where(time0_mask, preserved_x0, pred_x0)
        preserved_x0 = torch.where((s == 0) != time0_mask, pred_x0, preserved_x0)
        time0_mask |= (s == 0)
        return {
            "sample": sample,
            "pred_x0": pred_x0,
            "memory_dict": {
                "time0_mask": time0_mask,
                "preserved_x0": preserved_x0,
            }
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
        memory_dict = None
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
                    memory_dict=memory_dict,
                )
                yield out
                xt = out["sample"]
                memory_dict = out["memory_dict"]
