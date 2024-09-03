import torch
from einops import rearrange


class PsfAsImage(torch.nn.Module):
    def __init__(self, num_psf_params):
        super().__init__()
        self.num_psf_params = num_psf_params

    def num_channels_per_band(self):
        """Number of input channels for model based on this input normalizer."""
        return self.num_psf_params

    def get_input_tensor(self, batch):
        assert "psf_params" in batch, "PsfAsImage specified but psf params not provided"
        n, c, h, w = batch["images"].shape
        psf_params = batch["psf_params"]
        psf_params = psf_params.view(n, c, self.num_psf_params, 1, 1)
        return psf_params.expand(n, c, self.num_psf_params, h, w)


class ClaheNormalizer(torch.nn.Module):
    def __init__(self, min_stdev, kernel_size=9, padding=4):
        self.min_stdev = min_stdev
        self.kernel_size = kernel_size
        self.padding = padding

        super().__init__()

    def num_channels_per_band(self):
        """Number of input channels for model based on this input normalizer."""
        return 1

    def get_input_tensor(self, batch):
        """Perform Contrast Limited Adaptive Histogram Equalization (CLAHE) on input images."""
        images4d = batch["images"].squeeze(2)
        orig_shape = images4d.shape

        # Padding for borders in image
        padding4d = (self.padding, self.padding, self.padding, self.padding)
        pad_images = torch.nn.functional.pad(images4d, pad=padding4d, mode="reflect")
        # Unfold image, compute means
        f = torch.nn.Unfold(kernel_size=(self.kernel_size, self.kernel_size), padding=0, stride=1)
        out = f(pad_images)
        reshape_val = int(out.shape[1] / orig_shape[1])
        out = torch.reshape(
            out, (orig_shape[0], orig_shape[1], reshape_val, orig_shape[2], orig_shape[3])
        )
        # Compute residuals
        res_img = images4d - torch.mean(out, dim=2)
        # Pad residuals, compute squared residuals
        pad_res_img = torch.nn.functional.pad(res_img, pad=padding4d, mode="reflect")
        # Unfold squared residuals
        sqr_res = f(pad_res_img**2)
        reshape_sqr_res = torch.reshape(
            sqr_res, (orig_shape[0], orig_shape[1], reshape_val, orig_shape[2], orig_shape[3])
        )
        # Find rolling std
        stdev = torch.sqrt(torch.mean(reshape_sqr_res, dim=2))
        # Output rolling z-score
        normalized_img = res_img / torch.clamp(stdev, min=self.min_stdev)
        return normalized_img.unsqueeze(2)


class AsinhQuantileNormalizer(torch.nn.Module):
    def __init__(self, q, sample_every_n=None):
        super().__init__()
        self.register_buffer("q", torch.tensor(q))
        self.register_buffer("quantiles", torch.zeros(len(q)))
        self.register_buffer("sample_every_n", torch.tensor(sample_every_n))
        self.num_updates = 0

    def num_channels_per_band(self):
        """Number of input channels for model based on this input normalizer."""
        return len(self.q)

    def get_input_tensor(self, batch):
        ss_images = batch["images"]  # assumes images are already sky subtracted
        if self.sample_every_n:
            ss_images = ss_images[
                :, :, :: int(self.sample_every_n.item()), :: int(self.sample_every_n.item())
            ]
        if self.training and self.num_updates < 100:
            self.num_updates += 1
            cur_quantiles = torch.quantile(ss_images, q=self.q)
            lr = 1.0 / self.num_updates
            self.quantiles *= 1 - lr
            self.quantiles += lr * cur_quantiles

        quantiles5d = rearrange(self.quantiles, "n -> 1 1 n 1 1")
        ss_images5d = rearrange(ss_images, "b bands h w -> b bands 1 h w")
        centered_images = ss_images5d - quantiles5d
        # asinh seems to saturate beyond 5 or so
        scaled_images = centered_images * (5.0 / quantiles5d.abs().clamp(1e-6))
        return torch.asinh(scaled_images)


class NullNormalizer(torch.nn.Module):
    def num_channels_per_band(self):
        return 1

    def get_input_tensor(self, batch):
        return rearrange((batch["images"] + 0.5).clamp(1e-6) * 100, "b bands h w -> b bands 1 h w")
