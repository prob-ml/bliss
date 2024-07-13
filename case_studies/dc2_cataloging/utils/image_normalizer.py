from typing import Dict

import torch


class DynamicAsinhImageNormalizer:
    def __init__(
        self,
        bands: list,
        asinh_params: Dict[str, float],
    ):
        self.bands = bands
        self.asinh_params = asinh_params

        assert self.asinh_params, "asinh_params can't be None"
        assert (
            not self.include_background
        ), "if you want to use asinh, please don't include background"

        thresholds_num = len(self.asinh_params["thresholds"])
        thresholds = torch.tensor(self.asinh_params["thresholds"]).view(1, -1)
        thresholds = thresholds.expand(len(self.bands), thresholds_num)
        thresholds = thresholds.view(1, len(self.bands), thresholds_num, 1, 1).clone()

        self.asinh_thresholds_tensor = torch.nn.Parameter(thresholds, requires_grad=True)

    def num_channels_per_band(self):
        return len(self.asinh_params["thresholds"])

    def get_input_tensor(self, batch):
        raw_images = batch["images"][:, self.bands].unsqueeze(2)

        asinh_thresholds_tensor = self.asinh_thresholds_tensor.detach()
        filtered_images = raw_images - asinh_thresholds_tensor
        processed_images = filtered_images * self.asinh_params["scale"]
        processed_images = torch.asinh(processed_images)

        return processed_images, self.asinh_thresholds_tensor.squeeze().unsqueeze(0)


class MovingAvgAsinhImageNormalizer:
    def __init__(
        self,
        bands: list,
        asinh_params: Dict[str, float],
    ):
        self.bands = bands
        self.asinh_params = asinh_params

        assert self.asinh_params, "asinh_params can't be None"
        assert (
            not self.include_background
        ), "if you want to use asinh, please don't include background"

        thresholds_num = len(self.asinh_params["thresholds"])
        thresholds = torch.tensor(self.asinh_params["thresholds"]).view(1, 1, thresholds_num, 1, 1)
        self.asinh_thresholds_tensor = thresholds

        self.asinh_buffer_size = 1000
        self.asinh_buffer_ptr = 0
        self.asinh_quantiles_tensor = torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        buffer_shape = (self.asinh_buffer_size, *self.asinh_thresholds_tensor.squeeze().shape)
        self.asinh_thresholds_buffer = torch.full(buffer_shape, torch.nan)

    def num_channels_per_band(self):
        return len(self.asinh_params["thresholds"])

    def get_input_tensor(self, batch):
        raw_images = batch["images"][:, self.bands].unsqueeze(2)

        self.asinh_thresholds_tensor = self.asinh_thresholds_tensor.to(device=raw_images.device)
        self.asinh_quantiles_tensor = self.asinh_quantiles_tensor.to(device=raw_images.device)
        self.asinh_thresholds_buffer = self.asinh_thresholds_buffer.to(device=raw_images.device)

        if self.training:
            self.asinh_buffer_ptr %= self.asinh_buffer_size
            self.asinh_thresholds_buffer[self.asinh_buffer_ptr] = torch.quantile(
                raw_images, q=self.asinh_quantiles_tensor
            )
            buffer_median, _ = torch.nanmedian(self.asinh_thresholds_buffer, dim=0)
            self.asinh_thresholds_tensor = buffer_median.view(
                *self.asinh_thresholds_tensor.shape,
            )
            self.asinh_buffer_ptr += 1

        filtered_images = raw_images - self.asinh_thresholds_tensor
        processed_images = filtered_images * self.asinh_params["scale"]
        return torch.asinh(processed_images)


class PerbandMovingAvgAsinhImageNormalizer:
    def __init__(
        self,
        bands: list,
        asinh_params: Dict[str, float],
    ):
        self.bands = bands
        self.asinh_params = asinh_params

        assert self.asinh_params, "asinh_params can't be None"
        assert (
            not self.include_background
        ), "if you want to use asinh, please don't include background"

        thresholds_num = len(self.asinh_params["thresholds"])
        thresholds = torch.tensor(self.asinh_params["thresholds"]).view(1, -1)
        thresholds = thresholds.expand(len(self.bands), thresholds_num)
        thresholds = thresholds.view(1, len(self.bands), thresholds_num, 1, 1).clone()
        self.asinh_thresholds_tensor = thresholds

        self.asinh_buffer_size = 1000
        self.asinh_buffer_ptr = 0
        self.asinh_quantiles_tensor = torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        buffer_shape = (self.asinh_buffer_size, *self.asinh_thresholds_tensor.squeeze().shape)
        self.asinh_thresholds_buffer = torch.full(buffer_shape, torch.nan)

    def num_channels_per_band(self):
        return len(self.asinh_params["thresholds"])

    def get_input_tensor(self, batch):
        raw_images = batch["images"][:, self.bands].unsqueeze(2)

        self.asinh_thresholds_tensor = self.asinh_thresholds_tensor.to(device=raw_images.device)
        self.asinh_quantiles_tensor = self.asinh_quantiles_tensor.to(device=raw_images.device)
        self.asinh_thresholds_buffer = self.asinh_thresholds_buffer.to(device=raw_images.device)

        if self.training:
            self.asinh_buffer_ptr %= self.asinh_buffer_size
            reshaped_raw_images = (
                raw_images.permute((1, 0, 2, 3, 4)).contiguous().view(raw_images.shape[1], -1)
            )
            raw_images_quantiles = torch.quantile(
                reshaped_raw_images,
                q=self.asinh_quantiles_tensor,
                dim=1,
            )
            self.asinh_thresholds_buffer[self.asinh_buffer_ptr] = raw_images_quantiles.permute(
                (1, 0)
            ).contiguous()
            buffer_median, _ = torch.nanmedian(self.asinh_thresholds_buffer, dim=0)
            self.asinh_thresholds_tensor = buffer_median.view(
                *self.asinh_thresholds_tensor.shape,
            )
            self.asinh_buffer_ptr += 1

        filtered_images = raw_images - self.asinh_thresholds_tensor
        processed_images = filtered_images * self.asinh_params["scale"]
        return torch.asinh(processed_images)


class FixedThresholdsAsinhImageNormalizer:
    def __init__(
        self,
        bands: list,
        asinh_params: Dict[str, float],
    ):
        self.bands = bands
        self.asinh_params = asinh_params

        assert self.asinh_params, "asinh_params can't be None"
        assert (
            not self.include_background
        ), "if you want to use asinh, please don't include background"

        self.asinh_thresholds_tensor = torch.tensor(self.asinh_params["thresholds"]).view(
            1, 1, -1, 1, 1
        )

    def num_channels_per_band(self):
        return len(self.asinh_params["thresholds"])

    def get_input_tensor(self, batch):
        raw_images = batch["images"][:, self.bands].unsqueeze(2)
        self.asinh_thresholds_tensor = self.asinh_thresholds_tensor.to(device=raw_images.device)
        filtered_images = raw_images - self.asinh_thresholds_tensor
        processed_images = filtered_images * self.asinh_params["scale"]
        return torch.asinh(processed_images)
