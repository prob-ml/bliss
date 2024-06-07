from typing import Dict

import torch

from bliss.encoder.image_normalizer import ImageNormalizer


class BasicAsinhImageNormalizer(ImageNormalizer):
    def __init__(
        self,
        bands: list,
        include_original: bool,
        include_background: bool,
        concat_psf_params: bool,
        num_psf_params: int,
        log_transform_stdevs: list,
        use_clahe: bool,
        clahe_min_stdev: float,
        asinh_params: Dict[str, float],
    ):
        super().__init__(
            bands,
            include_original,
            include_background,
            concat_psf_params,
            num_psf_params,
            log_transform_stdevs,
            use_clahe,
            clahe_min_stdev,
        )
        self.asinh_params = asinh_params

        assert self.asinh_params, "asinh_params can't be None"
        assert (
            not self.include_background
        ), "if you want to use asinh, please don't include background"

        thresholds_num = len(self.asinh_params["thresholds"])
        thresholds = torch.tensor(self.asinh_params["thresholds"]).view(1, -1)
        thresholds = thresholds.expand(len(self.bands), thresholds_num)
        thresholds = thresholds.view(1, len(self.bands), thresholds_num, 1, 1).clone()

        scales = torch.log(torch.tensor([self.asinh_params["scale"]]))
        scales = scales.expand(1, len(self.bands), thresholds_num, 1, 1).clone()

        self.asinh_thresholds_tensor = torch.nn.Parameter(thresholds, requires_grad=True)
        self.asinh_scales_tensor = torch.nn.Parameter(scales, requires_grad=True)

    def num_channels_per_band(self):
        pre_nch = super().num_channels_per_band()
        return pre_nch + len(self.asinh_params["thresholds"])

    def get_input_tensor(self, batch):
        pre_input_tensor = super().get_input_tensor(batch)
        raw_images = batch["images"][:, self.bands].unsqueeze(2)

        filtered_images = raw_images - self.asinh_thresholds_tensor
        processed_images = filtered_images * torch.exp(self.asinh_scales_tensor)
        processed_images = torch.asinh(processed_images)

        if pre_input_tensor is not None:
            input_tensor = torch.cat((pre_input_tensor, processed_images), dim=2)
        else:
            input_tensor = processed_images

        stacked_asinh_params = torch.stack(
            (self.asinh_thresholds_tensor.squeeze(), self.asinh_scales_tensor.squeeze())
        )

        return input_tensor, stacked_asinh_params


class MovingAvgAsinhImageNormalizer(ImageNormalizer):
    def __init__(
        self,
        bands: list,
        include_original: bool,
        include_background: bool,
        concat_psf_params: bool,
        num_psf_params: int,
        log_transform_stdevs: list,
        use_clahe: bool,
        clahe_min_stdev: float,
        asinh_params: Dict[str, float],
    ):
        super().__init__(
            bands,
            include_original,
            include_background,
            concat_psf_params,
            num_psf_params,
            log_transform_stdevs,
            use_clahe,
            clahe_min_stdev,
        )
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
        pre_nch = super().num_channels_per_band()
        return pre_nch + len(self.asinh_params["thresholds"])

    def get_input_tensor(self, batch):
        pre_input_tensor = super().get_input_tensor(batch)
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
        processed_images = torch.asinh(processed_images)

        if pre_input_tensor is not None:
            input_tensor = torch.cat((pre_input_tensor, processed_images), dim=2)
        else:
            input_tensor = processed_images

        return input_tensor


class PerbandMovingAvgAsinhImageNormalizer(ImageNormalizer):
    def __init__(
        self,
        bands: list,
        include_original: bool,
        include_background: bool,
        concat_psf_params: bool,
        num_psf_params: int,
        log_transform_stdevs: list,
        use_clahe: bool,
        clahe_min_stdev: float,
        asinh_params: Dict[str, float],
    ):
        super().__init__(
            bands,
            include_original,
            include_background,
            concat_psf_params,
            num_psf_params,
            log_transform_stdevs,
            use_clahe,
            clahe_min_stdev,
        )
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
        pre_nch = super().num_channels_per_band()
        return pre_nch + len(self.asinh_params["thresholds"])

    def get_input_tensor(self, batch):
        pre_input_tensor = super().get_input_tensor(batch)
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
        processed_images = torch.asinh(processed_images)

        if pre_input_tensor is not None:
            input_tensor = torch.cat((pre_input_tensor, processed_images), dim=2)
        else:
            input_tensor = processed_images

        return input_tensor
