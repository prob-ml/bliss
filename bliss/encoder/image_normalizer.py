# flake8: noqa: WPS348
import warnings
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt  # noqa: WPS301
import numpy as np
import torch


class ImageNormalizer(torch.nn.Module):
    def __init__(
        self,
        bands: list,
        include_original: bool,
        concat_psf_params: bool,
        num_psf_params: int,
        log_transform_stdevs: list,
        use_clahe: bool,
        clahe_min_stdev: float,
        asinh_params: Dict[str, float],
    ):
        """Initializes DetectionEncoder.

        Args:
            bands: list of bands to use for input
            include_original: whether to include the original image as an input channel
            concat_psf_params: whether to include the PSF parameters as input channels
            num_psf_params: number of PSF parameters
            log_transform_stdevs: list of thresholds to apply log transform to (can be empty)
            use_clahe: whether to apply Contrast Limited Adaptive Histogram Equalization to images
            clahe_min_stdev: minimum standard deviation for CLAHE
            asinh_params: parameters for asinh normalization
        """
        super().__init__()

        self.bands = bands
        self.include_original = include_original
        self.concat_psf_params = concat_psf_params
        self.num_psf_params = num_psf_params
        self.log_transform_stdevs = log_transform_stdevs
        self.use_clahe = use_clahe
        self.clahe_min_stdev = clahe_min_stdev
        self.asinh_params = asinh_params

        if self.asinh_params:
            thresholds_num = len(self.asinh_params["thresholds"])
            thresholds = (
                torch.tensor(self.asinh_params["thresholds"])
                .view(1, -1)
                .expand(len(self.bands), thresholds_num)
                .view(1, len(self.bands), thresholds_num, 1, 1)
            ).clone()
            scales = (
                torch.log(torch.tensor([self.asinh_params["scale"]]))
                .expand(1, len(self.bands), thresholds_num, 1, 1)
                .clone()
            )
            self.asinh_thresholds_tensor = torch.nn.Parameter(thresholds, requires_grad=True)
            self.asinh_scales_tensor = torch.nn.Parameter(scales, requires_grad=True)
        else:
            self.asinh_thresholds_tensor = None
            self.asinh_scales_tensor = None

        if not (log_transform_stdevs or use_clahe or asinh_params):
            warnings.warn("Normalization should be enabled (you could use log/clahe/asinh).")

    def num_channels_per_band(self):
        """Determine number of input channels for model based on desired input transforms."""
        nch = 1  # background is always included (except for asinh)
        if self.include_original:
            nch += 1
        if self.concat_psf_params:
            nch += self.num_psf_params
        if self.log_transform_stdevs:
            nch += len(self.log_transform_stdevs)
        if self.use_clahe:
            nch += 1
        if self.asinh_params:
            nch += len(self.asinh_params["thresholds"])
            nch -= 1
        return nch

    def get_input_tensor(self, batch):
        """Extracts data from batch and concatenates into a single tensor to be input into model.

        Args:
            batch: input batch (as dictionary)

        Returns:
            Tensor: b x c x 2 x h x w tensor, where the number of input channels `c` is based on the
                input transformations to use
        """
        assert batch["images"].size(2) % 16 == 0, "image dims must be multiples of 16"
        assert batch["images"].size(3) % 16 == 0, "image dims must be multiples of 16"

        if self.log_transform_stdevs:
            assert batch["background"].min() > 1e-6, "background must be positive"

        input_bands = batch["images"].shape[1]
        if input_bands < len(self.bands):
            msg = f"Expected >= {len(self.bands)} bands in the input but found only {input_bands}"
            warnings.warn(msg)

        raw_images = batch["images"][:, self.bands].unsqueeze(2)
        backgrounds = batch["background"][:, self.bands].unsqueeze(2)
        inputs = [] if self.asinh_params else [backgrounds]

        if self.include_original:
            inputs.insert(0, raw_images)  # add extra dim for 5d input

        if self.concat_psf_params:
            msg = "concat_psf_params specified but psf params not present in data"
            assert "psf_params" in batch, msg
            n, c, i, h, w = raw_images.shape
            psf_params = batch["psf_params"][:, self.bands]
            psf_params = psf_params.view(n, c, self.num_psf_params * i, 1, 1)
            psf_params = psf_params.expand(n, c, self.num_psf_params * i, h, w)
            inputs.append(psf_params)

        for threshold in self.log_transform_stdevs:
            image_offsets = (raw_images - backgrounds) / backgrounds.sqrt() - threshold
            transformed_img = torch.log(torch.clamp(image_offsets + 1.0, min=1.0))
            inputs.append(transformed_img)

        # we should revisit normalizing the whole 80x80 image to see if that still performs
        # better than CLAHE. if so, we can remove CLAHE and for large images partition them
        # into 80x80 images before prediction

        if self.use_clahe:
            renormalized_img = self.clahe(raw_images, self.clahe_min_stdev)
            inputs.append(renormalized_img)
            if not self.asinh_params:
                inputs[0] = self.clahe(backgrounds, self.clahe_min_stdev)

        if self.asinh_params:
            inputs.append(
                torch.asinh(
                    (raw_images - self.asinh_thresholds_tensor)
                    * torch.exp(self.asinh_scales_tensor),
                )
            )

        return torch.cat(inputs, dim=2)

    @classmethod
    def clahe(cls, images, min_stdev, kernel_size=9, padding=4):
        """Perform Contrast Limited Adaptive Histogram Equalization (CLAHE) on input images."""
        images4d = images.squeeze(2)
        orig_shape = images4d.shape

        # Padding for borders in image
        padding4d = (padding, padding, padding, padding)
        pad_images = torch.nn.functional.pad(images4d, pad=padding4d, mode="reflect")
        # Unfold image, compute means
        f = torch.nn.Unfold(kernel_size=(kernel_size, kernel_size), padding=0, stride=1)
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
        normalized_img = res_img / torch.clamp(stdev, min=min_stdev)
        return normalized_img.unsqueeze(2)

    # just for debug
    def plot_image(self, inputs, asinh_thresholds, raw_images, split_ids):
        figure_output_path = Path("./input_images/")
        figure_output_path.mkdir(exist_ok=True)
        batch_size = raw_images.shape[0]
        for i in range(batch_size):
            for j in range(1, len(asinh_thresholds) + 1):
                image_matrix = inputs[-j].squeeze(2)[i]
                self._plot_image_matrix(
                    image_matrix,
                    figure_output_path / f"{split_ids[i]}_threshold_{asinh_thresholds[-j]}.png",
                )

            image_matrix = raw_images.squeeze(2)[i]
            self._plot_image_matrix(
                image_matrix, figure_output_path / f"{split_ids[i]}_raw_image.png"
            )

    # just for debug
    def _plot_image_matrix(self, image_matrix, output_path):
        image = image_matrix.detach().cpu().numpy()
        image = np.sum(image, axis=0)
        vmin = image.min().item()
        vmax = image.max().item()
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        ax.matshow(image, vmin=vmin, vmax=vmax, cmap="viridis")
        fig.savefig(output_path)
        plt.close(fig)
