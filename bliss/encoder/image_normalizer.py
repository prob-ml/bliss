import warnings

import torch


class ImageNormalizer(torch.nn.Module):
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
        asinh_cutoffs: list,
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
            include_background: whether to include background as an input channel
            asinh_cutoffs: list of asinh cutoffs to apply to images
        """
        super().__init__()

        self.bands = bands
        self.include_original = include_original
        self.include_background = include_background
        self.concat_psf_params = concat_psf_params
        self.num_psf_params = num_psf_params
        self.log_transform_stdevs = log_transform_stdevs
        self.use_clahe = use_clahe
        self.clahe_min_stdev = clahe_min_stdev
        self.asinh_cutoffs = asinh_cutoffs

    def num_channels_per_band(self):
        """Determine number of input channels for model based on desired input transforms."""
        nch = 0
        if self.include_background:
            nch += 1
        if self.include_original:
            nch += 1
        if self.concat_psf_params:
            nch += self.num_psf_params
        if self.log_transform_stdevs:
            nch += len(self.log_transform_stdevs)
        if self.use_clahe:
            nch += 1
        if self.asinh_cutoffs:
            nch += len(self.asinh_cutoffs)
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

        input_bands = batch["images"].shape[1]
        if input_bands < len(self.bands):
            msg = f"Expected >= {len(self.bands)} bands in the input but found only {input_bands}"
            warnings.warn(msg)

        raw_images = batch["images"][:, self.bands].unsqueeze(2)
        backgrounds = batch["background"][:, self.bands].unsqueeze(2)
        inputs = []

        if self.include_background:
            inputs.append(backgrounds)

        if self.include_original:
            inputs.append(raw_images)

        if self.concat_psf_params:
            msg = "concat_psf_params specified but psf params not present in data"
            assert "psf_params" in batch, msg
            n, c, i, h, w = raw_images.shape
            psf_params = batch["psf_params"][:, self.bands]
            psf_params = psf_params.view(n, c, self.num_psf_params * i, 1, 1)
            psf_params = psf_params.expand(n, c, self.num_psf_params * i, h, w)
            inputs.append(psf_params)

        if self.log_transform_stdevs:
            assert batch["background"].min() > 1e-6, "background must be positive"
            for threshold in self.log_transform_stdevs:
                image_offsets = (raw_images - backgrounds) / backgrounds.sqrt() - threshold
                transformed_img = torch.log(torch.clamp(image_offsets + 1.0, min=1.0))
                inputs.append(transformed_img)

        if self.asinh_cutoffs:
            ss_images = raw_images - backgrounds
            for cutoff in self.asinh_cutoffs:
                asinh_img = torch.asinh((ss_images - cutoff) / (0.1 * cutoff)) * 10
                inputs.append(asinh_img)

        # we should revisit normalizing the whole 80x80 image to see if that still performs
        # better than CLAHE. if so, we can remove CLAHE and for large images partition them
        # into 80x80 images before prediction

        if self.use_clahe:
            renormalized_img = self.clahe(raw_images, self.clahe_min_stdev)
            inputs.append(renormalized_img)
            # also normalize background
            if self.include_background:
                inputs[0] = self.clahe(backgrounds, self.clahe_min_stdev)

        return torch.cat(inputs, dim=2) if inputs else None

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
