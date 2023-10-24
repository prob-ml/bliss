import warnings

import torch


class ImageNormalizer(torch.nn.Module):
    def __init__(
        self,
        bands: list,
        include_original: bool,
        use_deconv_channel: bool,
        concat_psf_params: bool,
        log_transform_stdevs: list,
        use_clahe: bool,
    ):
        """Initializes DetectionEncoder.

        Args:
            bands: list of bands to use for input
            include_original: whether to include the original image as an input channel
            use_deconv_channel: whether to include the deconvolved image as an input channel
            concat_psf_params: whether to include the PSF parameters as input channels
            log_transform_stdevs: list of thresholds to apply log transform to (can be empty)
            use_clahe: whether to apply Contrast Limited Adaptive Histogram Equalization to images
        """
        super().__init__()

        self.bands = bands
        self.include_original = include_original
        self.use_deconv_channel = use_deconv_channel
        self.concat_psf_params = concat_psf_params
        self.log_transform_stdevs = log_transform_stdevs
        self.use_clahe = use_clahe

        if not (log_transform_stdevs or use_clahe):
            warnings.warn("Either log transform or clahe should be enabled.")

    def num_channels_per_band(self):
        """Determine number of input channels for model based on desired input transforms."""
        nch = 1  # background is always included
        if self.include_original:
            nch += 1
        if self.use_deconv_channel:
            nch += 1
        if self.concat_psf_params:
            nch += 6  # number of PSF parameters for SDSS, may vary for other surveys
        if self.log_transform_stdevs:
            nch += len(self.log_transform_stdevs)
        if self.use_clahe:
            nch += 1
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
        inputs = [backgrounds]

        if self.include_original:
            inputs.insert(0, raw_images)  # add extra dim for 5d input

        if self.use_deconv_channel:
            msg = "use_deconv_channel specified but deconvolution not present in data"
            assert "deconvolution" in batch, msg
            inputs.append(batch["deconvolution"][:, self.bands].unsqueeze(2))

        if self.concat_psf_params:
            msg = "concat_psf_params specified but psf params not present in data"
            assert "psf_params" in batch, msg
            n, c, i, h, w = raw_images.shape
            psf_params = batch["psf_params"][:, self.bands]
            inputs.append(psf_params.view(n, c, 6 * i, 1, 1).expand(n, c, 6 * i, h, w))

        for threshold in self.log_transform_stdevs:
            image_offsets = (raw_images - backgrounds) / backgrounds.sqrt() - threshold
            transformed_img = torch.log(torch.clamp(image_offsets + 1.0, min=1.0))
            inputs.append(transformed_img)

        if self.use_clahe:
            renormalized_img = self.clahe(raw_images, 9, 200, 4)
            inputs.append(renormalized_img)
            inputs[0] = self.clahe(backgrounds, 9, 200, 4)

        return torch.cat(inputs, dim=2)

    @classmethod
    def clahe(cls, imgs, s, c, p):
        """Perform Contrast Limited Adaptive Histogram Equalization (CLAHE) on input images."""
        imgs4d = torch.squeeze(imgs, dim=2)
        padding = (p, p, p, p)
        orig_shape = imgs4d.shape

        # Padding for borders in image
        pad_images = torch.nn.functional.pad(imgs4d, pad=padding, mode="reflect")
        # Unfold image, compute means
        f = torch.nn.Unfold(kernel_size=(s, s), padding=0, stride=1)
        out = f(pad_images)
        reshape_val = int(out.shape[1] / orig_shape[1])
        out = torch.reshape(
            out, (orig_shape[0], orig_shape[1], reshape_val, orig_shape[2], orig_shape[3])
        )
        # Compute residuals
        res_img = imgs4d - torch.mean(out, dim=2)
        # Pad residuals, compute squared residuals
        pad_res_img = torch.nn.functional.pad(res_img, pad=padding, mode="reflect")
        # Unfold squared residuals
        sqr_res = f(pad_res_img**2)
        reshape_sqr_res = torch.reshape(
            sqr_res, (orig_shape[0], orig_shape[1], reshape_val, orig_shape[2], orig_shape[3])
        )
        # Find rolling std
        std = torch.sqrt(torch.mean(reshape_sqr_res, dim=2))
        # Output rolling z-score
        rolling_z = res_img / torch.clamp(std, min=c)
        return torch.unsqueeze(rolling_z, dim=2)
