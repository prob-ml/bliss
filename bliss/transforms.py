import torch


def rolling_z_score(imgs, s, c, p):
    """Perform a rolling z_score transform on input images."""
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


def pixelwise_norm(img, bg):
    """Perform pixel-wise normalization of foreground via background."""
    return (img - bg) / torch.sqrt(bg)


def pixelwise_norm_source(img):
    """Perform pixel-wise normalization of full image."""
    return img / torch.sqrt(img)


def log_transform(img):
    """Perform pixel-wise log transformation of full image."""
    return torch.log(img)
