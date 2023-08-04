import cv2 as cv
import numpy as np
import torch


def z_score(imgs):
    """Perform z_score standardization on input images."""
    mu = torch.mean(imgs, dim=(3, 4), keepdim=True)
    sig = torch.std(imgs, dim=(3, 4), keepdim=True)
    return (imgs - mu) / sig


def clahe(imgs):
    """Perform contrast-limited adaptive histogram equalization on images."""
    clahe_imgs = torch.zeros_like(imgs)
    clahe = cv.createCLAHE(climLimit=2.0, tileGridSize=(10, 10))  # pylint: disable=E1101
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[1]):
            np_img = imgs[i, j, 0].cpu().numpy()
            np_img /= np_img.max()
            rescaled_np_img = (np_img * 255).astype(np.uint8)
            clahe_imgs[i, j, 0] = clahe.apply(rescaled_np_img)
    return torch.from_numpy(clahe_imgs.to("cuda:0"))


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


def tanh_mod(img, scale, shift):
    """Perform pixel-wise hyperbolic tangent of full image."""
    return scale * torch.nn.Tanh(img + shift)  # pylint: disable=E1121
