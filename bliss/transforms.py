import torch


def z_score(imgs):
    """Perform z_score standardization on input images."""
    mu = torch.mean(imgs, dim=(2, 3), keepdim=True)
    sig = torch.std(imgs, dim=(2, 3), keepdim=True)
    return (imgs - mu) / sig


def pixelwise_norm(img, bg):
    """Perform pixel-wise normalization of foreground via background."""
    return (img - bg) / torch.sqrt(bg)


def pixelwise_norm_source(img):
    """Perform pixel-wise normalization of full image."""
    return img / torch.sqrt(img)


def log_transform(img):
    """Perform pixel-wise log transformation of full image."""
    return torch.log(img)
