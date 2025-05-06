import torch
from einops import rearrange


def compute_weighted_avg_ellip(tile_cat, kernel_size, kernel_sigma):
    # construct gaussian kernel (https://github.com/cubiq/ComfyUI_essentials/issues/41)
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    x = torch.exp(-(x**2) / (2 * kernel_sigma**2))
    kernel1d = x / x.sum()
    kernel = rearrange(kernel1d.unsqueeze(-1) * kernel1d.unsqueeze(0), "h w -> 1 1 h w")

    e1_lsst_sum = rearrange(tile_cat["ellip1_lsst_sum"], "nth ntw 1 -> 1 1 nth ntw")
    e1_lsst_count = rearrange(tile_cat["ellip1_lsst_count"], "nth ntw 1 -> 1 1 nth ntw")
    e2_lsst_sum = rearrange(tile_cat["ellip2_lsst_sum"], "nth ntw 1 -> 1 1 nth ntw")
    e2_lsst_count = rearrange(tile_cat["ellip2_lsst_count"], "nth ntw 1 -> 1 1 nth ntw")

    e1_lsst_wavg = (
        (torch.nn.functional.conv2d(e1_lsst_sum, kernel, padding=kernel_size // 2))
        / (torch.nn.functional.conv2d(e1_lsst_count, kernel, padding=kernel_size // 2))
    ).squeeze([0, 1])
    e2_lsst_wavg = (
        (torch.nn.functional.conv2d(e2_lsst_sum, kernel, padding=kernel_size // 2))
        / (torch.nn.functional.conv2d(e2_lsst_count, kernel, padding=kernel_size // 2))
    ).squeeze([0, 1])

    return torch.stack((e1_lsst_wavg, e2_lsst_wavg), dim=-1)
