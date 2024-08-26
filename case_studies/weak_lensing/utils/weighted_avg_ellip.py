import torch
from einops import rearrange


def compute_weighted_avg_ellip(tile_cat, kernel_size, kernel_sigma):
    # construct gaussian kernel (https://github.com/cubiq/ComfyUI_essentials/issues/41)
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    x = torch.exp(-(x**2) / (2 * kernel_sigma**2))
    kernel1d = x / x.sum()
    kernel = rearrange(kernel1d.unsqueeze(-1) * kernel1d.unsqueeze(0), "h w -> 1 1 h w")

    e1_lensed_sum = rearrange(tile_cat["ellip1_lensed_sum"], "nth ntw 1 -> 1 1 nth ntw")
    e1_lensed_count = rearrange(tile_cat["ellip1_lensed_count"], "nth ntw 1 -> 1 1 nth ntw")
    e2_lensed_sum = rearrange(tile_cat["ellip2_lensed_sum"], "nth ntw 1 -> 1 1 nth ntw")
    e2_lensed_count = rearrange(tile_cat["ellip2_lensed_count"], "nth ntw 1 -> 1 1 nth ntw")

    e1_lensed = (
        (torch.nn.functional.conv2d(e1_lensed_sum, kernel, padding=kernel_size // 2))
        / (torch.nn.functional.conv2d(e1_lensed_count, kernel, padding=kernel_size // 2))
    ).squeeze([0, 1])
    e2_lensed = (
        (torch.nn.functional.conv2d(e2_lensed_sum, kernel, padding=kernel_size // 2))
        / (torch.nn.functional.conv2d(e2_lensed_count, kernel, padding=kernel_size // 2))
    ).squeeze([0, 1])

    return torch.stack((e1_lensed, e2_lensed), dim=-1)
