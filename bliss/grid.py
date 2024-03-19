import torch
from einops import pack


def get_mgrid(slen: int, device: torch.device):
    assert slen >= 3 and slen % 2 == 1
    offset = (slen - 1) // 2
    offsets = torch.arange(-offset, offset + 1, 1, device=device)
    x, y = torch.meshgrid(offsets, offsets, indexing="ij")  # same as numpy default indexing.
    mgrid_not_normalized, _ = pack([y, x], "h w *")
    # normalize to (-1, 1) and scale slightly because of the way f.grid_sample
    # parameterizes the edges: (0, 0) is center of edge pixel
    return (mgrid_not_normalized / offset).float() * (slen - 1) / slen
