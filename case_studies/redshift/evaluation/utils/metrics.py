import torch

from bliss.encoder.metrics import CatFilter


def _plocs_are_out_boundary(plocs: torch.Tensor, tile_slen: int, boundary_width: float):
    plocs_in_tile = plocs % tile_slen
    return torch.all(
        (plocs_in_tile >= boundary_width) & (plocs_in_tile <= (tile_slen - boundary_width)), dim=-1
    )


def _is_in_boundary(
    plocs: torch.Tensor, is_on_mask: torch.Tensor, tile_slen: int, boundary_width: float
):
    plocs_out_boundary = _plocs_are_out_boundary(
        plocs, tile_slen=tile_slen, boundary_width=boundary_width
    )
    plocs_in_boundary = ~plocs_out_boundary
    plocs_in_boundary &= is_on_mask

    return plocs_in_boundary


def _is_not_in_boundary(
    plocs: torch.Tensor, is_on_mask: torch.Tensor, tile_slen: int, boundary_width: float
):
    plocs_out_boundary = _plocs_are_out_boundary(
        plocs, tile_slen=tile_slen, boundary_width=boundary_width
    )
    plocs_out_boundary &= is_on_mask

    return plocs_out_boundary


class InBoundaryFilter(CatFilter):
    def __init__(self, tile_slen: int, boundary_width: float) -> None:
        super().__init__()
        self.tile_slen = tile_slen
        self.boundary_width = boundary_width

    def get_cur_filter_bools(self, true_cat, est_cat):
        true_filter_bools = _is_in_boundary(
            true_cat["plocs"], true_cat.is_on_mask, self.tile_slen, self.boundary_width
        )
        est_filter_bools = _is_in_boundary(
            est_cat["plocs"], est_cat.is_on_mask, self.tile_slen, self.boundary_width
        )

        return true_filter_bools, est_filter_bools

    def get_cur_postfix(self):
        return "in_boundary"


class OutBoundaryFilter(CatFilter):
    def __init__(self, tile_slen: int, boundary_width: float) -> None:
        super().__init__()
        self.tile_slen = tile_slen
        self.boundary_width = boundary_width

    def get_cur_filter_bools(self, true_cat, est_cat):
        true_filter_bools = _is_not_in_boundary(
            true_cat["plocs"], true_cat.is_on_mask, self.tile_slen, self.boundary_width
        )
        est_filter_bools = _is_not_in_boundary(
            est_cat["plocs"], est_cat.is_on_mask, self.tile_slen, self.boundary_width
        )

        return true_filter_bools, est_filter_bools

    def get_cur_postfix(self):
        return "out_boundary"
