import torch

class RandomCropTransform(torch.nn.Module):
    def __init__(self, box_slen: int, boundary_pad: int):
        super().__init__()
        self.box_slen = box_slen
        self.boundary_pad = boundary_pad
        assert self.boundary_pad >= 1

    def __call__(self, datum_in):
        datum_out = {k: v for k, v in datum_in.items() if k != "full_catalog"}

        h_pixels, w_pixels = datum_in["images"].shape[1:]
        assert h_pixels == w_pixels
        box_origin = torch.randint(low=self.boundary_pad - 1, 
                                    high=h_pixels - self.boundary_pad - self.box_slen,
                                    size=(1, 2))
        box_end = box_origin + self.box_slen
        assert (box_end < h_pixels).all()
        full_cat_dict = datum_in["full_catalog"]
        plocs_mask = torch.all(
            (full_cat_dict["plocs"] < box_end) & \
            (full_cat_dict["plocs"] > box_origin), 
            dim=1
        )
        new_full_cat_dict = {}
        for k, v in full_cat_dict.items():
            if k == "n_sources":
                new_full_cat_dict[k] = plocs_mask.sum().to(dtype=full_cat_dict["n_sources"].dtype)
                continue
            new_full_cat_dict[k] = v[plocs_mask]
            if k == "plocs":
                new_full_cat_dict[k] -= box_origin
        datum_out["images"] = datum_out["images"][:, 
                                                  box_origin[0, 0].item():box_end[0, 0].item(), 
                                                  box_origin[0, 1].item():box_end[0, 1].item()]
        
        datum_out["full_catalog"] = new_full_cat_dict
        return datum_out
