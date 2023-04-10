import torch
from hydra.utils import instantiate


def prepare_image(x):
    x = torch.from_numpy(x).cuda().unsqueeze(0)
    # image dimensions must be a multiple of 16
    height = x.size(2) - (x.size(2) % 16)
    width = x.size(3) - (x.size(3) % 16)
    return x[:, :, :height, :width]


def predict(cfg):
    encoder = instantiate(cfg.encoder).cuda()
    enc_state_dict = torch.load(cfg.training.weight_save_path)
    encoder.load_state_dict(enc_state_dict)
    encoder.eval()

    sdss = instantiate(cfg.predict.dataset)
    batch = {
        "images": prepare_image(sdss[0]["image"]),
        "background": prepare_image(sdss[0]["background"]),
    }

    with torch.no_grad():
        pred = encoder.encode_batch(batch)
        est_cat = encoder.variational_mode(pred)

    # TODO: plot the original image, the reconstruction, and the residual image using plotly's
    # interactive (zoomable) plots
    # TODO: display accuracy metrics if ground truth, or a proxy for ground truth, is available
    print("{} light sources detected".format(est_cat.n_sources.item()))
