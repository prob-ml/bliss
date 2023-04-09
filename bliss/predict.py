import torch
from hydra.utils import instantiate


def prepare_image(x):
    x = torch.from_numpy(x).cuda().unsqueeze(0)
    return x[:, :, :1488, :720]


def predict(cfg):
    sdss = instantiate(cfg.inference.dataset)
    encoder = instantiate(cfg.encoder).cuda()
    # TODO: load saved weights
    encoder.eval()
    batch = {
        "images": prepare_image(sdss[0]["image"]),
        "background": prepare_image(sdss[0]["background"]),
    }
    pred = encoder.encode_batch(batch)
    est_cat = encoder.variational_mode(pred)
    print(est_cat)
