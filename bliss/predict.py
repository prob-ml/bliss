import torch
from hydra.utils import instantiate


def prepare_image(x, device):
    x = torch.from_numpy(x).unsqueeze(0)
    x = x.to(device=device)
    # image dimensions must be a multiple of 16
    height = x.size(2) - (x.size(2) % 16)
    width = x.size(3) - (x.size(3) % 16)
    return x[:, :, :height, :width]


def predict(cfg, image, background):
    encoder = instantiate(cfg.encoder).to(cfg.predict.device)
    enc_state_dict = torch.load(cfg.predict.weight_save_path)
    encoder.load_state_dict(enc_state_dict)
    encoder.eval()

    batch = {"images": image, "background": background}

    with torch.no_grad():
        pred = encoder.encode_batch(batch)
        est_cat = encoder.variational_mode(pred)

    print("{} light sources detected".format(est_cat.n_sources.item()))

    return est_cat


def predict_sdss(cfg):
    sdss = instantiate(cfg.predict.dataset)
    idx0, idx1 = cfg.predict.crop[0], cfg.predict.crop[1]

    crop_img = sdss[0]["image"][:, idx0:idx1, idx0:idx1]
    crop_bg = sdss[0]["background"][:, idx0:idx1, idx0:idx1]

    prepare_img = prepare_image(crop_img, cfg.predict.device)
    prepare_bg = prepare_image(crop_bg, cfg.predict.device)

    return predict(cfg, prepare_img, prepare_bg), crop_img, crop_bg
