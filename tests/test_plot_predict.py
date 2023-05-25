from bliss.predict import plot_predict, predict_sdss, prepare_image


def test_plot_predict(cfg):
    est_cat, true_plocs, crop_img, crop_bg = predict_sdss(cfg)
    image = prepare_image(crop_img, cfg.predict.device)
    background = prepare_image(crop_bg, cfg.predict.device)

    ttc = cfg.encoder.tiles_to_crop
    ts = cfg.encoder.tile_slen
    ptc = ttc * ts
    cropped_image = image[0, 0, ptc:-ptc, ptc:-ptc]
    cropped_background = background[0, 0, ptc:-ptc, ptc:-ptc]
    plot_predict(cfg, cropped_image, cropped_background, true_plocs, est_cat)
