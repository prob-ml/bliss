from bliss.predict import plot_predict, predict_sdss


def test_plot_predict(cfg):
    est_cat, image, background, true_plocs = predict_sdss(cfg)

    ttc = cfg.encoder.tiles_to_crop
    ts = cfg.encoder.tile_slen
    ptc = ttc * ts
    cropped_image = image[0, ptc:-ptc, ptc:-ptc]
    cropped_background = background[0, ptc:-ptc, ptc:-ptc]
    plot_predict(cfg, cropped_image, cropped_background, true_plocs, est_cat)
