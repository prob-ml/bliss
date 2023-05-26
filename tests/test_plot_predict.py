from bliss.predict import predict_sdss


def test_plot_predict(cfg):
    predict_sdss(cfg, if_plot=True)
