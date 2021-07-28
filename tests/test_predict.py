from bliss import predict


def test_predict_run(devices, get_config):
    overrides = {
        "mode": "predict",
        "predict": "sdss",
        "predict.output_file": "null",
        "predict.device": f"cuda:{devices.device.index}" if devices.use_cuda else "cpu",
        "predict.testing": True,
    }
    cfg = get_config(overrides, devices)
    predict.predict(cfg)
