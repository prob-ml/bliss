import shutil

from bliss.generate import generate
from bliss.predict import predict_sdss
from bliss.train import train


class TestEndToEnd:
    def test_generate(self, cfg):
        generate(cfg)

    def test_train(self, cfg):
        train(cfg)

    def test_predict(self, cfg, tmpdir_factory):
        the_cfg = cfg.copy()
        temp_dir = tmpdir_factory.mktemp("plot")
        temp_file = str(temp_dir) + "/" + the_cfg.predict.plot.out_file_name
        the_cfg.predict.plot.out_file_name = temp_file
        the_cfg.predict.plot.show_plot = True
        predict_sdss(the_cfg)

        # delete the tmpdir
        shutil.rmtree(str(temp_dir))
