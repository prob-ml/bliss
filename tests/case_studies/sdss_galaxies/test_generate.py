from pathlib import Path

from hydra.utils import instantiate

from bliss import generate


class TestGenerate:
    def test_generate_run(self, devices, get_sdss_galaxies_config):
        cfg = get_sdss_galaxies_config({}, devices)
        dataset = instantiate(
            cfg.datasets.simulated,
            generate_device="cuda:0" if devices.use_cuda else "cpu",
        )
        filepath = Path(cfg.paths.root) / "example.pt"
        imagepath = Path(cfg.paths.root) / "example_images.jpg"
        generate.generate(
            dataset, filepath, imagepath, n_plots=25, global_params=("background", "slen")
        )
        filepath.unlink()
        imagepath.unlink()
