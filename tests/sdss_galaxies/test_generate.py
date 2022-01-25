import os

import pytest

from bliss import generate


class TestGenerate:
    @pytest.fixture(scope="class")
    def overrides(self, devices, paths):
        return {
            "generate.dataset": "${datasets.simulated}",
            "datasets.simulated.generate_device": "cuda:0" if devices.use_cuda else "cpu",
            "generate.file": f"{paths['root']}/example.pt",
            "generate.common": ["background", "slen"],
        }

    def test_generate_run(self, devices, paths, overrides, get_config):
        cfg = get_config(overrides, devices)
        generate.generate(cfg)

        os.remove(f"{paths['root']}/example.pt")
        os.remove(f"{paths['root']}/example_images.pdf")
