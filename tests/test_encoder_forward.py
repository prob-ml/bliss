"""Test that we can run one loop of encoder training."""

import torch
from astropy.table import Table
from torch.utils.data import DataLoader

from bliss.datasets.galsim_blends import SavedGalsimBlends, generate_dataset, parse_dataset
from bliss.datasets.lsst import get_default_lsst_psf
from bliss.datasets.table_utils import column_to_tensor
from bliss.encoders.binary import BinaryEncoder
from bliss.encoders.deblend import GalaxyEncoder
from bliss.encoders.detection import DetectionEncoder


def test_encoder_forward(home_dir, tmp_path):
    ae_state_dict = home_dir / "case_studies" / "galsim_galaxies" / "models" / "autoencoder.pt"

    catsim_table = Table.read(home_dir / "data" / "OneDegSq.fits")
    all_star_mags = column_to_tensor(
        Table.read(home_dir / "data" / "stars_med_june2018.fits"), "i_ab"
    )
    psf = get_default_lsst_psf()
    blends_ds = generate_dataset(32, catsim_table, all_star_mags, psf, 10)

    saved_ds_path = tmp_path / "train_ds.pt"
    torch.save(blends_ds, saved_ds_path)
    saved_ds1 = SavedGalsimBlends(saved_ds_path, 32, keep_padding=False)
    saved_ds2 = SavedGalsimBlends(saved_ds_path, 32, keep_padding=True)

    dl1 = DataLoader(saved_ds1, batch_size=32, num_workers=0)
    dl2 = DataLoader(saved_ds2, batch_size=32, num_workers=0)

    binary_encoder = BinaryEncoder()
    detection_encoder = DetectionEncoder()
    galaxy_encoder = GalaxyEncoder(ae_state_dict)

    with torch.no_grad():
        for b in dl1:
            im, bg, tc, _ = parse_dataset(b)
            binary_encoder.get_loss(im, bg, tc)
            detection_encoder.get_loss(im, bg, tc)

        for b in dl2:
            im, bg, tc, pd = parse_dataset(b)
            galaxy_encoder.get_loss(im, pd, bg, tc)
