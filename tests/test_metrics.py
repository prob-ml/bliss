import torch

from bliss.catalog import FullCatalog
from bliss.reporting import compute_batch_tp_fp


def test_metrics():
    slen = 50
    slack = 1.0

    true_locs = torch.tensor(
        [[[0.5, 0.5], [0.3, 0.3], [0.0, 0.0]], [[0.1, 0.1], [0.2, 0.2], [0.0, 0.0]]]
    ).reshape(2, 3, 2)
    est_locs = torch.tensor(
        [[[0.49, 0.49], [0.1, 0.1], [0.29, 0.29]], [[0.19, 0.19], [0.01, 0.01], [0.0, 0.0]]]
    ).reshape(2, 3, 2)
    true_galaxy_bools = torch.tensor([[1, 0, 0], [1, 1, 0]]).reshape(2, 3, 1)
    est_galaxy_bools = torch.tensor([[0, 1, 0], [1, 0, 0]]).reshape(2, 3, 1)

    true_params = FullCatalog(
        slen,
        slen,
        {
            "n_sources": torch.tensor([2, 2]),
            "plocs": true_locs * slen,
            "galaxy_bools": true_galaxy_bools,
        },
    )
    est_params = FullCatalog(
        slen,
        slen,
        {
            "n_sources": torch.tensor([3, 2]),
            "plocs": est_locs * slen,
            "galaxy_bools": est_galaxy_bools,
        },
    )

    tp, fp, ntrue = compute_batch_tp_fp(true_params, est_params, slack=slack)
    precision = tp.sum() / (tp.sum() + fp.sum())
    recall = tp.sum() / ntrue.sum()

    assert precision == 3 / (3 + 2)
    assert recall == 3 / 4
