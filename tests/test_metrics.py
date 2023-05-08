import torch

from bliss.catalog import FullCatalog
from bliss.metrics import BlissMetrics


def test_metrics():
    slen = 50
    slack = 1.0
    bliss_metrics = BlissMetrics(slack)

    true_locs = torch.tensor([[[0.5, 0.5], [0.0, 0.0]], [[0.2, 0.2], [0.1, 0.1]]])
    est_locs = torch.tensor([[[0.49, 0.49], [0.1, 0.1]], [[0.19, 0.19], [0.01, 0.01]]])
    true_galaxy_bools = torch.tensor([[[1], [0]], [[1], [1]]])
    est_galaxy_bools = torch.tensor([[[0], [1]], [[1], [0]]])

    d_true = {
        "n_sources": torch.tensor([1, 2]),
        "plocs": true_locs * slen,
        "galaxy_bools": true_galaxy_bools,
    }
    true_params = FullCatalog(slen, slen, d_true)

    d_est = {
        "n_sources": torch.tensor([2, 2]),
        "plocs": est_locs * slen,
        "galaxy_bools": est_galaxy_bools,
    }
    est_params = FullCatalog(slen, slen, d_est)

    results_metrics = bliss_metrics(true_params, est_params)
    precision = results_metrics["precision"]
    recall = results_metrics["recall"]
    avg_distance = results_metrics["avg_distance"]

    class_acc = results_metrics["class_acc"]
    gal_tp = results_metrics["gal_tp"]
    gal_fp = results_metrics["gal_fp"]
    gal_tn = results_metrics["gal_tn"]
    gal_fn = results_metrics["gal_fn"]

    assert precision == 2 / (2 + 2)
    assert recall == 2 / 3
    assert class_acc == 1 / 2
    assert gal_tp == torch.tensor([1])
    assert gal_fp == torch.tensor([1])
    assert gal_fn == torch.tensor([0])
    assert gal_tn == torch.tensor([0])
    assert avg_distance.item() == 50 * (0.01 + (0.01 + 0.09) / 2) / 2
