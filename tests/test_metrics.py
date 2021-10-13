import torch

from bliss.metrics import ClassificationMetrics, DetectionMetrics


def test_metrics():
    slen = 50
    slack = 1.0
    detect = DetectionMetrics(slack)
    classify = ClassificationMetrics(slack)

    true_locs = torch.tensor([[[0.5, 0.5], [0.0, 0.0]], [[0.2, 0.2], [0.1, 0.1]]]).reshape(2, 2, 2)
    est_locs = torch.tensor([[[0.49, 0.49], [0.1, 0.1]], [[0.19, 0.19], [0.01, 0.01]]]).reshape(
        2, 2, 2
    )
    true_galaxy_bool = torch.tensor([[1, 0], [1, 1]]).reshape(2, 2, 1)
    est_galaxy_bool = torch.tensor([[0, 1], [1, 0]]).reshape(2, 2, 1)

    true_params = {
        "n_sources": torch.tensor([1, 2]),
        "plocs": true_locs * slen,
        "galaxy_bool": true_galaxy_bool,
    }
    est_params = {
        "n_sources": torch.tensor([2, 2]),
        "plocs": est_locs * slen,
        "galaxy_bool": est_galaxy_bool,
    }

    results_detection = detect(true_params, est_params)
    precision = results_detection["precision"]
    recall = results_detection["recall"]
    avg_distance = results_detection["avg_distance"]

    results_classify = classify(true_params, est_params)
    class_acc = results_classify["class_acc"]
    conf_matrix = results_classify["conf_matrix"]

    assert precision == 2 / (2 + 2)
    assert recall == 2 / 3
    assert class_acc == 1 / 2
    assert conf_matrix.eq(torch.tensor([[1, 1], [0, 0]])).all()
    assert avg_distance.item() == 50 * (0.01 + (0.01 + 0.09) / 2) / 2
