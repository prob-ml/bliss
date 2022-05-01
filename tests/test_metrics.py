import torch
from sklearn.metrics import confusion_matrix

from bliss.catalog import FullCatalog
from bliss.reporting import ClassificationMetrics, DetectionMetrics, match_by_locs_kdtree


def kdtree_test(true_loc, est_loc, true_bool, est_bool, slen, slack):

    tp_kdtree = torch.tensor(0)
    fp_kdtree = torch.tensor(0)
    n_source_tru = torch.tensor([1, 2])
    n_source_est = torch.tensor([2, 2])
    plocs_tru = true_loc * slen
    plocs_est = est_loc * slen
    total_true_n_sources = torch.tensor(0)
    avg_distance_kdtree = torch.tensor(0).float()
    total_correct_class = torch.tensor(0)
    total_n_matches = torch.tensor(0)
    conf_matrix = torch.tensor([[0, 0], [0, 0]])

    count = 0
    for b in range(2):
        ntrue, nest = n_source_tru[b].int().item(), n_source_est[b].int().item()
        tlocs, elocs = plocs_tru[b], plocs_est[b]
        tgbool, egbool = true_bool[b].reshape(-1), est_bool[b].reshape(-1)
        if ntrue > 0 and nest > 0:
            mtrue, mest, dkeep, avg_distance = match_by_locs_kdtree(tlocs, elocs, slack)
            tp = len(elocs[mest][dkeep])
            fp = nest - tp
            assert fp >= 0
            tp_kdtree += tp
            fp_kdtree += fp
            avg_distance_kdtree += avg_distance
            total_true_n_sources += ntrue
            count += 1
            tgbool = tgbool[mtrue][dkeep].reshape(-1)
            egbool = egbool[mest][dkeep].reshape(-1)
            total_n_matches += len(egbool)
            total_correct_class += tgbool.eq(egbool).sum().int()
            conf_matrix += confusion_matrix(tgbool, egbool, labels=[1, 0])

    avg_distance_kdtree /= count
    precision_kdtree = tp_kdtree / (tp_kdtree + fp_kdtree)
    recall = tp_kdtree / total_true_n_sources
    class_acc = total_correct_class / total_n_matches

    return precision_kdtree, recall, class_acc, avg_distance_kdtree, conf_matrix


def test_metrics():
    slen = 50
    slack = 1.0
    detect = DetectionMetrics(slack)
    classify = ClassificationMetrics(slack)

    true_locs = torch.tensor([[[0.5, 0.5], [0.0, 0.0]], [[0.2, 0.2], [0.1, 0.1]]]).reshape(2, 2, 2)
    est_locs = torch.tensor([[[0.49, 0.49], [0.1, 0.1]], [[0.19, 0.19], [0.01, 0.01]]]).reshape(
        2, 2, 2
    )
    true_galaxy_bools = torch.tensor([[1, 0], [1, 1]]).reshape(2, 2, 1)
    est_galaxy_bools = torch.tensor([[0, 1], [1, 0]]).reshape(2, 2, 1)

    true_params = FullCatalog(
        slen,
        slen,
        {
            "n_sources": torch.tensor([1, 2]),
            "plocs": true_locs * slen,
            "galaxy_bools": true_galaxy_bools,
        },
    )
    est_params = FullCatalog(
        slen,
        slen,
        {
            "n_sources": torch.tensor([2, 2]),
            "plocs": est_locs * slen,
            "galaxy_bools": est_galaxy_bools,
        },
    )

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

    # test new methods, test 1
    precision_kdtree, recall, class_acc, avg_distance_kdtree, conf_matrix = kdtree_test(
        true_locs, est_locs, true_galaxy_bools, est_galaxy_bools, slen, slack
    )

    assert precision_kdtree == 2 / (2 + 2)
    assert recall == 2 / 3
    assert class_acc == 1 / 2
    assert conf_matrix.eq(torch.tensor([[1, 1], [0, 0]])).all()
    assert torch.round(avg_distance_kdtree, decimals=3) == 0.707

    # test 2
    true_locs = torch.tensor([[[0.5, 0.5], [0.0, 0.0]], [[0.2, 0.2], [0.1, 0.1]]]).reshape(2, 2, 2)
    est_locs = torch.tensor([[[0.5, 0.45], [0.1, 0.1]], [[0.25, 0.2], [0.1, 0.05]]]).reshape(
        2, 2, 2
    )
    true_galaxy_bools = torch.tensor([[1, 0], [1, 1]]).reshape(2, 2, 1)
    est_galaxy_bools = torch.tensor([[1, 1], [1, 0]]).reshape(2, 2, 1)

    precision_kdtree, recall, class_acc, avg_distance_kdtree, conf_matrix = kdtree_test(
        true_locs, est_locs, true_galaxy_bools, est_galaxy_bools, 1, slack
    )

    assert precision_kdtree == 3 / (2 + 2)
    assert recall == 3 / 3
    assert class_acc == 2 / 3
    assert conf_matrix.eq(torch.tensor([[2, 1], [0, 0]])).all()
    assert torch.round(avg_distance_kdtree, decimals=3) == 0.05
