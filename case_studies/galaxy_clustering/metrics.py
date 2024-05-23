import torch
from torchmetrics import Metric


class ClusterMembershipAccuracy(Metric):
    def __init__(self):
        super().__init__()

        self.add_state("membership_tp", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("membership_tn", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("membership_fp", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("membership_fn", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("n_matches", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            self.n_matches += tcat_matches.size(0)

            true_membership = true_cat["membership"][i][tcat_matches]
            est_membership = est_cat["membership"][i][ecat_matches]

            self.membership_tp += (true_membership * est_membership).sum()
            self.membership_tn += (~true_membership * ~est_membership).sum()
            self.membership_fp += (~true_membership * est_membership).sum()
            self.membership_fn += (true_membership * ~est_membership).sum()

    def compute(self):
        precision = self.membership_tp / (self.membership_tp + self.membership_fp)
        recall = self.membership_tp / (self.membership_tp + self.membership_fn)
        accuracy = (self.membership_tp + self.membership_tn) / (
            self.membership_tp + self.membership_tn + self.membership_fp + self.membership_fn
        )
        f1 = 2 * precision * recall / (precision + recall)
        return {
            "membership_accuracy": accuracy,
            "membership_precision": precision,
            "membership_recall": recall,
            "membership_f1": f1,
        }
