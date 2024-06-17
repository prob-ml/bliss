import torch
from torchmetrics import Metric


class RedshiftMeanSquaredError(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_squared_error", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat):
        for i in range(true_cat.batch_size):
            true_red = true_cat["redshifts"][i]
            est_red = est_cat["redshifts"][i]
            red_err = ((true_red - est_red).abs() ** 2).sum()
            self.total += torch.flatten(true_cat["redshifts"][i]).shape[0]

            self.sum_squared_error += red_err

    def compute(self):
        mse = self.sum_squared_error / self.total
        return {"redshifts/mse": mse.item()}
