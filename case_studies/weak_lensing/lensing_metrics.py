import torch
from torchmetrics import Metric


class LensingMapMSE(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("shear1_sum_squared_err", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("shear2_sum_squared_err", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("convergence_sum_squared_err", default=torch.zeros(1), dist_reduce_fx="sum")
        # potentially throws a division by zero error if true_idx is empty and uncaught
        self.add_state("total", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching) -> None:
        for i, match in enumerate(matching):
            true_idx, est_idx = match
            true_shear = true_cat["shear"][i, true_idx, :].flatten(end_dim=-2)
            pred_shear = est_cat["shear"][i, est_idx, :].flatten(end_dim=-2)
            true_convergence = true_cat["convergence"][i, true_idx, :].flatten()
            pred_convergence = est_cat["convergence"][i, est_idx, :].flatten()

            shear1_sq_err = ((true_shear[:, 0].flatten() - pred_shear[:, 0].flatten()) ** 2).sum()
            shear2_sq_err = ((true_shear[:, 1].flatten() - pred_shear[:, 1].flatten()) ** 2).sum()
            convergence_sq_err = ((true_convergence - pred_convergence) ** 2).sum()

            self.total += true_idx.size(0)
            self.shear1_sum_squared_err += shear1_sq_err
            self.shear2_sum_squared_err += shear2_sq_err
            self.convergence_sum_squared_err += convergence_sq_err

    def compute(self):
        shear1_mse = self.shear1_sum_squared_err / self.total
        shear2_mse = self.shear2_sum_squared_err / self.total
        convergence_mse = self.convergence_sum_squared_err / self.total

        return {
            "Shear 1 MSE": shear1_mse,
            "Shear 2 MSE": shear2_mse,
            "Convergence MSE": convergence_mse,
        }
