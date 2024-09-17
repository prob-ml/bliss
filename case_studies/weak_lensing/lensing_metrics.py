import torch
from torchmetrics import Metric


class LensingMapMSE(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("shear1_sum_squared_err", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state(
            "baseline_shear1_sum_squared_err", default=torch.zeros(1), dist_reduce_fx="sum"
        )
        self.add_state("shear2_sum_squared_err", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state(
            "baseline_shear2_sum_squared_err", default=torch.zeros(1), dist_reduce_fx="sum"
        )
        self.add_state("convergence_sum_squared_err", default=torch.zeros(1), dist_reduce_fx="sum")
        # potentially throws a division by zero error if true_idx is empty and uncaught
        self.total = 1

    def update(self, true_cat, est_cat, matching) -> None:
        true_shear_1 = true_cat["shear_1"]
        true_shear_2 = true_cat["shear_2"]
        pred_shear_1  = est_cat["shear_1"]
        pred_shear_2 = est_cat["shear_2"]
        true_shear = torch.cat((true_shear_1, true_shear_2), dim=-1)
        pred_shear = torch.cat((pred_shear_1, pred_shear_2), dim=-1)
        true_shear = true_shear.flatten(1, 2)
        pred_shear = pred_shear.flatten(1, 2)
        baseline_pred_shear = true_cat["ellip_lensed"].flatten(1, 2)
        if "convergence" not in est_cat:
            true_convergence = torch.zeros_like(true_shear_1).flatten(1, 2)
            pred_convergence = torch.zeros_like(true_convergence).flatten(1, 2)
        else:
            true_convergence = true_cat["convergence"].flatten(1, 2)
            pred_convergence = est_cat["convergence"].flatten(1, 2)

        shear1_sq_err = ((true_shear[:, :, 0] - pred_shear[:, :, 0]) ** 2).sum()
        baseline_shear1_sq_err = ((true_shear[:, :, 0] - baseline_pred_shear[:, :, 0]) ** 2).sum()
        shear2_sq_err = ((true_shear[:, :, 1] - pred_shear[:, :, 1]) ** 2).sum()
        baseline_shear2_sq_err = ((true_shear[:, :, 1] - baseline_pred_shear[:, :, 1]) ** 2).sum()
        convergence_sq_err = ((true_convergence - pred_convergence) ** 2).sum()

        self.shear1_sum_squared_err += shear1_sq_err
        self.baseline_shear1_sum_squared_err += baseline_shear1_sq_err
        self.shear2_sum_squared_err += shear2_sq_err
        self.baseline_shear2_sum_squared_err += baseline_shear2_sq_err
        self.convergence_sum_squared_err += convergence_sq_err

        self.total = torch.tensor(true_convergence.shape[1])

    def compute(self):
        shear1_mse = self.shear1_sum_squared_err / self.total
        baseline_shear1_mse = self.baseline_shear1_sum_squared_err / self.total
        shear2_mse = self.shear2_sum_squared_err / self.total
        baseline_shear2_mse = self.baseline_shear2_sum_squared_err / self.total
        convergence_mse = self.convergence_sum_squared_err / self.total

        return {
            "Shear 1 MSE": shear1_mse,
            "Baseline shear 1 MSE": baseline_shear1_mse,
            "Shear 2 MSE": shear2_mse,
            "Baseline shear 2 MSE": baseline_shear2_mse,
            "Convergence MSE": convergence_mse,
        }
