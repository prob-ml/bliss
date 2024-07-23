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
        self.add_state(
            "baseline_convergence_sum_squared_err", default=torch.zeros(1), dist_reduce_fx="sum"
        )
        # potentially throws a division by zero error if true_idx is empty and uncaught
        self.total = None

    def update(self, true_cat, est_cat, matching) -> None:
        # along dim 2
        true_shear = true_cat["shear"].flatten(1, 2)
        pred_shear = est_cat["shear"].flatten(1, 2)
        baseline_pred_shear = true_shear.mean(1).unsqueeze(1) * torch.ones_like(true_shear)
        true_convergence = true_cat["convergence"].flatten(1, 2)
        pred_convergence = est_cat["convergence"].flatten(1, 2)
        baseline_pred_convergence = true_convergence.mean(1).unsqueeze(-1) * torch.ones_like(
            true_convergence
        )

        shear1_sq_err = ((true_shear[:, :, 0] - pred_shear[:, :, 0]) ** 2).sum()
        baseline_shear1_sq_err = ((true_shear[:, :, 0] - baseline_pred_shear[:, :, 0]) ** 2).sum()
        shear2_sq_err = ((true_shear[:, :, 1] - pred_shear[:, :, 1]) ** 2).sum()
        baseline_shear2_sq_err = ((true_shear[:, :, 0] - baseline_pred_shear[:, :, 1]) ** 2).sum()
        convergence_sq_err = ((true_convergence - pred_convergence) ** 2).sum()
        baseline_convergence_sq_err = ((true_convergence - baseline_pred_convergence) ** 2).sum()

        self.shear1_sum_squared_err += shear1_sq_err
        self.baseline_shear1_sum_squared_err += baseline_shear1_sq_err
        self.shear2_sum_squared_err += shear2_sq_err
        self.baseline_shear2_sum_squared_err += baseline_shear2_sq_err
        self.convergence_sum_squared_err += convergence_sq_err
        self.baseline_convergence_sum_squared_err += baseline_convergence_sq_err

        self.total = torch.tensor(true_cat["shear"].shape[1])

    def compute(self):
        shear1_mse = self.shear1_sum_squared_err / self.total
        baseline_shear1_mse = self.baseline_shear1_sum_squared_err / self.total
        shear2_mse = self.shear2_sum_squared_err / self.total
        baseline_shear2_mse = self.baseline_shear2_sum_squared_err / self.total
        convergence_mse = self.convergence_sum_squared_err / self.total
        baseline_convergence_mse = self.baseline_convergence_sum_squared_err / self.total

        return {
            "Shear 1 MSE": shear1_mse,
            "Baseline shear 1 MSE": baseline_shear1_mse,
            "Shear 2 MSE": shear2_mse,
            "Baseline shear 2 MSE": baseline_shear2_mse,
            "Convergence MSE": convergence_mse,
            "Baseline convergence MSE": baseline_convergence_mse,
        }
