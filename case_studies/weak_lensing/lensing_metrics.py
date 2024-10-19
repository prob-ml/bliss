import torch
from torchmetrics import Metric


class LensingMSE(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("shear1_sum_squared_err", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state(
            "zero_baseline_shear1_sum_squared_err", default=torch.zeros(1), dist_reduce_fx="sum"
        )
        self.add_state(
            "ellip_baseline_shear1_sum_squared_err", default=torch.zeros(1), dist_reduce_fx="sum"
        )
        self.add_state("shear2_sum_squared_err", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state(
            "zero_baseline_shear2_sum_squared_err", default=torch.zeros(1), dist_reduce_fx="sum"
        )
        self.add_state(
            "ellip_baseline_shear2_sum_squared_err", default=torch.zeros(1), dist_reduce_fx="sum"
        )
        self.add_state("convergence_sum_squared_err", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching) -> None:
        true_shear1 = true_cat["shear_1"].flatten(1, 2)
        true_shear2 = true_cat["shear_2"].flatten(1, 2)
        pred_shear1 = est_cat["shear_1"].flatten(1, 2)
        pred_shear2 = est_cat["shear_2"].flatten(1, 2)
        zero_baseline_pred_shear1 = torch.zeros_like(true_shear1)
        zero_baseline_pred_shear2 = torch.zeros_like(true_shear2)
        ellip_baseline_pred_shear1 = (
            true_cat["ellip_lensed_wavg"][..., 0].unsqueeze(-1).flatten(1, 2)
        )
        ellip_baseline_pred_shear2 = (
            true_cat["ellip_lensed_wavg"][..., 1].unsqueeze(-1).flatten(1, 2)
        )

        if "convergence" not in est_cat:
            true_convergence = torch.zeros_like(true_shear1).flatten(1, 2)
            pred_convergence = torch.zeros_like(true_convergence)
        else:
            true_convergence = true_cat["convergence"].flatten(1, 2)
            pred_convergence = est_cat["convergence"].flatten(1, 2)

        shear1_sq_err = ((true_shear1 - pred_shear1) ** 2).sum()
        zero_baseline_shear1_sq_err = ((true_shear1 - zero_baseline_pred_shear1) ** 2).sum()
        ellip_baseline_shear1_sq_err = ((true_shear1 - ellip_baseline_pred_shear1) ** 2).sum()
        shear2_sq_err = ((true_shear2 - pred_shear2) ** 2).sum()
        zero_baseline_shear2_sq_err = ((true_shear2 - zero_baseline_pred_shear2) ** 2).sum()
        ellip_baseline_shear2_sq_err = ((true_shear2 - ellip_baseline_pred_shear2) ** 2).sum()
        convergence_sq_err = ((true_convergence - pred_convergence) ** 2).sum()

        self.shear1_sum_squared_err += shear1_sq_err
        self.zero_baseline_shear1_sum_squared_err += zero_baseline_shear1_sq_err
        self.ellip_baseline_shear1_sum_squared_err += ellip_baseline_shear1_sq_err
        self.shear2_sum_squared_err += shear2_sq_err
        self.zero_baseline_shear2_sum_squared_err += zero_baseline_shear2_sq_err
        self.ellip_baseline_shear2_sum_squared_err += ellip_baseline_shear2_sq_err
        self.convergence_sum_squared_err += convergence_sq_err

        self.total += torch.tensor(true_convergence.shape[1])

    def compute(self):
        shear1_mse = self.shear1_sum_squared_err / self.total
        zero_baseline_shear1_mse = self.zero_baseline_shear1_sum_squared_err / self.total
        ellip_baseline_shear1_mse = self.ellip_baseline_shear1_sum_squared_err / self.total
        shear2_mse = self.shear2_sum_squared_err / self.total
        zero_baseline_shear2_mse = self.zero_baseline_shear2_sum_squared_err / self.total
        ellip_baseline_shear2_mse = self.ellip_baseline_shear2_sum_squared_err / self.total
        convergence_mse = self.convergence_sum_squared_err / self.total

        return {
            "Shear 1 MSE": shear1_mse,
            "Zero baseline shear 1 MSE": zero_baseline_shear1_mse,
            "Ellip baseline shear 1 MSE": ellip_baseline_shear1_mse,
            "Shear 2 MSE": shear2_mse,
            "Zero baseline shear 2 MSE": zero_baseline_shear2_mse,
            "Ellip baseline shear 2 MSE": ellip_baseline_shear2_mse,
            "Convergence MSE": convergence_mse,
        }
