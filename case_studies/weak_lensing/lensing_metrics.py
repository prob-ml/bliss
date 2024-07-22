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
        # along dim 2
        true_shear = true_cat["shear"]
        pred_shear = est_cat["shear"]

        true_convergence = true_cat["convergence"]
        pred_convergence = est_cat["convergence"]

        shear1_sq_err = ((true_shear[:, :, 0].flatten() - pred_shear[:, :, 0].flatten()) ** 2).sum()
        shear2_sq_err = ((true_shear[:, :, 1].flatten() - pred_shear[:, :, 1].flatten()) ** 2).sum()

        true_convergence = true_cat["convergence"].flatten()
        pred_convergence = est_cat["convergence"].flatten()
        convergence_sq_err = ((true_convergence - pred_convergence) ** 2).sum()

        self.shear1_sum_squared_err += shear1_sq_err
        self.shear2_sum_squared_err += shear2_sq_err
        self.convergence_sum_squared_err += convergence_sq_err

        self.total = torch.tensor(  # pylint: disable=attribute-defined-outside-init
            true_cat["shear"].shape[1]
        )

    def compute(self):
        shear1_mse = self.shear1_sum_squared_err / self.total
        shear2_mse = self.shear2_sum_squared_err / self.total
        convergence_mse = self.convergence_sum_squared_err / self.total

        return {
            "Shear 1 MSE": shear1_mse,
            "Shear 2 MSE": shear2_mse,
            "Convergence MSE": convergence_mse,
        }
