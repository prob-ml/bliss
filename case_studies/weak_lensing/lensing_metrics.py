from torchmetrics import Metric, PearsonCorrCoef
import torch

class LensingMapCorrCoef(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds_shear", default=torch.Tensor(), dist_reduce_fx="cat")
        self.add_state("target_shear", default=torch.Tensor(), dist_reduce_fx="cat")
        self.add_state("preds_convergence", default=torch.Tensor(), dist_reduce_fx="cat")
        self.add_state("target_convergence", default=torch.Tensor(), dist_reduce_fx="cat")

        self.compute_corrcoef = PearsonCorrCoef()
    
    def update(self, true_cat, est_cat, matching) -> None:
        for i in range(len(matching)):
            true_idx, est_idx = matching[i]

            self.target_shear = torch.cat((self.target_shear, true_cat["shear"][i,true_idx,:].flatten(end_dim=-2)),)
            self.preds_shear = torch.cat((self.preds_shear, est_cat["shear"][i, est_idx, :].flatten(end_dim=-2)),)

            self.target_convergence = torch.cat((self.target_convergence, true_cat["convergence"][i,true_idx,:].flatten()),)
            self.preds_convergence = torch.cat((self.preds_convergence, est_cat["convergence"][i,est_idx,:].flatten()),)

    def compute(self,):
        posterior_shear1 = self.preds_shear[:,0]
        posterior_shear2 = self.preds_shear[:,1]
        posterior_convergence = self.preds_convergence

        true_shear1 = self.target_shear[:,0]
        true_shear2 = self.target_shear[:,1]
        true_convergence = self.target_convergence

        shear_1_corr_coef = self.compute_corrcoef(posterior_shear1, true_shear1)
        shear_2_corr_coef = self.compute_corrcoef(posterior_shear2, true_shear2)
        convergence_corr_coef = self.compute_corrcoef(posterior_convergence, true_convergence)

        return {
            "Shear 1 corr_coef" : shear_1_corr_coef,
            "Shear 2 corr_coef" : shear_2_corr_coef,
            "Convergence corr_coef" : convergence_corr_coef
        }

class LensingMapRMSE(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds_shear", default=torch.Tensor(), dist_reduce_fx="cat")
        self.add_state("target_shear", default=torch.Tensor(), dist_reduce_fx="cat")
        self.add_state("preds_convergence", default=torch.Tensor(), dist_reduce_fx="cat")
        self.add_state("target_convergence", default=torch.Tensor(), dist_reduce_fx="cat")

    def update(self, true_cat, est_cat, matching) -> None:
        for i in range(len(matching)):
            true_idx, est_idx = matching[i]

            self.target_shear = torch.cat((self.target_shear, true_cat["shear"][i,true_idx,:].flatten(end_dim=-2)),)
            self.preds_shear = torch.cat((self.preds_shear, est_cat["shear"][i, est_idx, :].flatten(end_dim=-2)),)

            self.target_convergence = torch.cat((self.target_convergence, true_cat["convergence"][i,true_idx,:].flatten()),)
            self.preds_convergence = torch.cat((self.preds_convergence, est_cat["convergence"][i,est_idx,:].flatten()),)

    def compute(self,):
        posterior_shear1 = self.preds_shear[:,0]
        posterior_shear2 = self.preds_shear[:,1]
        posterior_convergence = self.preds_convergence

        true_shear1 = self.target_shear[:,0]
        true_shear2 = self.target_shear[:,1]
        true_convergence = self.target_convergence

        shear_1_RMSE = torch.sqrt(torch.mean((true_shear1.flatten()-posterior_shear1.flatten())**2))
        shear_2_RMSE = torch.sqrt(torch.mean((true_shear2.flatten()-posterior_shear2.flatten())**2))
        convergence_RMSE = torch.sqrt(torch.mean((true_convergence-posterior_convergence)**2))

        return {
            "Shear 1 RMSE" : shear_1_RMSE,
            "Shear 2 RMSE" : shear_2_RMSE,
            "Convergence RMSE" : convergence_RMSE
        }
