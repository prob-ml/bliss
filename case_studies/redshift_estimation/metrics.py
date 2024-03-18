from torchmetrics import Metric

import torch

class RedshiftMeanSquaredError(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.add_state("preds", default=torch.tensor([]), dist_reduce_fx="cat")
        # self.add_state("truth", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("sum_squared_error", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total",  default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            self.total += tcat_matches.size(0)

            true_red = true_cat["redshift"].loc[i][tcat_matches]
            est_red = est_cat["redshift"].loc[i][ecat_matches]
            red_err = ((true_red - est_red).abs()** 2).sum()
            
            self.sum_squared_err += red_err

    def compute(self):
        mse = self.sum_squared_err / self.total
        return {"Mean squared error": mse.item()}



class RedshiftCatastrophicErrorRate(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.add_state("catastrophic_errors", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            self.total += tcat_matches.size(0)

            est_miu = est_cat["redshift"].loc[i][ecat_matches]
            est_sigma = est_cat["redshift"].scale[i][ecat_matches]
            true_miu = true_cat["redshift"].loc[i][tcat_matches]
            temp = abs(true_miu - est_miu ) > 5 * est_sigma
            self.catastrophic_errors += temp.sum()
        
            
    def compute(self):
        csr = self.catastrophic_errors.float() / self.total
        return {"Catastrophic error rate": csr.item()}
    


class RedshiftNLL(Metric):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        
        # self.add_state("preds_loc", default=torch.tensor([]), dist_reduce_fx="cat")
        # self.add_state("preds_scale", default=torch.tensor([]), dist_reduce_fx="cat")
        # self.add_state("truth", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("negative_loglikelihood", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, true_cat, est_cat, matching):
        for i in range(true_cat.batch_size):
            tcat_matches, ecat_matches = matching[i]
            self.total += tcat_matches.size(0)

            est_miu = est_cat["redshift"].loc[i][ecat_matches]
            est_sigma = est_cat["redshift"].scale[i][ecat_matches]
            true_miu = true_cat["redshift"].loc[i][tcat_matches]
            distribution = torch.distributions.Normal(est_miu, est_sigma)
            nll = -distribution.log_prob(true_miu)
            self.negative_loglikelihood += sum(nll)
            
    
    def compute(self):

        nll = self.negative_loglikelihood / self.total
        return {"negative loglikelihood": nll.item()}
    
    
# preds = torch.tensor([0.9924, 0.9927, 0.9909,  0.9837, 0.9853, 0.9877])
# preds_scale = torch.tensor([0.0416, 0.0426, 0.0425,  0.0421, 0.0418, 0.0419])
# truth = torch.tensor([0.9955, 0.9297, 0.949,  0.9387, 0.9852, 0.9857])

# mse_metric = RedshiftMeanSquaredError()
# cer_metric = RedshiftCatastrophicErrorRate(threshold=0.01)  # Set your threshold for catastrophic errors
# nll_metric = RedshiftNLL()
# # Update the metrics with the prediction and truth tensors
# mse_metric.update(preds, truth)
# cer_metric.update(preds, truth)
# nll_metric.update(preds, preds_scale, truth)

# # Compute the metrics
# mse = mse_metric.compute()
# cer = cer_metric.compute()
# nll = nll_metric.compute()
# print(f"Mean Squared Error: {mse.item()}")
# print(f"Catastrophic Error Rate: {cer.item()}")
# print(f"Negative log likelihood: {nll.item()}")