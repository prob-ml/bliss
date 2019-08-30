import time
import numpy as np

import hungarian_alg

from scipy.optimize import linear_sum_assignment

import torch

batchsize = 2048
dim = 20
X = torch.randn(batchsize, dim, dim)

true_n_stars = torch.randint(0, dim+1, (batchsize, ))

t0 = time.time()
perm_array1 = hungarian_alg.run_batch_hungarian_alg(X, true_n_stars)
print('non-parallel time: {0:.05f}sec'.format(time.time() - t0))

t0 = time.time()
perm_array2 = hungarian_alg.run_batch_hungarian_alg_parallel(X, true_n_stars)
print('parallel time: {0:.05f}sec'.format(time.time() - t0))


assert torch.all(perm_array1 == perm_array2)

# def run_batch_hungarian_alg(log_probs_all, n_stars):
#     # log_probs_all should be a tensor of size
#     # (batchsize x estimated_param x true param)
#     # giving for each N, the log prob of the estimated parameter
#     # against the target parameter
#
#     # this finds the MAXIMAL permutation of log_probs_all
#
#     batchsize = log_probs_all.shape[0]
#     max_detections = log_probs_all.shape[1]
#     perm = np.zeros((batchsize,max_detections))
#
#     # This is done in numpy ...
#     log_probs_all_np = log_probs_all.to('cpu').detach().numpy()
#
#     for i in range(batchsize):
#         n_stars_i = int(n_stars[i])
#         row_indx, col_indx = linear_sum_assignment(\
#                                 -log_probs_all_np[i, 0:n_stars_i, 0:n_stars_i])
#
#         perm[i, :] = np.concatenate((col_indx, np.arange(n_stars_i, max_detections)))
#
#     return torch.LongTensor(perm)
#
# t0 = time.time()
# run_batch_hungarian_alg(X, true_n_stars)
# print('old algorithm time: {0:.05f}sec'.format(time.time() - t0))
