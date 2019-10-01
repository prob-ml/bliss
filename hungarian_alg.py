import torch

from torch import multiprocessing
import ctypes
import numpy as np

from scipy.optimize import linear_sum_assignment

from itertools import permutations

# parallelism adapted from
# https://stackoverflow.com/questions/36596805/python-multiprocessing-claims-too-many-open-files-when-no-files-are-even-opened
def _run_linear_sum_assignment(i, perm_array, is_on, X):
    is_on_i = is_on[i]
    n_stars_i = int(np.sum(is_on_i))
    assert n_stars_i <= X.shape[1]

    row_indx, col_indx = linear_sum_assignment(-X[i, is_on_i, 0:n_stars_i])

    # perm_array[i, :] = np.concatenate((col_indx, np.arange(n_stars_i, X.shape[-1])))
    perm_array[i, :] = np.zeros(is_on.shape[1])
    perm_array[i, is_on_i] = col_indx

def _pool_init(_perm_array, _X, _is_on):
    global perm_array, X, is_on
    perm_array = _perm_array
    X = _X
    is_on = _is_on

def pool_linear_sum_assignment(i):
    _run_linear_sum_assignment(i, perm_array, is_on, X)

def run_batch_hungarian_alg_parallel(log_probs_all, is_on):

    batchsize = log_probs_all.shape[0]
    max_stars = log_probs_all.shape[1]

    # This is done in numpy ...
    log_probs_all_np = log_probs_all.to('cpu').detach().numpy()
    is_on_np = is_on.to('cpu').detach().numpy()


    perm_array_base = multiprocessing.Array(ctypes.c_double, batchsize*max_stars)
    perm_array = np.ctypeslib.as_array(perm_array_base.get_obj())
    perm_array = perm_array.reshape(batchsize, max_stars)

    pool = multiprocessing.Pool(multiprocessing.cpu_count(),
                                _pool_init,
                                (perm_array, log_probs_all_np, is_on_np))

    pool.map(pool_linear_sum_assignment, range(batchsize)); pool.close()

    return(torch.LongTensor(perm_array))


# this function runs it sequentially
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


# This is the O(n!) implementation
# used only for testing
def find_min_col_permutation(X):
    dim = X.shape[1]

    # X should be square
    assert X.shape[0] == X.shape[1]

    # enumerate all possibilities
    l = list(permutations(range(0, dim)))

    # vector just for indexing
    seq_tensor = torch.LongTensor([i for i in range(dim)])

    # loop through all possibliities, get loss
    loss = torch.zeros(len(l))
    for i in range(len(l)):
        loss[i] = torch.sum(X[seq_tensor, l[i]])

    # find best permutation
    return l[torch.argmin(loss)]
