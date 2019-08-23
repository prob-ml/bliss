import torch
from itertools import permutations

import time

# my implementation for the hungarian algorithm in pytorch
# adapted from
# https://github.com/scipy/scipy/blob/v0.18.1/scipy/optimize/_hungarian.py#L13-L107

def linear_sum_assignment(cost_matrix):

    # should be two dimensional square matrix
    assert len(cost_matrix.shape) == 2
    assert cost_matrix.shape[0] == cost_matrix.shape[1]

    state = _Hungary(cost_matrix)

    step = _step1

    i = 0
    while step is not None:
        t0 = time.time()
        step = step(state)
        i += 1

    marked = state.marked

    return (marked == 1).nonzero()[:, 1]


class _Hungary(object):
    """State of the Hungarian algorithm.
    Parameters
    ----------
    cost_matrix : 2D matrix
        The cost matrix. Must have shape[1] >= shape[0].
    """

    def __init__(self, cost_matrix):
        self.C = cost_matrix.clone()

        n, m = self.C.shape
        self.row_uncovered = torch.ones(n).type(torch.ByteTensor)
        self.col_uncovered = torch.ones(m).type(torch.ByteTensor)
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = torch.zeros((n + m, 2)).type(torch.LongTensor)
        self.marked = torch.zeros((n, m)).type(torch.LongTensor)

    def _clear_covers(self):
        """Clear all covered matrix cells"""
        self.row_uncovered[:] = 1
        self.col_uncovered[:] = 1


# Individual steps of the algorithm follow, as a state machine: they return
# the next step to be taken (function to be called), if any.

def _step1(state):
    """Steps 1 and 2 in the Wikipedia page."""

    # Step 1: For each row of the matrix, find the smallest element and
    # subtract it from every element in its row.
    state.C -= state.C.min(dim=1)[0].unsqueeze(dim = 1)
    # Step 2: Find a zero (Z) in the resulting matrix. If there is no
    # starred zero in its row or column, star Z. Repeat for each element
    # in the matrix.
    which_zero = torch.transpose((state.C == 0).nonzero(), 0, 1)

    for i, j in zip(*which_zero):
        if state.col_uncovered[j] and state.row_uncovered[i]:
            state.marked[i, j] = 1
            state.col_uncovered[j] = 0
            state.row_uncovered[i] = 0

    state._clear_covers()
    return _step3


def _step3(state):
    """
    Cover each column containing a starred zero. If n columns are covered,
    the starred zeros describe a complete set of unique assignments.
    In this case, Go to DONE, otherwise, Go to Step 4.
    """
    marked = (state.marked == 1)
    state.col_uncovered[torch.any(marked, dim=0)] = 0

    if marked.sum() < state.C.shape[0]:
        return _step4


def _step4(state):
    """
    Find a noncovered zero and prime it. If there is no starred zero
    in the row containing this primed zero, Go to Step 5. Otherwise,
    cover this row and uncover the column containing the starred
    zero. Continue in this manner until there are no uncovered zeros
    left. Save the smallest uncovered value and Go to Step 6.
    """
    # We convert to int as numpy operations are faster on int
    C = (state.C == 0).type(torch.LongTensor)
    covered_C = C * state.row_uncovered.unsqueeze(dim = 1).type(torch.LongTensor)
    covered_C *= state.col_uncovered.type(torch.LongTensor)
    n = state.C.shape[0]
    m = state.C.shape[1]

    while True:
        # Find an uncovered zero
        indx = torch.argmax(covered_C)
        row, col = (indx / covered_C.shape[1], indx % covered_C.shape[1])
        if covered_C[row, col] == 0:
            return _step6
        else:
            state.marked[row, col] = 2
            # Find the first starred element in the row
            star_col = torch.argmax(state.marked[row] == 1)
            if state.marked[row, star_col] != 1:
                # Could not find one
                state.Z0_r = row
                state.Z0_c = col
                return _step5
            else:
                col = star_col
                state.row_uncovered[row] = 0
                state.col_uncovered[col] = 1
                covered_C[:, col] = C[:, col] * \
                        state.row_uncovered.type(torch.LongTensor)
                covered_C[row] = 0


def _step5(state):
    """
    Construct a series of alternating primed and starred zeros as follows.
    Let Z0 represent the uncovered primed zero found in Step 4.
    Let Z1 denote the starred zero in the column of Z0 (if any).
    Let Z2 denote the primed zero in the row of Z1 (there will always be one).
    Continue until the series terminates at a primed zero that has no starred
    zero in its column. Unstar each starred zero of the series, star each
    primed zero of the series, erase all primes and uncover every line in the
    matrix. Return to Step 3
    """
    count = 0
    path = state.path
    path[count, 0] = state.Z0_r
    path[count, 1] = state.Z0_c

    while True:
        # Find the first starred element in the col defined by
        # the path.
        row = torch.argmax(state.marked[:, path[count, 1]] == 1)
        if state.marked[row, path[count, 1]] != 1:
            # Could not find one
            break
        else:
            count += 1
            path[count, 0] = row
            path[count, 1] = path[count - 1, 1]

        # Find the first prime element in the row defined by the
        # first path step
        col = torch.argmax(state.marked[path[count, 0]] == 2)
        if state.marked[row, col] != 2:
            col = -1
        count += 1
        path[count, 0] = path[count - 1, 0]
        path[count, 1] = col

    # Convert paths
    for i in range(count + 1):
        if state.marked[path[i, 0], path[i, 1]] == 1:
            state.marked[path[i, 0], path[i, 1]] = 0
        else:
            state.marked[path[i, 0], path[i, 1]] = 1

    state._clear_covers()
    # Erase all prime markings
    state.marked[state.marked == 2] = 0
    return _step3


def _step6(state):
    """
    Add the value found in Step 4 to every element of each covered row,
    and subtract it from every element of each uncovered column.
    Return to Step 4 without altering any stars, primes, or covered lines.
    """
    # the smallest uncovered value in the matrix
    if torch.any(state.row_uncovered) and torch.any(state.col_uncovered):
        minval = torch.min(state.C[state.row_uncovered], dim=0)[0]
        minval = torch.min(minval[state.col_uncovered])

        state.C[~state.row_uncovered] += minval
        state.C[:, state.col_uncovered] -= minval

    return _step4

# This is the O(n!) implementation
# used only for testing
def find_min_permutation(X):
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
