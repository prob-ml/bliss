import math

import torch

from case_studies.redshift.splinex.bspline import DeclanBSpline

"""
The port seems successful based on this test case at least. 
Let's give it a whirl within redshift family.
"""

min_val = 0.0
max_val = 2 * math.pi
nknots = 40
degree = 3
n_grid_points = 100
my_spline = DeclanBSpline(
    min_val=min_val,
    max_val=max_val,
    nknots=nknots,
    n_grid_points=n_grid_points,
    degree=degree,
)


coeffs = torch.zeros((5, nknots), dtype=torch.float32)
coeffs[0, 3] = 1.0

spline_curve_vals, _, _ = my_spline(coeffs.T)
