import torch


def compute_knots(n: int, degree: int = 3) -> torch.Tensor:
    """Get the knot vector for a given number of points.

    Args:
        n: Number of evaluation points.
        degree: Degree of the B-spline.

    Returns:
        torch.Tensor: Knot vector.
    """
    u = torch.linspace(0.0, 1.0, n, dtype=torch.float32)
    if degree % 2 == 0:
        middle_knots = (u[:-1] + u[1:]) / 2.0
        middle_knots = torch.cat((middle_knots, u[-1:]))
    else:
        middle_knots = u
    kv = torch.cat((u[0:1].repeat(degree), middle_knots, u[-1:].repeat(degree)))
    return kv


def clamp_ctrls(x: torch.Tensor, degree: int = 2) -> torch.Tensor:
    """Clamp a vector of controls for inference.

    Clamping guarantees that the spline passes through start and endpoints.

    Args:
        x (torch.Tensor): Control vector.
        degree (int): Degree of the B-spline.

    Returns:
        torch.Tensor: Clamped control vector with multiplicities at start and end.
    """
    if x.ndimension() == 2:
        x = x.unsqueeze(0)

    nrep = degree // 2
    first_points = x[:, 0:1, :].repeat(1, nrep, 1)
    last_points = x[:, -1:, :].repeat(1, nrep, 1)
    ctrl_points = torch.cat([first_points, x, last_points], dim=1)

    return torch.cat(ctrl_points, dim=1)
