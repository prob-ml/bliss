import torch


class DeclanBSpline:
    """B-Spline interpolation in PyTorch."""

    def __init__(
        self,
        min_val: float,
        max_val: float,
        nknots: int,
        n_grid_points: int,
        degree: int = 3,
        order: int = 2,
        device: str = "cpu",
    ):
        """Initialize the B-spline."""
        self.min_val = min_val
        self.max_val = max_val
        self.knots = torch.linspace(min_val, max_val, nknots)
        self.knots = torch.cat(
            [
                self.knots.new_full((degree,), min_val),
                self.knots,
                self.knots.new_full((degree,), max_val),
            ]
        )

        self.t_values = torch.linspace(min_val, max_val, n_grid_points)

        self.degree = degree
        self.n = len(self.knots) - degree - 1
        self.order = order
        self.Phi = self.compute_basis_matrix()
        self.dPhi = self.compute_derivative_matrix(1) if order > 0 else None
        self.ddPhi = self.compute_derivative_matrix(2) if order > 1 else None

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please use CPU instead.")
        self.device = device
        self.Phi = self.Phi.to(device)
        self.dPhi = self.dPhi.to(device) if self.dPhi is not None else None
        self.ddPhi = self.ddPhi.to(device) if self.ddPhi is not None else None

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the B-spline and its derivatives at the given control points.

        Supports evaluation in batches, i.e., for a (m, n, l) matrix the input
         will be treated as m batches of n x l inputs, where n is the number
         of control points and l the target dimension.

        Args:
            x: Control points for the B-spline.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Evaluated spline, first derivative, and second derivative.

        """
        xc = self.clamp_ctrls(x, self.degree)

        y = torch.matmul(self.Phi, xc)
        y_dot = torch.matmul(self.dPhi, xc) if self.order > 0 else None
        y_ddot = torch.matmul(self.ddPhi, xc) if self.order > 1 else None

        if x.ndimension() > 2:
            y = y.view(-1, x.shape[0], x.shape[-1]).transpose(0, 1)
            if self.order > 0:
                y_dot = y_dot.view(-1, x.shape[0], x.shape[-1]).transpose(0, 1)
            if self.order > 1:
                y_ddot = y_ddot.view(-1, x.shape[0], x.shape[-1]).transpose(0, 1)

        return y, y_dot, y_ddot

    def compute_basis_matrix(self) -> torch.Tensor:
        """Compute the basis matrix for the B-spline.

        Returns:
            torch.Tensor: Basis matrix.
        """
        basis_matrix = torch.stack(
            [
                torch.stack([self.basis_function(t, self.degree, i) for t in self.t_values])
                for i in range(self.n)
            ]
        )

        return basis_matrix.T

    def compute_derivative_matrix(self, order: int = 1) -> torch.Tensor:
        """Compute the derivative matrix for the B-spline.

        Args:
            order: Order of the derivative.

        Returns:
            torch.Tensor: Derivative matrix.
        """
        derivative_matrix = torch.stack(
            [
                torch.stack(
                    [
                        self.derivative_basis_function(t, self.degree, i, order)
                        for t in self.t_values
                    ]
                )
                for i in range(self.n)
            ]
        )

        return derivative_matrix.T

    def basis_function(self, t: float, k: int, i: int) -> float:
        """Cox-de Boor recursion formula for B-spline basis function.

        Args:
            t: Parameter value.
            k: Degree of the basis function.
            i: Knot span index.

        Returns:
            float: Basis function value at t.

        """
        if k == 0:
            return 1.0 if self.knots[i] <= t < self.knots[i + 1] else 0.0

        left_term = 0.0
        if self.knots[i + k] != self.knots[i]:
            left_term = (
                (t - self.knots[i])
                / (self.knots[i + k] - self.knots[i])
                * self.basis_function(t, k - 1, i)
            )

        right_term = 0.0
        if self.knots[i + k + 1] != self.knots[i + 1]:
            right_term = (
                (self.knots[i + k + 1] - t)
                / (self.knots[i + k + 1] - self.knots[i + 1])
                * self.basis_function(t, k - 1, i + 1)
            )

        special_case = (
            1.0
            if not ((i == 0 and t == self.knots[0]) or (i == self.n - 1 and t == self.knots[-1]))
            else 0.0
        )
        return (left_term + right_term) * special_case + 1.0 - special_case

    def derivative_basis_function(self, t: float, k: int, i: int, order: int = 1) -> float:
        """Compute the derivative of the B-spline basis function.

        Args:
            t: Parameter value.
            k: Degree of the basis function.
            i: Knot span index.
            order: Order of the derivative.

        Returns:
            float: Derivative of the basis function at t.

        """
        if order == 0:
            return self.basis_function(t, k, i)
        if k == 0:
            return 0.0

        left_term = 0.0
        if self.knots[i + k] != self.knots[i]:
            left_term = (
                k
                / (self.knots[i + k] - self.knots[i])
                * self.derivative_basis_function(t, k - 1, i, order - 1)
            )

        right_term = 0.0
        if self.knots[i + k + 1] != self.knots[i + 1]:
            right_term = (
                k
                / (self.knots[i + k + 1] - self.knots[i + 1])
                * self.derivative_basis_function(t, k - 1, i + 1, order - 1)
            )

        return left_term - right_term

    def clamp_ctrls(self, x: torch.Tensor, degree: int = 2) -> torch.Tensor:
        """Clamp a vector of controls for inference.

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

        return ctrl_points.squeeze(0)
