import numpy as np
import torch
import torch.nn as nn
import random
from typing import Tuple
import matplotlib.pyplot as plt

########################
# KERNEL DEFINITIONS
########################

class Kernel(nn.Module):
    """
    Base Kernel class to define the interface.
    """
    def forward(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel matrix between X and Z.
        :param X: Tensor of shape (N, D)
        :param Z: Tensor of shape (M, D)
        :return: Tensor of shape (N, M)
        """
        raise NotImplementedError("Kernel forward method not implemented.")


class RBFKernel(Kernel):
    """
    Radial Basis Function (RBF) or Gaussian kernel.
    K(x, x') = amplitude * exp(-0.5 / length_scale^2 * ||x - x'||^2)
    """
    def __init__(self, length_scale: float = 1.0, amplitude: float = 1.0) -> None:
        super().__init__()
        self.log_length_scale = nn.Parameter(torch.log(torch.tensor(length_scale)))
        self.log_amplitude = nn.Parameter(torch.log(torch.tensor(amplitude)))

    @property
    def length_scale(self) -> torch.Tensor:
        return torch.exp(self.log_length_scale)

    @property
    def amplitude(self) -> torch.Tensor:
        return torch.exp(self.log_amplitude)

    def forward(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        # Compute pairwise squared distances.
        X_sq = (X ** 2).sum(dim=1, keepdim=True)
        Z_sq = (Z ** 2).sum(dim=1, keepdim=True)
        sqdist = X_sq + Z_sq.T - 2.0 * X.mm(Z.T)
        # RBF kernel.
        return self.amplitude * torch.exp(-0.5 * sqdist / self.length_scale**2)


class MaternKernel(Kernel):
    """
    Matern kernel with nu=1.5, as an example.
    K(x, x') = amplitude * (1 + sqrt(3)*r/length_scale) * exp(-sqrt(3)*r/length_scale)
    """
    def __init__(self, length_scale: float = 1.0, amplitude: float = 1.0) -> None:
        super().__init__()
        self.log_length_scale = nn.Parameter(torch.log(torch.tensor(length_scale)))
        self.log_amplitude = nn.Parameter(torch.log(torch.tensor(amplitude)))

    @property
    def length_scale(self) -> torch.Tensor:
        return torch.exp(self.log_length_scale)

    @property
    def amplitude(self) -> torch.Tensor:
        return torch.exp(self.log_amplitude)

    def forward(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        # Pairwise distances.
        X_sq = (X ** 2).sum(dim=1, keepdim=True)
        Z_sq = (Z ** 2).sum(dim=1, keepdim=True)
        sqdist = X_sq + Z_sq.T - 2.0 * X.mm(Z.T)
        r = torch.sqrt(torch.clamp(sqdist, min=1e-12))
        # Matern 3/2 kernel.
        factor = (1.0 + (np.sqrt(3) * r / self.length_scale))
        return self.amplitude * factor * torch.exp(-np.sqrt(3) * r / self.length_scale)


class PolynomialKernel(Kernel):
    """
    Polynomial kernel.
    K(x, x') = (alpha + x^T x')^degree
    """
    def __init__(self, degree: int = 2, alpha: float = 1.0) -> None:
        super().__init__()
        self.degree = degree
        self.alpha = alpha

    def forward(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        return (self.alpha + X.mm(Z.T)) ** self.degree


########################
# GAUSSIAN PROCESS MODEL
########################

class GaussianProcess(nn.Module):
    """
    A Gaussian Process (GP) model for regression with learnable hyperparameters.
    """

    def __init__(
        self,
        kernel: Kernel = None,
        noise_scale: float = 1.0
    ) -> None:
        """
        Initialize the GP model with a chosen kernel and noise level.

        :param kernel: A kernel object (e.g., RBFKernel, MaternKernel, etc.).
        :param noise_scale: Measurement noise level.
        """
        super().__init__()
        # Default to RBF kernel if none is provided.
        if kernel is None:
            kernel = RBFKernel()
        self.kernel = kernel

        # Log-scale noise.
        self.log_noise_scale = nn.Parameter(torch.tensor(np.log(noise_scale)))

        # Internal storage for training data.
        self.X = None
        self.y = None
        self.L = None
        self.alpha = None

    @property
    def noise_scale(self) -> torch.Tensor:
        return torch.exp(self.log_noise_scale)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Fit the GP model to training data.

        :param X: Tensor of shape (N, D) containing N data points of dimension D.
        :param y: Tensor of shape (N, 1) containing target values.
        :return: Marginal log-likelihood (negative of the negative log-likelihood).
        """
        assert X.dim() == 2, "X should be of shape (N, D)"
        assert y.dim() == 2, "y should be of shape (N, 1)"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"

        # Compute training kernel matrix.
        K = self.kernel(X, X)
        # Add noise on the diagonal.
        K += self.noise_scale * torch.eye(len(X), device=X.device)
        # Factorize via Cholesky.
        L = torch.linalg.cholesky(K)

        # Solve for alpha.
        alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, y))

        # Compute negative log-likelihood.
        D = X.shape[1]
        neg_log_likelihood = (
            -0.5 * y.T.mm(alpha)
            - torch.log(torch.diag(L)).sum()
            - D * 0.5 * np.log(2.0 * np.pi)
        )

        # Store.
        self.X = X
        self.y = y
        self.L = L
        self.alpha = alpha

        return neg_log_likelihood

    def predict(self, X_test: torch.Tensor, full_cov: bool = False):
        """
        Generate mean and variance (or full covariance) predictions for test inputs.
        Assumes that the model has been fit to training data.

        :param X_test: Tensor of shape (N, D) for N new data points of dimension D.
        :param full_cov: Whether to return the full predictive covariance matrix.
        :return: (mean, variance) if full_cov=False, else (mean, covariance)
        """
        assert X_test.dim() == 2, "X_test should be of shape (N, D)"
        L = self.L
        alpha = self.alpha
        # Compute cross-kernel.
        k_test = self.kernel(self.X, X_test)

        # Solve.
        v = torch.linalg.solve(L, k_test)

        # Mean.
        mean = k_test.T.mm(alpha)

        # Predictive covariance.
        k_xx = self.kernel(X_test, X_test)
        cov = k_xx - v.T.mm(v)

        if not full_cov:
            # Return just the diagonal.
            # Clip negative values from numerical issues.
            var = torch.diag(cov).clamp_min(0.0) + self.noise_scale
            return mean, var
        else:
            # Return the full covariance.
            cov_diag = torch.diag(cov)
            cov_diag_clamped = torch.diag(cov_diag.clamp_min(0.0))
            cov = cov.clone()
            cov[torch.eye(X_test.size(0), device=X_test.device).bool()] = cov_diag_clamped
            cov += self.noise_scale * torch.eye(X_test.size(0), device=X_test.device)
            return mean, cov

    def train_step(self, X: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """
        Single training step for hyperparameter optimization.
        :param X: Training inputs.
        :param y: Training targets.
        :param optimizer: torch.optim optimizer.
        :return: The negative log-likelihood as a float.
        """
        optimizer.zero_grad()
        nll = -self.fit(X, y).sum()
        nll.backward()
        optimizer.step()
        return nll.item()


def plot_gp_predictions(X_train: torch.Tensor, y_train: torch.Tensor,
                        X_test: torch.Tensor, mean: torch.Tensor, var: torch.Tensor,
                        title: str = "GP Regression - Uncertainty") -> None:
    """
    Plots the training data, mean predictions, and confidence intervals.
    """
    # Move data to CPU and convert to NumPy.
    X_train_cpu = X_train.detach().cpu().numpy()
    y_train_cpu = y_train.detach().cpu().numpy()
    X_test_cpu = X_test.detach().cpu().numpy().squeeze()  # shape (N,)

    mu_cpu = mean.detach().cpu().numpy().squeeze()        # shape (N,)
    var_cpu = var.detach().cpu().numpy()
    var_cpu = np.clip(var_cpu, a_min=0.0, a_max=None)     # ensure no negatives
    std_cpu = np.sqrt(var_cpu)

    plt.figure(figsize=(8, 5))
    # Scatter training data.
    plt.scatter(X_train_cpu, y_train_cpu, color='red', s=20, alpha=0.6, label='Train Data')
    # Plot predicted mean.
    plt.plot(X_test_cpu, mu_cpu, 'b-', label='GP Mean')
    # Confidence intervals at ±2 std.
    lower = mu_cpu - 2.0 * std_cpu
    upper = mu_cpu + 2.0 * std_cpu

    # Ensure X_test_cpu is 1D for fill_between.
    if X_test_cpu.ndim > 1:
        X_test_cpu = X_test_cpu.ravel()

    plt.fill_between(X_test_cpu, lower, upper, alpha=0.2, color='blue', label='95% CI')

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()