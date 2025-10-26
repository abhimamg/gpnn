"""
GPNN - Gaussian Process Neural Networks with PyTorch

A Python package for Gaussian Process regression with various kernel implementations
and uncertainty quantification.

Main Components:
    - Kernels: RBFKernel, MaternKernel, PolynomialKernel
    - Models: GaussianProcess for regression
    - Utilities: Plotting and visualization functions
    - Configuration: Centralized constants and defaults
    - Exceptions: Custom error classes for better debugging

Quick Start:
    >>> import torch
    >>> from gpnn import GaussianProcess, RBFKernel, plot_gp_predictions
    >>> 
    >>> # Create model
    >>> kernel = RBFKernel(length_scale=1.0, amplitude=1.0)
    >>> gp = GaussianProcess(kernel=kernel, noise_scale=0.1)
    >>> 
    >>> # Train
    >>> X_train = torch.randn(100, 2)
    >>> y_train = torch.sin(X_train[:, 0:1])
    >>> optimizer = torch.optim.Adam(gp.parameters(), lr=0.01)
    >>> 
    >>> for epoch in range(50):
    >>>     loss = gp.train_step(X_train, y_train, optimizer)
    >>> 
    >>> # Predict
    >>> X_test = torch.randn(20, 2)
    >>> mean, var = gp.predict(X_test)
"""

from .config import (
    DEFAULT_AMPLITUDE,
    DEFAULT_LENGTH_SCALE,
    DEFAULT_NOISE_SCALE,
    get_default_config,
    get_kernel_defaults,
)
from .exceptions import (
    DimensionMismatchError,
    GPNNException,
    KernelError,
    NotFittedError,
    NumericalInstabilityError,
)
from .kernels import Kernel, MaternKernel, PolynomialKernel, RBFKernel
from .models import GaussianProcess
from .utils import (
    plot_gp_predictions,
    plot_training_metrics,
    plot_training_metrics_detailed,
)

__version__ = "0.1.0"
__author__ = "Abhishek Mamgain"
__email__ = "abhi.mamg@gmail.com"

__all__ = [
    # Kernels
    "Kernel",
    "RBFKernel",
    "MaternKernel",
    "PolynomialKernel",
    # Models
    "GaussianProcess",
    # Utilities
    "plot_gp_predictions",
    "plot_training_metrics",
    "plot_training_metrics_detailed",
    # Configuration
    "get_default_config",
    "get_kernel_defaults",
    "DEFAULT_LENGTH_SCALE",
    "DEFAULT_AMPLITUDE",
    "DEFAULT_NOISE_SCALE",
    # Exceptions
    "GPNNException",
    "KernelError",
    "DimensionMismatchError",
    "NotFittedError",
    "NumericalInstabilityError",
]


def main() -> None:
    """Entry point for the gpnn CLI."""
    print("Hello from gpnn!")
