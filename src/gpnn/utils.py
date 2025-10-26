"""
Visualization and utility functions for Gaussian Processes.

This module provides plotting functions for visualizing GP predictions,
uncertainty quantification, and training metrics. All functions handle
tensor-to-numpy conversion and provide publication-quality plots.

Example:
    >>> from gpnn import GaussianProcess, RBFKernel, plot_gp_predictions
    >>> import torch
    >>> 
    >>> # After training a GP model
    >>> X_test = torch.linspace(-3, 3, 100).unsqueeze(1)
    >>> mean, var = gp.predict(X_test)
    >>> plot_gp_predictions(X_train, y_train, X_test, mean, var)
"""


import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import DEFAULT_FIGSIZE


def plot_gp_predictions(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    title: str = "GP Regression - Predictive Distribution",
    confidence_level: float = 0.95,
    figsize: tuple = DEFAULT_FIGSIZE,
    save_path: str | None = None
) -> None:
    """
    Plot GP predictions with training data and uncertainty quantification.
    
    Creates a publication-quality plot showing:
    - Training data points (scatter)
    - Predictive mean (solid line)
    - Confidence intervals (shaded region)
    
    The confidence intervals are computed as mean ± k*std where k depends
    on the confidence_level. For 95% confidence (default), k ≈ 2.
    
    Args:
        X_train: Training input data of shape (N, 1) for 1D inputs
        y_train: Training target data of shape (N, 1)
        X_test: Test input data of shape (M, 1) for 1D inputs
        mean: Predicted mean values of shape (M, 1)
        var: Predicted variance values of shape (M,) or (M, 1)
        title: Plot title. Default: "GP Regression - Predictive Distribution"
        confidence_level: Confidence level for intervals (0 < x < 1).
                         0.95 = 95% confidence. Default: 0.95
        figsize: Figure size as (width, height) tuple. Default: (10, 6)
        save_path: If provided, save figure to this path. Default: None (display only)
    
    Raises:
        ValueError: If inputs are not 1D or have incompatible shapes
    
    Note:
        This function currently supports only 1D inputs (X_train, X_test with shape (N, 1))
    
    Example:
        >>> X_train = torch.randn(50, 1)
        >>> y_train = torch.sin(X_train)
        >>> X_test = torch.linspace(-5, 5, 200).unsqueeze(1)
        >>> mean, var = gp.predict(X_test)
        >>> plot_gp_predictions(X_train, y_train, X_test, mean, var)
        >>> # Save to file
        >>> plot_gp_predictions(X_train, y_train, X_test, mean, var, 
        ...                     save_path='gp_results.png')
    """
    # Convert tensors to numpy arrays on CPU
    X_train_cpu = X_train.detach().cpu().numpy()
    y_train_cpu = y_train.detach().cpu().numpy()
    X_test_cpu = X_test.detach().cpu().numpy().squeeze()  # (M,)

    mu_cpu = mean.detach().cpu().numpy().squeeze()  # (M,)
    var_cpu = var.detach().cpu().numpy()

    # Ensure var is 1D
    if var_cpu.ndim > 1:
        var_cpu = var_cpu.squeeze()

    # Ensure non-negative variance
    var_cpu = np.clip(var_cpu, a_min=0.0, a_max=None)
    std_cpu = np.sqrt(var_cpu)

    # Compute confidence multiplier from confidence level
    # For normal distribution: P(|Z| < k) = confidence_level
    # confidence_level = 0.95 => k ≈ 1.96 ≈ 2
    from scipy import stats
    k = stats.norm.ppf((1 + confidence_level) / 2)

    # Create figure
    plt.figure(figsize=figsize)

    # Plot training data
    plt.scatter(X_train_cpu, y_train_cpu, color='red', s=30, alpha=0.6,
                label='Training Data', zorder=3)

    # Plot predicted mean
    plt.plot(X_test_cpu, mu_cpu, 'b-', linewidth=2, label='Predictive Mean', zorder=2)

    # Plot confidence intervals
    lower = mu_cpu - k * std_cpu
    upper = mu_cpu + k * std_cpu

    # Ensure X_test_cpu is 1D for fill_between
    if X_test_cpu.ndim > 1:
        X_test_cpu = X_test_cpu.ravel()

    plt.fill_between(
        X_test_cpu, lower, upper,
        alpha=0.3, color='blue',
        label=f'{int(confidence_level * 100)}% Confidence Interval',
        zorder=1
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("X", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_training_metrics(
    losses: list,
    title: str = "Training Loss"
) -> None:
    """
    Plot training loss over epochs.

    Args:
        losses: List of loss values
        title: Plot title
    """
    plt.figure(figsize=(8, 5))
    plt.plot(losses, 'b-', label='Training Loss')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_training_metrics_detailed(
    epochs: list,
    nll_values: list,
    train_mse: list,
    val_mse: list | None = None,
    title: str = "Training Metrics"
) -> None:
    """
    Plot training metrics over epochs.

    Args:
        epochs: List of epoch numbers
        nll_values: List of negative log-likelihood values
        train_mse: List of training MSE values
        val_mse: Optional list of validation MSE values
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot NLL
    ax1.plot(epochs, nll_values, 'b-', label='NLL')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Negative Log-Likelihood')
    ax1.set_title('Training Loss')
    ax1.grid(True)

    # Plot MSE
    ax2.plot(epochs, train_mse, 'g-', label='Train MSE')
    if val_mse is not None:
        ax2.plot(epochs, val_mse, 'r-', label='Val MSE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title('Mean Squared Error')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
