#!/usr/bin/env python3
"""
Example usage of the GPNN package for Gaussian Process regression.

This script demonstrates how to use the package with synthetic data.
"""

import torch
import numpy as np
import random

from gpnn import GaussianProcess, RBFKernel, plot_gp_predictions


def main():
    """Main example function."""
    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate synthetic data
    print("Generating synthetic data...")
    N = 100
    X_full = torch.randn(N, 1)
    y_full = torch.sin(X_full * 2.0 * np.pi / 4.0) + torch.randn(N, 1) * 0.1

    # Split data
    train_size = 80
    X_train = X_full[:train_size].to(device)
    y_train = y_full[:train_size].to(device)
    X_val = X_full[train_size:].to(device)
    y_val = y_full[train_size:].to(device)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Initialize GP model
    rbf_kernel = RBFKernel(length_scale=1.0, amplitude=1.0)
    gp_rbf = GaussianProcess(kernel=rbf_kernel, noise_scale=0.1).to(device)

    # Train the model
    print("Training GP model...")
    optimizer = torch.optim.Adam(gp_rbf.parameters(), lr=0.05)
    epochs = 30

    for epoch in range(epochs):
        # Perform a single optimization step.
        nll = gp_rbf.train_step(X_train, y_train, optimizer)

        # Compute train and validation MSE.
        train_pred, _ = gp_rbf.predict(X_train)
        train_mse = (train_pred - y_train).pow(2).mean().item()

        val_pred, _ = gp_rbf.predict(X_val)
        val_mse = (val_pred - y_val).pow(2).mean().item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, NLL = {nll:.4f}, "
                  f"Train MSE = {train_mse:.4f}, Val MSE = {val_mse:.4f}")

    # Create test grid for visualization
    grid = torch.linspace(-3, 3, 100).unsqueeze(1).to(device)
    
    # Make predictions
    print("Making predictions...")
    mu_rbf, var_rbf = gp_rbf.predict(grid)

    # Plot results
    print("Plotting results...")
    plot_gp_predictions(
        X_train.cpu(), y_train.cpu(), grid.cpu(), mu_rbf.cpu(), var_rbf.cpu(),
        title="GP Regression with RBF Kernel - Uncertainty Visualization"
    )

    print("Example completed successfully!")


if __name__ == "__main__":
    main()