#!/usr/bin/env python3
"""
Advanced example: Comparing multiple kernels and hyperparameter tuning.

This script demonstrates:
1. Training GPs with different kernels
2. Comparing model performance
3. Hyperparameter evolution tracking
4. Model selection based on validation performance
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from gpnn import (
    GaussianProcess, 
    RBFKernel, 
    MaternKernel, 
    PolynomialKernel,
    plot_gp_predictions
)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def generate_data(n_samples: int = 150, noise_level: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic 1D data with multiple frequency components.
    
    Args:
        n_samples: Number of data points
        noise_level: Standard deviation of observation noise
    
    Returns:
        Tuple of (X, y) tensors
    """
    X = torch.linspace(-5, 5, n_samples).unsqueeze(1)
    
    # Create a complex function with multiple components
    y = (
        torch.sin(X * 2.0) +           # Low frequency
        0.5 * torch.sin(X * 6.0) +     # Medium frequency  
        0.3 * torch.cos(X * 0.5)       # Very low frequency
    )
    
    # Add noise
    y += torch.randn_like(y) * noise_level
    
    return X, y


def train_model(
    gp: GaussianProcess,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.05,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train a GP model and track metrics.
    
    Args:
        gp: GaussianProcess model
        X_train: Training inputs
        y_train: Training targets
        X_val: Validation inputs
        y_val: Validation targets
        epochs: Number of training epochs
        lr: Learning rate
        verbose: Whether to print progress
    
    Returns:
        Dictionary containing training history
    """
    optimizer = torch.optim.Adam(gp.parameters(), lr=lr)
    
    history = {
        'train_nll': [],
        'train_mse': [],
        'val_mse': [],
        'length_scale': [],
        'amplitude': [],
        'noise_scale': []
    }
    
    for epoch in range(epochs):
        # Training step
        nll = gp.train_step(X_train, y_train, optimizer)
        history['train_nll'].append(nll)
        
        # Compute metrics
        with torch.no_grad():
            # Training MSE
            train_pred, _ = gp.predict(X_train)
            train_mse = ((train_pred - y_train) ** 2).mean().item()
            history['train_mse'].append(train_mse)
            
            # Validation MSE
            val_pred, _ = gp.predict(X_val)
            val_mse = ((val_pred - y_val) ** 2).mean().item()
            history['val_mse'].append(val_mse)
        
        # Track hyperparameters
        params = gp.get_hyperparameters()
        history['noise_scale'].append(params['noise_scale'])
        if 'length_scale' in params:
            history['length_scale'].append(params['length_scale'])
        if 'amplitude' in params:
            history['amplitude'].append(params['amplitude'])
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train NLL: {nll:.4f}, Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}")
            print(f"  Params: {params}")
    
    return history


def plot_comparison(
    models: Dict[str, GaussianProcess],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    save_path: str = None
) -> None:
    """
    Plot predictions from multiple models side by side.
    
    Args:
        models: Dictionary mapping model names to GaussianProcess objects
        X_train: Training inputs
        y_train: Training targets
        X_test: Test inputs for visualization
        save_path: Optional path to save figure
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, gp) in zip(axes, models.items()):
        # Make predictions
        mean, var = gp.predict(X_test)
        
        # Convert to numpy
        X_train_np = X_train.cpu().numpy()
        y_train_np = y_train.cpu().numpy()
        X_test_np = X_test.cpu().numpy().squeeze()
        mean_np = mean.cpu().numpy().squeeze()
        std_np = torch.sqrt(var).cpu().numpy()
        
        # Plot
        ax.scatter(X_train_np, y_train_np, c='red', s=20, alpha=0.6, label='Training Data')
        ax.plot(X_test_np, mean_np, 'b-', linewidth=2, label='Mean')
        ax.fill_between(
            X_test_np,
            mean_np - 2 * std_np,
            mean_np + 2 * std_np,
            alpha=0.3, color='blue', label='95% CI'
        )
        
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_training_history(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: str = None
) -> None:
    """
    Plot training histories for multiple models.
    
    Args:
        histories: Dictionary mapping model names to history dictionaries
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Validation MSE
    ax = axes[0, 0]
    for name, history in histories.items():
        ax.plot(history['val_mse'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation MSE')
    ax.set_title('Validation Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Training NLL
    ax = axes[0, 1]
    for name, history in histories.items():
        ax.plot(history['train_nll'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Negative Log-Likelihood')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Length Scale Evolution
    ax = axes[1, 0]
    for name, history in histories.items():
        if 'length_scale' in history and history['length_scale']:
            ax.plot(history['length_scale'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Length Scale')
    ax.set_title('Length Scale Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Noise Scale Evolution
    ax = axes[1, 1]
    for name, history in histories.items():
        ax.plot(history['noise_scale'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Noise Scale')
    ax.set_title('Noise Scale Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def main():
    """Main function demonstrating advanced GP usage."""
    print("=" * 70)
    print("Advanced GP Example: Kernel Comparison")
    print("=" * 70)
    
    # Set seed
    set_seed(42)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")
    
    # Generate data
    print("Generating synthetic data...")
    X_full, y_full = generate_data(n_samples=150, noise_level=0.15)
    
    # Split data: 70% train, 15% validation, 15% test
    n_train = 105
    n_val = 23
    
    X_train = X_full[:n_train].to(device)
    y_train = y_full[:n_train].to(device)
    X_val = X_full[n_train:n_train+n_val].to(device)
    y_val = y_full[n_train:n_train+n_val].to(device)
    X_test = X_full[n_train+n_val:].to(device)
    y_test = y_full[n_train+n_val:].to(device)
    
    print(f"Data splits: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test\n")
    
    # Create models with different kernels
    models = {
        'RBF': GaussianProcess(
            kernel=RBFKernel(length_scale=1.0, amplitude=1.0),
            noise_scale=0.1
        ).to(device),
        'Matérn': GaussianProcess(
            kernel=MaternKernel(length_scale=1.0, amplitude=1.0),
            noise_scale=0.1
        ).to(device),
        'Polynomial': GaussianProcess(
            kernel=PolynomialKernel(degree=3, alpha=0.5),
            noise_scale=0.1
        ).to(device)
    }
    
    # Train all models
    histories = {}
    
    for name, gp in models.items():
        print(f"\n{'='*70}")
        print(f"Training {name} Kernel")
        print(f"{'='*70}")
        
        history = train_model(
            gp, X_train, y_train, X_val, y_val,
            epochs=100, lr=0.05, verbose=True
        )
        histories[name] = history
        
        # Final evaluation
        with torch.no_grad():
            test_pred, _ = gp.predict(X_test)
            test_mse = ((test_pred - y_test) ** 2).mean().item()
            test_rmse = np.sqrt(test_mse)
        
        print(f"\nFinal Test RMSE: {test_rmse:.6f}")
        print(f"Final Hyperparameters: {gp.get_hyperparameters()}")
    
    # Select best model based on validation performance
    print(f"\n{'='*70}")
    print("Model Selection")
    print(f"{'='*70}")
    
    best_model_name = min(
        histories.keys(),
        key=lambda name: min(histories[name]['val_mse'])
    )
    
    print(f"\nBest model based on validation MSE: {best_model_name}")
    
    for name in histories.keys():
        best_val_mse = min(histories[name]['val_mse'])
        print(f"  {name}: Best Val MSE = {best_val_mse:.6f}")
    
    # Visualizations
    print(f"\n{'='*70}")
    print("Generating Visualizations")
    print(f"{'='*70}\n")
    
    # Create test grid for smooth plotting
    X_grid = torch.linspace(-5, 5, 200).unsqueeze(1).to(device)
    
    # Plot model comparisons
    plot_comparison(models, X_train, y_train, X_grid)
    
    # Plot training histories
    plot_training_history(histories)
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
