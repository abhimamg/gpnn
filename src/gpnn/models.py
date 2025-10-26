"""
Gaussian Process model implementation.

This module contains the main GaussianProcess class for regression tasks.
It implements the standard GP regression algorithm with automatic hyperparameter
optimization via gradient descent.

Mathematical Background:
    A Gaussian Process defines a distribution over functions. Given training data
    (X, y), the posterior distribution at test points X* is:
    
    f(X*) ~ N(μ*, Σ*)
    
    where:
    - μ* = K(X*, X)[K(X, X) + σ²I]⁻¹y
    - Σ* = K(X*, X*) - K(X*, X)[K(X, X) + σ²I]⁻¹K(X, X*)
    
    The marginal log-likelihood (used for training) is:
    
    log p(y|X) = -½y^T[K + σ²I]⁻¹y - ½log|K + σ²I| - (n/2)log(2π)

Example:
    >>> from gpnn import GaussianProcess, RBFKernel
    >>> import torch
    >>> 
    >>> # Create and train a GP model
    >>> kernel = RBFKernel(length_scale=1.0, amplitude=1.0)
    >>> gp = GaussianProcess(kernel=kernel, noise_scale=0.1)
    >>> 
    >>> # Fit to training data
    >>> X_train = torch.randn(100, 2)
    >>> y_train = torch.randn(100, 1)
    >>> gp.fit(X_train, y_train)
    >>> 
    >>> # Make predictions
    >>> X_test = torch.randn(20, 2)
    >>> mean, var = gp.predict(X_test)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .kernels import Kernel, RBFKernel
from .config import (
    DEFAULT_NOISE_SCALE, 
    JITTER_DEFAULT, 
    MIN_VARIANCE, 
    MAX_CHOLESKY_ATTEMPTS
)
from .exceptions import NotFittedError, DimensionMismatchError, NumericalInstabilityError


class GaussianProcess(nn.Module):
    """
    A Gaussian Process (GP) model for regression with learnable hyperparameters.
    
    This class implements exact GP regression using Cholesky decomposition for
    efficient computation. Hyperparameters (kernel parameters and noise) can be
    learned via gradient descent by maximizing the marginal log-likelihood.
    
    The model maintains training data and precomputed matrices for efficient
    predictions. After fitting, it can make predictions at new test points
    along with uncertainty estimates.
    
    Attributes:
        kernel: Covariance function (e.g., RBFKernel, MaternKernel)
        log_noise_scale: Learnable noise parameter (in log-space)
        X: Training input data (set after fitting)
        y: Training target data (set after fitting)
        L: Cholesky factor of kernel matrix (set after fitting)
        alpha: Precomputed vector for predictions (set after fitting)
    
    Properties:
        noise_scale: Actual noise level (exp of log parameter)
        is_fitted: Whether the model has been fitted to data
    
    Example:
        >>> import torch
        >>> from gpnn import GaussianProcess, RBFKernel
        >>> 
        >>> # Initialize
        >>> kernel = RBFKernel(length_scale=1.0)
        >>> gp = GaussianProcess(kernel=kernel, noise_scale=0.1)
        >>> 
        >>> # Fit to data
        >>> X = torch.randn(50, 2)
        >>> y = torch.randn(50, 1)
        >>> nll = gp.fit(X, y)
        >>> 
        >>> # Predict
        >>> X_new = torch.randn(10, 2)
        >>> mean, var = gp.predict(X_new)
    """

    def __init__(
        self,
        kernel: Optional[Kernel] = None,
        noise_scale: float = DEFAULT_NOISE_SCALE,
        jitter: float = JITTER_DEFAULT
    ) -> None:
        """
        Initialize the GP model with a kernel and noise level.

        Args:
            kernel: A kernel object (e.g., RBFKernel, MaternKernel).
                   If None, defaults to RBFKernel with default parameters.
            noise_scale: Observation noise standard deviation (σ > 0).
                        Represents measurement noise or model uncertainty. Default: 0.1
            jitter: Small positive value added to diagonal for numerical stability.
                   Default: 1e-6
        
        Raises:
            ValueError: If noise_scale or jitter are non-positive
        """
        super().__init__()
        
        # Validate inputs
        if noise_scale <= 0:
            raise ValueError(f"noise_scale must be positive, got {noise_scale}")
        if jitter <= 0:
            raise ValueError(f"jitter must be positive, got {jitter}")
        
        # Default to RBF kernel if none is provided
        if kernel is None:
            kernel = RBFKernel()
        self.kernel = kernel

        # Store noise in log-space to ensure positivity during optimization
        self.log_noise_scale = nn.Parameter(torch.tensor(np.log(noise_scale)))
        self.jitter = jitter

        # Internal storage for training data and precomputed values
        self.X: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.L: Optional[torch.Tensor] = None
        self.alpha: Optional[torch.Tensor] = None

    @property
    def noise_scale(self) -> torch.Tensor:
        """Get the current noise scale value."""
        return torch.exp(self.log_noise_scale)
    
    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted to training data."""
        return self.X is not None and self.y is not None

    def _compute_cholesky(self, K: torch.Tensor) -> torch.Tensor:
        """
        Compute Cholesky decomposition with automatic jitter addition.
        
        Attempts Cholesky decomposition, adding increasing jitter if needed
        to handle numerical issues.
        
        Args:
            K: Kernel matrix (must be square and symmetric)
            
        Returns:
            Lower triangular Cholesky factor L where K = LL^T
            
        Raises:
            NumericalInstabilityError: If Cholesky fails even with maximum jitter
        """
        current_jitter = self.jitter
        
        for attempt in range(MAX_CHOLESKY_ATTEMPTS):
            try:
                # Add jitter to diagonal for numerical stability
                K_stable = K + current_jitter * torch.eye(
                    K.shape[0], device=K.device, dtype=K.dtype
                )
                L = torch.linalg.cholesky(K_stable)
                
                if attempt > 0:
                    # Warn if we needed extra jitter
                    import warnings
                    warnings.warn(
                        f"Cholesky decomposition required jitter={current_jitter:.2e} "
                        f"(attempt {attempt + 1}/{MAX_CHOLESKY_ATTEMPTS})",
                        stacklevel=2
                    )
                return L
                
            except RuntimeError as e:
                if attempt < MAX_CHOLESKY_ATTEMPTS - 1:
                    # Increase jitter and try again
                    current_jitter *= 10
                    continue
                else:
                    # Failed even with maximum jitter
                    raise NumericalInstabilityError(
                        f"Cholesky decomposition failed after {MAX_CHOLESKY_ATTEMPTS} "
                        f"attempts with maximum jitter {current_jitter:.2e}. "
                        f"The kernel matrix may be ill-conditioned. "
                        f"Try: (1) Increase noise_scale, (2) Scale your inputs, "
                        f"(3) Check for duplicate data points. Original error: {e}"
                    ) from e
    
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Fit the GP model to training data.
        
        This method:
        1. Validates input shapes
        2. Computes the kernel matrix K = k(X, X)
        3. Adds observation noise: K_y = K + σ²I
        4. Performs Cholesky decomposition: K_y = LL^T
        5. Computes α = L^T \\ (L \\ y) for efficient predictions
        6. Returns the marginal log-likelihood
        
        The marginal log-likelihood is:
            log p(y|X) = -½y^T K_y^{-1} y - ½ log|K_y| - (n/2) log(2π)

        Args:
            X: Training input data of shape (N, D) where N is the number of
               samples and D is the feature dimensionality
            y: Training target data of shape (N, 1) containing scalar outputs
        
        Returns:
            Marginal log-likelihood value (scalar tensor). Higher values indicate
            better fit. This can be used as a loss when negated.
        
        Raises:
            DimensionMismatchError: If X or y have incorrect dimensions or incompatible shapes
            NumericalInstabilityError: If matrix operations fail due to ill-conditioning
        
        Example:
            >>> gp = GaussianProcess(kernel=RBFKernel())
            >>> X_train = torch.randn(100, 3)
            >>> y_train = torch.randn(100, 1)
            >>> log_likelihood = gp.fit(X_train, y_train)
            >>> print(f"Log-likelihood: {log_likelihood.item():.2f}")
        """
        # Validate input dimensions
        if X.dim() != 2:
            raise DimensionMismatchError(
                expected="2D tensor (N, D)",
                got=f"{X.dim()}D tensor with shape {X.shape}"
            )
        if y.dim() != 2:
            raise DimensionMismatchError(
                expected="2D tensor (N, 1)",
                got=f"{y.dim()}D tensor with shape {y.shape}"
            )
        if X.shape[0] != y.shape[0]:
            raise DimensionMismatchError(
                expected=f"{X.shape[0]} samples in y to match X",
                got=f"{y.shape[0]} samples"
            )
        if y.shape[1] != 1:
            raise DimensionMismatchError(
                expected="single output dimension (N, 1)",
                got=f"shape {y.shape}"
            )

        # Compute training kernel matrix
        K = self.kernel(X, X)
        
        # Add observation noise to diagonal
        K_y = K + (self.noise_scale ** 2) * torch.eye(
            len(X), device=X.device, dtype=X.dtype
        )
        
        # Cholesky decomposition with automatic jitter handling
        L = self._compute_cholesky(K_y)

        # Solve for alpha = K_y^{-1} y via forward and backward substitution
        # First solve: L v = y
        v = torch.linalg.solve_triangular(L, y, upper=False)
        # Then solve: L^T alpha = v
        alpha = torch.linalg.solve_triangular(L.T, v, upper=True)

        # Compute marginal log-likelihood
        # -0.5 * y^T * alpha - sum(log(diag(L))) - 0.5 * n * log(2π)
        n = X.shape[0]
        data_fit = -0.5 * y.T.mm(alpha)
        complexity_penalty = -torch.log(torch.diag(L)).sum()
        constant = -0.5 * n * np.log(2.0 * np.pi)
        
        marginal_log_likelihood = data_fit + complexity_penalty + constant

        # Store for predictions
        self.X = X
        self.y = y
        self.L = L
        self.alpha = alpha

        return marginal_log_likelihood.squeeze()

    def predict(
        self, 
        X_test: torch.Tensor, 
        full_cov: bool = False,
        return_std: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions for test inputs with uncertainty estimates.
        
        Given test points X*, computes the posterior predictive distribution:
            f(X*) ~ N(μ*, Σ*)
        
        where:
            μ* = K(X*, X) α
            Σ* = K(X*, X*) - K(X*, X) K_y^{-1} K(X, X*) + σ²I
        
        The model must be fitted before calling this method.

        Args:
            X_test: Test input tensor of shape (M, D) where M is the number of
                   test points and D must match training dimensionality
            full_cov: If True, return full predictive covariance matrix (M, M).
                     If False, return only diagonal variances (M,). Default: False
            return_std: If True and full_cov=False, return standard deviations
                       instead of variances. Default: False
        
        Returns:
            Tuple of (mean, variance/covariance):
            - mean: Predictive mean of shape (M, 1)
            - variance: If full_cov=False, diagonal variances/std of shape (M,)
                       If full_cov=True, full covariance matrix of shape (M, M)
        
        Raises:
            NotFittedError: If model hasn't been fitted to training data
            DimensionMismatchError: If X_test has wrong dimensions
        
        Example:
            >>> # After fitting the model
            >>> X_test = torch.randn(20, 3)
            >>> mean, var = gp.predict(X_test)
            >>> # Get standard deviations instead
            >>> mean, std = gp.predict(X_test, return_std=True)
            >>> # Get full covariance
            >>> mean, cov = gp.predict(X_test, full_cov=True)
        """
        if not self.is_fitted:
            raise NotFittedError(
                "Model must be fitted before making predictions. "
                "Call fit(X_train, y_train) first."
            )
        
        # Validate input dimensions
        if X_test.dim() != 2:
            raise DimensionMismatchError(
                expected="2D tensor (M, D)",
                got=f"{X_test.dim()}D tensor with shape {X_test.shape}"
            )
        if X_test.shape[1] != self.X.shape[1]:
            raise DimensionMismatchError(
                expected=f"feature dimension {self.X.shape[1]} to match training data",
                got=f"{X_test.shape[1]}"
            )

        # Compute cross-kernel k(X, X_test)
        k_test = self.kernel(self.X, X_test)  # (N, M)

        # Compute predictive mean: μ* = k(X*, X)^T α
        mean = k_test.T.mm(self.alpha)  # (M, 1)

        # Compute predictive covariance
        # First solve: L v = k(X, X_test)
        v = torch.linalg.solve_triangular(self.L, k_test, upper=False)  # (N, M)

        # Prior covariance at test points
        k_test_test = self.kernel(X_test, X_test)  # (M, M)
        
        # Posterior covariance: K** - v^T v
        cov = k_test_test - v.T.mm(v)  # (M, M)

        if not full_cov:
            # Return only diagonal (marginal variances)
            var = torch.diag(cov)  # (M,)
            
            # Add observation noise and ensure non-negative
            var = torch.clamp(var, min=0.0) + self.noise_scale ** 2
            
            if return_std:
                return mean, torch.sqrt(var)
            else:
                return mean, var
        else:
            # Return full covariance matrix
            # Clamp diagonal to ensure positive definiteness
            diag_indices = torch.arange(X_test.shape[0], device=X_test.device)
            cov[diag_indices, diag_indices] = torch.clamp(
                cov[diag_indices, diag_indices], 
                min=0.0
            )
            
            # Add observation noise to diagonal
            cov = cov + (self.noise_scale ** 2) * torch.eye(
                X_test.shape[0], device=X_test.device, dtype=X_test.dtype
            )
            
            return mean, cov

    def train_step(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Perform a single training step for hyperparameter optimization.
        
        This method:
        1. Zeros gradients
        2. Computes marginal log-likelihood via fit()
        3. Negates it (to maximize becomes minimize)
        4. Backpropagates gradients
        5. Updates parameters via optimizer
        
        The hyperparameters being optimized include:
        - Kernel parameters (e.g., length_scale, amplitude)
        - Noise scale
        
        Args:
            X: Training input tensor of shape (N, D)
            y: Training target tensor of shape (N, 1)
            optimizer: PyTorch optimizer (e.g., Adam, SGD) initialized with
                      self.parameters()
        
        Returns:
            Negative log-likelihood value (float). Lower is better during training.
        
        Example:
            >>> gp = GaussianProcess(kernel=RBFKernel())
            >>> optimizer = torch.optim.Adam(gp.parameters(), lr=0.01)
            >>> 
            >>> for epoch in range(100):
            >>>     loss = gp.train_step(X_train, y_train, optimizer)
            >>>     if epoch % 10 == 0:
            >>>         print(f"Epoch {epoch}, Loss: {loss:.4f}")
        """
        optimizer.zero_grad()
        
        # Compute marginal log-likelihood
        log_likelihood = self.fit(X, y)
        
        # Negate to convert maximization to minimization
        nll = -log_likelihood.sum()
        
        # Compute gradients
        nll.backward()
        
        # Update parameters
        optimizer.step()
        
        return nll.item()
    
    def get_hyperparameters(self) -> dict:
        """
        Get current hyperparameter values.
        
        Returns:
            Dictionary containing current hyperparameter values
        
        Example:
            >>> gp = GaussianProcess(kernel=RBFKernel())
            >>> params = gp.get_hyperparameters()
            >>> print(params)
            {'noise_scale': 0.1, 'kernel_type': 'RBFKernel', ...}
        """
        params = {
            'noise_scale': self.noise_scale.item(),
            'kernel_type': type(self.kernel).__name__,
        }
        
        # Add kernel-specific parameters
        if hasattr(self.kernel, 'length_scale'):
            params['length_scale'] = self.kernel.length_scale.item()
        if hasattr(self.kernel, 'amplitude'):
            params['amplitude'] = self.kernel.amplitude.item()
        if hasattr(self.kernel, 'degree'):
            params['degree'] = self.kernel.degree
        if hasattr(self.kernel, 'alpha'):
            params['alpha'] = self.kernel.alpha
            
        return params
    
    def __repr__(self) -> str:
        """String representation of the model."""
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return (f"GaussianProcess(kernel={type(self.kernel).__name__}, "
                f"noise_scale={self.noise_scale.item():.4f}, {fitted_str})")