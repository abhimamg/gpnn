"""
Kernel implementations for Gaussian Processes.

This module provides various kernel (covariance) functions used in Gaussian Process
regression. Kernels encode assumptions about the function being modeled, such as
smoothness, periodicity, and other properties.

Available Kernels:
    - RBFKernel: Radial Basis Function (Gaussian) kernel for smooth functions
    - MaternKernel: Matérn kernel for functions with controlled smoothness
    - PolynomialKernel: Polynomial kernel for modeling polynomial trends

Mathematical Background:
    A kernel function k(x, x') measures the similarity between two inputs x and x'.
    It must be positive semi-definite to ensure valid covariance matrices.
    
Example:
    >>> import torch
    >>> from gpnn import RBFKernel
    >>> 
    >>> # Create a kernel with specific hyperparameters
    >>> kernel = RBFKernel(length_scale=1.0, amplitude=1.0)
    >>> 
    >>> # Compute kernel matrix between two sets of points
    >>> X = torch.randn(5, 2)  # 5 points in 2D
    >>> Z = torch.randn(3, 2)  # 3 points in 2D
    >>> K = kernel(X, Z)       # Returns (5, 3) kernel matrix
    >>> print(K.shape)
    torch.Size([5, 3])
"""


import numpy as np
import torch
import torch.nn as nn

from .config import DEFAULT_AMPLITUDE, DEFAULT_LENGTH_SCALE, MIN_VARIANCE
from .exceptions import DimensionMismatchError


class Kernel(nn.Module):
    """
    Base class for all kernel functions.
    
    All kernel implementations should inherit from this class and implement
    the forward method to compute the kernel matrix between two sets of points.
    
    The kernel function k(x, x') should be:
        - Symmetric: k(x, x') = k(x', x)
        - Positive semi-definite: for any set of points, the kernel matrix K
          with K[i,j] = k(x_i, x_j) must have non-negative eigenvalues
    
    Attributes:
        None in base class (subclasses define learnable parameters)
    """

    def forward(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel matrix between two sets of points.
        
        Args:
            X: Input tensor of shape (N, D) where N is the number of points
               and D is the dimensionality
            Z: Input tensor of shape (M, D) where M is the number of points
               and D is the dimensionality (must match X's dimensionality)
        
        Returns:
            Kernel matrix of shape (N, M) where element (i,j) is k(X[i], Z[j])
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses
            DimensionMismatchError: If X and Z have different feature dimensions
        """
        if X.shape[1] != Z.shape[1]:
            raise DimensionMismatchError(
                expected="X and Z with same feature dimension",
                got=f"X: {X.shape}, Z: {Z.shape}"
            )
        raise NotImplementedError("Kernel forward method must be implemented by subclass.")

    def _validate_inputs(self, X: torch.Tensor, Z: torch.Tensor) -> None:
        """
        Validate input tensors for kernel computation.
        
        Args:
            X: First input tensor
            Z: Second input tensor
            
        Raises:
            DimensionMismatchError: If inputs have incorrect dimensions
        """
        if X.dim() != 2:
            raise DimensionMismatchError(
                expected="2D tensor (N, D)",
                got=f"{X.dim()}D tensor with shape {X.shape}"
            )
        if Z.dim() != 2:
            raise DimensionMismatchError(
                expected="2D tensor (M, D)",
                got=f"{Z.dim()}D tensor with shape {Z.shape}"
            )
        if X.shape[1] != Z.shape[1]:
            raise DimensionMismatchError(
                expected="matching feature dimensions",
                got=f"X: {X.shape[1]}, Z: {Z.shape[1]}"
            )


class RBFKernel(Kernel):
    """
    Radial Basis Function (RBF) or Squared Exponential kernel.
    
    The RBF kernel is one of the most commonly used kernels for Gaussian Processes.
    It assumes the function being modeled is infinitely differentiable (very smooth).
    
    Mathematical Formula:
        k(x, x') = σ² * exp(-||x - x'||² / (2 * ℓ²))
        
        where:
        - σ² (amplitude²) controls the vertical scale of variation
        - ℓ (length_scale) controls how quickly correlation decays with distance
        - ||x - x'|| is the Euclidean distance between points
    
    The kernel produces smooth, continuous functions with correlation that
    decays exponentially with distance. Points closer than the length_scale
    are highly correlated, while distant points are nearly independent.
    
    Attributes:
        log_length_scale: Learnable parameter (log-space for positivity)
        log_amplitude: Learnable parameter (log-space for positivity)
    
    Properties:
        length_scale: Actual length scale value (exp of log parameter)
        amplitude: Actual amplitude value (exp of log parameter)
    
    Example:
        >>> kernel = RBFKernel(length_scale=1.0, amplitude=1.0)
        >>> X = torch.tensor([[0.0], [1.0], [2.0]])
        >>> K = kernel(X, X)
        >>> # Points at same location have correlation = amplitude²
        >>> # Points far apart have correlation ≈ 0
    """

    def __init__(
        self,
        length_scale: float = DEFAULT_LENGTH_SCALE,
        amplitude: float = DEFAULT_AMPLITUDE
    ) -> None:
        """
        Initialize the RBF kernel.
        
        Args:
            length_scale: Characteristic length scale (ℓ > 0). Larger values
                         mean the function varies more slowly. Default: 1.0
            amplitude: Amplitude parameter (σ > 0). Controls the overall scale
                      of the function. Default: 1.0
        
        Raises:
            ValueError: If length_scale or amplitude are non-positive
        """
        super().__init__()
        if length_scale <= 0:
            raise ValueError(f"length_scale must be positive, got {length_scale}")
        if amplitude <= 0:
            raise ValueError(f"amplitude must be positive, got {amplitude}")

        # Store in log-space to ensure positivity during optimization
        self.log_length_scale = nn.Parameter(torch.log(torch.tensor(length_scale)))
        self.log_amplitude = nn.Parameter(torch.log(torch.tensor(amplitude)))

    @property
    def length_scale(self) -> torch.Tensor:
        """Get the current length scale value."""
        return torch.exp(self.log_length_scale)

    @property
    def amplitude(self) -> torch.Tensor:
        """Get the current amplitude value."""
        return torch.exp(self.log_amplitude)

    def forward(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel matrix between X and Z.
        
        Args:
            X: Input tensor of shape (N, D)
            Z: Input tensor of shape (M, D)
        
        Returns:
            Kernel matrix of shape (N, M)
        
        Raises:
            DimensionMismatchError: If X and Z have incompatible dimensions
        """
        self._validate_inputs(X, Z)

        # Compute pairwise squared Euclidean distances efficiently
        # ||x - z||² = ||x||² + ||z||² - 2x·z
        X_sq = (X ** 2).sum(dim=1, keepdim=True)  # (N, 1)
        Z_sq = (Z ** 2).sum(dim=1, keepdim=True)  # (M, 1)
        sqdist = X_sq + Z_sq.T - 2.0 * X.mm(Z.T)  # (N, M)

        # Clamp to ensure numerical stability (handle floating point errors)
        sqdist = torch.clamp(sqdist, min=0.0)

        # Apply RBF formula: σ² * exp(-||x - x'||² / (2ℓ²))
        return self.amplitude ** 2 * torch.exp(-0.5 * sqdist / self.length_scale ** 2)


class MaternKernel(Kernel):
    """
    Matérn covariance kernel.
    
    The Matérn kernel is a generalization of the RBF kernel that allows control
    over the smoothness of the resulting function. This implementation uses ν=3/2,
    which produces functions that are once differentiable.
    
    Mathematical Formula (ν = 3/2):
        k(x, x') = σ² * (1 + √3·r/ℓ) * exp(-√3·r/ℓ)
        
        where:
        - σ² (amplitude²) controls the vertical scale
        - ℓ (length_scale) controls the decay rate
        - r = ||x - x'|| is the Euclidean distance
    
    Compared to RBF:
        - Matérn kernels allow modeling of rougher functions
        - ν=1/2: Produces rough, non-differentiable functions (Ornstein-Uhlenbeck)
        - ν=3/2: Produces once-differentiable functions (this implementation)
        - ν=5/2: Produces twice-differentiable functions
        - ν→∞: Converges to RBF kernel (infinitely differentiable)
    
    Attributes:
        log_length_scale: Learnable parameter (log-space for positivity)
        log_amplitude: Learnable parameter (log-space for positivity)
    
    Properties:
        length_scale: Actual length scale value
        amplitude: Actual amplitude value
    
    Example:
        >>> kernel = MaternKernel(length_scale=1.0, amplitude=1.0)
        >>> X = torch.tensor([[0.0], [1.0], [2.0]])
        >>> K = kernel(X, X)
    """

    def __init__(
        self,
        length_scale: float = DEFAULT_LENGTH_SCALE,
        amplitude: float = DEFAULT_AMPLITUDE
    ) -> None:
        """
        Initialize the Matérn kernel with ν=3/2.
        
        Args:
            length_scale: Characteristic length scale (ℓ > 0). Default: 1.0
            amplitude: Amplitude parameter (σ > 0). Default: 1.0
        
        Raises:
            ValueError: If length_scale or amplitude are non-positive
        """
        super().__init__()
        if length_scale <= 0:
            raise ValueError(f"length_scale must be positive, got {length_scale}")
        if amplitude <= 0:
            raise ValueError(f"amplitude must be positive, got {amplitude}")

        self.log_length_scale = nn.Parameter(torch.log(torch.tensor(length_scale)))
        self.log_amplitude = nn.Parameter(torch.log(torch.tensor(amplitude)))

    @property
    def length_scale(self) -> torch.Tensor:
        """Get the current length scale value."""
        return torch.exp(self.log_length_scale)

    @property
    def amplitude(self) -> torch.Tensor:
        """Get the current amplitude value."""
        return torch.exp(self.log_amplitude)

    def forward(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute the Matérn 3/2 kernel matrix between X and Z.
        
        Args:
            X: Input tensor of shape (N, D)
            Z: Input tensor of shape (M, D)
        
        Returns:
            Kernel matrix of shape (N, M)
        
        Raises:
            DimensionMismatchError: If X and Z have incompatible dimensions
        """
        self._validate_inputs(X, Z)

        # Compute pairwise squared distances
        X_sq = (X ** 2).sum(dim=1, keepdim=True)
        Z_sq = (Z ** 2).sum(dim=1, keepdim=True)
        sqdist = X_sq + Z_sq.T - 2.0 * X.mm(Z.T)

        # Compute Euclidean distances with numerical stability
        r = torch.sqrt(torch.clamp(sqdist, min=MIN_VARIANCE))

        # Matérn 3/2 formula
        sqrt3_r_over_l = np.sqrt(3) * r / self.length_scale
        return (self.amplitude ** 2 *
                (1.0 + sqrt3_r_over_l) *
                torch.exp(-sqrt3_r_over_l))


class PolynomialKernel(Kernel):
    """
    Polynomial (dot-product) kernel.
    
    The polynomial kernel measures similarity through the dot product of input
    vectors, raised to a power. It's useful for modeling polynomial relationships
    and is commonly used in classification tasks.
    
    Mathematical Formula:
        k(x, x') = (α + x·x')^d
        
        where:
        - α (alpha) is an offset/bias term (α ≥ 0)
        - d (degree) is the polynomial degree (positive integer)
        - x·x' is the dot product of inputs
    
    Properties:
        - Non-stationary: correlation depends on absolute position, not just distance
        - Degree controls complexity: higher degrees model more complex functions
        - Alpha adds flexibility: with α=0, only considers input magnitudes
    
    Attributes:
        degree: Polynomial degree (integer)
        alpha: Offset parameter (float)
    
    Note:
        Unlike RBF and Matérn kernels, this kernel is not stationary (translation
        invariant). The alpha parameter is not learnable in this implementation.
    
    Example:
        >>> kernel = PolynomialKernel(degree=2, alpha=1.0)
        >>> X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> K = kernel(X, X)
        >>> # K[i,j] = (1.0 + X[i]·X[j])²
    """

    def __init__(
        self,
        degree: int = 2,
        alpha: float = 1.0
    ) -> None:
        """
        Initialize the polynomial kernel.
        
        Args:
            degree: Polynomial degree (d ≥ 1). Higher degrees model more
                   complex interactions. Default: 2 (quadratic)
            alpha: Offset/bias term (α ≥ 0). Controls influence of lower-degree
                  terms. Default: 1.0
        
        Raises:
            ValueError: If degree < 1 or alpha < 0
        """
        super().__init__()
        if degree < 1:
            raise ValueError(f"degree must be at least 1, got {degree}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

        self.degree = degree
        self.alpha = alpha

    def forward(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute the polynomial kernel matrix between X and Z.
        
        Args:
            X: Input tensor of shape (N, D)
            Z: Input tensor of shape (M, D)
        
        Returns:
            Kernel matrix of shape (N, M)
        
        Raises:
            DimensionMismatchError: If X and Z have incompatible dimensions
        """
        self._validate_inputs(X, Z)

        # Compute dot products: X·Z^T
        dot_product = X.mm(Z.T)

        # Apply polynomial formula: (α + x·x')^d
        return (self.alpha + dot_product) ** self.degree

    def __repr__(self) -> str:
        """String representation of the kernel."""
        return f"PolynomialKernel(degree={self.degree}, alpha={self.alpha})"
