"""
Configuration constants and default settings for GPNN.

This module centralizes configuration parameters and constants used throughout
the package, making it easier to maintain and modify default behaviors.
"""

from typing import Any

# Numerical stability constants
JITTER_DEFAULT = 1e-6  # Default jitter for numerical stability
MIN_VARIANCE = 1e-12   # Minimum variance to prevent numerical issues
MAX_CHOLESKY_ATTEMPTS = 3  # Maximum attempts for Cholesky decomposition

# Default hyperparameters
DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_AMPLITUDE = 1.0
DEFAULT_NOISE_SCALE = 0.1
DEFAULT_POLYNOMIAL_DEGREE = 2
DEFAULT_POLYNOMIAL_ALPHA = 1.0

# Training defaults
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_OPTIMIZER = "adam"
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_CONVERGENCE_TOL = 1e-6

# Visualization defaults
DEFAULT_FIGSIZE = (10, 6)
DEFAULT_CONFIDENCE_LEVEL = 0.95  # 95% confidence interval (±2 std)
DEFAULT_CONFIDENCE_STD_MULTIPLIER = 2.0
DEFAULT_DPI = 100

# Matern kernel defaults
MATERN_NU_OPTIONS = {
    "1/2": 0.5,
    "3/2": 1.5,
    "5/2": 2.5,
}
DEFAULT_MATERN_NU = 1.5


def get_default_config() -> dict[str, Any]:
    """
    Get default configuration dictionary.
    
    Returns:
        Dictionary containing all default configuration parameters
    """
    return {
        "jitter": JITTER_DEFAULT,
        "min_variance": MIN_VARIANCE,
        "length_scale": DEFAULT_LENGTH_SCALE,
        "amplitude": DEFAULT_AMPLITUDE,
        "noise_scale": DEFAULT_NOISE_SCALE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "max_iterations": DEFAULT_MAX_ITERATIONS,
        "convergence_tol": DEFAULT_CONVERGENCE_TOL,
    }


def get_kernel_defaults(kernel_type: str) -> dict[str, Any]:
    """
    Get default parameters for a specific kernel type.
    
    Args:
        kernel_type: Type of kernel ("rbf", "matern", "polynomial")
        
    Returns:
        Dictionary containing default parameters for the specified kernel
        
    Raises:
        ValueError: If kernel_type is not recognized
    """
    defaults = {
        "rbf": {
            "length_scale": DEFAULT_LENGTH_SCALE,
            "amplitude": DEFAULT_AMPLITUDE,
        },
        "matern": {
            "length_scale": DEFAULT_LENGTH_SCALE,
            "amplitude": DEFAULT_AMPLITUDE,
            "nu": DEFAULT_MATERN_NU,
        },
        "polynomial": {
            "degree": DEFAULT_POLYNOMIAL_DEGREE,
            "alpha": DEFAULT_POLYNOMIAL_ALPHA,
        },
    }

    if kernel_type.lower() not in defaults:
        raise ValueError(
            f"Unknown kernel type '{kernel_type}'. "
            f"Available types: {', '.join(defaults.keys())}"
        )

    return defaults[kernel_type.lower()]
