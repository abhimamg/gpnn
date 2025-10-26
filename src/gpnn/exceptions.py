"""
Custom exceptions for the GPNN package.

This module defines custom exception classes to provide clearer error messages
and better error handling throughout the package.
"""


class GPNNException(Exception):
    """Base exception class for all GPNN-related errors."""

    pass


class KernelError(GPNNException):
    """Raised when there's an error with kernel computation."""

    pass


class DimensionMismatchError(GPNNException):
    """Raised when input dimensions don't match expected shapes."""

    def __init__(self, expected: str, got: str):
        """
        Initialize dimension mismatch error.
        
        Args:
            expected: Description of expected dimensions
            got: Description of actual dimensions
        """
        super().__init__(f"Dimension mismatch: expected {expected}, got {got}")


class NotFittedError(GPNNException):
    """Raised when attempting to use a model that hasn't been fitted."""

    def __init__(self, message: str = "Model must be fitted before making predictions"):
        """
        Initialize not fitted error.
        
        Args:
            message: Custom error message
        """
        super().__init__(message)


class NumericalInstabilityError(GPNNException):
    """Raised when numerical instability is detected in computations."""

    pass
