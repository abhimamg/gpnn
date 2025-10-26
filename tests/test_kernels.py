"""
Tests for kernel implementations.
"""

import torch

from gpnn.kernels import RBFKernel, MaternKernel, PolynomialKernel


class TestKernelCore:
    """Core kernel functionality tests."""
    
    def test_kernel_shapes(self):
        """Test that all kernels return correct shapes."""
        kernels = [RBFKernel(), MaternKernel(), PolynomialKernel()]
        X = torch.randn(10, 2)
        Z = torch.randn(15, 2)
        
        for kernel in kernels:
            K = kernel(X, Z)
            assert K.shape == (10, 15)
    
    def test_kernel_symmetry(self):
        """Test that kernels produce symmetric matrices."""
        kernels = [RBFKernel(), MaternKernel()]
        X = torch.randn(5, 2)
        
        for kernel in kernels:
            K = kernel(X, X)
            assert torch.allclose(K, K.T, atol=1e-6)
    
    def test_rbf_positive_definite(self):
        """Test that RBF kernel matrix is positive definite."""
        kernel = RBFKernel()
        X = torch.randn(5, 2)
        
        K = kernel(X, X)
        eigenvals = torch.linalg.eigvals(K)
        assert torch.all(eigenvals.real > -1e-6)
    
    def test_polynomial_kernel_computation(self):
        """Test polynomial kernel computes correctly."""
        kernel = PolynomialKernel(degree=3, alpha=0.0)
        X = torch.ones(2, 1)
        Z = torch.ones(2, 1) * 2
        
        # K(x, z) = (0 + x^T z)^3 = (x^T z)^3
        # For X=[1], Z=[2]: K = (1*2)^3 = 8
        K = kernel(X, Z)
        expected = torch.tensor([[8.0, 8.0], [8.0, 8.0]])
        assert torch.allclose(K, expected)
