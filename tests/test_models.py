"""
Tests for the GaussianProcess model.
"""

import torch
import pytest

from gpnn.models import GaussianProcess
from gpnn.kernels import RBFKernel, MaternKernel
from gpnn.exceptions import NotFittedError, DimensionMismatchError


class TestGaussianProcessCore:
    """Core functionality tests."""
    
    def test_fit_and_predict(self):
        """Test basic fit and predict workflow."""
        gp = GaussianProcess(kernel=RBFKernel())
        X_train = torch.randn(20, 2)
        y_train = torch.randn(20, 1)
        
        # Fit
        log_likelihood = gp.fit(X_train, y_train)
        assert torch.is_tensor(log_likelihood)
        assert gp.is_fitted
        
        # Predict
        X_test = torch.randn(10, 2)
        mean, var = gp.predict(X_test)
        assert mean.shape == (10, 1)
        assert var.shape == (10,)
        assert torch.all(var >= 0)
    
    def test_training_improves_fit(self):
        """Test that training decreases loss."""
        gp = GaussianProcess(kernel=RBFKernel())
        X = torch.randn(50, 1)
        y = torch.sin(X) + 0.1 * torch.randn(50, 1)
        optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)
        
        losses = []
        for _ in range(20):
            losses.append(gp.train_step(X, y, optimizer))
        
        assert losses[-1] < losses[0]
    
    def test_predict_not_fitted_error(self):
        """Test that predict raises error when not fitted."""
        gp = GaussianProcess()
        with pytest.raises(NotFittedError):
            gp.predict(torch.randn(5, 2))


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_invalid_initialization(self):
        """Test invalid initialization parameters."""
        with pytest.raises(ValueError):
            GaussianProcess(noise_scale=-0.1)
        with pytest.raises(ValueError):
            GaussianProcess(jitter=-1e-6)
    
    def test_dimension_mismatch_errors(self):
        """Test dimension mismatch detection."""
        gp = GaussianProcess()
        
        # Wrong X dimensions
        with pytest.raises(DimensionMismatchError):
            gp.fit(torch.randn(10), torch.randn(10, 1))
        
        # Wrong y dimensions
        with pytest.raises(DimensionMismatchError):
            gp.fit(torch.randn(10, 2), torch.randn(10))
        
        # Mismatched samples
        with pytest.raises(DimensionMismatchError):
            gp.fit(torch.randn(10, 2), torch.randn(15, 1))
        
        # Multiple outputs
        with pytest.raises(DimensionMismatchError):
            gp.fit(torch.randn(10, 2), torch.randn(10, 3))
    
    def test_predict_dimension_mismatch(self):
        """Test predict with wrong dimensions."""
        gp = GaussianProcess()
        gp.fit(torch.randn(20, 2), torch.randn(20, 1))
        
        with pytest.raises(DimensionMismatchError):
            gp.predict(torch.randn(10, 3))  # Wrong feature dim


class TestAdvancedFeatures:
    """Test advanced features."""
    
    def test_predict_with_full_covariance(self):
        """Test full covariance prediction."""
        gp = GaussianProcess()
        gp.fit(torch.randn(20, 2), torch.randn(20, 1))
        
        X_test = torch.randn(10, 2)
        mean, cov = gp.predict(X_test, full_cov=True)
        
        assert mean.shape == (10, 1)
        assert cov.shape == (10, 10)
        assert torch.allclose(cov, cov.T, atol=1e-5)
    
    def test_numerical_stability_with_duplicates(self):
        """Test handling of duplicate points with jitter."""
        gp = GaussianProcess(jitter=1e-4)
        X = torch.ones(5, 2)  # All duplicate
        y = torch.randn(5, 1)
        
        log_likelihood = gp.fit(X, y)
        assert torch.is_tensor(log_likelihood)
    
    def test_different_kernels(self):
        """Test with different kernel types."""
        for kernel in [RBFKernel(), MaternKernel()]:
            gp = GaussianProcess(kernel=kernel)
            X = torch.randn(20, 2)
            y = torch.randn(20, 1)
            
            gp.fit(X, y)
            mean, var = gp.predict(torch.randn(10, 2))
            
            assert mean.shape == (10, 1)
            assert var.shape == (10,)
