from ..utils import BaseTestClass
import pytest


class TestBernsteinFlowClass(BaseTestClass):
    """Test class for BernsteinFlow distribution"""
    
    def test_init_bernstein(self):
        """Test initialization parameters specific to BernsteinFlow"""
        from lightgbmlss.distributions.BernsteinFlow import BernsteinFlow
        
        # Test valid initialization
        dist = BernsteinFlow()
        assert dist.degree == 8
        assert dist.support_bounds == (-5.0, 5.0)
        assert dist.n_dist_param == 9  # degree + 1
        
        # Test target_support validation
        with pytest.raises(ValueError, match="target_support must be a string."):
            BernsteinFlow(target_support=1)
        with pytest.raises(ValueError, match="Invalid target_support."):
            BernsteinFlow(target_support="invalid_target_support")
            
        # Test degree validation
        with pytest.raises(ValueError, match="degree must be an integer."):
            BernsteinFlow(degree=1.0)
        with pytest.raises(ValueError, match="degree must be a positive integer > 0."):
            BernsteinFlow(degree=0)
        with pytest.raises(ValueError, match="degree should be <= 20 for numerical stability."):
            BernsteinFlow(degree=25)
            
        # Test bound validation  
        with pytest.raises(ValueError, match="bound must be positive."):
            BernsteinFlow(bound=-1.0)
            
        # Test stabilization validation
        with pytest.raises(ValueError, match="stabilization must be a string."):
            BernsteinFlow(stabilization=1)
        with pytest.raises(ValueError, match="Invalid stabilization method."):
            BernsteinFlow(stabilization="invalid_stabilization")
            
        # Test loss_fn validation
        with pytest.raises(ValueError, match="loss_fn must be a string."):
            BernsteinFlow(loss_fn=1)
        with pytest.raises(ValueError, match="Invalid loss_fn."):
            BernsteinFlow(loss_fn="invalid_loss_fn")
    
    def test_parameter_dictionary(self):
        """Test that parameter dictionary is correctly set up"""
        from lightgbmlss.distributions.BernsteinFlow import BernsteinFlow
        
        dist = BernsteinFlow(degree=5)
        assert isinstance(dist.param_dict, dict)
        assert len(dist.param_dict) == 6  # degree + 1
        assert all(f"beta_{i}" in dist.param_dict for i in range(6))
        assert all(callable(func) for func in dist.param_dict.values())
        
    def test_different_degrees(self):
        """Test BernsteinFlow with different polynomial degrees"""
        from lightgbmlss.distributions.BernsteinFlow import BernsteinFlow
        
        degrees = [3, 5, 10, 15]
        for degree in degrees:
            dist = BernsteinFlow(degree=degree)
            assert dist.degree == degree
            assert dist.n_dist_param == degree + 1
            assert len(dist.param_dict) == degree + 1
    
    def test_target_supports(self):
        """Test different target supports"""
        from lightgbmlss.distributions.BernsteinFlow import BernsteinFlow
        
        supports = ["real", "positive", "positive_integer", "unit_interval"]
        for support in supports:
            dist = BernsteinFlow(target_support=support)
            assert dist.target_transform is not None
            
    def test_transform_properties(self):
        """Test that the BernsteinQuantileTransform is properly initialized"""
        from lightgbmlss.distributions.BernsteinFlow import BernsteinFlow, BernsteinQuantileTransform
        import torch
        
        # Create transform directly
        transform = BernsteinQuantileTransform(degree=5, support_bounds=(-3.0, 3.0))
        assert transform.degree == 5
        assert transform.support_bounds == (-3.0, 3.0)
        assert transform.raw_betas.shape[0] == 6  # degree + 1
        
        # Test binomial coefficients are computed
        assert transform.binomial_coeffs is not None
        assert len(transform.binomial_coeffs) == 6
        
        # Test monotonicity property
        betas = transform.betas
        assert torch.all(betas[1:] >= betas[:-1])  # Should be non-decreasing
        
    def test_bernstein_basis_functions(self):
        """Test Bernstein basis function computation"""
        from lightgbmlss.distributions.BernsteinFlow import BernsteinQuantileTransform
        import torch
        
        transform = BernsteinQuantileTransform(degree=3)
        u = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        
        # Test that all basis functions sum to 1
        total = torch.zeros_like(u)
        for k in range(4):  # degree + 1
            basis_k = transform._bernstein_basis(u, k)
            total += basis_k
            
        # Should sum to 1 (with some numerical tolerance)
        assert torch.allclose(total, torch.ones_like(u), atol=1e-6)
        
    def test_transform_forward_inverse(self):
        """Test forward and inverse transform"""
        from lightgbmlss.distributions.BernsteinFlow import BernsteinQuantileTransform
        import torch
        
        transform = BernsteinQuantileTransform(degree=5, support_bounds=(-2.0, 2.0))
        
        # Test points in unit interval
        u = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        
        # Forward transform
        y = transform(u)
        assert y.shape == u.shape
        assert torch.all(y >= -2.0) and torch.all(y <= 2.0)  # Should be in support bounds
        
        # Inverse transform (approximate due to numerical method)
        u_reconstructed = transform._inverse(y)
        assert u_reconstructed.shape == u.shape
        assert torch.allclose(u, u_reconstructed, atol=1e-3)  # Allow some numerical error
        
    def test_jacobian_computation(self):
        """Test log absolute determinant of Jacobian"""
        from lightgbmlss.distributions.BernsteinFlow import BernsteinQuantileTransform
        import torch
        
        transform = BernsteinQuantileTransform(degree=5)
        
        u = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        y = transform(u)
        
        # Should be able to compute Jacobian
        log_det_jac = transform.log_abs_det_jacobian(u, y)
        assert log_det_jac.shape == u.shape
        assert torch.all(torch.isfinite(log_det_jac))  # Should not have NaN or inf
    
    def test_defaults(self):
        """Test default values match expected behavior"""
        from lightgbmlss.distributions.BernsteinFlow import BernsteinFlow
        
        dist = BernsteinFlow()
        assert isinstance(dist.univariate, bool)
        assert dist.univariate is True
        assert isinstance(dist.discrete, bool)
        assert dist.discrete is False
        
    def test_distribution_class_integration(self):
        """Test integration with LightGBMLSS model class"""
        from lightgbmlss.distributions.BernsteinFlow import BernsteinFlow
        from lightgbmlss.model import LightGBMLSS
        import numpy as np
        
        # Should be able to create a model with BernsteinFlow
        dist = BernsteinFlow(degree=5)
        model = LightGBMLSS(dist)
        
        # Test that model is properly initialized
        assert model.dist.n_dist_param == 6
        assert model.dist.param_dict is not None
        
        # Test with synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        
        # Should be able to compute starting values
        start_values = model.dist.calculate_start_values(y, max_iter=10)
        assert len(start_values) == 2  # loss and start_values
        assert len(start_values[1]) == 6  # Should match n_dist_param