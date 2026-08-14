from ..utils import BaseTestClass
import pytest
import torch
import inspect


class TestClass(BaseTestClass):
    @staticmethod
    def supports_natural_gradient(dist_class):
        """Check if distribution supports natural_gradient parameter."""
        sig = inspect.signature(dist_class.__init__)
        return 'natural_gradient' in sig.parameters
    
    def test_natural_gradient_parameter(self, univariate_cont_dist):
        """Test that natural_gradient parameter is properly handled."""
        if not self.supports_natural_gradient(univariate_cont_dist):
            pytest.skip(f"{univariate_cont_dist.__name__} doesn't support natural_gradient parameter")
        
        # Test default value
        dist_default = univariate_cont_dist()
        assert isinstance(dist_default.natural_gradient, bool)
        assert dist_default.natural_gradient is False
        
        # Test explicit True/False
        dist_true = univariate_cont_dist(natural_gradient=True)
        assert dist_true.natural_gradient is True
        
        dist_false = univariate_cont_dist(natural_gradient=False)
        assert dist_false.natural_gradient is False
    
    def test_natural_gradient_affects_gradient_computation(self, univariate_cont_dist):
        """Test that natural_gradient enables FIM computation."""
        if not self.supports_natural_gradient(univariate_cont_dist):
            pytest.skip(f"{univariate_cont_dist.__name__} doesn't support natural_gradient parameter")
        
        # Create test data
        n_params = univariate_cont_dist().n_dist_param
        predt = [torch.randn(100, requires_grad=True) for _ in range(n_params)]
        
        # Test that natural gradient distribution can compute FIM
        dist_natural = univariate_cont_dist(natural_gradient=True)
        fim = dist_natural.compute_fisher_information_matrix(predt)
        
        # Verify FIM is computed correctly
        assert fim is not None
        assert len(fim) == n_params
        
        # Verify each FIM element has correct shape and is positive
        for i, fim_i in enumerate(fim):
            assert fim_i.shape == predt[i].shape
            assert torch.all(fim_i > 0), f"FIM element {i} should be positive"
            assert torch.all(torch.isfinite(fim_i)), f"FIM element {i} should be finite"