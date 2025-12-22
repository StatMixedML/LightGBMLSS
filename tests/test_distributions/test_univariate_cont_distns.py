from ..utils import BaseTestClass
import pytest
import torch


class TestClass(BaseTestClass):
    def test_init(self, univariate_cont_dist):
        assert isinstance(univariate_cont_dist().stabilization, str)
        assert univariate_cont_dist().stabilization is not None
        with pytest.raises(ValueError, match="Invalid stabilization method."):
            univariate_cont_dist(stabilization="invalid_stabilization")

        with pytest.raises(ValueError, match="Invalid response function."):
            univariate_cont_dist(response_fn="invalid_response_fn")

        assert isinstance(univariate_cont_dist().loss_fn, str)
        assert univariate_cont_dist().loss_fn is not None
        with pytest.raises(ValueError, match="Invalid loss function."):
            univariate_cont_dist(loss_fn="invalid_loss_fn")

        # Test initialize parameter validation (TYPE CHECKS)
        with pytest.raises(ValueError, match="Invalid initialize. Please choose from True or False."):
            univariate_cont_dist(initialize="True")
        with pytest.raises(ValueError, match="Invalid initialize. Please choose from True or False."):
            univariate_cont_dist(initialize=1)
        with pytest.raises(ValueError, match="Invalid initialize. Please choose from True or False."):
            univariate_cont_dist(initialize=0)
        with pytest.raises(ValueError, match="Invalid initialize. Please choose from True or False."):
            univariate_cont_dist(initialize=None)

    def test_distribution_parameters(self, univariate_cont_dist):
        assert isinstance(univariate_cont_dist().param_dict, dict)
        assert set(univariate_cont_dist().param_dict.keys()) == set(univariate_cont_dist().distribution_arg_names)
        assert all(callable(func) for func in univariate_cont_dist().param_dict.values())
        assert univariate_cont_dist().n_dist_param == len(univariate_cont_dist().distribution_arg_names)
        assert isinstance(univariate_cont_dist().n_dist_param, int)
        assert isinstance(univariate_cont_dist().distribution_arg_names, list)
        assert univariate_cont_dist().distribution_arg_names == list(univariate_cont_dist().distribution.arg_constraints.keys())

    def test_defaults(self, univariate_cont_dist):
        assert isinstance(univariate_cont_dist().univariate, bool)
        assert univariate_cont_dist().univariate is True
        assert isinstance(univariate_cont_dist().discrete, bool)
        assert univariate_cont_dist().discrete is False
        assert univariate_cont_dist().tau is None
        assert isinstance(univariate_cont_dist().penalize_crossing, bool)

        # Test initialize default value
        assert isinstance(univariate_cont_dist().initialize, bool)
        assert univariate_cont_dist().initialize is False
        assert univariate_cont_dist(initialize=True).initialize is True
        assert univariate_cont_dist(initialize=False).initialize is False

    def test_fisher_information_matrix_exp(self, univariate_cont_dist):
        """Test FIM computation with exp response function."""
        dist = univariate_cont_dist(response_fn="exp")
        
        # Create test data with appropriate dimensions
        n_params = dist.n_dist_param
        eta_tensors = [torch.tensor([0.0, 1.0, -1.0]) for _ in range(n_params)]
        
        # Compute FIM
        fim = dist.compute_fisher_information_matrix(eta_tensors)
        
        # Assertions
        assert len(fim) == n_params
        for i, f in enumerate(fim):
            assert f.shape == eta_tensors[i].shape
            # All FIM values should be positive
            assert torch.all(f > 0), f"FIM values for parameter {i} should be positive"
            # Should not contain NaN or Inf
            assert not torch.any(torch.isnan(f)), f"FIM contains NaN for parameter {i}"
            assert not torch.any(torch.isinf(f)), f"FIM contains Inf for parameter {i}"

    def test_fisher_information_matrix_softplus(self, univariate_cont_dist):
        """Test FIM computation with softplus response function."""
        dist = univariate_cont_dist(response_fn="softplus")
        
        # Create test data with appropriate dimensions
        n_params = dist.n_dist_param
        eta_tensors = [torch.tensor([0.5, 1.0, 1.5]) for _ in range(n_params)]
        
        # Compute FIM
        fim = dist.compute_fisher_information_matrix(eta_tensors)
        
        # Assertions
        assert len(fim) == n_params
        for i, f in enumerate(fim):
            assert f.shape == eta_tensors[i].shape
            # All FIM values should be positive
            assert torch.all(f > 0), f"FIM values for parameter {i} should be positive"
            # Should not contain NaN or Inf
            assert not torch.any(torch.isnan(f)), f"FIM contains NaN for parameter {i}"
            assert not torch.any(torch.isinf(f)), f"FIM contains Inf for parameter {i}"

    def test_fisher_information_matrix_numerical_stability(self, univariate_cont_dist):
        """Test FIM computation with extreme values."""
        dist = univariate_cont_dist(response_fn="exp")
        
        # Test with very small and large values
        n_params = dist.n_dist_param
        eta_tensors = [torch.tensor([-10.0, 0.0, 10.0]) for _ in range(n_params)]
        
        # Compute FIM
        fim = dist.compute_fisher_information_matrix(eta_tensors)
        
        # Assertions
        for i, f in enumerate(fim):
            # Should not contain NaN or Inf
            assert not torch.any(torch.isnan(f)), f"FIM contains NaN for parameter {i}"
            assert not torch.any(torch.isinf(f)), f"FIM contains Inf for parameter {i}"
            # All values should be positive
            assert torch.all(f > 0), f"FIM values for parameter {i} should be positive"

    def test_fisher_information_matrix_batch_dimensions(self, univariate_cont_dist):
        """Test FIM computation with different batch dimensions."""
        dist = univariate_cont_dist(response_fn="exp")
        
        # Test with different batch sizes
        for batch_size in [1, 10, 100]:
            eta_tensors = [torch.randn(batch_size) for _ in range(dist.n_dist_param)]
            
            fim = dist.compute_fisher_information_matrix(eta_tensors)
            
            assert len(fim) == dist.n_dist_param
            for i, f in enumerate(fim):
                assert f.shape == (batch_size,)
                assert torch.all(f > 0), f"FIM values for parameter {i} should be positive"
