from ..utils import BaseTestClass
import pytest
from lightgbmlss.distributions.Mixture import *
from lightgbmlss.distributions.Gaussian import *
from lightgbmlss.distributions.SplineFlow import *


@pytest.fixture
def example_component():
    return Gaussian()


class TestClass(BaseTestClass):

    def test_get_component_distributions(self, mixture_dist):
        assert isinstance(get_component_distributions(), list)
        assert all(isinstance(dist, str) for dist in get_component_distributions())

    def test_init(self, mixture_dist, example_component):
        with pytest.raises(ValueError, match="component_distribution must be one of the following:"):
            mixture_dist(component_distribution=SplineFlow())

        with pytest.raises(ValueError, match="M must be an integer."):
            mixture_dist(example_component, M=2.0)
            mixture_dist(example_component, M="2.0")

        with pytest.raises(ValueError, match="M must be greater than 1."):
            mixture_dist(example_component, M=1)

        with pytest.raises(ValueError, match="Loss for component_distribution must be 'nll'."):
            mixture_dist(component_distribution=Gaussian(loss_fn="crps"))

        with pytest.raises(ValueError, match="hessian_mode must be a string."):
            mixture_dist(example_component, hessian_mode=1)

        with pytest.raises(ValueError, match="hessian_mode must be either 'individual' or 'grouped'."):
            mixture_dist(example_component, hessian_mode="invalid_hessian_mode")

        with pytest.raises(ValueError, match="tau must be a float."):
            mixture_dist(example_component, tau=1)

        with pytest.raises(ValueError, match="tau must be greater than 0."):
            mixture_dist(example_component, tau=0.0)

        # Test initialize parameter validation
        with pytest.raises(ValueError, match="Invalid initialize. Please choose from True or False."):
            mixture_dist(example_component, initialize="True")
        with pytest.raises(ValueError, match="Invalid initialize. Please choose from True or False."):
            mixture_dist(example_component, initialize=1)
        with pytest.raises(ValueError, match="Invalid initialize. Please choose from True or False."):
            mixture_dist(example_component, initialize=0)

        # Test default and valid values
        assert isinstance(mixture_dist(example_component).initialize, bool)
        assert mixture_dist(example_component).initialize is False
        assert mixture_dist(example_component, initialize=True).initialize is True

    def test_distribution_parameters(self, mixture_dist, example_component):
        assert isinstance(mixture_dist(example_component).param_dict, dict)
        assert all(callable(func) for func in mixture_dist(example_component).param_dict.values())

    def test_defaults(self, mixture_dist, example_component):
        assert isinstance(mixture_dist(example_component).univariate, bool)
        assert mixture_dist(example_component).univariate is True
        assert isinstance(mixture_dist(example_component).discrete, bool)
