from ..utils import BaseTestClass
import pandas as pd
import numpy as np


class TestClass(BaseTestClass):
    def test_draw_samples(self, dist_class):
        # Create data for testing
        predt_params = pd.DataFrame(np.array([0.5 for _ in range(dist_class.dist.n_dist_param)], dtype="float32")).T

        # Call the function
        dist_samples = dist_class.dist.draw_samples(predt_params)

        # Assertions
        if str(dist_class.dist).split(".")[2] != "Expectile":
            assert isinstance(dist_samples, (pd.DataFrame, type(None)))
            assert not dist_samples.isna().any().any()
            assert not np.isinf(dist_samples).any().any()

    def test_draw_samples_mixture(self, mixture_class):
        # Create data for testing
        predt_params = pd.DataFrame(np.array([0.5 for _ in range(mixture_class.dist.n_dist_param)], dtype="float32")).T

        # Call the function
        dist_samples = mixture_class.dist.draw_samples(predt_params)

        # Assertions
        assert isinstance(dist_samples, (pd.DataFrame, type(None)))
        assert not dist_samples.isna().any().any()
        assert not np.isinf(dist_samples).any().any()
