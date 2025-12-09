import torch
from ..utils import BaseTestClass
import pytest
import pandas as pd
import numpy as np


class TestClass(BaseTestClass):
    @pytest.mark.parametrize("n_obs", [1, 5])
    def test_draw_samples(self, dist_class, n_obs):
        # Create data for testing with n_obs observations
        predt_params = pd.DataFrame(
            np.array(
                [[0.5 for _ in range(dist_class.dist.n_dist_param)] for _ in range(n_obs)],
                dtype="float32",
            )
        )
        # Call the function
        dist_samples = dist_class.dist.draw_samples(predt_params)

        # Assertions
        if str(dist_class.dist).split(".")[2] != "Expectile":
            assert isinstance(dist_samples, (pd.DataFrame, type(None)))
            # row count must match number of observations
            assert dist_samples.shape[0] == predt_params.shape[0]
            assert not dist_samples.isna().any().any()
            assert not np.isinf(dist_samples).any().any()

    @pytest.mark.parametrize("n_obs", [1, 5])
    def test_draw_samples_mixture(self, mixture_class, n_obs):
        # Create data for testing with n_obs observations
        predt_params = pd.DataFrame(
            np.array(
                [[0.5 for _ in range(mixture_class.dist.n_dist_param)] for _ in range(n_obs)],
                dtype="float32",
            )
        )
        # Call the function
        dist_samples = mixture_class.dist.draw_samples(predt_params)

        # Assertions
        assert isinstance(dist_samples, (pd.DataFrame, type(None)))
        # row count must match number of observations
        assert dist_samples.shape[0] == predt_params.shape[0]
        assert not dist_samples.isna().any().any()
        assert not np.isinf(dist_samples).any().any()
