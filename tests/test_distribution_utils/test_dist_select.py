import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from ..utils import BaseTestClass

from lightgbmlss.distributions import (
    Beta,
    Gaussian,
    StudentT,
    Gamma,
    Cauchy,
    LogNormal,
    Weibull,
    Gumbel,
    Laplace)
from lightgbmlss.distributions.Mixture import *
from lightgbmlss.distributions.SplineFlow import *
from lightgbmlss.distributions.distribution_utils import DistributionClass as univariate_dist_class
from lightgbmlss.distributions.flow_utils import NormalizingFlowClass as flow_dist_class
from lightgbmlss.distributions.mixture_distribution_utils import MixtureDistributionClass as mixture_dist_class

import numpy as np
import pandas as pd


class TestClass(BaseTestClass):
    ####################################################################################################################
    # Univariate Distribution
    ####################################################################################################################
    def test_univar_dist_select(self):
        # Create data for testing
        target = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        candidate_distributions = [Beta, Gaussian, StudentT, Gamma, Cauchy, LogNormal, Weibull, Gumbel, Laplace]

        # Call the function
        dist_df = univariate_dist_class().dist_select(
            target, candidate_distributions, plot=False, max_iter=2
        ).reset_index(drop=True)

        # Assertions
        assert isinstance(dist_df, pd.DataFrame)
        assert not dist_df.isna().any().any()
        assert isinstance(dist_df["distribution"].values[0], str)
        assert np.issubdtype(dist_df["nll"].dtype, np.float64)
        assert not np.isnan(dist_df["nll"].values).any()
        assert not np.isinf(dist_df["nll"].values).any()

    @pytest.mark.skipif(
        not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
        reason="matplotlib and seaborn are required to run this test."
    )
    def test_univar_dist_select_plot(self):
        # Create data for testing
        target = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        candidate_distributions = [Beta, Gaussian, StudentT, Gamma, Cauchy, LogNormal, Weibull, Gumbel, Laplace]

        # Call the function
        dist_df = univariate_dist_class().dist_select(
            target, candidate_distributions, plot=True, max_iter=2
        ).reset_index(drop=True)

        # Assertions
        assert isinstance(dist_df, pd.DataFrame)
        assert not dist_df.isna().any().any()
        assert isinstance(dist_df["distribution"].values[0], str)
        assert np.issubdtype(dist_df["nll"].dtype, np.float64)
        assert not np.isnan(dist_df["nll"].values).any()
        assert not np.isinf(dist_df["nll"].values).any()

    ####################################################################################################################
    # Normalizing Flows
    ####################################################################################################################
    def test_flow_select(self):
        # Create data for testing
        target = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        bound = np.max([np.abs(target.min()), target.max()])
        target_support = "real"

        candidate_flows = [
            SplineFlow(target_support=target_support, count_bins=2, bound=bound, order="linear"),
            SplineFlow(target_support=target_support, count_bins=2, bound=bound, order="quadratic")
        ]

        # Call the function
        dist_df = flow_dist_class().flow_select(
            target, candidate_flows, plot=False, max_iter=2
        ).reset_index(drop=True)

        # Assertions
        assert isinstance(dist_df, pd.DataFrame)
        assert not dist_df.isna().any().any()
        assert isinstance(dist_df["NormFlow"].values[0], str)
        assert np.issubdtype(dist_df["nll"].dtype, np.float64)
        assert not np.isnan(dist_df["nll"].values).any()
        assert not np.isinf(dist_df["nll"].values).any()

    @pytest.mark.skipif(
        not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
        reason="matplotlib and seaborn are required to run this test."
    )
    def test_flow_select_plot(self):
        # Create data for testing
        target = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        bound = np.max([np.abs(target.min()), target.max()])
        target_support = "real"

        candidate_flows = [
            SplineFlow(target_support=target_support, count_bins=2, bound=bound, order="linear"),
            SplineFlow(target_support=target_support, count_bins=2, bound=bound, order="quadratic")
        ]

        # Call the function
        dist_df = flow_dist_class().flow_select(
            target, candidate_flows, plot=True, max_iter=2
        ).reset_index(drop=True)

        # Assertions
        assert isinstance(dist_df, pd.DataFrame)
        assert not dist_df.isna().any().any()
        assert isinstance(dist_df["NormFlow"].values[0], str)
        assert np.issubdtype(dist_df["nll"].dtype, np.float64)
        assert not np.isnan(dist_df["nll"].values).any()
        assert not np.isinf(dist_df["nll"].values).any()

    ####################################################################################################################
    # Mixture Distributions
    ####################################################################################################################
    def test_mixture_dist_select(self):
        # Create data for testing
        target = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        candidate_distributions = [
            Mixture(Beta.Beta()),
            Mixture(Gaussian.Gaussian()),
            Mixture(StudentT.StudentT()),
            Mixture(Gamma.Gamma()),
            Mixture(Cauchy.Cauchy()),
            Mixture(LogNormal.LogNormal()),
            Mixture(Weibull.Weibull()),
            Mixture(Gumbel.Gumbel()),
            Mixture(Laplace.Laplace())
        ]

        # Call the function
        dist_df = mixture_dist_class().dist_select(
            target, candidate_distributions, plot=False, max_iter=2
        ).reset_index(drop=True)

        # Assertions
        assert isinstance(dist_df, pd.DataFrame)
        assert not dist_df.isna().any().any()
        assert isinstance(dist_df["distribution"].values[0], str)
        assert np.issubdtype(dist_df["nll"].dtype, np.float64)
        assert not np.isnan(dist_df["nll"].values).any()
        assert not np.isinf(dist_df["nll"].values).any()

    @pytest.mark.skipif(
        not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
        reason="matplotlib and seaborn are required to run this test."
    )
    def test_mixture_dist_select_plot(self):
        # Create data for testing
        target = np.array([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
        candidate_distributions = [
            Mixture(Beta.Beta()),
            Mixture(Gaussian.Gaussian()),
            Mixture(StudentT.StudentT()),
            Mixture(Gamma.Gamma()),
            Mixture(Cauchy.Cauchy()),
            Mixture(LogNormal.LogNormal()),
            Mixture(Weibull.Weibull()),
            Mixture(Gumbel.Gumbel()),
            Mixture(Laplace.Laplace())
        ]

        # Call the function
        dist_df = mixture_dist_class().dist_select(
            target, candidate_distributions, plot=True, max_iter=2
        ).reset_index(drop=True)

        # Assertions
        assert isinstance(dist_df, pd.DataFrame)
        assert not dist_df.isna().any().any()
        assert isinstance(dist_df["distribution"].values[0], str)
        assert np.issubdtype(dist_df["nll"].dtype, np.float64)
        assert not np.isnan(dist_df["nll"].values).any()
        assert not np.isinf(dist_df["nll"].values).any()
