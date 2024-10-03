import numpy as np
import pandas as pd

from lightgbmlss.model import *
from lightgbmlss.distributions.Gaussian import *
from lightgbmlss.distributions.Mixture import *
from lightgbmlss.distributions.Expectile import *
from lightgbmlss.distributions.SplineFlow import *
from lightgbmlss.datasets.data_loader import load_simulated_gaussian_data
import pytest
from pytest import approx
from lightgbmlss.utils import identity_fn


@pytest.fixture
def univariate_data():
    train, test = load_simulated_gaussian_data()
    X_train, y_train = train.filter(regex="x"), train["y"].values
    X_test, y_test = test.filter(regex="x"), test["y"].values
    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test)
    deval = lgb.Dataset(X_test, label=y_test)

    return dtrain, dtest, deval, X_test


@pytest.fixture
def univariate_lgblss():
    return LightGBMLSS(Gaussian())


@pytest.fixture
def mixture_lgblss():
    return LightGBMLSS(Mixture(Gaussian()))


@pytest.fixture
def flow_lgblss():
    return LightGBMLSS(
        SplineFlow(target_support="real",
                   count_bins=2
                   )
    )


@pytest.fixture
def expectile_lgblss():
    return LightGBMLSS(Expectile())


@pytest.fixture
def univariate_params():
    opt_params = {
        "eta": 0.06554395841226755,
        "max_depth": 3,
        "num_leaves": 255,
        "min_data_in_leaf": 20,
        "min_gain_to_split": 5.76808477078835,
        "min_sum_hessian_in_leaf": 9.446692680480123e-05,
        "subsample": 0.3022671193739115,
        "feature_fraction": 0.7869489723419915,
        "boosting": "gbdt"
    }
    n_rounds = 46

    return opt_params, n_rounds


@pytest.fixture
def expectile_params():
    opt_params = {
        "eta": 0.669098091972402,
        "max_depth": 2,
        "num_leaves": 255,
        "min_data_in_leaf": 20,
        "min_gain_to_split": 33.016324935465434,
        "min_sum_hessian_in_leaf": 60.4377077445418,
        "subsample": 0.8748337817075426,
        "feature_fraction": 0.9497140456000938,
        "boosting": "gbdt"
    }
    n_rounds = 2

    return opt_params, n_rounds


class TestClass:
    def test_model_univ_train(self, univariate_data, univariate_lgblss, univariate_params):
        # Unpack
        dtrain, _, _, _ = univariate_data
        opt_params, n_rounds = univariate_params
        lgblss = univariate_lgblss

        # Train the model
        lgblss.train(opt_params, dtrain, n_rounds)

        # Assertions
        assert isinstance(lgblss.booster, lgb.Booster)

    def test_model_univ_train_eval(self, univariate_data, univariate_lgblss, univariate_params):
        # Unpack
        dtrain, dtest, deval, _ = univariate_data
        opt_params, n_rounds = univariate_params
        lgblss = univariate_lgblss

        # Add evaluation set
        valid_sets = [dtrain, deval]
        valid_names = ["train", "evaluation"]

        # Train the model
        lgblss.train(opt_params, dtrain, n_rounds, valid_sets=valid_sets, valid_names=valid_names)

        # Assertions
        assert isinstance(lgblss.booster, lgb.Booster)

    def test_model_hpo(self, univariate_data, univariate_lgblss, ):
        # Unpack
        dtrain, _, _, _ = univariate_data
        lgblss = univariate_lgblss

        # Create hyperparameter dictionary
        param_dict = {
            "eta": ["float", {"low": 1e-5, "high": 1, "log": True}],
            "max_depth": ["int", {"low": 1, "high": 2, "log": False}],
            "device_type": ["categorical", ["cpu"]],
        }

        # Train the model
        np.random.seed(123)
        opt_param = lgblss.hyper_opt(
            param_dict,
            dtrain,
            num_boost_round=10,
            nfold=5,
            early_stopping_rounds=20,
            max_minutes=10,
            n_trials=5,
            silence=True,
            seed=123,
            hp_seed=123
        )

        # Assertions
        assert isinstance(opt_param, dict)

    def test_model_predict(self, univariate_data, univariate_lgblss, univariate_params):
        # Unpack
        dtrain, _, _, X_test = univariate_data
        opt_params, n_rounds = univariate_params
        lgblss = univariate_lgblss

        # Train the model
        lgblss.train(opt_params, dtrain, n_rounds)

        # Call the predict method
        n_samples = 100
        quantiles = [0.1, 0.5, 0.9]

        pred_params = lgblss.predict(X_test, pred_type="parameters")
        pred_samples = lgblss.predict(X_test, pred_type="samples", n_samples=n_samples)
        pred_quantiles = lgblss.predict(X_test, pred_type="quantiles", quantiles=quantiles)
        pred_contributions = lgblss.predict(X_test, pred_type="contributions")

        # Assertions
        assert isinstance(pred_params, (pd.DataFrame, type(None)))
        assert not pred_params.isna().any().any()
        assert not np.isinf(pred_params).any().any()
        assert pred_params.shape[1] == lgblss.dist.n_dist_param
        assert approx(pred_params["loc"].mean(), abs=0.2) == 10.0
        assert pred_params.columns.name == "parameters"

        assert isinstance(pred_samples, (pd.DataFrame, type(None)))
        assert not pred_samples.isna().any().any()
        assert not np.isinf(pred_samples).any().any()
        assert pred_samples.shape[1] == n_samples
        assert pred_samples.columns.name == "samples"

        assert isinstance(pred_quantiles, (pd.DataFrame, type(None)))
        assert not pred_quantiles.isna().any().any()
        assert not np.isinf(pred_quantiles).any().any()
        assert pred_quantiles.shape[1] == len(quantiles)
        assert pred_quantiles.columns.name == "quantiles"

        assert isinstance(pred_contributions, (pd.DataFrame, type(None)))
        assert not pred_contributions.isna().any().any()
        assert not np.isinf(pred_contributions).any().any()
        assert (pred_contributions.shape[1] ==
                lgblss.dist.n_dist_param * (X_test.shape[1] + 1)
                )

        assert pred_contributions.columns.names == ["parameters", "feature_contributions"]

        for key, response_func in lgblss.dist.param_dict.items():
            # Sum contributions for each parameter and apply response function
            pred_contributions_combined = (
                pd.Series(response_func(
                    torch.tensor(
                        pred_contributions.xs(key, level="parameters", axis=1).sum(axis=1).values)
                )))
            assert np.allclose(
                pred_contributions_combined,
                pred_params[key], atol=1e-5
            )


    def test_model_plot(self, univariate_data, univariate_lgblss, univariate_params):
        # Unpack
        dtrain, dtest, _, X_test = univariate_data
        opt_params, n_rounds = univariate_params
        lgblss = univariate_lgblss

        # Train the model
        lgblss.train(opt_params, dtrain, n_rounds)

        # Call the function
        lgblss.plot(X_test, parameter="scale", feature="x_true", plot_type="Partial_Dependence")
        lgblss.plot(X_test, parameter="scale", feature="x_true", plot_type="Feature_Importance")

    def test_model_expectile_plot(self, univariate_data, expectile_lgblss, expectile_params):
        # Unpack
        dtrain, dtest, _, X_test = univariate_data
        opt_params, n_rounds = expectile_params
        lgblss_expectile = expectile_lgblss

        # Train the model
        lgblss_expectile.train(opt_params, dtrain, n_rounds)

        # Call the function
        lgblss_expectile.expectile_plot(X_test,
                                        expectile="expectile_0.9",
                                        feature="x_true",
                                        plot_type="Partial_Dependence")

        lgblss_expectile.expectile_plot(X_test,
                                        expectile="expectile_0.9",
                                        feature="x_true",
                                        plot_type="Feature_Importance")

    def test_model_mixture_train(self, univariate_data, mixture_lgblss):
        # Unpack
        dtrain, _, _, _ = univariate_data
        params, n_rounds = {"eta": 0.1}, 10
        lgblss = mixture_lgblss

        # Train the model
        lgblss.train(params, dtrain, n_rounds)

        # Assertions
        assert isinstance(lgblss.booster, lgb.Booster)

    def test_model_flow_train(self, univariate_data, flow_lgblss):
        # Unpack
        dtrain, _, _, _ = univariate_data
        params, n_rounds = {"eta": 0.1}, 10
        lgblss = flow_lgblss

        # Train the model
        lgblss.train(params, dtrain, n_rounds)

        # Assertions
        assert isinstance(lgblss.booster, lgb.Booster)
