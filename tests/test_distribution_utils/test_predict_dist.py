from ..utils import BaseTestClass
import numpy as np
import pandas as pd
import lightgbm as lgb


class TestClass(BaseTestClass):
    ####################################################################################################################
    # Univariate Distribution
    ####################################################################################################################
    def test_predict_dist_univariate(self, dist_class, pred_type):
        if dist_class.dist.univariate and not hasattr(dist_class.dist, "base_dist"):
            # Create data for testing
            np.random.seed(123)
            X_dta = pd.DataFrame(np.random.rand(100).reshape(-1, 1))
            y_dta = np.random.rand(100)
            dtrain = lgb.Dataset(X_dta, label=y_dta)

            # Train the model
            params = {"eta": 0.01}
            dist_class.train(params, dtrain, num_boost_round=2)

            # Call the function
            if dist_class.dist.tau is not None and pred_type in ["quantiles", "samples"]:
                pred_type = "parameters"
            predt_df = dist_class.dist.predict_dist(dist_class.booster,
                                                    X_dta,
                                                    dist_class.start_values,
                                                    pred_type,
                                                    n_samples=100,
                                                    quantiles=[0.1, 0.5, 0.9]
                                                    )

            # Assertions
            assert isinstance(predt_df, pd.DataFrame)
            assert not predt_df.isna().any().any()
            assert not np.isinf(predt_df).any().any()
            if pred_type == "parameters" or pred_type == "expectiles":
                assert predt_df.shape[1] == dist_class.dist.n_dist_param
            if dist_class.dist.tau is None:
                if pred_type == "samples":
                    assert predt_df.shape[1] == 100
                elif pred_type == "quantiles":
                    assert predt_df.shape[1] == 3

    ####################################################################################################################
    # Normalizing Flow
    ####################################################################################################################
    def test_predict_dist_flow(self, flow_class, pred_type):
        # Create data for testing
        np.random.seed(123)
        X_dta = pd.DataFrame(np.random.rand(100).reshape(-1, 1))
        y_dta = np.random.rand(100)
        dtrain = lgb.Dataset(X_dta, label=y_dta)

        # Train the model
        params = {"eta": 0.01}
        flow_class.train(params, dtrain, num_boost_round=2)

        # Call the function
        if pred_type in ["expectiles"]:
            pred_type = "parameters"
        predt_df = flow_class.dist.predict_dist(flow_class.booster,
                                                X_dta,
                                                flow_class.start_values,
                                                pred_type,
                                                n_samples=100,
                                                quantiles=[0.1, 0.5, 0.9]
                                                )

        # Assertions
        assert isinstance(predt_df, pd.DataFrame)
        assert not predt_df.isna().any().any()
        assert not np.isinf(predt_df).any().any()
        if pred_type == "parameters" or pred_type == "expectiles":
            assert predt_df.shape[1] == flow_class.dist.n_dist_param
        if pred_type == "samples":
            assert predt_df.shape[1] == 100
        elif pred_type == "quantiles":
            assert predt_df.shape[1] == 3

    ####################################################################################################################
    # Mixture Distributions
    ####################################################################################################################
    def test_predict_dist_mixture(self, mixture_class, pred_type):
        # Create data for testing
        np.random.seed(123)
        X_dta = np.random.rand(100).reshape(-1, 1)
        y_dta = np.random.rand(100)
        dtrain = lgb.Dataset(X_dta, label=y_dta)

        # Train the model
        params = {"eta": 0.01}
        mixture_class.train(params, dtrain, num_boost_round=2)

        # Call the function
        if pred_type in ["expectiles"]:
            pred_type = "parameters"
        predt_df = mixture_class.dist.predict_dist(mixture_class.booster,
                                                   X_dta,
                                                   mixture_class.start_values,
                                                   pred_type,
                                                   n_samples=100,
                                                   quantiles=[0.1, 0.5, 0.9]
                                                   )

        # Assertions
        assert isinstance(predt_df, pd.DataFrame)
        assert not predt_df.isna().any().any()
        assert not np.isinf(predt_df).any().any()
        if pred_type == "parameters" or pred_type == "expectiles":
            assert predt_df.shape[1] == mixture_class.dist.n_dist_param
        if pred_type == "samples":
            assert predt_df.shape[1] == 100
        elif pred_type == "quantiles":
            assert predt_df.shape[1] == 3
