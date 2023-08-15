from ..utils import BaseTestClass
import numpy as np
import pandas as pd
import lightgbm as lgb


class TestClass(BaseTestClass):
    def test_predict_dist(self, dist_class_univariate_continuous_discrete):
        if dist_class_univariate_continuous_discrete.dist.tau is None:
            # Create data for testing
            np.random.seed(123)
            X_dta = pd.DataFrame(np.random.rand(100).reshape(-1, 1))
            y_dta = np.random.rand(100)
            dtrain = lgb.Dataset(X_dta, label=y_dta)

            # Train the model
            params = {"eta": 0.01}
            dist_class_univariate_continuous_discrete.train(params, dtrain, num_boost_round=2)

            # Call the function
            predt_df = dist_class_univariate_continuous_discrete.dist.generate_probabilities(
                dist_class_univariate_continuous_discrete.booster,
                X_dta,
                dist_class_univariate_continuous_discrete.start_values,
                y=y_dta)

            # Assertions
            assert isinstance(predt_df, pd.DataFrame)
            assert not predt_df.isna().any().any()
            assert not np.isinf(predt_df).any().any()
            assert predt_df.shape[1] == 1
            assert all(predt_df < 1)
            assert all(predt_df > 0)
