import lightgbm as lgb
import numpy as np
import torch
from torch.distributions import Normal
from sklearn.model_selection import KFold, train_test_split
from lightgbmlss.distributions.distribution_utils import DistributionClass
from lightgbmlss.model import *
from lightgbmlss.distributions.Gaussian import *


# Test function for LightGBMLSS with natural gradient
def test_lightgbmlss_with_natural_gradient():
    # Create a synthetic dataset
    np.random.seed(123)
    X, y = np.random.rand(1000, 10), np.random.normal(0, 1, 1000)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Define the distribution and model
    dist = Gaussian(stabilization="None",
                        response_fn = "exp",
                        loss_fn = "nll",
                        natural_gradient = True)
    model = LightGBMLSS(dist=dist)

    # Define training parameters
    params = {
        "verbosity": -1,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9
    }

    # Train the model
    model.train(
        params=params,
        train_set=train_data,
        num_boost_round=1,
        valid_sets=[train_data, test_data],
    )

    # Predict and evaluate
    y_pred = model.predict(pd.DataFrame(X_test))['loc'].values
    print(y_pred)

# Run the test
if __name__ == "__main__":
    test_lightgbmlss_with_natural_gradient()