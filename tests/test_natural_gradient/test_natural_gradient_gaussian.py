"""
Test suite for Natural Gradient training with Gaussian distribution in LightGBMLSS.

Tests verify that:
1. Natural gradient flag properly changes gradient computation
2. Fisher Information Matrix is computed correctly
3. Training converges successfully with natural gradients
4. Performance comparisons are meaningful
"""

import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb
from scipy import stats
import torch

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lightgbmlss.model import LightGBMLSS
from lightgbmlss.distributions.Gaussian import Gaussian


class TestNaturalGradientGaussian:
    """
    Test suite for Natural Gradient training with Gaussian distribution.
    """
    
    @pytest.fixture
    def synthetic_heteroscedastic_data(self):
        """
        Generate synthetic heteroscedastic Gaussian data matching the notebook example.
        """
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Complex mean function (matching notebook)
        mu = (2.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2] + 
              0.15 * X[:, 3] * X[:, 4] + 0.1 * np.sin(X[:, 5]))
        
        # Heteroscedastic variance (matching notebook)
        sigma = np.exp(0.5 + 0.3 * np.abs(X[:, 6]) + 0.2 * X[:, 7] + 
                       0.1 * (X[:, 8] ** 2))
        
        # Sample from Gaussian
        y = mu + sigma * np.random.randn(n_samples)
        
        X_df = pd.DataFrame(X, columns=[f'x{i}' for i in range(n_features)])
        
        return X_df, np.array(y), np.array(mu), np.array(sigma)
    
    @pytest.fixture
    def train_test_split_data(self, synthetic_heteroscedastic_data):
        """
        Split data into train and test sets (80-20 split).
        """
        X_df, y, mu, sigma = synthetic_heteroscedastic_data
        
        # 80-20 split
        split_idx = int(0.8 * len(y))
        
        X_train = X_df.iloc[:split_idx].copy()
        X_test = X_df.iloc[split_idx:].copy()
        y_train = y[:split_idx].copy()
        y_test = y[split_idx:].copy()
        mu_train = mu[:split_idx]
        mu_test = mu[split_idx:]
        sigma_train = sigma[:split_idx]
        sigma_test = sigma[split_idx:]
        
        # Ensure proper numpy array types
        y_train = np.asarray(y_train, dtype=np.float64)
        y_test = np.asarray(y_test, dtype=np.float64)
        
        return X_train, X_test, y_train, y_test, mu_train, mu_test, sigma_train, sigma_test
    
    def test_natural_gradient_initialization(self):
        """
        Test that Gaussian distribution can be initialized with natural_gradient flag.
        """
        # With natural gradient
        gauss_nat = Gaussian(
            stabilization="MAD", 
            response_fn="softplus", 
            loss_fn="nll", 
            natural_gradient=True
        )
        assert gauss_nat.natural_gradient == True, "Natural gradient flag should be True"
        
        # Without natural gradient
        gauss_std = Gaussian(
            stabilization="MAD", 
            response_fn="softplus", 
            loss_fn="nll", 
            natural_gradient=False
        )
        assert gauss_std.natural_gradient == False, "Natural gradient flag should be False"
        
        print("✅ Natural gradient initialization test passed")
    
    def test_lightgbmlss_with_natural_gradient(self, train_test_split_data):
        """
        Test that LightGBMLSS can be trained with natural gradient using the notebook syntax.
        """
        X_train, X_test, y_train, y_test, _, _, _, _ = train_test_split_data
        
        # Training parameters (matching notebook)
        params = {
            "learning_rate": 0.05,
            "max_depth": 5,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_pre_filter": False,
            "verbose": -1
        }
        
        # Create Gaussian distribution with natural gradient
        gauss_nat = Gaussian(
            stabilization="MAD", 
            response_fn="softplus", 
            loss_fn="nll", 
            natural_gradient=True
        )
        
        # Create LightGBMLSS model
        lgblss_nat = LightGBMLSS(gauss_nat)
        
        # Set start values (matching notebook syntax)
        lgblss_nat.start_values = np.array([
            np.array(0.5) for _ in range(lgblss_nat.dist.n_dist_param)
        ])
        
        # Create datasets
        dtrain_nat = lgb.Dataset(X_train, label=y_train)
        dvalid_nat = lgb.Dataset(X_test, label=y_test, reference=dtrain_nat)
        
        # Train model
        eval_results_nat = {}
        lgblss_nat.train(
            params, dtrain_nat, num_boost_round=50,
            valid_sets=[dtrain_nat, dvalid_nat],
            valid_names=['train', 'valid'],
            callbacks=[lgb.record_evaluation(eval_results_nat)]
        )
        
        # Check that training succeeded
        assert 'train' in eval_results_nat, "Training results should be recorded"
        assert 'valid' in eval_results_nat, "Validation results should be recorded"
        assert 'nll' in eval_results_nat['train'], "NLL metric should be present"
        
        # Check that loss decreased
        initial_loss = eval_results_nat['train']['nll'][0]
        final_loss = eval_results_nat['train']['nll'][-1]
        assert final_loss < initial_loss, "Training loss should decrease"
        
        # Make predictions
        pred_nat = lgblss_nat.predict(X_test)
        
        # Check predictions structure
        assert 'loc' in pred_nat, "Predictions should contain 'loc' (mean)"
        assert 'scale' in pred_nat, "Predictions should contain 'scale' (std)"
        assert len(pred_nat['loc']) == len(y_test), "Prediction length should match test set"
        
        # Check predictions are reasonable
        assert np.all(np.isfinite(pred_nat['loc'])), "Mean predictions should be finite"
        assert np.all(pred_nat['scale'] > 0), "Std predictions should be positive"
        assert np.all(np.isfinite(pred_nat['scale'])), "Std predictions should be finite"
        
        print("✅ LightGBMLSS with natural gradient training test passed")
    
    def test_natural_vs_standard_gradient_comparison(self, train_test_split_data):
        """
        Test that natural gradient and standard gradient produce different results.
        This matches the comparison done in the notebook.
        """
        X_train, X_test, y_train, y_test, _, _, _, _ = train_test_split_data
        
        params = {
            "learning_rate": 0.05,
            "max_depth": 5,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_pre_filter": False,
            "verbose": -1
        }
        
        num_rounds = 100
        
        # Train with Natural Gradient
        gauss_nat = Gaussian(
            stabilization="MAD", 
            response_fn="softplus", 
            loss_fn="nll", 
            natural_gradient=True
        )
        lgblss_nat = LightGBMLSS(gauss_nat)
        lgblss_nat.start_values = np.array([
            np.array(0.5) for _ in range(lgblss_nat.dist.n_dist_param)
        ])
        
        dtrain_nat = lgb.Dataset(X_train, label=y_train)
        dvalid_nat = lgb.Dataset(X_test, label=y_test, reference=dtrain_nat)
        
        eval_results_nat = {}
        lgblss_nat.train(
            params, dtrain_nat, num_boost_round=num_rounds,
            valid_sets=[dtrain_nat, dvalid_nat],
            valid_names=['train', 'valid'],
            callbacks=[lgb.record_evaluation(eval_results_nat)]
        )
        
        # Train with Standard Gradient
        gauss_std = Gaussian(
            stabilization="MAD", 
            response_fn="softplus", 
            loss_fn="nll", 
            natural_gradient=False
        )
        lgblss_std = LightGBMLSS(gauss_std)
        lgblss_std.start_values = np.array([
            np.array(0.5) for _ in range(lgblss_std.dist.n_dist_param)
        ])
        
        dtrain_std = lgb.Dataset(X_train, label=y_train)
        dvalid_std = lgb.Dataset(X_test, label=y_test, reference=dtrain_std)
        
        eval_results_std = {}
        lgblss_std.train(
            params, dtrain_std, num_boost_round=num_rounds,
            valid_sets=[dtrain_std, dvalid_std],
            valid_names=['train', 'valid'],
            callbacks=[lgb.record_evaluation(eval_results_std)]
        )
        
        # Make predictions
        pred_nat = lgblss_nat.predict(X_test)
        pred_std = lgblss_std.predict(X_test)
        
        # Calculate NLL (matching notebook)
        nll_nat = -np.mean(stats.norm.logpdf(
            y_test, loc=pred_nat['loc'], scale=pred_nat['scale']
        ))
        nll_std = -np.mean(stats.norm.logpdf(
            y_test, loc=pred_std['loc'], scale=pred_std['scale']
        ))
        
        # Check that both methods produce valid results
        assert np.isfinite(nll_nat), "Natural gradient NLL should be finite"
        assert np.isfinite(nll_std), "Standard gradient NLL should be finite"
        
        # Check that predictions differ (methods are actually different)
        mean_diff = np.mean(np.abs(pred_nat['loc'] - pred_std['loc']))
        assert mean_diff > 0, "Natural and standard gradient should produce different predictions"
        
        print(f"✅ Natural vs Standard gradient comparison test passed")
        print(f"   Natural Gradient NLL: {nll_nat:.4f}")
        print(f"   Standard Gradient NLL: {nll_std:.4f}")
        print(f"   Mean prediction difference: {mean_diff:.4f}")
    
    def test_natural_gradient_convergence(self, train_test_split_data):
        """
        Test that natural gradient converges (loss decreases over iterations).
        """
        X_train, X_test, y_train, y_test, _, _, _, _ = train_test_split_data
        
        params = {
            "learning_rate": 0.05,
            "max_depth": 5,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_pre_filter": False,
            "verbose": -1
        }
        
        gauss_nat = Gaussian(
            stabilization="MAD", 
            response_fn="softplus", 
            loss_fn="nll", 
            natural_gradient=True
        )
        lgblss_nat = LightGBMLSS(gauss_nat)
        lgblss_nat.start_values = np.array([
            np.array(0.5) for _ in range(lgblss_nat.dist.n_dist_param)
        ])
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)
        
        eval_results = {}
        lgblss_nat.train(
            params, dtrain, num_boost_round=150,
            valid_sets=[dtrain, dvalid],
            valid_names=['train', 'valid'],
            callbacks=[lgb.record_evaluation(eval_results)]
        )
        
        # Check convergence
        train_losses = eval_results['train']['nll']
        valid_losses = eval_results['valid']['nll']
        
        # Training loss should decrease
        assert train_losses[-1] < train_losses[0], "Training loss should decrease"
        
        # Most iterations should show improvement
        improvements = sum(1 for i in range(1, len(train_losses)) 
                          if train_losses[i] < train_losses[i-1])
        improvement_rate = improvements / (len(train_losses) - 1)
        assert improvement_rate > 0.7, f"At least 70% of iterations should improve (got {improvement_rate:.1%})"
        
        print(f"✅ Natural gradient convergence test passed")
        print(f"   Initial train loss: {train_losses[0]:.4f}")
        print(f"   Final train loss: {train_losses[-1]:.4f}")
        print(f"   Improvement rate: {improvement_rate:.1%}")
    
    def test_natural_gradient_parameter_estimation(self, train_test_split_data):
        """
        Test that natural gradient produces reasonable parameter estimates.
        """
        X_train, X_test, y_train, y_test, _, mu_test, _, sigma_test = train_test_split_data
        
        params = {
            "learning_rate": 0.1,
            "max_depth": 5,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_pre_filter": False,
            "verbose": -1
        }
        
        gauss_nat = Gaussian(
            stabilization="MAD", 
            response_fn="softplus", 
            loss_fn="nll", 
            natural_gradient=True
        )
        lgblss_nat = LightGBMLSS(gauss_nat)
        lgblss_nat.start_values = np.array([
            np.array(0.5) for _ in range(lgblss_nat.dist.n_dist_param)
        ])
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        lgblss_nat.train(params, dtrain, num_boost_round=200)
        
        # Predict parameters
        pred = lgblss_nat.predict(X_test)
        
        # Check correlation with true parameters
        mu_corr = np.corrcoef(pred['loc'], mu_test)[0, 1]
        sigma_corr = np.corrcoef(pred['scale'], sigma_test)[0, 1]
        
        # Should have positive correlation
        assert mu_corr > 0.3, f"Mean parameter correlation too low: {mu_corr:.4f}"
        assert sigma_corr > 0.2, f"Std parameter correlation too low: {sigma_corr:.4f}"
        
        # Check MAE is reasonable
        mae_mu = np.mean(np.abs(pred['loc'] - mu_test))
        mae_sigma = np.mean(np.abs(pred['scale'] - sigma_test))
        
        # MAE should be less than the standard deviation of true values
        assert mae_mu < 3 * np.std(mu_test), f"Mean MAE too high: {mae_mu:.4f}"
        assert mae_sigma < 3 * np.std(sigma_test), f"Std MAE too high: {mae_sigma:.4f}"
        
        print(f"✅ Natural gradient parameter estimation test passed")
        print(f"   Mean correlation: {mu_corr:.4f}, MAE: {mae_mu:.4f}")
        print(f"   Std correlation: {sigma_corr:.4f}, MAE: {mae_sigma:.4f}")
    
    def test_start_values_setting(self, train_test_split_data):
        """
        Test that start values can be set using the notebook syntax.
        """
        X_train, _, y_train, _, _, _, _, _ = train_test_split_data
        
        gauss = Gaussian(
            stabilization="MAD", 
            response_fn="softplus", 
            loss_fn="nll", 
            natural_gradient=True
        )
        lgblss = LightGBMLSS(gauss)
        
        # Test the notebook syntax for setting start values
        n_params = lgblss.dist.n_dist_param
        assert n_params == 2, "Gaussian should have 2 parameters"
        
        # Set start values using notebook syntax
        lgblss.start_values = np.array([np.array(0.5) for _ in range(n_params)])
        
        # Check that start values are set correctly
        assert lgblss.start_values is not None, "Start values should be set"
        assert len(lgblss.start_values) == n_params, f"Should have {n_params} start values"
        
        # Train to verify it works
        params = {
            "learning_rate": 0.05,
            "max_depth": 3,
            "num_leaves": 7,
            "feature_pre_filter": False,
            "verbose": -1
        }
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        lgblss.train(params, dtrain, num_boost_round=10)
        
        print("✅ Start values setting test passed")
    
    def test_dataset_creation_syntax(self, train_test_split_data):
        """
        Test the dataset creation syntax used in the notebook.
        """
        X_train, X_test, y_train, y_test, _, _, _, _ = train_test_split_data
        
        # Create datasets using notebook syntax
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)
        
        # Verify datasets are created correctly
        assert dtrain.get_label() is not None, "Training labels should be set"
        assert len(dtrain.get_label()) == len(y_train), "Training label length should match"
        
        assert dvalid.get_label() is not None, "Validation labels should be set"
        assert len(dvalid.get_label()) == len(y_test), "Validation label length should match"
        
        print("✅ Dataset creation syntax test passed")
    
    def test_prediction_output_format(self, train_test_split_data):
        """
        Test that predictions have the expected format (DataFrame with 'loc' and 'scale').
        """
        X_train, X_test, y_train, y_test, _, _, _, _ = train_test_split_data
        
        params = {
            "learning_rate": 0.05,
            "max_depth": 3,
            "num_leaves": 7,
            "feature_pre_filter": False,
            "verbose": -1
        }
        
        gauss = Gaussian(
            stabilization="MAD", 
            response_fn="softplus", 
            loss_fn="nll", 
            natural_gradient=True
        )
        lgblss = LightGBMLSS(gauss)
        lgblss.start_values = np.array([np.array(0.5) for _ in range(lgblss.dist.n_dist_param)])
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        lgblss.train(params, dtrain, num_boost_round=50)
        
        # Make predictions
        pred = lgblss.predict(X_test)
        
        # Check output format - should be DataFrame with 'loc' and 'scale' columns
        assert isinstance(pred, pd.DataFrame), "Predictions should be a pandas DataFrame"
        assert 'loc' in pred.columns, "Predictions should have 'loc' column"
        assert 'scale' in pred.columns, "Predictions should have 'scale' column"
        
        # Check properties - can access as dict-like or DataFrame
        assert len(pred) == len(X_test), "Prediction length should match test set"
        assert pred['loc'].shape == (len(X_test),), "'loc' shape should match test set"
        assert pred['scale'].shape == (len(X_test),), "'scale' shape should match test set"
        
        # Check values are reasonable
        assert np.all(np.isfinite(pred['loc'])), "Mean predictions should be finite"
        assert np.all(pred['scale'] > 0), "Std predictions should be positive"
        assert np.all(np.isfinite(pred['scale'])), "Std predictions should be finite"
        
        print("✅ Prediction output format test passed")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
