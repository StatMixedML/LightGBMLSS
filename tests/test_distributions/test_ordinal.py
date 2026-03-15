from lightgbmlss.model import LightGBMLSS
from lightgbmlss.distributions.OrderedLogistic import OrderedLogistic

import pytest
import torch
import numpy as np
import pandas as pd
import lightgbm as lgb


class TestOrderedLogistic:
    """Tests for the OrderedLogistic distribution."""

    # ── Init validation ───────────────────────────────────────────────
    def test_init_valid(self):
        dist = OrderedLogistic(n_classes=5)
        assert dist.n_classes == 5
        assert dist.n_dist_param == 5  # 1 predictor + 4 cutpoints
        assert dist.discrete is True
        assert dist.univariate is True
        assert dist.loss_fn == "nll"

    def test_init_n_classes_validation(self):
        with pytest.raises(ValueError, match="n_classes must be an integer"):
            OrderedLogistic(n_classes=1)
        with pytest.raises(ValueError, match="n_classes must be an integer"):
            OrderedLogistic(n_classes=2.5)

    def test_init_stabilization_validation(self):
        with pytest.raises(ValueError, match="Invalid stabilization method."):
            OrderedLogistic(stabilization="invalid")

    def test_init_loss_fn_validation(self):
        with pytest.raises(ValueError, match="Invalid loss function."):
            OrderedLogistic(loss_fn="crps")

    def test_init_initialize_validation(self):
        with pytest.raises(ValueError, match="Invalid initialize."):
            OrderedLogistic(initialize="yes")

    def test_defaults(self):
        dist = OrderedLogistic()
        assert dist.n_classes == 3
        assert dist.n_dist_param == 3  # 1 predictor + 2 cutpoints
        assert dist.initialize is False
        assert dist.stabilization == "None"
        assert dist.tau is None

    # ── Parameter dict structure ──────────────────────────────────────
    def test_param_dict_structure(self):
        dist = OrderedLogistic(n_classes=4)
        assert "predictor" in dist.param_dict
        assert "cutpoint_raw_1" in dist.param_dict
        assert "cutpoint_raw_2" in dist.param_dict
        assert "cutpoint_raw_3" in dist.param_dict
        assert len(dist.param_dict) == 4
        assert dist.n_dist_param == len(dist.distribution_arg_names)
        assert all(callable(fn) for fn in dist.param_dict.values())

    # ── Cutpoint ordering ─────────────────────────────────────────────
    def test_raw_to_ordered_cutpoints_monotonic(self):
        """Ordered cutpoints must be strictly increasing."""
        raw = [
            torch.tensor([[0.5], [-1.0], [2.0], [0.1]]),
            torch.tensor([[-0.3], [0.5], [-0.8], [1.5]]),
            torch.tensor([[1.0], [0.2], [0.3], [-0.5]]),
        ]
        ordered = OrderedLogistic._raw_to_ordered_cutpoints(raw)
        diffs = torch.diff(ordered, dim=1)
        assert (diffs > 0).all(), "Cutpoints are not strictly increasing"

    def test_raw_to_ordered_cutpoints_single(self):
        """K=2: single cutpoint, no ordering needed."""
        raw = [torch.tensor([[1.0], [2.0]])]
        ordered = OrderedLogistic._raw_to_ordered_cutpoints(raw)
        assert ordered.shape == (2, 1)

    def test_raw_to_ordered_cutpoints_shape(self):
        """Output shape must be (n_obs, K-1)."""
        n_obs, K = 8, 5
        raw = [torch.randn(n_obs, 1) for _ in range(K - 1)]
        ordered = OrderedLogistic._raw_to_ordered_cutpoints(raw)
        assert ordered.shape == (n_obs, K - 1)

    # ── Start values ──────────────────────────────────────────────────
    def test_calculate_start_values(self):
        dist = OrderedLogistic(n_classes=3)
        target = np.array([0, 0, 1, 1, 2, 2], dtype=float)
        loss, start_values = dist.calculate_start_values(target)
        assert len(start_values) == 3  # 1 predictor + 2 cutpoints
        assert np.isfinite(loss)
        assert np.all(np.isfinite(start_values))

    # ── Forward pass ──────────────────────────────────────────────────
    def test_get_params_loss_runs(self):
        K = 3
        dist = OrderedLogistic(n_classes=K)
        n_obs = 10
        np.random.seed(42)
        predt = np.random.randn(n_obs * K)
        target = torch.tensor(
            np.random.randint(0, K, size=(n_obs, 1)), dtype=torch.float32
        )
        start_values = [0.0, 0.5, 0.3]

        params, loss = dist.get_params_loss(predt, target, start_values, requires_grad=True)
        assert np.isfinite(loss.item())
        assert len(params) == K

    # ── Gradient and Hessian ──────────────────────────────────────────
    def test_gradients_finite(self):
        K = 3
        dist = OrderedLogistic(n_classes=K)
        n_obs = 10
        np.random.seed(42)
        predt = np.random.randn(n_obs * K)
        target = torch.tensor(
            np.random.randint(0, K, size=(n_obs, 1)), dtype=torch.float32
        )
        start_values = [0.0, 0.5, 0.3]

        params, loss = dist.get_params_loss(predt, target, start_values, requires_grad=True)
        grad, hess = dist.compute_gradients_and_hessians(loss, params, np.ones((n_obs, 1)))
        assert np.all(np.isfinite(grad))
        assert np.all(np.isfinite(hess))

    # ── Draw samples ──────────────────────────────────────────────────
    def test_draw_samples_valid_range(self):
        K = 4
        dist = OrderedLogistic(n_classes=K)
        params_df = pd.DataFrame({
            "predictor": [0.0, 1.0],
            "cutpoint_1": [-1.0, -1.0],
            "cutpoint_2": [0.0, 0.0],
            "cutpoint_3": [1.0, 1.0],
        })
        samples = dist.draw_samples(params_df, n_samples=100)
        assert samples.shape == (2, 100)
        assert samples.min().min() >= 0
        assert samples.max().max() <= K - 1

    def test_draw_samples_dtype(self):
        dist = OrderedLogistic(n_classes=3)
        params_df = pd.DataFrame({
            "predictor": [0.0],
            "cutpoint_1": [-0.5],
            "cutpoint_2": [0.5],
        })
        samples = dist.draw_samples(params_df, n_samples=50)
        assert samples.dtypes.apply(lambda d: np.issubdtype(d, np.integer)).all()

    # ── Integration: objective_fn with lgb.Dataset ─────────────────────
    def test_objective_fn_with_dataset(self):
        K = 3
        dist = OrderedLogistic(n_classes=K)
        model = LightGBMLSS(dist)

        n_obs = 20
        np.random.seed(42)
        X = np.random.randn(n_obs, 3)
        y = np.random.randint(0, K, size=n_obs).astype(float)

        dtrain = lgb.Dataset(X, label=y)
        model.set_init_score(dtrain)
        dtrain.construct()

        predt = np.random.randn(n_obs * K)
        grad, hess = dist.objective_fn(predt, dtrain)
        assert grad.shape == (n_obs * K,)
        assert hess.shape == (n_obs * K,)
        assert np.all(np.isfinite(grad))
        assert np.all(np.isfinite(hess))

    # ── K=2 (binary) edge case ────────────────────────────────────────
    def test_binary_case(self):
        """K=2 collapses to logistic regression with one cutpoint."""
        dist = OrderedLogistic(n_classes=2)
        assert dist.n_dist_param == 2
        assert dist.n_classes == 2

        n_obs = 8
        predt = np.random.randn(n_obs * 2)
        target = torch.tensor(
            np.random.randint(0, 2, size=(n_obs, 1)), dtype=torch.float32
        )
        params, loss = dist.get_params_loss(predt, target, [0.0, 0.0], requires_grad=True)
        assert np.isfinite(loss.item())
