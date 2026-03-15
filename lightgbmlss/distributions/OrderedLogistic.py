import torch
from torch.nn.functional import softplus
from torch.autograd import grad as autograd
from torch.optim import LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pyro.distributions import OrderedLogistic as OrderedLogistic_Pyro

import lightgbm as lgb
import numpy as np
import pandas as pd

from typing import List, Tuple

from .distribution_utils import DistributionClass
from ..utils import identity_fn


class OrderedLogistic(DistributionClass):
    """
    Ordered Logistic (cumulative link) distribution for ordinal regression.

    Models P(y = k | x) via the cumulative logistic link:
        P(y ≤ k | η) = σ(cₖ - η)

    where η is a latent predictor score and c₁ < c₂ < ... < cₖ₋₁ are
    ordered cutpoints.

    Distributional Parameters
    -------------------------
    predictor: torch.Tensor
        Latent score η. Unconstrained real-valued.
    cutpoint_raw_1 ... cutpoint_raw_{K-1}: torch.Tensor
        Raw (unconstrained) cutpoint parameters. Internally transformed
        to ordered cutpoints via cumulative softplus.

    Source
    -------------------------
    https://docs.pyro.ai/en/stable/distributions.html#orderedlogistic

    Parameters
    -------------------------
    n_classes: int
        Number of ordinal categories K (e.g. 5 for a 1-5 Likert scale).
        Labels must be integers in {0, 1, ..., K-1}.
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are
        "None", "MAD", "L2".
    loss_fn: str
        Loss function. Currently only "nll" (negative log-likelihood) is
        supported for ordinal regression.
    initialize: bool
        Whether to initialize the distributional parameters with
        unconditional start values.
    """
    def __init__(
        self,
        n_classes: int = 3,
        stabilization: str = "None",
        loss_fn: str = "nll",
        initialize: bool = False,
    ):
        # Input Checks
        if not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError("n_classes must be an integer >= 2.")
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError(
                "Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'."
            )
        if loss_fn not in ["nll"]:
            raise ValueError("Invalid loss function. Please select 'nll'.")
        if not isinstance(initialize, bool):
            raise ValueError("Invalid initialize. Please choose from True or False.")

        self.n_classes = n_classes
        n_cutpoints = n_classes - 1

        # First parameter is the predictor (identity transform).
        # Remaining K-1 parameters are raw cutpoint values.
        # The ordering constraint is enforced inside get_params_loss, not here.
        param_dict = {"predictor": identity_fn}
        for i in range(1, n_cutpoints + 1):
            param_dict[f"cutpoint_raw_{i}"] = identity_fn

        torch.distributions.Distribution.set_default_validate_args(False)

        super().__init__(
            distribution=OrderedLogistic_Pyro,
            univariate=True,
            discrete=True,
            n_dist_param=len(param_dict),
            stabilization=stabilization,
            param_dict=param_dict,
            distribution_arg_names=list(param_dict.keys()),
            loss_fn=loss_fn,
            initialize=initialize,
        )

    @staticmethod
    def _raw_to_ordered_cutpoints(raw_cutpoints: List[torch.Tensor]) -> torch.Tensor:
        """
        Transform K-1 unconstrained raw parameters into ordered cutpoints.

        Arguments
        ---------
        raw_cutpoints: List[torch.Tensor]
            List of K-1 tensors, each of shape (n_obs, 1).

        Returns
        -------
        cutpoints: torch.Tensor
            Tensor of shape (n_obs, K-1) with c₁ < c₂ < ... < cₖ₋₁.
        """
        base = raw_cutpoints[0]  # c₁ = raw₁ (unconstrained)
        if len(raw_cutpoints) == 1:
            return base  # K=2: single cutpoint, no ordering needed

        # Softplus-transform the gaps to ensure strict positivity
        gaps = torch.cat(
            [softplus(rc) + 1e-6 for rc in raw_cutpoints[1:]], dim=1
        )
        # Cumulative sum of gaps gives the offsets from the base cutpoint
        offsets = torch.cumsum(gaps, dim=1)
        cutpoints = torch.cat([base, base + offsets], dim=1)

        return cutpoints

    def get_params_loss(
        self,
        predt: np.ndarray,
        target: torch.Tensor,
        start_values: List[float],
        requires_grad: bool = False,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute predicted parameters and NLL loss for ordinal regression.

        The first column of predt is the predictor η.
        Columns 1..K-1 are raw cutpoint parameters, transformed here into
        ordered cutpoints via cumulative softplus.

        Arguments
        ---------
        predt: np.ndarray
            Predicted values, shape (n_obs * n_dist_param,).
        target: torch.Tensor
            Target values, shape (n_obs, 1).
        start_values: List[float]
            Starting values for each distributional parameter.
        requires_grad: bool
            Whether to add to the computational graph.

        Returns
        -------
        predt: List[torch.Tensor]
            Per-parameter tensors with gradient tracking.
        loss: torch.Tensor
            Negative log-likelihood.
        """
        # Reshape: (n_obs * n_params,) → (n_obs, n_params)
        predt = predt.reshape(-1, self.n_dist_param, order="F")

        # Replace NaNs/Infs with start values
        nan_inf_mask = np.isnan(predt) | np.isinf(predt)
        predt[nan_inf_mask] = np.take(start_values, np.where(nan_inf_mask)[1])

        # Convert to tensors with gradient tracking
        predt = [
            torch.tensor(
                predt[:, i].reshape(-1, 1),
                requires_grad=requires_grad,
            )
            for i in range(self.n_dist_param)
        ]

        # Split: predictor vs raw cutpoints
        predictor = predt[0].squeeze(-1)  # (n_obs,)
        raw_cutpoints = predt[1:]         # list of K-1 tensors of shape (n_obs, 1)

        # Apply the ordering transform
        cutpoints = self._raw_to_ordered_cutpoints(raw_cutpoints)  # (n_obs, K-1)

        # Construct distribution and compute NLL
        dist_fit = OrderedLogistic_Pyro(predictor=predictor, cutpoints=cutpoints)
        loss = -torch.nansum(dist_fit.log_prob(target.squeeze(-1).long()))

        return predt, loss

    def loss_fn_start_values(
        self,
        params: List[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """NLL for start-value optimization over unconditional parameters."""
        # Replace NaNs/Infs
        nan_inf_idx = (
            torch.isnan(torch.stack(params)) | torch.isinf(torch.stack(params))
        )
        params = torch.where(nan_inf_idx, torch.tensor(0.5), torch.stack(params))
        params = [params[i].reshape(-1, 1) for i in range(self.n_dist_param)]

        # Broadcast scalar params to (n_obs, 1) for batched construction
        predictor = params[0].squeeze(-1).expand(target.shape[0])
        raw_cutpoints = [rc.expand(target.shape[0], 1) for rc in params[1:]]

        cutpoints = self._raw_to_ordered_cutpoints(raw_cutpoints)

        dist = OrderedLogistic_Pyro(predictor=predictor, cutpoints=cutpoints)
        loss = -torch.nansum(dist.log_prob(target.squeeze(-1).long()))

        return loss

    def calculate_start_values(
        self,
        target: np.ndarray,
        max_iter: int = 50,
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate starting values for ordinal regression parameters.

        Initializes the predictor at 0 and spaces cutpoints at the empirical
        quantile boundaries of the ordinal labels, then refines via L-BFGS.

        Arguments
        ---------
        target: np.ndarray
            Target labels in {0, ..., K-1}.
        max_iter: int
            Maximum number of L-BFGS iterations.

        Returns
        -------
        loss: float
            Final NLL value.
        start_values: np.ndarray
            Starting values for each distributional parameter.
        """
        target_t = torch.tensor(target, dtype=torch.float32).reshape(-1, 1)
        n_cutpoints = self.n_classes - 1

        # Place cutpoints at roughly equal-probability boundaries
        boundaries = np.linspace(0, self.n_classes - 1, n_cutpoints + 2)[1:-1]

        init_raw = [boundaries[0]]
        for i in range(1, n_cutpoints):
            gap = boundaries[i] - boundaries[i - 1]
            init_raw.append(np.log(np.exp(gap) - 1))  # inverse softplus

        params = [
            torch.tensor(0.0, requires_grad=True, dtype=torch.float32),
        ] + [
            torch.tensor(float(v), requires_grad=True, dtype=torch.float32)
            for v in init_raw
        ]

        optimizer = LBFGS(
            params,
            lr=0.1,
            max_iter=min(int(max_iter / 4), 20),
            line_search_fn="strong_wolfe",
        )
        lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

        def closure():
            optimizer.zero_grad()
            loss = self.loss_fn_start_values(params, target_t)
            loss.backward()
            return loss

        loss_vals = []
        for _ in range(max_iter):
            loss = optimizer.step(closure)
            lr_scheduler.step(loss)
            loss_vals.append(loss.item())

        loss = np.array(loss_vals[-1])
        start_values = np.array([p.detach().numpy() for p in params])
        start_values = np.nan_to_num(start_values, nan=0.5, posinf=0.5, neginf=0.5)

        return loss, start_values

    def draw_samples(
        self,
        predt_params: pd.DataFrame,
        n_samples: int = 1000,
        seed: int = 123,
    ) -> pd.DataFrame:
        """
        Draw ordinal class samples from the predicted distribution.

        Arguments
        ---------
        predt_params: pd.DataFrame
            DataFrame with columns ["predictor", "cutpoint_1", ..., "cutpoint_{K-1}"].
            Cutpoints must already be ordered (output of predict_dist).
        n_samples: int
            Number of samples to draw per observation.
        seed: int
            Random seed.

        Returns
        -------
        pd.DataFrame
            Shape (n_obs, n_samples), integer-valued in {0, ..., K-1}.
        """
        torch.manual_seed(seed)

        pred_vals = torch.tensor(predt_params.values, dtype=torch.float32)
        predictor = pred_vals[:, 0]          # (n_obs,)
        cutpoints = pred_vals[:, 1:]         # (n_obs, K-1) — already ordered

        dist = OrderedLogistic_Pyro(predictor=predictor, cutpoints=cutpoints)
        samples = dist.sample((n_samples,)).detach().numpy()  # (n_samples, n_obs)
        samples = samples.T                                    # (n_obs, n_samples)

        df = pd.DataFrame(samples)
        df.columns = [f"y_sample{i}" for i in range(n_samples)]

        return df.astype(int)

    def predict_dist(
        self,
        booster: lgb.Booster,
        data: pd.DataFrame,
        start_values: np.ndarray,
        pred_type: str = "parameters",
        n_samples: int = 1000,
        quantiles: list = [0.1, 0.5, 0.9],
        seed: int = 123,
    ) -> pd.DataFrame:
        """
        Predict from the trained ordinal regression model.

        Arguments
        ---------
        booster: lgb.Booster
            Trained model.
        data: pd.DataFrame
            Data to predict from.
        start_values: np.ndarray
            Starting values for each distributional parameter.
        pred_type: str
            - "parameters"  — predictor η and ordered cutpoints c₁...cₖ₋₁
            - "class_probs" — P(y=k) for each class k ∈ {0, ..., K-1}
            - "samples"     — ordinal class samples
            - "quantiles"   — quantiles from the sample-based distribution
        n_samples: int
            Number of samples (for "samples" and "quantiles").
        quantiles: list
            Quantile levels (for "quantiles").
        seed: int
            Random seed for sampling.

        Returns
        -------
        pd.DataFrame
            Predictions in the requested format.
        """
        predt = torch.tensor(
            booster.predict(data, raw_score=True),
            dtype=torch.float32,
        ).reshape(-1, self.n_dist_param)

        # Add init_score offsets (one per distributional parameter)
        init_score = torch.tensor(
            np.ones(shape=(data.shape[0], 1)) * start_values,
            dtype=torch.float32,
        )
        predt = predt + init_score

        # Split predictor and raw cutpoints, then apply ordering transform
        predictor = predt[:, 0]
        raw_cutpoints = [predt[:, i].reshape(-1, 1) for i in range(1, self.n_dist_param)]
        cutpoints = self._raw_to_ordered_cutpoints(raw_cutpoints)  # (n_obs, K-1)

        # Build parameter DataFrame with ordered cutpoints
        param_names = ["predictor"] + [
            f"cutpoint_{i}" for i in range(1, self.n_classes)
        ]
        dist_params = torch.cat(
            [predictor.unsqueeze(1), cutpoints], dim=1
        ).detach().numpy()
        dist_params_df = pd.DataFrame(dist_params, columns=param_names)

        if pred_type == "parameters":
            return dist_params_df

        elif pred_type == "class_probs":
            dist = OrderedLogistic_Pyro(predictor=predictor, cutpoints=cutpoints)
            probs = dist.probs.detach().numpy()
            return pd.DataFrame(
                probs,
                columns=[f"P(y={k})" for k in range(self.n_classes)],
            )

        elif pred_type == "samples":
            return self.draw_samples(
                predt_params=dist_params_df,
                n_samples=n_samples,
                seed=seed,
            )

        elif pred_type == "quantiles":
            pred_samples_df = self.draw_samples(
                predt_params=dist_params_df,
                n_samples=n_samples,
                seed=seed,
            )
            pred_quant_df = pred_samples_df.quantile(quantiles, axis=1).T
            pred_quant_df.columns = [f"quant_{q}" for q in quantiles]
            return pred_quant_df.astype(int)
