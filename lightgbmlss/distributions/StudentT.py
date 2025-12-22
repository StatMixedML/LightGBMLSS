import torch
from torch.distributions import StudentT as StudentT_Torch
from .distribution_utils import DistributionClass
from ..utils import *
from typing import List


class StudentT(DistributionClass):
    """
    Student-T Distribution Class

    Distributional Parameters
    -------------------------
    df: torch.Tensor
        Degrees of freedom.
    loc: torch.Tensor
        Mean of the distribution.
    scale: torch.Tensor
        Scale of the distribution.

    Source
    -------------------------
    https://pytorch.org/docs/stable/distributions.html#studentt

     Parameters
    -------------------------
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD", "L2".
    response_fn: str
        Response function for transforming the distributional parameters to the correct support. Options are
        "exp" (exponential) or "softplus" (softplus).
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood) or "crps" (continuous ranked probability score).
        Note that if "crps" is used, the Hessian is set to 1, as the current CRPS version is not twice differentiable.
        Hence, using the CRPS disregards any variation in the curvature of the loss function.
    initialize: bool
        Whether to initialize the distributional parameters with unconditional start values. Initialization can help
        to improve speed of convergence in some cases. However, it may also lead to early stopping or suboptimal
        solutions if the unconditional start values are far from the optimal values.
    """
    def __init__(self,
                 stabilization: str = "None",
                 response_fn: str = "exp",
                 loss_fn: str = "nll",
                 initialize: bool = False,
                 ):

        # Input Checks
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError("Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'.")
        if loss_fn not in ["nll", "crps"]:
            raise ValueError("Invalid loss function. Please choose from 'nll' or 'crps'.")
        if not isinstance(initialize, bool):
            raise ValueError("Invalid initialize. Please choose from True or False.")

        # Specify Response Functions
        response_functions = {
            "exp": (exp_fn, exp_fn_df),
            "softplus": (softplus_fn, softplus_fn_df)
        }
        if response_fn in response_functions:
            response_fn, response_fn_df = response_functions[response_fn]
        else:
            raise ValueError(
                "Invalid response function. Please choose from 'exp' or 'softplus'.")

        # Set the parameters specific to the distribution
        distribution = StudentT_Torch
        param_dict = {"df": response_fn_df, "loc": identity_fn, "scale": response_fn}
        torch.distributions.Distribution.set_default_validate_args(False)

        # Specify Distribution Class
        super().__init__(distribution=distribution,
                         univariate=True,
                         discrete=False,
                         n_dist_param=len(param_dict),
                         stabilization=stabilization,
                         param_dict=param_dict,
                         distribution_arg_names=list(param_dict.keys()),
                         loss_fn=loss_fn,
                         initialize=initialize,
                         )

    def compute_fisher_information_matrix(self, predt: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute Fisher Information Matrix diagonal for Student-T distribution.

        For Student-T(ν, μ, σ):
        - Fisher Information w.r.t. μ: I(μ) = (ν+1)/((ν+3)σ²)
        - Fisher Information w.r.t. natural parameter η_σ where σ = g(η_σ):
          I(η_σ) = (2ν/((ν+3)σ²)) * (g'(η_σ))²
        - Fisher Information w.r.t. ν: Using simplified constant approximation

        Parameters
        ----------
        predt : List[torch.Tensor]
            [eta_df, eta_loc, eta_scale] - raw parameters before response functions

        Returns
        -------
        fim : List[torch.Tensor]
            [FIM_df, FIM_loc, FIM_scale]
        """
        eta_df, eta_loc, eta_scale = predt[0], predt[1], predt[2]
        
        # Apply response functions
        df = self.param_dict["df"](eta_df)
        loc = self.param_dict["loc"](eta_loc)
        scale = self.param_dict["scale"](eta_scale)
        
        # FIM for loc (μ) - location parameter uses identity
        fim_loc = (df + 1.0) / ((df + 3.0) * scale ** 2 + 1e-12)
        
        # FIM for scale (σ) natural parameter
        response_fn_scale = self.param_dict["scale"]
        if response_fn_scale == exp_fn:
            # For exp: g'(η) = σ, so I(η_σ) = (2ν/((ν+3)σ²)) * σ² = 2ν/(ν+3)
            fim_scale = (2.0 * df) / (df + 3.0 + 1e-12)
        else:
            # For other response functions: compute derivative
            eta_scale_grad = eta_scale.detach().requires_grad_(True)
            scale_grad = response_fn_scale(eta_scale_grad)
            
            g_prime = torch.autograd.grad(
                outputs=scale_grad.sum(),
                inputs=eta_scale_grad,
                create_graph=False,
                retain_graph=False
            )[0]
            
            # Fisher Information: I(η_σ) = (2ν/((ν+3)σ²)) * (g'(η))²
            fim_scale = (2.0 * df.detach() / ((df.detach() + 3.0) * scale.detach() ** 2 + 1e-12)) * (g_prime ** 2)
        
        # FIM for df (ν) - using simplified constant approximation
        # The exact FIM for df is complex and not provided in standard references
        # Using a conservative constant value
        fim_df = torch.ones_like(eta_df) * 0.5
        
        return [fim_df, fim_loc, fim_scale]

