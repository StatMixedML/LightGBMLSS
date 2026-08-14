from torch.distributions import Laplace as Laplace_Torch
from .distribution_utils import DistributionClass
from ..utils import *
from typing import List


class Laplace(DistributionClass):
    """
    Laplace distribution class.

    Distributional Parameters
    -------------------------
    loc: torch.Tensor
        Mean of the distribution.
    scale: torch.Tensor
        Scale of the distribution.

    Source
    -------------------------
    https://pytorch.org/docs/stable/distributions.html#laplace

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
        response_functions = {"exp": exp_fn, "softplus": softplus_fn}
        if response_fn in response_functions:
            response_fn = response_functions[response_fn]
        else:
            raise ValueError(
                "Invalid response function. Please choose from 'exp' or 'softplus'.")

        # Set the parameters specific to the distribution
        distribution = Laplace_Torch
        param_dict = {"loc": identity_fn, "scale": response_fn}
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
        Compute Fisher Information Matrix diagonal for Laplace distribution.

        For Laplace L(μ, b):
        - Fisher Information w.r.t. μ: I(μ) = 1/b²
        - Fisher Information w.r.t. natural parameter η_b where b = g(η_b):
          I(η_b) = (1/b²) * (g'(η_b))²

        For common response functions:
        - exp: g'(η) = b, so I(η_b) = 1
        - softplus: g'(η) = sigmoid(η), so I(η_b) = (1/b²) * sigmoid(η)²

        Parameters
        ----------
        predt : List[torch.Tensor]
            [eta_mu, eta_b] - raw parameters before response functions

        Returns
        -------
        fim : List[torch.Tensor]
            [FIM_mu, FIM_b]
        """
        eta_mu, eta_b = predt[0], predt[1]

        # Apply response functions
        response_fn_b = self.param_dict["scale"]
        b = response_fn_b(eta_b)

        # FIM for μ (location parameter uses identity)
        fim_mu = 1.0 / (b ** 2 + 1e-12)

        # FIM for b natural parameter - optimize for exp case
        if response_fn_b == exp_fn:
            # For exp: g'(η) = b, so I(η_b) = 1
            fim_b = torch.ones_like(eta_b)
        else:
            # For other response functions: compute derivative
            eta_b_grad = eta_b.detach().requires_grad_(True)
            b_grad = response_fn_b(eta_b_grad)

            g_prime = torch.autograd.grad(
                outputs=b_grad.sum(),
                inputs=eta_b_grad,
                create_graph=False,
                retain_graph=False
            )[0]

            # Fisher Information: I(η_b) = (1/b²) * (g'(η))²
            fim_b = (1.0 / (b.detach() ** 2 + 1e-12)) * (g_prime ** 2)

        return [fim_mu, fim_b]
