from torch.distributions import Poisson as Poisson_Torch
from .distribution_utils import DistributionClass
from ..utils import *
from typing import List


class Poisson(DistributionClass):
    """
    Poisson distribution class.

    Distributional Parameters
    -------------------------
    rate: torch.Tensor
        Rate parameter of the distribution (often referred to as lambda).

    Source
    -------------------------
    https://pytorch.org/docs/stable/distributions.html#poisson

    Parameters
    -------------------------
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD", "L2".
    response_fn: str
        Response function for transforming the distributional parameters to the correct support. Options are
        "exp" (exponential), "softplus" (softplus) or "relu" (rectified linear unit).
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood).
    initialize: bool
        Whether to initialize the distributional parameters with unconditional start values. Initialization can help
        to improve speed of convergence in some cases. However, it may also lead to early stopping or suboptimal
        solutions if the unconditional start values are far from the optimal values.
    """
    def __init__(self,
                 stabilization: str = "None",
                 response_fn: str = "relu",
                 loss_fn: str = "nll",
                 initialize: bool = False,
                 ):

        # Input Checks
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError("Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'.")
        if loss_fn not in ["nll"]:
            raise ValueError("Invalid loss function. Please select 'nll'.")
        if not isinstance(initialize, bool):
            raise ValueError("Invalid initialize. Please choose from True or False.")

        # Specify Response Functions
        response_functions = {"exp": exp_fn, "softplus": softplus_fn, "relu": relu_fn}
        if response_fn in response_functions:
            response_fn = response_functions[response_fn]
        else:
            raise ValueError(
                "Invalid response function for total_count. Please choose from 'exp', 'softplus' or 'relu'.")

        # Set the parameters specific to the distribution
        distribution = Poisson_Torch
        param_dict = {"rate": response_fn}
        torch.distributions.Distribution.set_default_validate_args(False)

        # Specify Distribution Class
        super().__init__(distribution=distribution,
                         univariate=True,
                         discrete=True,
                         n_dist_param=len(param_dict),
                         stabilization=stabilization,
                         param_dict=param_dict,
                         distribution_arg_names=list(param_dict.keys()),
                         loss_fn=loss_fn,
                         initialize=initialize,
                         )

    def compute_fisher_information_matrix(self, predt: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute Fisher Information Matrix diagonal for Poisson distribution.
        
        For Poisson with rate parameter λ:
        - The variance equals the mean: Var(Y) = λ
        - Fisher Information with respect to the natural parameter η (where λ = g(η))
          and g is the response function (exp, softplus, or relu):
          FIM_η = λ * (g'(η))^2
        
        For common response functions:
        - exp: g'(η) = exp(η) = λ, so FIM = λ^2
        - softplus: g'(η) = sigmoid(η), so FIM = λ * sigmoid(η)^2
        - relu: g'(η) = 1 if η > 0 else 0, so FIM ≈ λ (for η > 0)
        
        Parameters
        ----------
        predt : List[torch.Tensor]
            [eta] - raw parameter before response function
        
        Returns
        -------
        fim : List[torch.Tensor]
            [FIM_eta] - Fisher Information for the natural parameter
        """
        # Raw parameter (before response function)
        eta = predt[0]
        
        # Apply response function to get λ
        response_fn = list(self.param_dict.values())[0]
        lam = response_fn(eta)
        
        # FIM for rate parameter - optimize for exp case
        if response_fn == exp_fn:
            # For exp: g'(η) = λ, so I(η) = λ²
            fim_eta = lam.detach() ** 2 + 1e-8
        else:
            # For other response functions: compute derivative
            eta_grad = eta.detach().requires_grad_(True)
            lam_grad = response_fn(eta_grad)
            
            # Get derivative g'(η)
            g_prime = torch.autograd.grad(
                outputs=lam_grad.sum(),
                inputs=eta_grad,
                create_graph=False,
                retain_graph=False
            )[0]
            
            # Fisher Information: I(η) = λ * (g'(η))^2
            fim_eta = lam.detach() * (g_prime ** 2) + 1e-8
        
        return [fim_eta]