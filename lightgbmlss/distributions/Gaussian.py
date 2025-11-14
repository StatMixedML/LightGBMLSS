import torch
from torch.distributions import Normal as Gaussian_Torch
from .distribution_utils import DistributionClass
from ..utils import *
from typing import List


class Gaussian(DistributionClass):
    """
    Gaussian distribution class.

    Distributional Parameters
    -------------------------
    loc: torch.Tensor
        Mean of the distribution (often referred to as mu).
    scale: torch.Tensor
        Standard deviation of the distribution (often referred to as sigma).

    Source
    -------------------------
    https://pytorch.org/docs/stable/distributions.html#normal

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
    """
    def __init__(self,
                 stabilization: str = "None",
                 response_fn: str = "exp",
                 loss_fn: str = "nll",
                 natural_gradient: bool = False,
                 clip_value: float = None,
                 ):

        # Input Checks
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError("Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'.")
        if loss_fn not in ["nll", "crps"]:
            raise ValueError("Invalid loss function. Please choose from 'nll' or 'crps'.")

        # Specify Response Functions
        response_functions = {"exp": exp_fn, "softplus": softplus_fn}
        if response_fn in response_functions:
            response_fn = response_functions[response_fn]
        else:
            raise ValueError(
                "Invalid response function. Please choose from 'exp' or 'softplus'.")

        # Set the parameters specific to the distribution
        distribution = Gaussian_Torch
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
                         natural_gradient=natural_gradient,
                         clip_value=clip_value,  
                         )
        
    def compute_fisher_information_matrix(self, predt: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute Fisher Information Matrix diagonal for Gaussian distribution.
        
        For Gaussian with parameters [mu, log(sigma)]:
        - FIM_mu = 1 / sigma^2
        - FIM_log_sigma = 2 (constant for log-parameterization)
        
        Parameters
        ----------
        predt : List[torch.Tensor]
            [mu_raw, log_sigma_raw]
        
        Returns
        -------
        fim : List[torch.Tensor]
            [FIM_mu, FIM_log_sigma]
        """
        mu_raw, log_sigma_raw = predt[0], predt[1]
        
        # Transform to response scale
        mu = self.param_dict["loc"](mu_raw)
        sigma = self.param_dict["scale"](log_sigma_raw)
        
        # Compute FIM diagonal elements
        fim_mu = 1.0 / (sigma ** 2 + 1e-12)
        fim_log_sigma = torch.ones_like(log_sigma_raw) * 2.0
        
        return [fim_mu, fim_log_sigma]
