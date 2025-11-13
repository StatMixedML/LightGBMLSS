from torch.distributions import Normal, TransformedDistribution, SigmoidTransform
from .distribution_utils import DistributionClass
from ..utils import *


class LogitNormal(DistributionClass):
    """
    Logit-Normal distribution class.

    Distributional Parameters
    -------------------------
    loc: torch.Tensor
        Mean of the normal distribution before applying the logit transformation.
    scale: torch.Tensor
        Standard deviation of the normal distribution before applying the logit transformation.

    Source
    -------------------------
    https://pytorch.org/docs/stable/distributions.html#normal

    Parameters
    -------------------------
    stabilization: str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD", "L2".
    response_fn: str
        Response function for transforming the distributional parameters to the correct support. Options are
        "identity" (no transformation) or "softplus" (softplus to ensure positivity).
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood) or "crps" (continuous ranked probability score).
        Note that if "crps" is used, the Hessian is set to 1, as the current CRPS version is not twice differentiable.
    """
    
    def __init__(self,
                 stabilization: str = "None",
                 response_fn: str = "identity",
                 loss_fn: str = "nll"
                 ):

        # Input Checks
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError("Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'.")
        if loss_fn not in ["nll", "crps"]:
            raise ValueError("Invalid loss function. Please choose from 'nll' or 'crps'.")

        # Specify Response Functions
        response_functions = {"identity": identity_fn, "softplus": softplus_fn}
        if response_fn in response_functions:
            response_fn = response_functions[response_fn]
        else:
            raise ValueError("Invalid response function. Please choose from 'identity' or 'softplus'.")

        # Define Logit-Normal as a transformed distribution
        base_distribution = Normal
        transformed_distribution = lambda loc, scale: TransformedDistribution(
            base_distribution(loc, scale), [SigmoidTransform()]
        )

        # Define Parameter Mapping
        param_dict = {"loc": identity_fn, "scale": response_fn}
        torch.distributions.Distribution.set_default_validate_args(False)

        # Specify Distribution Class
        super().__init__(distribution=transformed_distribution,
                         univariate=True,
                         discrete=False,
                         n_dist_param=len(param_dict),
                         stabilization=stabilization,
                         param_dict=param_dict,
                         distribution_arg_names=list(param_dict.keys()),
                         loss_fn=loss_fn
                         )