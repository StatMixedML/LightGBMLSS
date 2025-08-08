import torch
import torch.nn as nn
import numpy as np
from torch.distributions import identity_transform, SigmoidTransform, SoftplusTransform
from pyro.distributions import Normal
from pyro.distributions.transforms import Transform
from .flow_utils import NormalizingFlowClass
from ..utils import identity_fn


class BernsteinQuantileTransform(Transform):
    """
    Bernstein polynomial quantile transform.
    
    This transform uses Bernstein polynomials to parameterize a quantile function (inverse CDF).
    The Bernstein polynomial of degree n is defined as:
    
    Q(u) = sum_{k=0}^n beta_k * B_{n,k}(u)
    
    where B_{n,k}(u) = C(n,k) * u^k * (1-u)^{n-k} are the Bernstein basis functions,
    C(n,k) is the binomial coefficient, and beta_k are the learnable parameters that
    represent quantile values at specific points.
    
    The monotonicity constraint is enforced by requiring beta_k <= beta_{k+1}.
    """
    
    domain = torch.distributions.constraints.unit_interval
    codomain = torch.distributions.constraints.real
    bijective = True
    sign = +1
    
    def __init__(self, degree, support_bounds=(-5.0, 5.0)):
        super().__init__()
        self.degree = degree
        self.support_bounds = support_bounds
        
        # Initialize parameters with sorted values to ensure monotonicity
        init_values = torch.linspace(support_bounds[0], support_bounds[1], degree + 1)
        self.raw_betas = nn.Parameter(init_values)
        
        # Precompute binomial coefficients
        self.register_buffer('binomial_coeffs', self._compute_binomial_coefficients(degree))
    
    def _compute_binomial_coefficients(self, n):
        """Compute binomial coefficients C(n,k) for k=0,...,n"""
        coeffs = torch.zeros(n + 1)
        for k in range(n + 1):
            # Use scipy.special.comb for compatibility, fallback to manual calculation
            try:
                from scipy.special import comb
                coeffs[k] = torch.tensor(float(comb(n, k)))
            except ImportError:
                # Manual binomial coefficient calculation: C(n,k) = n! / (k! * (n-k)!)
                if k == 0 or k == n:
                    coeffs[k] = 1.0
                else:
                    coeff = 1.0
                    for i in range(min(k, n-k)):
                        coeff = coeff * (n - i) / (i + 1)
                    coeffs[k] = torch.tensor(float(coeff))
        return coeffs
    
    @property
    def betas(self):
        """Ensure monotonicity by using cumulative sum"""
        return torch.cumsum(torch.cat([self.raw_betas[:1], torch.nn.functional.softplus(self.raw_betas[1:] - self.raw_betas[:-1])]), dim=0)
    
    def _bernstein_basis(self, u, k):
        """Compute k-th Bernstein basis polynomial of degree n at u"""
        n = self.degree
        # B_{n,k}(u) = C(n,k) * u^k * (1-u)^{n-k}
        if k == 0:
            return self.binomial_coeffs[k] * torch.pow(1 - u, n - k)
        elif k == n:
            return self.binomial_coeffs[k] * torch.pow(u, k)
        else:
            return self.binomial_coeffs[k] * torch.pow(u, k) * torch.pow(1 - u, n - k)
    
    def _bernstein_polynomial(self, u):
        """Evaluate Bernstein polynomial quantile function at u"""
        u = torch.clamp(u, 1e-7, 1 - 1e-7)  # Avoid boundary issues
        
        result = torch.zeros_like(u)
        betas = self.betas
        
        for k in range(self.degree + 1):
            basis = self._bernstein_basis(u, k)
            result += betas[k] * basis
            
        return result
    
    def _bernstein_derivative(self, u):
        """Compute derivative of Bernstein polynomial (needed for Jacobian)"""
        u = torch.clamp(u, 1e-7, 1 - 1e-7)
        
        if self.degree == 0:
            return torch.zeros_like(u)
            
        # Derivative using the property: d/du B_{n,k}(u) = n * [B_{n-1,k-1}(u) - B_{n-1,k}(u)]
        result = torch.zeros_like(u)
        betas = self.betas
        n = self.degree
        
        for k in range(self.degree + 1):
            if k > 0:
                # B_{n-1,k-1}(u) term
                if k-1 <= n-1:
                    prev_basis = self._bernstein_basis_degree(u, k-1, n-1)
                    result += n * betas[k] * prev_basis
            
            if k < self.degree:
                # -B_{n-1,k}(u) term  
                if k <= n-1:
                    curr_basis = self._bernstein_basis_degree(u, k, n-1)
                    result -= n * betas[k] * curr_basis
        
        return torch.clamp(result, 1e-7, float('inf'))  # Ensure positive for monotonicity
    
    def _bernstein_basis_degree(self, u, k, degree):
        """Compute k-th Bernstein basis polynomial of given degree at u"""
        if degree < k or k < 0:
            return torch.zeros_like(u)
        
        # Compute binomial coefficient with fallback
        try:
            from scipy.special import comb
            binomial_coeff = float(comb(degree, k))
        except ImportError:
            if k == 0 or k == degree:
                binomial_coeff = 1.0
            else:
                coeff = 1.0
                for i in range(min(k, degree-k)):
                    coeff = coeff * (degree - i) / (i + 1)
                binomial_coeff = float(coeff)
        
        if k == 0:
            return binomial_coeff * torch.pow(1 - u, degree - k)
        elif k == degree:
            return binomial_coeff * torch.pow(u, k)
        else:
            return binomial_coeff * torch.pow(u, k) * torch.pow(1 - u, degree - k)
    
    def __call__(self, x):
        """Transform from uniform [0,1] to target distribution"""
        return self._bernstein_polynomial(x)
    
    def _inverse(self, y):
        """Inverse transform: find u such that Q(u) = y"""
        # This requires numerical inversion since no analytical inverse exists
        # We use binary search for robustness
        return self._numerical_inverse(y)
    
    def _numerical_inverse(self, y, max_iter=50, tol=1e-6):
        """Numerical inverse using binary search"""
        # Clamp y to support bounds
        y = torch.clamp(y, self.support_bounds[0] + 1e-6, self.support_bounds[1] - 1e-6)
        
        batch_shape = y.shape
        y_flat = y.flatten()
        
        # Initialize bounds
        lower = torch.zeros_like(y_flat) + 1e-7
        upper = torch.ones_like(y_flat) - 1e-7
        
        for _ in range(max_iter):
            mid = (lower + upper) / 2
            f_mid = self._bernstein_polynomial(mid.reshape(batch_shape)).flatten()
            
            # Update bounds based on comparison
            mask = f_mid < y_flat
            lower = torch.where(mask, mid, lower)
            upper = torch.where(~mask, mid, upper)
            
            # Check convergence
            if torch.max(upper - lower) < tol:
                break
        
        result = (lower + upper) / 2
        return result.reshape(batch_shape)
    
    def log_abs_det_jacobian(self, x, y):
        """Compute log absolute determinant of Jacobian"""
        derivative = self._bernstein_derivative(x)
        return torch.log(derivative)


class BernsteinFlow(NormalizingFlowClass):
    """
    Bernstein Flow class.
    
    The Bernstein flow is a normalizing flow based on Bernstein polynomial quantile functions.
    It uses Bernstein polynomials as basis functions to construct flexible, monotonic transformations
    that naturally preserve the ordering required for valid probability distributions.
    
    Key features:
    - Shape-constrained modeling with natural monotonicity preservation
    - Interpretable parameters (each coefficient represents a quantile value)
    - Computational efficiency with simple polynomial evaluation
    - Flexibility to approximate any monotonic quantile function
    
    Parameters
    ----------
    target_support : str
        The target support. Options are:
        - "real": [-inf, inf]
        - "positive": [0, inf]  
        - "positive_integer": [0, 1, 2, 3, ...]
        - "unit_interval": [0, 1]
    degree : int
        The degree of the Bernstein polynomial. Higher degree provides more flexibility
        but requires more parameters. Typical values: 5-15.
    bound : float
        The support bounds for the distribution. The quantile function will map
        [0,1] to approximately [-bound, bound].
    stabilization : str
        Stabilization method for the Gradient and Hessian. Options are "None", "MAD" or "L2".
    loss_fn : str
        Loss function. Options are "nll" (negative log-likelihood) or "crps" 
        (continuous ranked probability score). Note that if "crps" is used, the Hessian 
        is set to 1, as the current CRPS version is not twice differentiable.
    """
    
    def __init__(self,
                 target_support: str = "real",
                 degree: int = 8,
                 bound: float = 5.0,
                 stabilization: str = "None", 
                 loss_fn: str = "nll"
                 ):
        
        # Input validation
        if not isinstance(target_support, str):
            raise ValueError("target_support must be a string.")
            
        # Specify Target Transform
        transforms = {
            "real": (identity_transform, False),
            "positive": (SoftplusTransform(), False),
            "positive_integer": (SoftplusTransform(), True),
            "unit_interval": (SigmoidTransform(), False)
        }
        
        if target_support in transforms:
            target_transform, discrete = transforms[target_support]
        else:
            raise ValueError(
                "Invalid target_support. Options are 'real', 'positive', 'positive_integer', or 'unit_interval'.")
        
        # Check if degree is valid
        if not isinstance(degree, int):
            raise ValueError("degree must be an integer.")
        if degree <= 0:
            raise ValueError("degree must be a positive integer > 0.")
        if degree > 20:
            raise ValueError("degree should be <= 20 for numerical stability.")
            
        # Check if bound is valid
        if not isinstance(bound, float):
            bound = float(bound)
        if bound <= 0:
            raise ValueError("bound must be positive.")
            
        # Number of parameters (degree + 1 coefficients)
        n_params = degree + 1
        
        # Check if stabilization method is valid
        if not isinstance(stabilization, str):
            raise ValueError("stabilization must be a string.")
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError("Invalid stabilization method. Options are 'None', 'MAD' or 'L2'.")
            
        # Check if loss function is valid
        if not isinstance(loss_fn, str):
            raise ValueError("loss_fn must be a string.")
        if loss_fn not in ["nll", "crps"]:
            raise ValueError("Invalid loss_fn. Options are 'nll' or 'crps'.")
        
        # Specify parameter dictionary
        param_dict = {f"beta_{i}": identity_fn for i in range(n_params)}
        torch.distributions.Distribution.set_default_validate_args(False)
        
        # Support bounds for the transform
        support_bounds = (-bound, bound)
        
        # Specify Normalizing Flow Class
        super().__init__(base_dist=Normal,
                         flow_transform=BernsteinQuantileTransform,
                         degree=degree,
                         support_bounds=support_bounds,
                         n_dist_param=n_params,
                         param_dict=param_dict,
                         distribution_arg_names=list(param_dict.keys()),
                         target_transform=target_transform,
                         discrete=discrete,
                         univariate=True,
                         stabilization=stabilization,
                         loss_fn=loss_fn
                         )