<h4 align="center">

|  | **[Documentation](https://statmixedml.github.io/LightGBMLSS/)** · **[Release Notes](https://github.com/StatMixedML/LightGBMLSS/releases)** |
|---|---|
| **Open&#160;Source** | [![Apache 2.0](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/StatMixedML/LightGBMLSS/blob/master/LICENSE.txt) |
| **CI/CD** | [![github-actions](https://img.shields.io/github/actions/workflow/status/statmixedml/lightgbmlss/unit-tests.yml?logo=github)](https://github.com/statmixedml/lightgbmlss/actions/workflows/unit-tests.yml) <img src="https://github.com/StatMixedML/LightGBMLSS/actions/workflows/mkdocs.yaml/badge.svg" alt="Documentation status badge"> |
| **Code** |  [![!pypi](https://img.shields.io/pypi/v/lightgbmlss?color=orange)](https://pypi.org/project/lightgbmlss/) [![!python-versions](https://img.shields.io/pypi/pyversions/lightgbmlss)](https://www.python.org/) <img src="https://codecov.io/gh/StatMixedML/LightGBMLSS/branch/master/graph/badge.svg" alt="Code coverage status badge"> |
| **Downloads** | ![Pepy Total Downlods](https://img.shields.io/pepy/dt/lightgbmlss?label=PyPI%20Downloads%2FMonth&color=green) |
| **Citation** | [![Arxiv link](https://img.shields.io/badge/arXiv-XGBoostLSS%3A%20An%20extension%20of%20XGBoost%20to%20probabilistic%20forecasting-color=brightgreen)](https://arxiv.org/abs/1907.03178) |

<!---
![Python Version](https://img.shields.io/badge/python-3.10%20|%20%203.11-lightblue.svg)
![GitHub tag (with filter)](https://img.shields.io/github/v/tag/StatMixedML/LightGBMLSS?label=release&color=lightblue)
<img src="https://github.com/StatMixedML/LightGBMLSS/actions/workflows/mkdocs.yaml/badge.svg" alt="Documentation status badge">
<img src="https://github.com/StatMixedML/LightGBMLSS/workflows/unit-tests/badge.svg" alt="Unit test status badge">
<img src="https://codecov.io/gh/StatMixedML/LightGBMLSS/branch/master/graph/badge.svg" alt="Code coverage status badge">
![Pepy Total Downlods](https://img.shields.io/pepy/dt/lightgbmlss?label=PyPI%20Downloads%2FMonth&color=green)
-->

</h4>

#
<img align="right" width="156.5223" height="181.3" src="figures/LightGBMLSS.png">


# LightGBMLSS - An extension of LightGBM to probabilistic modelling
We introduce a comprehensive framework that models and predicts the full conditional distribution of a univariate target as a function of covariates. Choosing from a wide range of continuous, discrete, and mixed discrete-continuous distributions, modelling and predicting the entire conditional distribution greatly enhances the flexibility of LightGBM, as it allows to create probabilistic forecasts from which prediction intervals and quantiles of interest can be derived.

## `Features`
:white_check_mark: Estimation of all distributional parameters. <br/>
:white_check_mark: Normalizing Flows allow modelling of complex and multi-modal distributions. <br/>
:white_check_mark: Mixture-Densities can model a diverse range of data characteristics. <br/>
:white_check_mark: Zero-Adjusted and Zero-Inflated Distributions for modelling excess of zeros in the data. <br/>
:white_check_mark: Automatic derivation of Gradients and Hessian of all distributional parameters using [PyTorch](https://pytorch.org/docs/stable/autograd.html). <br/>
:white_check_mark: Automated hyper-parameter search, including pruning, is done via [Optuna](https://optuna.org/). <br/>
:white_check_mark: The output of LightGBMLSS is explained using [SHapley Additive exPlanations](https://github.com/dsgibbons/shap). <br/>
:white_check_mark: LightGBMLSS provides full compatibility with all the features and functionality of LightGBM. <br/>
:white_check_mark: LightGBMLSS is available in Python. <br/>

## `News`
:boom: [2025-12-11] Release of v0.6.0 LightGBMLSS to [PyPI](https://pypi.org/project/lightgbmlss/). <br/>
:boom: [2024-01-19] Release of LightGBMLSS to [PyPI](https://pypi.org/project/lightgbmlss/). <br/>
:boom: [2023-08-28] Release of v0.4.0 introduces Mixture-Densities. See the [release notes](https://github.com/StatMixedML/LightGBMLSS/releases) for an overview. <br/>
:boom: [2023-07-20] Release of v0.3.0 introduces Normalizing Flows. See the [release notes](https://github.com/StatMixedML/LightGBMLSS/releases) for an overview. <br/>
:boom: [2023-06-22] Release of v0.2.2. See the [release notes](https://github.com/StatMixedML/LightGBMLSS/releases) for an overview. <br/>
:boom: [2023-06-15] LightGBMLSS now supports Zero-Inflated and Zero-Adjusted Distributions. <br/>
:boom: [2023-05-26] Release of v0.2.1. See the [release notes](https://github.com/StatMixedML/LightGBMLSS/releases) for an overview. <br/>
:boom: [2023-05-23] Release of v0.2.0. See the [release notes](https://github.com/StatMixedML/LightGBMLSS/releases) for an overview. <br/>
:boom: [2022-01-05] LightGBMLSS now supports estimating the full predictive distribution via [Expectile Regression](https://epub.ub.uni-muenchen.de/31542/1/1471082x14561155.pdf). <br/>
:boom: [2022-01-05] LightGBMLSS now supports automatic derivation of Gradients and Hessians. <br/>
:boom: [2022-01-04] LightGBMLSS is initialized with suitable starting values to improve convergence of estimation. <br/>
:boom: [2022-01-04] LightGBMLSS v0.1.0 is released!

## `Installation`
To install the development version, please use
```python
pip install git+https://github.com/StatMixedML/LightGBMLSS.git
```
For the PyPI version, please use
```python
pip install lightgbmlss
```

## `Available Distributions`
Our framework is built upon PyTorch and Pyro, enabling users to harness a diverse set of distributional families. LightGBMLSS currently supports the [following distributions](https://statmixedml.github.io/LightGBMLSS/distributions/).

## `How to Use`
Please visit the [example section](https://statmixedml.github.io/LightGBMLSS/examples/Gaussian_Regression/) for guidance on how to use the framework.

## `Documentation`
For more information and context, please visit the [documentation](https://statmixedml.github.io/LightGBMLSS/).

## `Feedback`
We encourage you to provide feedback on how to enhance LightGBMLSS or request the implementation of additional distributions by opening a [new discussion](https://github.com/StatMixedML/LightGBMLSS/discussions).

## `How to Cite`
If you use LightGBMLSS in your research, please cite it as:

```bibtex
@misc{Maerz2023,
  author = {Alexander M\"arz},
  title = {{LightGBMLSS: An Extension of LightGBM to Probabilistic Modelling}},
  year = {2023},
  note = {GitHub repository, Version 0.6.0},
  howpublished = {\url{https://github.com/StatMixedML/LightGBMLSS}}
}
```

## `Reference Paper`
[![Arxiv link](https://img.shields.io/badge/arXiv-Distributional%20Gradient%20Boosting%20Machines-color=brightgreen)](https://arxiv.org/abs/2204.00778) <br/>
[![Arxiv link](https://img.shields.io/badge/arXiv-XGBoostLSS%3A%20An%20extension%20of%20XGBoost%20to%20probabilistic%20forecasting-color=brightgreen)](https://arxiv.org/abs/1907.03178) <br/>

## `Star History`
<a href="https://star-history.com/#StatMixedML/LightGBMLSS&Date">
    <img src="https://api.star-history.com/svg?repos=StatMixedML/LightGBMLSS&type=Date" width="450">
</a>

