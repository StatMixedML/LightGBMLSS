from setuptools import setup, find_packages

setup(
    name="lightgbmlss",
    version="0.4.0",
    description="LightGBMLSS - An extension of LightGBM to probabilistic modelling and prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Alexander März",
    author_email="alex.maerz@gmx.net",
    url="https://github.com/StatMixedML/LightGBMLSS",
    license="Apache License 2.0",
    packages=find_packages(exclude=["docs", "tests*"]),
    include_package_data=True,
    package_data={'': ['datasets/*.csv']},
    zip_safe=True,
    python_requires=">=3.9",
    install_requires=[
        "lightgbm~=4.1",
        "torch~=2.0.1",
        "pyro-ppl~=1.8.5",
        "optuna~=3.3.0",
        "properscoring~=0.1",
        "scikit-learn~=1.2.2",
        "numpy~=1.24.3",
        "pandas~=2.0.3",
        "plotnine~=0.12.1",
        "scipy~=1.11.1",
        "seaborn~=0.12.2",
        "tqdm~=4.65.0",
        "matplotlib~=3.7.2",
        "ipython~=8.14.0",
    ],
    extras_require={
        "docs": ["mkdocs", "mkdocstrings[python]", "mkdocs-jupyter"]
    },
    test_suite="tests",
    tests_require=["flake8", "pytest"],
)
