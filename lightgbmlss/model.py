import pandas as pd
import numpy as np
import collections
import copy
import json
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


import lightgbm as lgb

from lightgbmlss.distributions.distribution_utils import DistributionClass
from lightgbmlss.utils import *
import optuna
from optuna.samplers import TPESampler
from optuna.integration import LightGBMPruningCallback
import shap

from lightgbm.engine import CVBooster
from lightgbm.basic import (Booster, Dataset)

from sklearn.model_selection import BaseCrossValidator, GroupKFold, StratifiedKFold
from lightgbm.compat import SKLEARN_INSTALLED, _LGBMGroupKFold, _LGBMStratifiedKFold

import re
import pickle

_LGBM_EvalFunctionResultType = Tuple[str, float, bool]
_LGBM_BoosterBestScoreType = Dict[str, Dict[str, float]]
_LGBM_BoosterEvalMethodResultType = Tuple[str, str, float, bool]
_LGBM_CategoricalFeatureConfiguration = Union[List[str], List[int], "Literal['auto']"]
_LGBM_FeatureNameConfiguration = Union[List[str], "Literal['auto']"]
_LGBMBaseCrossValidator = BaseCrossValidator

_LGBM_CustomMetricFunction = Union[
    Callable[
        [np.ndarray, Dataset],
        _LGBM_EvalFunctionResultType,
    ],
    Callable[
        [np.ndarray, Dataset],
        List[_LGBM_EvalFunctionResultType]
    ],
]

_LGBM_PreprocFunction = Callable[
    [Dataset, Dataset, Dict[str, Any]],
    Tuple[Dataset, Dataset, Dict[str, Any]]
]


class LightGBMLSS:
    """
    LightGBMLSS model class

    Parameters
    ----------
    dist : Distribution
        DistributionClass object.
     start_values : np.ndarray
        Starting values for each distributional parameter.
    """
    def __init__(self, dist: DistributionClass):
        self.dist = dist             # Distribution object
        self.start_values = None     # Starting values for distributional parameters

    def set_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set parameters for distributional model.

        Arguments
        ---------
        params : Dict[str, Any]
            Parameters for model.

        Returns
        -------
        params : Dict[str, Any]
            Updated Parameters for model.
        """
        params_adj = {"num_class": self.dist.n_dist_param,
                      "metric": "None",
                      "objective": "None",
                      "random_seed": 123,
                      "verbose": -1}
        params.update(params_adj)

        return params

    def set_init_score(self, dmatrix: Dataset) -> None:
        """
        Set init_score for distributions.

        Arguments
        ---------
        dmatrix : Dataset
            Dataset to set base margin for.

        Returns
        -------
        None
        """
        if self.start_values is None:
            _, self.start_values = self.dist.calculate_start_values(dmatrix.get_label())
        init_score = (np.ones(shape=(dmatrix.get_label().shape[0], 1))) * self.start_values
        dmatrix.set_init_score(init_score.ravel(order="F"))

    def train(self,
              params: Dict[str, Any],
              train_set: Dataset,
              num_boost_round: int = 100,
              valid_sets: Optional[List[Dataset]] = None,
              valid_names: Optional[List[str]] = None,
              init_model: Optional[Union[str, Path, Booster]] = None,
              feature_name: _LGBM_FeatureNameConfiguration = 'auto',
              categorical_feature: _LGBM_CategoricalFeatureConfiguration = 'auto',
              keep_training_booster: bool = False,
              callbacks: Optional[List[Callable]] = None
              ) -> Booster:
        """Function to perform the training of a LightGBMLSS model with given parameters.

        Parameters
        ----------
        params : dict
            Parameters for training. Values passed through ``params`` take precedence over those
            supplied via arguments.
        train_set : Dataset
            Data to be trained on.
        num_boost_round : int, optional (default=100)
            Number of boosting iterations.
        valid_sets : list of Dataset, or None, optional (default=None)
            List of data to be evaluated on during training.
        valid_names : list of str, or None, optional (default=None)
            Names of ``valid_sets``.
        init_model : str, pathlib.Path, Booster or None, optional (default=None)
            Filename of LightGBM model or Booster instance used for continue training.
        feature_name : list of str, or 'auto', optional (default="auto")
            Feature names.
            If 'auto' and data is pandas DataFrame, data columns names are used.
        categorical_feature : list of str or int, or 'auto', optional (default="auto")
            Categorical features.
            If list of int, interpreted as indices.
            If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
            All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
            Floating point numbers in categorical features will be rounded towards 0.
        keep_training_booster : bool, optional (default=False)
            Whether the returned Booster will be used to keep training.
            If False, the returned value will be converted into _InnerPredictor before returning.
            This means you won't be able to use ``eval``, ``eval_train`` or ``eval_valid`` methods of the returned Booster.
            When your model is very large and cause the memory error,
            you can try to set this param to ``True`` to avoid the model conversion performed during the internal call of ``model_to_string``.
            You can still use _InnerPredictor as ``init_model`` for future continue training.
        callbacks : list of callable, or None, optional (default=None)
            List of callback functions that are applied at each iteration.
            See Callbacks in Python API for more information.

        Returns
        -------
        booster : Booster
            The trained Booster model.
        """
        self.set_params(params)
        self.set_init_score(train_set)

        if valid_sets is not None:
            valid_sets = self.set_valid_margin(valid_sets, self.start_values)

        params['objective'] = self.dist.objective_fn
        self.booster = lgb.train(params,
                                 train_set,
                                 num_boost_round=num_boost_round,
                                 feval=self.dist.metric_fn,
                                 valid_sets=valid_sets,
                                 valid_names=valid_names,
                                 init_model=init_model,
                                 feature_name=feature_name,
                                 categorical_feature=categorical_feature,
                                 keep_training_booster=keep_training_booster,
                                 callbacks=callbacks)

    def cv(self,
           params: Dict[str, Any],
           train_set: Dataset,
           num_boost_round: int = 100,
           folds: Optional[Union[Iterable[Tuple[np.ndarray, np.ndarray]], _LGBMBaseCrossValidator]] = None,
           nfold: int = 5,
           stratified: bool = True,
           shuffle: bool = True,
           init_model: Optional[Union[str, Path, Booster]] = None,
           feature_name: _LGBM_FeatureNameConfiguration = 'auto',
           categorical_feature: _LGBM_CategoricalFeatureConfiguration = 'auto',
           fpreproc: Optional[_LGBM_PreprocFunction] = None,
           seed: int = 123,
           callbacks: Optional[List[Callable]] = None,
           eval_train_metric: bool = False,
           return_cvbooster: bool = False
           ) -> Dict[str, Union[List[float], CVBooster]]:
        """Function to cross-validate a LightGBMLSS model with given parameters.

        Parameters
        ----------
        params : dict
            Parameters for training. Values passed through ``params`` take precedence over those
            supplied via arguments.
        train_set : Dataset
            Data to be trained on.
        num_boost_round : int, optional (default=100)
            Number of boosting iterations.
        folds : generator or iterator of (train_idx, test_idx) tuples, scikit-learn splitter object or None, optional (default=None)
            If generator or iterator, it should yield the train and test indices for each fold.
            If object, it should be one of the scikit-learn splitter classes
            (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
            and have ``split`` method.
            This argument has highest priority over other data split arguments.
        nfold : int, optional (default=5)
            Number of folds in CV.
        stratified : bool, optional (default=True)
            Whether to perform stratified sampling.
        shuffle : bool, optional (default=True)
            Whether to shuffle before splitting data.
        init_model : str, pathlib.Path, Booster or None, optional (default=None)
            Filename of LightGBM model or Booster instance used for continue training.
        feature_name : list of str, or 'auto', optional (default="auto")
            Feature names.
            If 'auto' and data is pandas DataFrame, data columns names are used.
        categorical_feature : list of str or int, or 'auto', optional (default="auto")
            Categorical features.
            If list of int, interpreted as indices.
            If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
            If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
            All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
            Floating point numbers in categorical features will be rounded towards 0.
        fpreproc : callable or None, optional (default=None)
            Preprocessing function that takes (dtrain, dtest, params)
            and returns transformed versions of those.
        seed : int, optional (default=0)
            Seed used to generate the folds (passed to numpy.random.seed).
        callbacks : list of callable, or None, optional (default=None)
            List of callback functions that are applied at each iteration.
            See Callbacks in Python API for more information.
        eval_train_metric : bool, optional (default=False)
            Whether to display the train metric in progress.
            The score of the metric is calculated again after each training step, so there is some impact on performance.
        return_cvbooster : bool, optional (default=False)
            Whether to return Booster models trained on each fold through ``CVBooster``.

        Returns
        -------
        eval_hist : dict
            Evaluation history.
            The dictionary has the following format:
            {'metric1-mean': [values], 'metric1-stdv': [values],
            'metric2-mean': [values], 'metric2-stdv': [values],
            ...}.
            If ``return_cvbooster=True``, also returns trained boosters wrapped in a ``CVBooster`` object via ``cvbooster`` key.
        """
        self.set_params(params)
        self.set_init_score(train_set)

        params['objective'] = self.dist.objective_fn
        self.bstLSS_cv = lgb.cv(params,
                                train_set,
                                feval=self.dist.metric_fn,
                                num_boost_round=num_boost_round,
                                folds=folds,
                                nfold=nfold,
                                stratified=False,
                                shuffle=False,
                                metrics=None,
                                init_model=init_model,
                                feature_name=feature_name,
                                categorical_feature=categorical_feature,
                                fpreproc=fpreproc,
                                seed=seed,
                                callbacks=callbacks,
                                eval_train_metric=eval_train_metric,
                                return_cvbooster=return_cvbooster)

        return self.bstLSS_cv

    def hyper_opt(
            self,
            hp_dict: Dict,
            train_set: lgb.Dataset,
            num_boost_round=500,
            nfold=10,
            early_stopping_rounds=20,
            max_minutes=10,
            n_trials=None,
            study_name=None,
            silence=False,
            seed=None,
            hp_seed=None
    ):
        """
        Function to tune hyperparameters using optuna.

        Arguments
        ----------
        hp_dict: dict
            Dictionary of hyperparameters to tune.
        train_set: lgb.Dataset
            Training data.
        num_boost_round: int
            Number of boosting iterations.
        nfold: int
            Number of folds in CV.
        early_stopping_rounds: int
            Activates early stopping. Cross-Validation metric (average of validation
            metric computed over CV folds) needs to improve at least once in
            every **early_stopping_rounds** round(s) to continue training.
            The last entry in the evaluation history will represent the best iteration.
            If there's more than one metric in the **eval_metric** parameter given in
            **params**, the last metric will be used for early stopping.
        max_minutes: int
            Time budget in minutes, i.e., stop study after the given number of minutes.
        n_trials: int
            The number of trials. If this argument is set to None, there is no limitation on the number of trials.
        study_name: str
            Name of the hyperparameter study.
        silence: bool
            Controls the verbosity of the trail, i.e., user can silence the outputs of the trail.
        seed: int
            Seed used to generate the folds (passed to numpy.random.seed).
        hp_seed: int
            Seed for random number generator used in the Bayesian hyper-parameter search.

        Returns
        -------
        opt_params : dict
            Optimal hyper-parameters.
        """

        def objective(trial):

            hyper_params = {}

            for param_name, param_value in hp_dict.items():

                param_type = param_value[0]

                if param_type == "categorical" or param_type == "none":
                    hyper_params.update({param_name: trial.suggest_categorical(param_name, param_value[1])})

                elif param_type == "float":
                    param_constraints = param_value[1]
                    param_low = param_constraints["low"]
                    param_high = param_constraints["high"]
                    param_log = param_constraints["log"]
                    hyper_params.update(
                        {param_name: trial.suggest_float(param_name,
                                                         low=param_low,
                                                         high=param_high,
                                                         log=param_log
                                                         )
                         })

                elif param_type == "int":
                    param_constraints = param_value[1]
                    param_low = param_constraints["low"]
                    param_high = param_constraints["high"]
                    param_log = param_constraints["log"]
                    hyper_params.update(
                        {param_name: trial.suggest_int(param_name,
                                                       low=param_low,
                                                       high=param_high,
                                                       log=param_log
                                                       )
                         })

            # Add booster if not included in dictionary
            if "boosting" not in hyper_params.keys():
                hyper_params.update({"boosting": trial.suggest_categorical("boosting", ["gbdt"])})

            # Add pruning and early stopping
            pruning_callback = LightGBMPruningCallback(trial, self.dist.loss_fn)
            early_stopping_callback = lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)

            lgblss_param_tuning = self.cv(hyper_params,
                                          train_set,
                                          num_boost_round=num_boost_round,
                                          nfold=nfold,
                                          callbacks=[pruning_callback, early_stopping_callback],
                                          seed=seed,
                                          )
            print(lgblss_param_tuning)
            # Extract the optimal number of boosting rounds
            opt_rounds = np.argmin(np.array(lgblss_param_tuning[f"valid {self.dist.loss_fn}-mean"])) + 1
            trial.set_user_attr("opt_round", int(opt_rounds))

            # Extract the best score
            best_score = np.min(np.array(lgblss_param_tuning[f"valid {self.dist.loss_fn}-mean"]))

            return best_score

        if study_name is None:
            study_name = "LightGBMLSS Hyper-Parameter Optimization"

        if silence:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        if hp_seed is not None:
            sampler = TPESampler(seed=hp_seed)
        else:
            sampler = TPESampler()

        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize", study_name=study_name)
        study.optimize(objective, n_trials=n_trials, timeout=60 * max_minutes, show_progress_bar=True)

        print("\nHyper-Parameter Optimization successfully finished.")
        print("  Number of finished trials: ", len(study.trials))
        print("  Best trial:")
        opt_param = study.best_trial

        # Add optimal stopping round
        opt_param.params["opt_rounds"] = study.trials_dataframe()["user_attrs_opt_round"][
            study.trials_dataframe()["value"].idxmin()]
        opt_param.params["opt_rounds"] = int(opt_param.params["opt_rounds"])

        print("    Value: {}".format(opt_param.value))
        print("    Params: ")
        for key, value in opt_param.params.items():
            print("    {}: {}".format(key, value))

        return opt_param.params

    def predict(self,
                data: pd.DataFrame,
                pred_type: str = "parameters",
                n_samples: int = 1000,
                quantiles: list = [0.1, 0.5, 0.9],
                seed: str = 123):
        """
        Function that predicts from the trained model.

        Arguments
        ---------
        data : pd.DataFrame
            Data to predict from.
        pred_type : str
            Type of prediction:
            - "samples" draws n_samples from the predicted distribution.
            - "quantiles" calculates the quantiles from the predicted distribution.
            - "parameters" returns the predicted distributional parameters.
            - "expectiles" returns the predicted expectiles.
        n_samples : int
            Number of samples to draw from the predicted distribution.
        quantiles : List[float]
            List of quantiles to calculate from the predicted distribution.
        seed : int
            Seed for random number generator used to draw samples from the predicted distribution.

        Returns
        -------
        predt_df : pd.DataFrame
            Predictions.
        """

        # Predict
        predt_df = self.dist.predict_dist(booster=self.booster,
                                          data=data,
                                          start_values=self.start_values,
                                          pred_type=pred_type,
                                          n_samples=n_samples,
                                          quantiles=quantiles,
                                          seed=seed)

        return predt_df

    def plot(self,
             X: pd.DataFrame,
             feature: str = "x",
             parameter: str = "loc",
             max_display: int = 15,
             plot_type: str = "Partial_Dependence"):
        """
        LightGBMLSS SHap plotting function.

        Arguments:
        ---------
        X: pd.DataFrame
            Train/Test Data
        feature: str
            Specifies which feature is to be plotted.
        parameter: str
            Specifies which distributional parameter is to be plotted.
        max_display: int
            Specifies the maximum number of features to be displayed.
        plot_type: str
            Specifies the type of plot:
                "Partial_Dependence" plots the partial dependence of the parameter on the feature.
                "Feature_Importance" plots the feature importance of the parameter.
        """
        shap.initjs()
        explainer = shap.TreeExplainer(self.booster)
        shap_values = explainer(X)

        param_pos = self.dist.distribution_arg_names.index(parameter)

        if plot_type == "Partial_Dependence":
            if self.dist.n_dist_param == 1:
                shap.plots.scatter(shap_values[:, feature], color=shap_values[:, feature])
            else:
                shap.plots.scatter(shap_values[:, feature][:, param_pos], color=shap_values[:, feature][:, param_pos])
        elif plot_type == "Feature_Importance":
            if self.dist.n_dist_param == 1:
                shap.plots.bar(shap_values, max_display=max_display if X.shape[1] > max_display else X.shape[1])
            else:
                shap.plots.bar(
                    shap_values[:, :, param_pos], max_display=max_display if X.shape[1] > max_display else X.shape[1]
                )

    def expectile_plot(self,
                       X: pd.DataFrame,
                       feature: str = "x",
                       expectile: str = "0.05",
                       plot_type: str = "Partial_Dependence"):
        """
        LightGBMLSS function for plotting expectile SHapley values.

        X: pd.DataFrame
            Train/Test Data
        feature: str
            Specifies which feature to use for plotting Partial_Dependence plot.
        expectile: str
            Specifies which expectile to plot.
        plot_type: str
            Specifies which SHapley-plot to visualize. Currently, "Partial_Dependence" and "Feature_Importance"
            are supported.
        """

        shap.initjs()
        explainer = shap.TreeExplainer(self.booster)
        shap_values = explainer(X)

        expect_pos = list(self.dist.param_dict.keys()).index(expectile)

        if plot_type == "Partial_Dependence":
            shap.plots.scatter(shap_values[:, feature][:, expect_pos], color=shap_values[:, feature][:, expect_pos])
        elif plot_type == "Feature_Importance":
            shap.plots.bar(shap_values[:, :, expect_pos], max_display=15 if X.shape[1] > 15 else X.shape[1])

    def set_valid_margin(self,
                         valid_sets: list,
                         start_values: np.ndarray
                         ) -> list:
        """
        Function that sets the base margin for the validation set.

        Arguments
        ---------
        valid_sets : list
            List of tuples containing the train and evaluation set.
        valid_names: list
            List of tuples containing the name of train and evaluation set.
        start_values : np.ndarray
            Array containing the start values for the distributional parameters.

        Returns
        -------
        valid_sets : list
            List of tuples containing the train and evaluation set.
        """
        valid_sets1 = valid_sets[0]
        init_score_val1 = (np.ones(shape=(valid_sets1.get_label().shape[0], 1))) * start_values
        valid_sets1.set_init_score(init_score_val1.ravel(order="F"))

        valid_sets2 = valid_sets[1]
        init_score_val2 = (np.ones(shape=(valid_sets2.get_label().shape[0], 1))) * start_values
        valid_sets2.set_init_score(init_score_val2.ravel(order="F"))

        valid_sets = [valid_sets1, valid_sets2]

        return valid_sets

    def save_model(self,
                   model_path: str
                   ) -> None:
        """
        Save the model to a file.

        Parameters
        ----------
        model_path : str
            The path to save the model.

        Returns
        -------
        None
        """
        with open(model_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(model_path: str):
        """
        Load the model from a file.

        Parameters
        ----------
        model_path : str
            The path to the saved model.

        Returns
        -------
        The loaded model.
        """
        with open(model_path, "rb") as f:
            return pickle.load(f)
