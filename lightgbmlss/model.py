import pandas as pd
import numpy as np
import collections
import copy
import json
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


import lightgbm as lgb
from lightgbmlss.utils import *
import optuna
from optuna.samplers import TPESampler
from optuna.integration import LightGBMPruningCallback
import shap

from lightgbm.engine import CVBooster
from lightgbm.basic import (Booster, Dataset)

from sklearn.model_selection import BaseCrossValidator, GroupKFold, StratifiedKFold
from lightgbm.compat import SKLEARN_INSTALLED, _LGBMGroupKFold, _LGBMStratifiedKFold

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
    """
    def __init__(self, dist):
        self.dist = dist.dist_class  # Distribution object

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

        params_adj = {"num_class": self.dist.n_dist_param,
                      "metric": "None",
                      "objective": "None",
                      "random_seed": 123,
                      "verbose": -1}

        params.update(params_adj)

        # Set init_score as starting point for each distributional parameter.
        _, self.start_values = self.dist.calculate_start_values(train_set.get_label())
        init_score = (np.ones(shape=(train_set.get_label().shape[0], 1))) * self.start_values
        train_set.set_init_score(init_score.ravel(order="F"))

        self.booster = lgb.train(params,
                                 train_set,
                                 num_boost_round=num_boost_round,
                                 fobj=self.dist.objective_fn,
                                 feval=self.dist.metric_fn,
                                 valid_sets=valid_sets,
                                 valid_names=valid_names,
                                 init_model=init_model,
                                 feature_name=feature_name,
                                 categorical_feature=categorical_feature,
                                 keep_training_booster=keep_training_booster,
                                 callbacks=callbacks)
        return self.booster

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

        params_adj = {"num_class": self.dist.n_dist_param,
                      "metric": "None",
                      "objective": "None",
                      "random_seed": 123,
                      "verbose": -1}

        params.update(params_adj)

        # Set init_score as starting point for each distributional parameter.
        _, self.start_values = self.dist.calculate_start_values(train_set.get_label())
        init_score = (np.ones(shape=(train_set.get_label().shape[0], 1))) * self.start_values
        train_set.set_init_score(init_score.ravel(order="F"))

        bstLSS_cv = lgb.cv(params,
                           train_set,
                           fobj=self.dist.objective_fn,
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

        return bstLSS_cv

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
            pruning_callback = LightGBMPruningCallback(trial, "NegLogLikelihood")
            early_stopping_callback = lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)

            lgblss_param_tuning = self.cv(hyper_params,
                                          train_set,
                                          num_boost_round=num_boost_round,
                                          nfold=nfold,
                                          callbacks=[pruning_callback, early_stopping_callback],
                                          seed=seed,
                                          )

            # Extract the optimal number of boosting rounds
            opt_rounds = np.argmin(np.array(lgblss_param_tuning["NegLogLikelihood-mean"])) + 1
            trial.set_user_attr("opt_round", int(opt_rounds))

            # Extract the best score
            best_score = np.min(np.array(lgblss_param_tuning["NegLogLikelihood-mean"]))

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
                test_set: pd.DataFrame,
                pred_type: str = "parameters",
                n_samples: int = 1000,
                quantiles: list = [0.1, 0.5, 0.9],
                seed: str = 123):
        """
        Function that predicts from the trained model.

        Arguments
        ---------
        test_set : pd.DataFrame
            Test data.
        pred_type : str
            Type of prediction:
            - "samples" draws n_samples from the predicted distribution.
            - "quantile" calculates the quantiles from the predicted distribution.
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

        # Set init_score as starting point for each distributional parameter.
        init_score_pred = (np.ones(shape=(test_set.shape[0], 1))) * self.start_values

        # Predict
        predt_df = self.dist.predict_dist(self.booster,
                                          test_set,
                                          init_score_pred,
                                          pred_type,
                                          n_samples,
                                          quantiles,
                                          seed)

        return predt_df

    def plot(self,
             X: pd.DataFrame,
             feature: str = "x",
             parameter: str = "loc",
             plot_type: str = "Partial_Dependence"):
        """
        XGBoostLSS SHap plotting function.

        Arguments:
        ---------
        X: pd.DataFrame
            Train/Test Data
        feature: str
            Specifies which feature is to be plotted.
        parameter: str
            Specifies which parameter is to be plotted. Valid parameters are "location", "scale", "df", "tau".
        plot_type: str
            Specifies the type of plot:
                "Partial_Dependence" plots the partial dependence of the parameter on the feature.
                "Feature_Importance" plots the feature importance of the parameter.
        """
        shap.initjs()
        explainer = shap.TreeExplainer(self.booster)
        shap_values = explainer(X)

        param_pos = list(self.dist.param_dict.keys()).index(parameter)

        if plot_type == "Partial_Dependence":
            if self.dist.n_dist_param == 1:
                shap.plots.scatter(shap_values[:, feature], color=shap_values[:, feature])
            else:
                shap.plots.scatter(shap_values[:, feature][:, param_pos], color=shap_values[:, feature][:, param_pos])
        elif plot_type == "Feature_Importance":
            if self.dist.n_dist_param == 1:
                shap.plots.bar(shap_values, max_display=15 if X.shape[1] > 15 else X.shape[1])
            else:
                shap.plots.bar(shap_values[:, :, param_pos], max_display=15 if X.shape[1] > 15 else X.shape[1])

    def expectile_plot(self,
                       X: pd.DataFrame,
                       feature: str = "x",
                       expectile: str = "0.05",
                       plot_type: str = "Partial_Dependence"):
        """
        XGBoostLSS function for plotting expectile SHapley values.

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
