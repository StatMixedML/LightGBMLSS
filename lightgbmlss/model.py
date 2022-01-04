import pandas as pd
import lightgbm as lgb
from lightgbmlss.distributions import *
from lightgbmlss.utils import *
import optuna
from optuna.samplers import TPESampler
from optuna.integration import LightGBMPruningCallback
import shap
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path


class lightgbmlss:
    """
    LightGBMLSS model class

    """

    def train(params: Dict[str, Any],
              dtrain: lgb.Dataset,
              dist,
              num_boost_round: int = 100,
              valid_sets: Optional[List[lgb.Dataset]] = None,
              valid_names: Optional[List[str]] = None,
              init_model: Optional[Union[str, Path, lgb.Booster]] = None,
              feature_name: Union[List[str], str] = 'auto',
              categorical_feature: Union[List[str], List[int], str] = 'auto',
              keep_training_booster: bool = False,
              callbacks: Optional[List[Callable]] = None) -> lgb.Booster:

        """Train a LightGBMLSS model with given parameters.

        Parameters
        ----------
        params : dict
            Parameters for training.
        dtrain : Dataset
            Data to be trained on.
        dist: lightgbmlss.distributions class.
            Specifies distributional assumption.
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
            All values in categorical features should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
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

        params_adj = {"num_class": dist.n_dist_param(),
                      "metric": "None",
                      "objective": "None",
                      "random_seed": 123,
                      "verbose": -1}

        params.update(params_adj)

        # Set init_score as starting point for each distributional parameter.
        dist.start_values = dist.initialize(dtrain.get_label())
        init_score = (np.ones(shape=(dtrain.get_label().shape[0], 1))) * dist.start_values
        dtrain.set_init_score(init_score.ravel(order="F"))

        bstLSS_train = lgb.train(params,
                                 dtrain,
                                 num_boost_round=num_boost_round,
                                 fobj=dist.Dist_Objective,
                                 feval=dist.Dist_Metric,
                                 valid_sets = valid_sets,
                                 valid_names=valid_names,
                                 init_model=init_model,
                                 feature_name=feature_name,
                                 categorical_feature=categorical_feature,
                                 keep_training_booster=keep_training_booster,
                                 callbacks=callbacks)
        return bstLSS_train


    def cv(params,
           dtrain,
           dist,
           num_boost_round=100,
           folds=None,
           nfold=10,
           stratified=False,
           shuffle=False,
           metrics=None,
           init_model=None,
           feature_name='auto',
           categorical_feature='auto',
           fpreproc=None,
           seed=123,
           callbacks=None,
           eval_train_metric=False,
           return_cvbooster=False):

        """Function to cross-validate a LightGBMLSS model with given parameters.

        Parameters
        ----------
        params : dict
            Parameters for Booster.
        dtrain : Dataset
            Data to be trained on.
        dist: lightgbm.distributions class
            Specifies distributional assumption.
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
        metrics : str, list of str, or None, optional (default=None)
            Evaluation metrics to be monitored while CV.
            If not None, the metric in ``params`` will be overridden.
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
            All values in categorical features should be less than int32 max value (2147483647).
            Large values could be memory consuming. Consider using consecutive integers starting from zero.
            All negative values in categorical features will be treated as missing values.
            The output cannot be monotonically constrained with respect to a categorical feature.
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
            If ``return_cvbooster=True``, also returns trained boosters via ``cvbooster`` key.
        """

        params_adj = {"num_class": dist.n_dist_param(),
                      "metric": "None",
                      "objective": "None",
                      "random_seed": 123,
                      "verbose": -1}

        params.update(params_adj)

        # Set init_score as starting point for each distributional parameter.
        dist.start_values = dist.initialize(dtrain.get_label())
        init_score = (np.ones(shape=(dtrain.get_label().shape[0], 1))) * dist.start_values
        dtrain.set_init_score(init_score.ravel(order="F"))

        bstLSS_cv = lgb.cv(params,
                           dtrain,
                           fobj=dist.Dist_Objective,
                           feval=dist.Dist_Metric,
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



    def hyper_opt(params, dtrain, dist, num_boost_round=500, nfold=10, early_stopping_rounds=20,
                  max_minutes=10, n_trials = None, study_name = "LightGBMLSS-HyperOpt", silence=False):
        """Function to tune hyperparameters using optuna.

        Parameters
        ----------
        params : dict
            Booster params in the form of "params_name": [min_val, max_val].
        dtrain : Dataset
            Data to be trained on.
        dist: lightgbmlss.distributions class
            Specifies distributional assumption.
        num_boost_round : int
            Number of boosting iterations.
        nfold : int
            Number of folds in CV.
        early_stopping_rounds: int
            Activates early stopping. Cross-Validation metric (average of validation
            metric computed over CV folds) needs to improve at least once in
            every **early_stopping_rounds** round(s) to continue training.
            The last entry in the evaluation history will represent the best iteration.
            If there's more than one metric in the **eval_metric** parameter given in
            **params**, the last metric will be used for early stopping.
        max_minutes : int
            Time budget in minutes, i.e., stop study after the given number of minutes.
        n_trials : int
            The number of trials. If this argument is set to None, there is no limitation on the number of trials.
        study_name : str
            Name of the hyperparameter study.
        silence : bool
            Controls the verbosity of the trail, i.e., user can silence the outputs of the trail.

        Returns
        -------
        opt_params : Dict() with optimal parameters.
        """

        def objective(trial):

            hyper_params = {
                "boosting": "gbdt",

                "eta": trial.suggest_loguniform("eta",
                                                params["eta"][0],
                                                params["eta"][1]),

                "max_depth": trial.suggest_int("max_depth",
                                               params["max_depth"][0],
                                               params["max_depth"][1]),

                "num_leaves": trial.suggest_int("num_leaves",
                                                params["num_leaves"][0],
                                                params["num_leaves"][1],
                                                step=16),

                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf",
                                                      params["min_data_in_leaf"][0],
                                                      params["min_data_in_leaf"][1],
                                                      step=100),

                "lambda_l1": trial.suggest_int("lambda_l1",
                                               params["lambda_l1"][0],
                                               params["lambda_l1"][1],
                                               step=1),

                "lambda_l2": trial.suggest_int("lambda_l2",
                                               params["lambda_l2"][0],
                                               params["lambda_l2"][1],
                                               step=1),

                "min_gain_to_split": trial.suggest_loguniform("min_gain_to_split",
                                                              params["min_gain_to_split"][0],
                                                              params["min_gain_to_split"][1]),

                "min_sum_hessian_in_leaf": trial.suggest_int("min_sum_hessian_in_leaf",
                                                             params["min_sum_hessian_in_leaf"][0],
                                                             params["min_sum_hessian_in_leaf"][1]),

                "subsample": trial.suggest_float("subsample",
                                                 params["subsample"][0],
                                                 params["subsample"][1]),

                "feature_fraction": trial.suggest_float("feature_fraction",
                                                        params["feature_fraction"][0],
                                                        params["feature_fraction"][1])
            }


            # Add pruning and early stopping
            pruning_callback = LightGBMPruningCallback(trial, "NegLogLikelihood")
            early_stopping_callback = lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)

            lgblss_param_tuning = lightgbmlss.cv(hyper_params,
                                                 dtrain,
                                                 dist,
                                                 num_boost_round=num_boost_round,
                                                 nfold=nfold,
                                                 callbacks=[pruning_callback, early_stopping_callback]
                                                 )

            # Add opt_rounds as a trial attribute, accessible via study.trials_dataframe(). # https://github.com/optuna/optuna/issues/1169
            opt_rounds = np.argmin(np.array(lgblss_param_tuning["NegLogLikelihood-mean"])) + 1
            trial.set_user_attr("opt_round", int(opt_rounds))

            # Extract the best score
            best_score = np.min(np.array(lgblss_param_tuning["NegLogLikelihood-mean"]))

            return best_score


        if silence:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = TPESampler(seed=123)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize", study_name=study_name)
        study.optimize(objective, n_trials=n_trials, timeout=60 * max_minutes, show_progress_bar=True)

        print("Hyper-Parameter Optimization successfully finished.")
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        opt_param = study.best_trial

        # Add optimal stopping round
        opt_param.params["opt_rounds"] = study.trials_dataframe()["user_attrs_opt_round"][
            study.trials_dataframe()["value"].idxmin()]
        opt_param.params["opt_rounds"] = int(opt_param.params["opt_rounds"])

        print("  Value: {}".format(opt_param.value))
        print("  Params: ")
        for key, value in opt_param.params.items():
            print("    {}: {}".format(key, value))

        return opt_param.params


    def predict(booster: lgb.Booster, dtest: pd.DataFrame, dist: str, pred_type: str,
                n_samples: int = 1000, quantiles: list = [0.1, 0.5, 0.9], seed: str = 123):
        '''A customized lightgbmlss prediction function.

        booster: lgb.Booster
            Trained LightGBMLSS-Model
        X: pd.DataFrame
            Test Data
        dist: str
            Specifies the distributional assumption.
        pred_type: str
            Specifies what is to be predicted:
                "response" draws n_samples from the predicted response distribution.
                "quantile" calculates the quantiles from the predicted response distribution.
                "parameters" returns the predicted distributional parameters.
                "expectiles" returns the predicted expectiles.
        n_samples: int
            If pred_type="response" specifies how many samples are drawn from the predicted response distribution.
        quantiles: list
            If pred_type="quantiles" calculates the quantiles from the predicted response distribution.
        seed: int
            If pred_type="response" specifies the seed for drawing samples from the predicted response distribution.

        '''

        dict_param = dist.param_dict()

        predt = booster.predict(dtest, predict_raw_score=True)

        dist_params_predts = []

        for i, (dist_param, response_fun) in enumerate(dict_param.items()):
            dist_params_predts.append(response_fun(predt[:, i] + dist.start_values[i]))

        dist_params_df = pd.DataFrame(dist_params_predts).T
        dist_params_df.columns = dict_param.keys()

        if pred_type == "parameters":
            return dist_params_df

        elif pred_type == "expectiles":
            return dist_params_df

        elif pred_type == "response":
            pred_resp_df = dist.pred_dist_rvs(pred_params=dist_params_df,
                                              n_samples=n_samples,
                                              seed=seed)

            pred_resp_df.columns = [str("y_pred_sample_") + str(i) for i in range(pred_resp_df.shape[1])]
            return pred_resp_df

        elif pred_type == "quantiles":
            pred_quant_df = dist.pred_dist_quantile(quantiles=quantiles,
                                                    pred_params=dist_params_df)

            pred_quant_df.columns = [str("quant_") + str(quantiles[i]) for i in range(len(quantiles))]
            return pred_quant_df


    def plot(booster: lgb.Booster, X: pd.DataFrame, feature: str = "x", parameter: str = "location", plot_type: str = "Partial_Dependence"):
        '''A customized LightGBMLSS plotting function.

        booster: lgb.Booster
            Trained lightgbmlss-Model
        X: pd.DataFrame
            Train/Test Data
        feature: str
            Specifies which feature to use for plotting Partial_Dependence plot.
        parameter: str
            Specifies which distributional parameter to plot. Valid parameters are "location", "scale", "nu", "tau".
        plot_type: str
            Specifies which SHapley-plot to visualize. Currently "Partial_Dependence" and "Feature_Importance" are supported.

        '''

        shap.initjs()
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer(X)

        if parameter == "location":
            param_pos = 0
        if parameter == "scale":
            param_pos = 1
        if parameter == "nu":
            param_pos = 2
        if parameter == "tau":
            param_pos = 3

        if plot_type == "Partial_Dependence":
            shap.plots.scatter(shap_values[:, feature][:, param_pos], color=shap_values[:, :, param_pos])
        elif plot_type == "Feature_Importance":
            shap.plots.bar(shap_values[:, :, param_pos], max_display = 15 if X.shape[1] > 15 else X.shape[1])



    def expectile_plot(booster: lgb.Booster, X: pd.DataFrame, dist, feature: str = "x", expectile: str = "0.05", plot_type: str = "Partial_Dependence"):
        '''A customized LightGBMLSS plotting function.

        booster: lgb.Booster
            Trained lightgbmlss-Model
        X: pd.DataFrame
            Train/Test Data
        dist: lightgbmlss.distributions class
            Specifies distributional assumption
        feature: str
            Specifies which feature to use for plotting Partial_Dependence plot.
        expectile: str
            Specifies which expectile to plot.
        plot_type: str
            Specifies which SHapley-plot to visualize. Currently "Partial_Dependence" and "Feature_Importance" are supported.

        '''

        shap.initjs()
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer(X)

        expect_pos = dist.expectiles.index(float(expectile))

        if plot_type == "Partial_Dependence":
            shap.plots.scatter(shap_values[:, feature][:, expect_pos], color=shap_values[:, :, expect_pos])
        elif plot_type == "Feature_Importance":
            shap.plots.bar(shap_values[:, :, expect_pos], max_display = 15 if X.shape[1] > 15 else X.shape[1])
