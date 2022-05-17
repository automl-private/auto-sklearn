# -*- encoding: utf-8 -*-
"""
===============
Custom Ensemble
===============

This example shows how to provide a custom ensemble to Auto-sklearn.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast
import warnings

import numpy as np
import sklearn.utils
import sklearn.utils.validation

from autosklearn.constants import TASK_TYPES, CLASSIFICATION_TASKS
from autosklearn.ensembles.abstract_ensemble import AbstractEnsemble
from autosklearn.metrics import Scorer, calculate_losses
from autosklearn.pipeline.base import BasePipeline


class WeightedMetricScorer(Scorer):
    def __init__(self, weights: np.ndarray | List[float], metrics: Sequence[Scorer]):
        self.name = "WeightedMetrics"
        self.weights = weights
        self.metrics = metrics
        self._sign = 1
        self._worst_possible_result = 0
        self._optimum = 0

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[List[float]] = None,
    ) -> float:
        scores = []
        for metric, weight in zip(self.metrics, self.weights):
            scores.append(metric(y_true, y_pred, sample_weight) * weight)
        return cast(float, np.sum(scores))


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:

    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)

            # And keep self
            is_efficient[i] = True
    return is_efficient


class MultiObjectiveEnsembleWrapper(AbstractEnsemble):
    def __init__(
        self,
        task_type: int,
        metrics: Sequence[Scorer] | Scorer,
        ensemble_class: Type[AbstractEnsemble],
        ensemble_kwargs: Dict,
        random_state: int | np.random.RandomState | None,
        n_weights: int = 100,
        fidelity: int = 10,
    ) -> None:
        self.task_type = task_type
        self.metrics = metrics if isinstance(metrics, Sequence) else [metrics]
        self.ensemble_class = ensemble_class
        self.ensemble_kwargs = ensemble_kwargs
        self.n_weights = n_weights
        self.fidelity = fidelity
        self.random_state = sklearn.utils.check_random_state(random_state)

    def __getstate__(self) -> Dict[str, Any]:
        # Cannot serialize a metric if
        # it is user defined.
        # That is, if doing pickle dump
        # the metric won't be the same as the
        # one in __main__. we don't use the metric
        # in the EnsembleSelection so this should
        # be fine
        self.metrics = None  # type: ignore
        return {key: value for key, value in self.__dict__.items() if key != "metrics"}

    def fit(
        self,
        base_models_predictions: List[np.ndarray] | np.ndarray,
        true_targets: np.ndarray,
        model_identifiers: List[Tuple[int, int, float]],
    ) -> AbstractEnsemble:
        ensembles = []
        losses = []

        # Generating weights according to Equation (1) from Knowles, 2005
        # https://www.cs.bham.ac.uk/~jdk/parego/emo2005parego.pdf
        all_weights = []
        # TODO increase this once we have a faster ensemble construction method
        fidelity = self.fidelity
        num_metrics = len(self.metrics)
        only_one_active = list(np.eye(num_metrics))

        for product in itertools.product(list(range(fidelity)), repeat=num_metrics):
            # This would set weight of all metrics to zero, not a good idea...
            if np.sum(product) == 0:
                continue
            # Drop all combinations in which only one metric has a positive value,
            # we'll add all of them later after taking a subset of all other weights
            elif num_metrics == np.sum(np.array(product) == 0) + 1:
                continue
            else:
                all_weights.append(product)

        # Make each weight vector sum up to 1
        all_weights = np.array(all_weights)
        all_weights = all_weights / np.sum(all_weights, axis=1).reshape((-1, 1))

        # Remove duplicate weight vectors
        all_weights = list(set(tuple(weights) for weights in all_weights))

        # Take a subsample of the weight vectors if we request less than generated
        if self.n_weights < len(all_weights):
            indices = self.random_state.choice(
                len(all_weights),
                replace=False,
                size=self.n_weights - len(only_one_active),
            )
            all_weights = [all_weights[i] for i in indices]

        # Add the weight vectors where only one metric is active to always sample that
        all_weights = only_one_active + all_weights

        # Sort such that we always start with the weight vector that assigns all
        # weights to the first metric
        all_weights = sorted(all_weights, key=lambda x: x[0], reverse=True)

        for weights in all_weights:
            metric = WeightedMetricScorer(metrics=self.metrics, weights=weights)

            ensemble = self.ensemble_class(
                task_type=self.task_type,
                metrics=metric,
                random_state=self.random_state,
                **self.ensemble_kwargs,
            )
            ensemble.fit(
                base_models_predictions=base_models_predictions,
                true_targets=true_targets,
                model_identifiers=model_identifiers,
            )
            prediction = ensemble.predict(
                base_models_predictions=base_models_predictions
            )
            losses.append(
                calculate_losses(
                    solution=true_targets,
                    prediction=prediction,
                    task_type=self.task_type,
                    metrics=self.metrics,
                )
            )
            ensembles.append(ensemble)

        # Prune to the Pareto front!
        losses_array = np.array(
            [[losses_[metric.name] for metric in self.metrics] for losses_ in losses]
        )
        pareto_front = is_pareto_efficient(losses_array)
        ensembles = [ensembles[i] for i, pareto in enumerate(pareto_front) if pareto]
        self.ensembles_ = ensembles
        return self

    def predict(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        return self.ensembles_[0].predict(predictions)

    def __str__(self) -> str:
        return ""

    @property
    def selected_ensemble(self) -> AbstractEnsemble:
        return self.ensembles_[0]

    def get_models_with_weights(
        self, models: Dict[Tuple[int, int, float], BasePipeline]
    ) -> List[Tuple[float, BasePipeline]]:
        return self.selected_ensemble.get_models_with_weights(models)

    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        return self.selected_ensemble.get_selected_model_identifiers()

    def get_validation_performance(self) -> float:
        return self.selected_ensemble.get_validation_performance()

    def get_identifiers_with_weights(
        self,
    ) -> List[Tuple[Tuple[int, int, float], float]]:
        return self.selected_ensemble.get_identifiers_with_weights()

    def get_pareto_front(self) -> Sequence[AbstractEnsemble]:
        return self.ensembles_


class ABLE(AbstractEnsemble):
    def __init__(
        self,
        task_type: int,
        metrics: Sequence[Scorer] | Scorer,
        random_state: int | np.random.RandomState | None,
        n_samples: int = 100,
        # TODO add a new regulator to limit the number of different models considered
    ) -> None:
        self.task_type = task_type
        if isinstance(metrics, Sequence):
            if len(metrics) > 1:
                warnings.warn(
                    "Ensemble selection can only optimize one metric, "
                    "but multiple metrics were passed, dropping all "
                    "except for the first metric."
                )
            self.metric = metrics[0]
        else:
            self.metric = metrics
        self.random_state = sklearn.utils.check_random_state(random_state)
        self.n_samples = n_samples

    def fit(
        self,
        base_models_predictions: List[np.ndarray],
        true_targets: np.ndarray,
        model_identifiers: List[Tuple[int, int, float]],
    ) -> AbstractEnsemble:
        self.n_samples = int(self.n_samples)
        if self.n_samples < 1:
            raise ValueError("Number of samples cannot be less than one!")
        if self.task_type not in TASK_TYPES:
            raise ValueError("Unknown task type %s." % self.task_type)
        if not isinstance(self.metric, Scorer):
            raise ValueError(
                "The provided metric must be an instance of Scorer, "
                "nevertheless it is {}({})".format(
                    self.metric,
                    type(self.metric),
                )
            )

        # First, set up the bootstrap samples
        n_datapoints = base_models_predictions[0].shape[0]
        base_indices = np.arange(n_datapoints, dtype=int)
        indices = []
        for i in range(self.n_samples):
            indices.append(
                sklearn.utils.resample(
                    base_indices,
                    replace=True,
                    n_samples=n_datapoints,
                    random_state=self.random_state,
                    stratify=true_targets
                    if self.task_type in CLASSIFICATION_TASKS
                    else None,
                )
            )

        scores = np.zeros((len(base_models_predictions), self.n_samples))
        for model_idx, predictions in enumerate(base_models_predictions):
            for i in range(self.n_samples):
                scores[model_idx, i] = self.metric(
                    y_true=true_targets[indices[i]],
                    y_pred=predictions[indices[i]],
                )

        num_wins = np.zeros(len(base_models_predictions))
        for i in range(self.n_samples):
            minimum = np.max(scores[:, i])
            all_minima = scores[:, i] == minimum
            minima_indices = np.where(all_minima > 0)[0]
            for mi in minima_indices:
                num_wins[mi] += 1 / len(minima_indices)

        self.weights_ = num_wins / np.sum(num_wins)
        self.identifiers_ = model_identifiers
        print(self.weights_)
        return self

    def __getstate__(self) -> Dict[str, Any]:
        # Cannot serialize a metric if
        # it is user defined.
        # That is, if doing pickle dump
        # the metric won't be the same as the
        # one in __main__. we don't use the metric
        # in the EnsembleSelection so this should
        # be fine
        return {key: value for key, value in self.__dict__.items() if key != "metrics"}

    def predict(
        self, base_models_predictions: np.ndarray | List[np.ndarray]
    ) -> np.ndarray:

        average = np.zeros_like(base_models_predictions[0], dtype=np.float64)
        tmp_predictions = np.empty_like(base_models_predictions[0], dtype=np.float64)

        # if predictions.shape[0] == len(self.weights_),
        # predictions include those of zero-weight models.
        if len(base_models_predictions) == len(self.weights_):
            for pred, weight in zip(base_models_predictions, self.weights_):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # if prediction model.shape[0] == len(non_null_weights),
        # predictions do not include those of zero-weight models.
        elif len(base_models_predictions) == np.count_nonzero(self.weights_):
            non_null_weights = [w for w in self.weights_ if w > 0]
            for pred, weight in zip(base_models_predictions, non_null_weights):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError(
                "The dimensions of ensemble predictions"
                " and ensemble weights do not match!"
            )
        del tmp_predictions
        return average

    def __str__(self) -> str:
        identifiers_str = " ".join(
            [
                f"{identifier}"
                for idx, identifier in enumerate(self.identifiers_)
                if self.weights_[idx] > 0
            ]
        )
        return (
            "Ensemble Selection:\n"
            f"\tOOB Score: {self.get_validation_performance()}\n"
            f"\tWeights: {self.weights_}\n"
            f"\tIdentifiers: {identifiers_str}\n"
        )

    def get_models_with_weights(
        self, models: Dict[Tuple[int, int, float], BasePipeline]
    ) -> List[Tuple[float, BasePipeline]]:
        output = []
        for i, weight in enumerate(self.weights_):
            if weight > 0.0:
                identifier = self.identifiers_[i]
                model = models[identifier]
                output.append((weight, model))

        output.sort(reverse=True, key=lambda t: t[0])

        return output

    def get_identifiers_with_weights(
        self,
    ) -> List[Tuple[Tuple[int, int, float], float]]:
        return list(zip(self.identifiers_, self.weights_))

    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            if weight > 0.0:
                output.append(identifier)

        return output

    def get_validation_performance(self) -> float:
        # TODO compute the OOB score!
        return 0


import sklearn.datasets
import sklearn.metrics
import autosklearn.classification

X, y = sklearn.datasets.fetch_openml(data_id=31, return_X_y=True, as_frame=True)
# Change the target to align with scikit-learn's convention that
# ``1`` is the minority class. In this example it is predicting
# that a credit is "bad", i.e. that it will default.
y = np.array([1 if val == "bad" else 0 for val in y])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    per_run_time_limit=10,
    tmp_folder="/tmp/autosklearn_classification_example_tmp",
    # ensemble_class=MultiObjectiveEnsembleWrapper,
    # ensemble_kwargs={
    #    'ensemble_class': ABLE,
    #    'fidelity': 12,
    #    'ensemble_kwargs': {
    #        'n_samples': 5,
    #    },
    # },
    # metric=[autosklearn.metrics.precision, autosklearn.metrics.recall],
    ensemble_class=ABLE,
    ensemble_kwargs={"n_samples": 200},
    metric=autosklearn.metrics.roc_auc,
)
automl.fit(X_train, y_train, dataset_name="breast_cancer")
print(automl.leaderboard())

predictions = automl.predict_proba(X_test)[:, 1]
print("Accuracy score:", sklearn.metrics.roc_auc_score(y_test, predictions))

# predictions = automl.predict(X_test)
# print("Precision", sklearn.metrics.precision_score(y_test, predictions))
# print("Recall", sklearn.metrics.recall_score(y_test, predictions))
