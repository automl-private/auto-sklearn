# -*- encoding: utf-8 -*-
"""
===============
Custom Ensemble
===============

This example shows how to provide a custom ensemble to Auto-sklearn.
"""
from __future__ import annotations

import itertools
import os
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast
import warnings

import numpy as np
import sklearn.utils
import sklearn.utils.validation

from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.constants import TASK_TYPES, CLASSIFICATION_TASKS
from autosklearn.ensembles.abstract_ensemble import AbstractEnsemble
from autosklearn.ensembles.abstract_ensemble import AbstractMultiObjectiveEnsemble
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
        # TODO: can we somehow add a normalization of the individual scores here?
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


def get_weights(
    fidelity: int,
    num_metrics: int,
    n_weights: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    # Generating weights according to Equation (1) from Knowles, 2005
    # https://www.cs.bham.ac.uk/~jdk/parego/emo2005parego.pdf
    all_weights = []
    # TODO increase this once we have a faster ensemble construction method
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
    if n_weights < len(all_weights):
        indices = random_state.choice(
            len(all_weights),
            replace=False,
            size=n_weights - len(only_one_active),
        )
        all_weights = [all_weights[i] for i in indices]
    # Add the weight vectors where only one metric is active to always sample that
    all_weights = only_one_active + all_weights
    # Sort such that we always start with the weight vector that assigns all
    # weights to the first metric
    all_weights = np.array(sorted(all_weights, key=lambda x: x[0], reverse=True))
    return all_weights


class MultiObjectiveEnsembleWrapper(AbstractMultiObjectiveEnsemble):
    def __init__(
        self,
        task_type: int,
        metrics: Sequence[Scorer] | Scorer,
        random_state: int | np.random.RandomState | None,
        backend: Backend,
        ensemble_class: Type[AbstractEnsemble],
        ensemble_kwargs: Dict,
        n_weights: int = 100,
        fidelity: int = 10,
    ) -> None:
        self.task_type = task_type
        self.metrics = metrics if isinstance(metrics, Sequence) else [metrics]
        self.random_state = sklearn.utils.check_random_state(random_state)
        self.backend = backend
        self.ensemble_class = ensemble_class
        self.ensemble_kwargs = ensemble_kwargs
        self.n_weights = n_weights
        self.fidelity = fidelity

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

        all_weights = get_weights(
            fidelity=self.fidelity,
            num_metrics=len(self.metrics),
            n_weights=self.n_weights,
            random_state=self.random_state,
        )

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


def get_bootstrap_indices(
    n_samples: int,
    n_datapoints: int,
    true_targets: np.ndarray,
    random_state: np.random.RandomState,
    task_type: int,
):
    base_indices = np.arange(n_datapoints, dtype=int)
    indices = []
    oob_indices = []
    for i in range(n_samples):
        indices_ = sklearn.utils.resample(
            base_indices,
            replace=True,
            n_samples=n_datapoints,
            random_state=random_state,
            stratify=true_targets if task_type in CLASSIFICATION_TASKS else None,
        )
        indices.append(indices_)
        indices_as_set = set(indices_)
        oob_indices.append(
            np.array([i for i in range(n_datapoints) if i not in indices_as_set])
        )
    return indices, oob_indices


def compute_weights_from_scores(
    n_samples: int,
    num_base_models: int,
    scores: np.ndarray,
) -> np.ndarray:
    num_wins = np.zeros(num_base_models)
    for i in range(n_samples):
        minimum = np.max(scores[:, i])
        all_minima = scores[:, i] == minimum
        minima_indices = np.where(all_minima > 0)[0]
        for mi in minima_indices:
            num_wins[mi] += 1 / len(minima_indices)
    model_weights = num_wins / np.sum(num_wins)
    return model_weights


class ABLE(AbstractEnsemble):
    def __init__(
        self,
        task_type: int,
        metrics: Sequence[Scorer] | Scorer,
        random_state: int | np.random.RandomState | None,
        backend: Backend,
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
        self.backend = backend
        self.random_state = sklearn.utils.check_random_state(random_state)
        self.n_samples = n_samples

        self._cache_file = os.path.join(
            self.backend.temporary_directory, "able_cache.pkl"
        )
        try:
            with open(self._cache_file, "rb") as fh:
                self._cache = pickle.load(fh)
        except:
            self._cache = dict()

    def fit(
        self,
        base_models_predictions: List[np.ndarray],
        true_targets: np.ndarray,
        model_identifiers: List[Tuple[int, int, float]],
    ) -> AbstractEnsemble:
        import time

        st = time.time()

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
        if "indices" not in self._cache:
            indices, _ = get_bootstrap_indices(
                n_samples=self.n_samples,
                n_datapoints=base_models_predictions[0].shape[0],
                true_targets=true_targets,
                task_type=self.task_type,
                random_state=self.random_state,
            )
            self._cache["indices"] = indices
        else:
            indices = self._cache["indices"]

        num_base_models = len(base_models_predictions)
        scores = np.zeros((num_base_models, self.n_samples))
        for model_idx, (predictions, identifier) in enumerate(
            zip(base_models_predictions, model_identifiers)
        ):
            if identifier in self._cache:
                model_scores = self._cache[identifier]
            else:
                model_scores = np.zeros(self.n_samples)
                for i in range(self.n_samples):
                    model_scores[i] = self.metric(
                        y_true=true_targets[indices[i]],
                        y_pred=predictions[indices[i]],
                    )
                    self._cache[identifier] = model_scores
            scores[model_idx, :] = model_scores

        self.weights_ = compute_weights_from_scores(
            n_samples=self.n_samples,
            num_base_models=num_base_models,
            scores=scores,
        )

        self.identifiers_ = model_identifiers
        print(self.weights_)

        with open(self._cache_file, "wb") as fh:
            pickle.dump(self._cache, fh)

        et = time.time()
        print(et - st)

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


class MultiObjectiveABLE(AbstractMultiObjectiveEnsemble):
    def __init__(
        self,
        task_type: int,
        metrics: Sequence[Scorer] | Scorer,
        random_state: int | np.random.RandomState | None,
        backend: Backend,
        n_samples: int = 100,
        # TODO add a new regulator to limit the number of different models considered
        n_weights: int = 100,
        fidelity: int = 15,
    ) -> None:
        self.task_type = task_type
        self.metrics = metrics
        self.random_state = sklearn.utils.check_random_state(random_state)
        self.backend = backend
        self.n_samples = n_samples
        self.n_weights = n_weights
        self.fidelity = fidelity

        self._cache_file = os.path.join(
            self.backend.temporary_directory, "able_cache.pkl"
        )
        try:
            with open(self._cache_file, "rb") as fh:
                self._cache = pickle.load(fh)
        except:
            self._cache = dict()

    def fit(
        self,
        base_models_predictions: List[np.ndarray],
        true_targets: np.ndarray,
        model_identifiers: List[Tuple[int, int, float]],
    ) -> AbstractEnsemble:
        import time

        st = time.time()

        self.n_samples = int(self.n_samples)
        if self.n_samples < 1:
            raise ValueError("Number of samples cannot be less than one!")
        if self.task_type not in TASK_TYPES:
            raise ValueError("Unknown task type %s." % self.task_type)

        num_basemodels = len(base_models_predictions)
        n_datapoints = base_models_predictions[0].shape[0]

        # First, set up the bootstrap samples
        if "indices" not in self._cache:
            indices, oob_indices = get_bootstrap_indices(
                n_samples=self.n_samples,
                n_datapoints=n_datapoints,
                true_targets=true_targets,
                task_type=self.task_type,
                random_state=self.random_state,
            )
            self._cache["indices"] = indices
            self._cache["oob_indices"] = oob_indices
        else:
            indices = self._cache["indices"]
            oob_indices = self._cache["oob_indices"]

        # Second, compute the scores per metric or load it from cache
        all_scores = {}
        for metric in self.metrics:
            scores = np.zeros((num_basemodels, self.n_samples))
            for model_idx, (predictions, identifier) in enumerate(
                zip(base_models_predictions, model_identifiers)
            ):
                key = (identifier, metric.name)
                if key in self._cache:
                    model_scores = self._cache[key]
                else:
                    model_scores = np.zeros(self.n_samples)
                    for i in range(self.n_samples):
                        model_scores[i] = metric(
                            y_true=true_targets[indices[i]],
                            y_pred=predictions[indices[i]],
                        )
                        self._cache[key] = model_scores
                scores[model_idx, :] = model_scores
            all_scores[metric.name] = scores
            # TODO: the paper by Knowles, 2005, demands that the scores are scaled to [0,
            #  1]. For Auto-sklearn we could use the observed best as the maximum and the dummy
            #  score as the minimum. However, this begs the question on how the dummy score
            #  should be passed to the Ensemble builder?

        # Third, iterate all weights and compute the weights and oob predictions (+ oob scores)
        all_weights = []
        all_ensemble_scores = []

        for metric_weights in get_weights(
            fidelity=self.fidelity,
            num_metrics=len(self.metrics),
            n_weights=self.n_weights,
            random_state=self.random_state,
        ):

            scores = np.zeros((num_basemodels, self.n_samples))
            for weight, metric in zip(metric_weights, self.metrics):
                scores += all_scores[metric.name] * weight

            model_weights = compute_weights_from_scores(
                n_samples=self.n_samples,
                num_base_models=num_basemodels,
                scores=scores,
            )

            # Under the assumption that ensembles do not change too much between the iterations,
            # this could be cached, too.
            # One would have to create a new key: ((identifier_0, weight_0), ..., (identifier_n,
            # (weight_n)) to address the out of bag score of the current ensemble that has been
            # computed in a previous iteration.

            # # compute the OOB predictions for this ensemble - slow version
            # ensemble_predictions_slow = np.zeros_like(base_models_predictions[0])
            # oob_count = np.zeros((base_models_predictions[0].shape[0], 1))
            # for oob_indices_ in oob_indices:
            #     for base_model_idx in range(len(base_models_predictions)):
            #         ensemble_predictions_slow[oob_indices_] += (
            #             base_models_predictions[base_model_idx][oob_indices_]
            #             * model_weights[base_model_idx]
            #         )
            #         oob_count[oob_indices_] += 1
            # oob_count /= len(base_models_predictions)
            # ensemble_predictions_slow /= oob_count

            # compute the OOB predictions for this ensemble - fast version
            ensemble_predictions = np.zeros_like(base_models_predictions[0])
            oob_count = np.zeros((n_datapoints, 1))
            oob_weights = np.zeros((n_datapoints, 1))
            for base_model_idx, base_model_weight in enumerate(model_weights):
                if base_model_weight == 0:
                    continue
                oob_count[:] = 0
                oob_weights[:] = 0
                for oob_indices_ in oob_indices:
                    oob_weights[oob_indices_] += base_model_weight
                    oob_count[oob_indices_] += 1
                oob_weights /= oob_count
                ensemble_predictions += (
                    base_models_predictions[base_model_idx] * oob_weights
                )

            # # Test that the hard-to-read fast version is reasonably close to the slow version in
            # # terms of the computed scores
            # np.testing.assert_array_almost_equal(
            #     ensemble_predictions_slow,
            #     ensemble_predictions,
            #     decimal=3,
            # )

            scores = []
            for weight, metric in zip(metric_weights, self.metrics):
                scores.append(metric(y_true=true_targets, y_pred=ensemble_predictions))

            all_weights.append(model_weights)
            all_ensemble_scores.append(scores)

        # Fourth: compute the pareto front based on the OOB scores
        pareto_front = is_pareto_efficient(np.array(all_ensemble_scores))
        ensembles = [all_weights[i] for i, pareto in enumerate(pareto_front) if pareto]
        self.ensembles_ = ensembles

        self.weights_ = self.ensembles_[0]
        self.validation_score_ = all_ensemble_scores[0]

        self.identifiers_ = model_identifiers
        print(np.sum(pareto_front))
        print(self.weights_)

        with open(self._cache_file, "wb") as fh:
            pickle.dump(self._cache, fh)

        et = time.time()
        print(et - st)

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
        return self.validation_score_[0]

    def get_pareto_front(self) -> Sequence[AbstractEnsemble]:
        return_value = []
        for i, ensemble in enumerate(self.ensembles_):
            able = ABLE(
                task_type=self.task_type,
                metrics=self.metrics,
                random_state=self.random_state,
                backend=self.backend,
                n_samples=self.n_samples,
            )
            able.weights_ = ensemble
            able.identifiers_ = self.identifiers_
            return_value.append(able)
        return return_value


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
    # ensemble_class=MultiObjectiveABLE,
    # ensemble_kwargs={"n_samples": 100, "n_weights": 1000},
    metric=autosklearn.metrics.roc_auc,
    # ensemble_class=ABLE,
    # ensemble_kwargs={"n_samples": 200},
)
automl.fit(X_train, y_train, dataset_name="German credit")
print(automl.leaderboard())

predictions = automl.predict_proba(X_test)[:, 1]
print("Accuracy score:", sklearn.metrics.roc_auc_score(y_test, predictions))

# predictions = automl.predict(X_test)
# print("Precision", sklearn.metrics.precision_score(y_test, predictions))
# print("Recall", sklearn.metrics.recall_score(y_test, predictions))
