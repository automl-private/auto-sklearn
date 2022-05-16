from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast

import itertools

import numpy as np
from sklearn.utils import check_random_state

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
    ) -> None:
        self.task_type = task_type
        self.metrics = metrics if isinstance(metrics, Sequence) else [metrics]
        self.ensemble_class = ensemble_class
        self.ensemble_kwargs = ensemble_kwargs
        self.n_weights = n_weights
        self.random_state = check_random_state(random_state)

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
        fidelity = 5
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
                task_type=self.task_type, metrics=metric, **self.ensemble_kwargs
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
