from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast

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
        return self.__dict__

    def fit(
        self,
        base_models_predictions: List[np.ndarray] | np.ndarray,
        true_targets: np.ndarray,
        model_identifiers: List[Tuple[int, int, float]],
    ) -> AbstractEnsemble:
        ensembles = []
        losses = []
        all_weights = self.random_state.rand(self.n_weights - 2, len(self.metrics))
        all_weights = all_weights / np.sum(all_weights, axis=1).reshape((-1, 1))
        all_weights = [[1.0, 0.0]] + list(all_weights) + [[0.0, 1.0]]
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

    def get_models_with_weights(
        self, models: Dict[Tuple[int, int, float], BasePipeline]
    ) -> List[Tuple[float, BasePipeline]]:
        return self.ensembles_[0].get_models_with_weights(models)

    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        return self.ensembles_[0].get_selected_model_identifiers()

    def get_validation_performance(self) -> float:
        return self.ensembles_[0].get_validation_performance()

    def get_identifiers_with_weights(
        self,
    ) -> List[Tuple[Tuple[int, int, float], float]]:
        return self.ensembles_[0].get_identifiers_with_weights()
