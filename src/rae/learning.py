"""Machine-learning selectors trained on the logs of past cohorts.

Two labelling regimes are supported (RQ1):

* ``decisions``: the label is the past selection decision (imitation of historical selectors);
* ``later_performance``: the label is the performance observed at the next stage. It is only
  available for athletes who are still in the system, which is a selective-labels problem.

Features are the observed performance, the number of earlier selections (training exposure)
and the stage. Birth information (position in the selection year) can be added or withheld,
to test "fairness through unawareness".
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rae.athletes import Athletes
from rae.pipeline import SimulationResult

Regime = Literal["decisions", "later_performance"]
ModelKind = Literal["linear", "gbm"]


def _features(
    athletes: Athletes,
    pool: np.ndarray,
    observed: np.ndarray,
    times_selected: np.ndarray,
    stage: int,
    use_birth_info: bool,
) -> np.ndarray:
    columns = [observed, times_selected.astype(float), np.full(pool.size, float(stage))]
    if use_birth_info:
        columns.append(athletes.birth_frac[pool])
    return np.column_stack(columns)


def build_training_data(
    results: Sequence[SimulationResult], regime: Regime, use_birth_info: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Turn the stage logs of past cohorts into a supervised dataset ``(X, y)``."""
    if regime not in ("decisions", "later_performance"):
        raise ValueError(f"unknown regime: {regime!r}")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for res in results:
        if not res.stages:
            raise ValueError("results carry no stage logs")
        for i, record in enumerate(res.stages):
            x = _features(
                res.athletes,
                record.pool,
                record.observed,
                record.times_selected_before,
                record.stage,
                use_birth_info,
            )
            if regime == "decisions":
                xs.append(x)
                ys.append(record.selected.astype(int))
                continue
            if i + 1 >= len(res.stages):
                continue  # the last stage has no later performance
            following = res.stages[i + 1]
            _, now, later = np.intersect1d(record.pool, following.pool, return_indices=True)
            xs.append(x[now])
            ys.append(following.observed[later])
    return np.vstack(xs), np.concatenate(ys)


def _make_model(kind: ModelKind, regime: Regime, seed: int):
    if kind == "linear":
        if regime == "decisions":
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        return make_pipeline(StandardScaler(), LinearRegression())
    if kind == "gbm":
        if regime == "decisions":
            return HistGradientBoostingClassifier(random_state=seed)
        return HistGradientBoostingRegressor(random_state=seed)
    raise ValueError(f"unknown model kind: {kind!r}")


class MLSelector:
    """Ranks athletes with a model trained on past cohorts."""

    def __init__(self, model, regime: Regime, use_birth_info: bool) -> None:
        self.model = model
        self.regime = regime
        self.use_birth_info = use_birth_info

    def scores(
        self,
        *,
        stage: int,
        pool: np.ndarray,
        athletes: Athletes,
        observed: np.ndarray,
        times_selected: np.ndarray,
    ) -> np.ndarray:
        x = _features(athletes, pool, observed, times_selected, stage, self.use_birth_info)
        if self.regime == "decisions":
            return self.model.predict_proba(x)[:, 1]
        return self.model.predict(x)


def train_selector(
    results: Sequence[SimulationResult],
    *,
    regime: Regime,
    use_birth_info: bool,
    model: ModelKind = "linear",
    seed: int = 0,
) -> MLSelector:
    """Fit a selector on the logs of past cohorts."""
    x, y = build_training_data(results, regime, use_birth_info)
    fitted = _make_model(model, regime, seed).fit(x, y)
    return MLSelector(fitted, regime, use_birth_info)
