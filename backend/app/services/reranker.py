"""Arm 3 — a hand-built logistic re-ranker (eval §2, decision D-5).

The **attribution control**: a learned linear model over the *identical* frozen
feature vocabulary the Stage-1 detector uses. If the LLM reviewer only ties this
on the same features, the accuracy claim is dead and the finding is governance,
not accuracy (eval §2). Built by hand — deterministic full-batch gradient descent,
L2 penalty, no sklearn/scipy — as ML warm-up and baseline discipline in one (D-5).

This module is **pure numeric math**: it trains on feature vectors + binary labels
handed to it and never touches the database or ground truth. The feature extraction
and the label join live in `evaluation.py` (the sole ground-truth reader). Fitting
is deterministic *by construction* — zero-initialised weights + full-batch descent —
so no seed is needed and two fits on identical inputs are byte-identical.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Baseline hyperparameters. Learning rate + iteration count are fixed and modest;
# on standardized inputs this converges well before `ITERATIONS` for this feature
# count. The L2 strength is *selected* per run on the tune split (evaluation.py),
# not fixed here.
DEFAULT_LR = 0.1
DEFAULT_ITERATIONS = 1000


def _sigmoid(z: float) -> float:
    """Numerically stable logistic function."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass(frozen=True)
class LogisticReRanker:
    """A fitted logistic model over a fixed-length feature vector. Inputs are
    standardized with the training means/stds captured at fit time, so callers pass
    raw feature values and the model applies the same transform it learned on."""

    weights: tuple[float, ...]
    bias: float
    means: tuple[float, ...]
    stds: tuple[float, ...]
    l2: float
    lr: float
    iterations: int

    @classmethod
    def fit(
        cls,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        *,
        l2: float,
        lr: float = DEFAULT_LR,
        iterations: int = DEFAULT_ITERATIONS,
    ) -> LogisticReRanker:
        """Fit by full-batch gradient descent on the penalized binary cross-entropy.
        The L2 term regularizes the weights only (never the bias). Deterministic:
        weights start at zero and every step sees the whole batch."""
        n = len(X)
        if n == 0:
            raise ValueError("cannot fit a re-ranker on an empty training set")
        if len(y) != n:
            raise ValueError(f"X/y length mismatch: {n} vs {len(y)}")
        d = len(X[0])

        means = [sum(row[j] for row in X) / n for j in range(d)]
        stds: list[float] = []
        for j in range(d):
            var = sum((row[j] - means[j]) ** 2 for row in X) / n
            s = math.sqrt(var)
            # A constant feature (std 0) centers to zero and contributes nothing;
            # a unit std keeps its standardized value at 0 rather than dividing by 0.
            stds.append(s if s > 1e-12 else 1.0)

        z_rows = [
            [(row[j] - means[j]) / stds[j] for j in range(d)] for row in X
        ]

        w = [0.0] * d
        b = 0.0
        for _ in range(iterations):
            grad_w = [0.0] * d
            grad_b = 0.0
            for i in range(n):
                zi = z_rows[i]
                s = b + sum(w[j] * zi[j] for j in range(d))
                err = _sigmoid(s) - y[i]
                grad_b += err
                for j in range(d):
                    grad_w[j] += err * zi[j]
            grad_b /= n
            for j in range(d):
                # mean data gradient + L2 penalty on the weight
                w[j] -= lr * (grad_w[j] / n + l2 * w[j])
            b -= lr * grad_b

        return cls(
            weights=tuple(w),
            bias=b,
            means=tuple(means),
            stds=tuple(stds),
            l2=l2,
            lr=lr,
            iterations=iterations,
        )

    def predict_proba(self, x: Sequence[float]) -> float:
        """P(abusive) for one raw feature vector, standardized with the fit-time
        statistics."""
        if len(x) != len(self.weights):
            raise ValueError(
                f"feature vector has length {len(x)}, expected {len(self.weights)}"
            )
        s = self.bias + sum(
            self.weights[j] * ((x[j] - self.means[j]) / self.stds[j])
            for j in range(len(x))
        )
        return _sigmoid(s)

    @property
    def weight_norm(self) -> float:
        """L2 norm of the learned weights (regularization-strength diagnostic)."""
        return math.sqrt(sum(wj * wj for wj in self.weights))
