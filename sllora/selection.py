"""Layer-selection rules and the memory-budgeted selector.

Rules (k = number of layers that get LoRA adapters):
  all            every layer (standard LoRA baseline)
  random         k random layers                       (Aggarwal et al. LF_H1)
  top            last k layers                         (cheapest: backprop depth = k)
  bottom         first k layers                        (control: most expensive placement)
  drift_protect  k LEAST-changed layers by base->instruct drift; the most-changed,
                 task-critical layers stay untouched   (LF_H2 logic, applied to adapters)
  importance     k layers with highest gradient norm on 256 Phase-2 samples (from GMT)
  budgeted       greedy: most important layers that fit a GB budget (SubTuning-style)

Memory cost of a layer set S is driven by its LOWEST layer, because autograd
must keep activations for every layer from min(S) up to the output:
    cost(S) = base + a * |S| + b * (L - min(S))
a, b, base are fitted from measured probes (CostModel.fit).
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

RULES = ["all", "random", "top", "bottom", "drift_protect", "importance", "budgeted"]


@dataclass
class CostModel:
    n_layers: int
    base_gb: float
    per_adapted_gb: float
    per_depth_gb: float

    def cost(self, layers: Sequence[int]) -> float:
        if not layers:
            return self.base_gb
        depth = self.n_layers - min(layers)
        return self.base_gb + self.per_adapted_gb * len(layers) + self.per_depth_gb * depth

    @classmethod
    def fit(cls, n_layers: int, measurements: List[Dict]) -> "CostModel":
        """measurements: [{"layers": [...], "peak_gb": float}, ...] (>= 3 distinct points)."""
        X, y = [], []
        for m in measurements:
            S = m["layers"]
            X.append([1.0, len(S), n_layers - min(S)])
            y.append(m["peak_gb"])
        X, y = np.array(X), np.array(y)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        base, a, b = (float(c) for c in coef)
        return cls(n_layers, base, max(a, 0.0), max(b, 0.0))

    def r2(self, measurements: List[Dict]) -> float:
        y = np.array([m["peak_gb"] for m in measurements])
        p = np.array([self.cost(m["layers"]) for m in measurements])
        ss = ((y - y.mean()) ** 2).sum()
        return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else 1.0

    def to_dict(self):
        return asdict(self)


def budgeted_select(importance: Sequence[float], cost: CostModel, budget_gb: float,
                    k_max: Optional[int] = None) -> List[int]:
    """Exact for this cost model: try every possible lowest layer l, take l plus
    the most important layers above it while the budget allows, keep the best."""
    L = len(importance)
    k_max = k_max or L
    best_val, best = -1.0, None
    for low in range(L):
        S = [low]
        if cost.cost(S) > budget_gb:
            continue
        for i in sorted(range(low + 1, L), key=lambda j: -importance[j]):
            if len(S) >= k_max:
                break
            if cost.cost(S + [i]) <= budget_gb:
                S.append(i)
        val = float(sum(importance[i] for i in S))
        if val > best_val:
            best_val, best = val, sorted(S)
    if best is None:
        raise ValueError(f"No layer set fits {budget_gb:.2f} GB (min possible {cost.cost([L - 1]):.2f} GB)")
    return best


def select_layers(rule: str, n_layers: int, k: Optional[int] = None, scores: Optional[Dict] = None,
                  budget_gb: Optional[float] = None, cost: Optional[CostModel] = None,
                  seed: int = 0) -> Optional[List[int]]:
    """Returns a sorted list of layer indices, or None for all layers."""
    L = n_layers
    if rule == "all":
        return None
    if rule == "budgeted":
        assert scores and "importance" in scores and cost and budget_gb, "budgeted needs importance, cost model, budget"
        return budgeted_select(scores["importance"], cost, budget_gb, k_max=k)
    assert k and 0 < k <= L, f"rule {rule} needs 0 < k <= {L}"
    if rule == "random":
        return sorted(random.Random(seed).sample(range(L), k))
    if rule == "top":
        return list(range(L - k, L))
    if rule == "bottom":
        return list(range(k))
    if rule == "drift_protect":
        d = scores["drift"]
        return sorted(sorted(range(L), key=lambda i: d[i])[:k])
    if rule == "importance":
        g = scores["importance"]
        return sorted(sorted(range(L), key=lambda i: -g[i])[:k])
    raise ValueError(f"unknown rule {rule}; choose from {RULES}")


def layer_strip(layers: Optional[Sequence[int]], n_layers: int) -> str:
    """'....####' style picture of which layers are adapted (layer 0 on the left)."""
    s = set(range(n_layers)) if layers is None else set(layers)
    return "".join("#" if i in s else "." for i in range(n_layers))
