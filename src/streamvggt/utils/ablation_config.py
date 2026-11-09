"""Process-wide switches for the Compactor comparison and the IDC ablations (rebuttal experiment 3).

Set once by the launcher (``configure``) and read by the eviction code, instead of threading every flag through
the model, aggregator, block and attention signatures. Defaults reproduce the existing behaviour exactly.

  attention_source         "causal_history": attention utility accumulated from later frames (IDC, existing).
                           "noncausal_current": at each eviction, the current frame's queries attend to all live
                           keys (anchor + candidates, before eviction); the column sum over queries, averaged over
                           heads, is the importance of every candidate, current-frame tokens included. No history.
  score_norm               "mean": (1-b)*lev/mean(lev) + b*util/mean(util) (IDC, existing).
                           "zscore": z(util) + lam*z(lev), population std (Compactor).
  compactor_lambda         lam above, and in the compactor policy (official code: 0.5).
  unobserved_lev_multiplier
                           None: tokens without attention yet keep the legacy score lev/mean(lev) (with b = 0.5
                           this is the paper's "2x leverage" rule). A number m: their score is m * w_lev * n(lev),
                           where w_lev and n() are the leverage weight and normalisation of score_norm (m = 2: the
                           paper's rule, m = 1: no doubling).
  compactor_ridge          ridge added to each head's key Gram matrix in the compactor policy (official: 5e-3).
  compactor_headwise_norm  compactor policy: unit-normalise each head's keys before leverage (IDC difference 2).
  compactor_sketch_dim     compactor policy: 0 = exact per-head leverage on the 64-d keys; k > 0 = right-sketch each
                           head's keys to k dims first (Gaussian N(0, 1/k), seed 42, shared by heads and layers, as
                           in the official code) and compute the ridge leverage on the k x k Gram matrix.
"""

from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class AblationConfig:
    attention_source: str = "causal_history"
    score_norm: str = "mean"
    compactor_lambda: float = 0.5
    unobserved_lev_multiplier: Optional[float] = None
    compactor_ridge: float = 5e-3
    compactor_headwise_norm: bool = False
    compactor_sketch_dim: int = 0


CONFIG = AblationConfig()

_CHOICES = {
    "attention_source": ("causal_history", "noncausal_current"),
    "score_norm": ("mean", "zscore"),
}


def configure(**kwargs) -> AblationConfig:
    names = {f.name for f in fields(AblationConfig)}
    for key, value in kwargs.items():
        if key not in names:
            raise ValueError(f"Unknown ablation option {key!r}")
        if key in _CHOICES and value not in _CHOICES[key]:
            raise ValueError(f"{key} must be one of {_CHOICES[key]}, got {value!r}")
        setattr(CONFIG, key, value)
    return CONFIG
