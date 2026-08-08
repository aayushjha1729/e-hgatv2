"""Quality indicators: hypervolume, GD+/IGD+, and spread.

All indicators are minimisation-oriented, depend only on objective vectors (no Torch),
and operate on the exact golden Pareto front PF*, which makes the effectiveness results
absolute rather than relative. They are cross-validated against pymoo in the test
suite.
"""

from __future__ import annotations

from ehgat.metrics.gdplus import gd_plus, igd_plus
from ehgat.metrics.hypervolume import hypervolume, nadir_reference
from ehgat.metrics.spread import spacing, spread

__all__ = [
    "gd_plus",
    "hypervolume",
    "igd_plus",
    "nadir_reference",
    "spacing",
    "spread",
]
