"""Deterministic fit scoring — no LLM call, no side effects.

All tunable parameters live here so they can be changed without touching
prompt logic or agent orchestration.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from .schemas import AntiPatternHit, DimensionJudgment

# ---------------------------------------------------------------------------
# Tunable starting weights (sum = 100)
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS: Dict[str, int] = {
    "industry":  25,
    "trigger":   25,
    "buyer":     25,
    "size":      15,
    "geography": 10,
}

# Fraction of weight earned per judgment level.
JUDGMENT_VALUES: Dict[str, float] = {
    "present": 1.0,
    "partial": 0.5,
    "absent":  0.0,
}

# Anti-pattern penalty: 20 points per hit, capped at 40.
ANTI_PATTERN_PENALTY_EACH: int = 20
ANTI_PATTERN_PENALTY_CAP:  int = 40

# Tier thresholds.
TIER_STRONG:   int = 70
TIER_MODERATE: int = 40


# ---------------------------------------------------------------------------

def score_fit(
    dimensions: "Dict[str, DimensionJudgment]",
    anti_patterns_hit: "List[AntiPatternHit]",
) -> Tuple[int, str]:
    """Compute (score 0-100, tier) from per-dimension judgments.

    Pure function: deterministic, no I/O, no randomness. Every score
    change is fully explained by which dimensions changed judgment.

    Formula:
        base    = sum(DIMENSION_WEIGHTS[d] * JUDGMENT_VALUES[judgment])
        penalty = min(ANTI_PATTERN_PENALTY_CAP,
                      ANTI_PATTERN_PENALTY_EACH * len(anti_patterns_hit))
        score   = max(0, min(100, round(base - penalty)))
        tier    = Strong if score >= 70 else Moderate if score >= 40 else Weak
    """
    base = sum(
        DIMENSION_WEIGHTS.get(name, 0) * JUDGMENT_VALUES.get(dim.judgment, 0.0)
        for name, dim in dimensions.items()
    )
    penalty = min(
        ANTI_PATTERN_PENALTY_CAP,
        ANTI_PATTERN_PENALTY_EACH * len(anti_patterns_hit),
    )
    score = max(0, min(100, round(base - penalty)))
    tier  = (
        "Strong"   if score >= TIER_STRONG   else
        "Moderate" if score >= TIER_MODERATE else
        "Weak"
    )
    return score, tier
