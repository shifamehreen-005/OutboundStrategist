"""Pure unit tests for backend.scoring.score_fit.

No LLM call, no network, no company name.  score_fit takes a dict of
DimensionJudgment objects and a list of AntiPatternHit objects and returns
(score, tier).  These tests prove the result is deterministic arithmetic
over the supplied evidence judgments, nothing else.
"""
from __future__ import annotations

import pytest

from backend.schemas import AntiPatternHit, DimensionJudgment
from backend.scoring import (
    ANTI_PATTERN_PENALTY_CAP,
    ANTI_PATTERN_PENALTY_EACH,
    DIMENSION_WEIGHTS,
    TIER_MODERATE,
    TIER_STRONG,
    score_fit,
)

# ---------------------------------------------------------------------------
# Fixtures — reusable dimension sets built from data, not company names
# ---------------------------------------------------------------------------

def _all_present() -> dict:
    return {k: DimensionJudgment(judgment="present", evidence_ids=["E1"])
            for k in DIMENSION_WEIGHTS}


def _all_absent() -> dict:
    return {k: DimensionJudgment(judgment="absent") for k in DIMENSION_WEIGHTS}


def _mixed_a() -> dict:
    """industry + trigger present, buyer partial, size + geography absent."""
    return {
        "industry":  DimensionJudgment(judgment="present", evidence_ids=["E2"]),
        "trigger":   DimensionJudgment(judgment="present", evidence_ids=["E3"]),
        "buyer":     DimensionJudgment(judgment="partial", evidence_ids=["E4"]),
        "size":      DimensionJudgment(judgment="absent"),
        "geography": DimensionJudgment(judgment="absent"),
    }


def _mixed_b() -> dict:
    """industry absent, buyer + geography present, trigger + size partial."""
    return {
        "industry":  DimensionJudgment(judgment="absent"),
        "trigger":   DimensionJudgment(judgment="partial", evidence_ids=["E1"]),
        "buyer":     DimensionJudgment(judgment="present", evidence_ids=["E2"]),
        "size":      DimensionJudgment(judgment="partial", evidence_ids=["E3"]),
        "geography": DimensionJudgment(judgment="present", evidence_ids=["E4"]),
    }


# ---------------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------------

def test_all_absent_scores_zero():
    """Every dimension absent and no anti-patterns → score must be exactly 0."""
    score, tier = score_fit(_all_absent(), [])
    assert score == 0
    assert tier == "Weak"


def test_all_present_scores_hundred():
    """Every dimension present and no anti-patterns → score must be exactly 100."""
    score, tier = score_fit(_all_present(), [])
    assert score == 100
    assert tier == "Strong"


# ---------------------------------------------------------------------------
# Two structurally different inputs → different scores
# ---------------------------------------------------------------------------

def test_different_inputs_produce_different_scores():
    """_mixed_a and _mixed_b have different dimension patterns; scores must differ."""
    score_a, _ = score_fit(_mixed_a(), [])
    score_b, _ = score_fit(_mixed_b(), [])

    # Confirm they actually differ so the test is meaningful.
    assert score_a != score_b, (
        f"Expected different scores for mixed_a and mixed_b, got {score_a} == {score_b}"
    )


def test_mixed_a_expected_value():
    """mixed_a: industry(25) + trigger(25) + buyer(12.5) = 62.5 → 63, Moderate."""
    score, tier = score_fit(_mixed_a(), [])
    # 25 + 25 + 25*0.5 + 0 + 0 = 62.5 → rounds to 62 (banker's rounding)
    assert score == round(
        DIMENSION_WEIGHTS["industry"]  * 1.0
        + DIMENSION_WEIGHTS["trigger"] * 1.0
        + DIMENSION_WEIGHTS["buyer"]   * 0.5
    )
    assert tier == "Moderate"


def test_mixed_b_expected_value():
    """mixed_b: buyer(25) + geo(10) + trigger(12.5) + size(7.5) = 55, Moderate."""
    score, tier = score_fit(_mixed_b(), [])
    expected = round(
        DIMENSION_WEIGHTS["buyer"]     * 1.0
        + DIMENSION_WEIGHTS["geography"] * 1.0
        + DIMENSION_WEIGHTS["trigger"]   * 0.5
        + DIMENSION_WEIGHTS["size"]      * 0.5
    )
    assert score == expected
    assert tier == "Moderate"


# ---------------------------------------------------------------------------
# Anti-pattern penalty
# ---------------------------------------------------------------------------

def test_anti_pattern_subtracts_fixed_amount():
    """Each AP costs exactly ANTI_PATTERN_PENALTY_EACH points."""
    no_ap, _  = score_fit(_all_present(), [])
    one_ap, _ = score_fit(_all_present(), [AntiPatternHit(name="x")])
    assert no_ap - one_ap == ANTI_PATTERN_PENALTY_EACH


def test_anti_pattern_penalty_is_capped():
    """Many APs cannot push the penalty past ANTI_PATTERN_PENALTY_CAP."""
    aps_lots = [AntiPatternHit(name=f"ap{i}") for i in range(10)]
    aps_cap  = [AntiPatternHit(name=f"ap{i}") for i in range(
        ANTI_PATTERN_PENALTY_CAP // ANTI_PATTERN_PENALTY_EACH
    )]
    score_lots, _ = score_fit(_all_present(), aps_lots)
    score_cap, _  = score_fit(_all_present(), aps_cap)
    assert score_lots == score_cap == 100 - ANTI_PATTERN_PENALTY_CAP


def test_score_never_goes_below_zero():
    """Floor is 0 — heavy penalties on a zero-base score stay at 0."""
    aps = [AntiPatternHit(name=f"ap{i}") for i in range(20)]
    score, _ = score_fit(_all_absent(), aps)
    assert score == 0


# ---------------------------------------------------------------------------
# Company-name independence
# ---------------------------------------------------------------------------

def test_same_dimensions_same_score_regardless_of_name():
    """score_fit has no concept of company name — label in note is irrelevant."""
    dims_x = {
        "industry":  DimensionJudgment(judgment="present", evidence_ids=["E1"],
                                       note="note about company X"),
        "trigger":   DimensionJudgment(judgment="partial", evidence_ids=["E2"],
                                       note="note about company X"),
        "buyer":     DimensionJudgment(judgment="absent",  note="note about company X"),
        "size":      DimensionJudgment(judgment="present", evidence_ids=["E3"],
                                       note="note about company X"),
        "geography": DimensionJudgment(judgment="present", evidence_ids=["E4"],
                                       note="note about company X"),
    }
    dims_y = {
        "industry":  DimensionJudgment(judgment="present", evidence_ids=["E9"],
                                       note="note about company Y"),
        "trigger":   DimensionJudgment(judgment="partial", evidence_ids=["E8"],
                                       note="note about company Y"),
        "buyer":     DimensionJudgment(judgment="absent",  note="note about company Y"),
        "size":      DimensionJudgment(judgment="present", evidence_ids=["E7"],
                                       note="note about company Y"),
        "geography": DimensionJudgment(judgment="present", evidence_ids=["E6"],
                                       note="note about company Y"),
    }
    score_x, tier_x = score_fit(dims_x, [])
    score_y, tier_y = score_fit(dims_y, [])
    assert score_x == score_y, "Same judgment pattern must yield identical score"
    assert tier_x  == tier_y
