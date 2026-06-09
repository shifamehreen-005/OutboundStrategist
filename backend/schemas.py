"""Data models shared across the pipeline.

Every piece of evidence carries its source URL so that grounding /
the claim map is a structural property of the data, not something
reconstructed after the fact.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Retrieval primitives
# ---------------------------------------------------------------------------
class Chunk(BaseModel):
    """A small passage of text retrieved from a single source URL."""
    id: str
    url: str
    title: str = ""
    text: str
    position: int = 0  # order within the source page


class Evidence(BaseModel):
    """A chunk that has been selected as relevant for a sub-task.

    `snippet` is a short, quotable span (<= ~240 chars) that the model
    actually used, so the claim map can show a tight citation.
    """
    id: str
    url: str
    snippet: str
    why: str = ""  # one line: why this is relevant


# ---------------------------------------------------------------------------
# Mode 1 outputs: ICP + value proposition
# ---------------------------------------------------------------------------
class SizeBand(BaseModel):
    label: str                 # e.g. "11-50", "Mid-market (200-1000)"
    rationale: str = ""


class Trigger(BaseModel):
    name: str                  # e.g. "Recent funding round"
    description: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    grounded: bool = True      # False if cited ids don't resolve to retrieved chunks


class Buyer(BaseModel):
    role: str                  # e.g. "VP of Sales"
    seniority: str = ""        # e.g. "VP / Director"
    why: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    grounded: bool = True      # False if cited ids don't resolve to retrieved chunks


class NamedCustomer(BaseModel):
    """A real customer extracted from the sender's site (logo wall / case study).

    The ICP is grounded in the shared pattern of these — who *actually* buys —
    not just the sender's aspirational marketing copy (BUILD_SPEC §0.6).
    """
    name: str
    evidence_ids: List[str] = Field(default_factory=list)
    source_url: str = ""       # URL the customer was found at (citation)
    provenance: str = ""       # "case-study" | "site-text" | "web"
    description: str = ""      # 1-sentence: what the customer company does
    industry: str = ""         # e.g. "Fintech", "B2B SaaS"
    size_hint: str = ""        # e.g. "Series B", "enterprise"


class ICP(BaseModel):
    industries: List[str] = Field(default_factory=list)
    size_bands: List[SizeBand] = Field(default_factory=list)
    triggers: List[Trigger] = Field(default_factory=list)
    buyers: List[Buyer] = Field(default_factory=list)
    geographies: List[str] = Field(default_factory=list)
    anti_patterns: List[str] = Field(default_factory=list)  # who is NOT a fit
    # 3-5 sharp questions derived from the real-customer pattern; Mode 1 writes
    # them, Mode 2 answers them ("is this target like the companies that buy?").
    qualifying_questions: List[str] = Field(default_factory=list)


class SenderProfile(BaseModel):
    company: str
    value_proposition: str
    one_liner: str = ""
    capabilities: List[str] = Field(default_factory=list)
    icp: ICP
    named_customers: List[NamedCustomer] = Field(default_factory=list)
    customer_pattern: str = ""   # "6/9 customers are B2B SaaS startups at Series A-C…"
    evidence: List[Evidence] = Field(default_factory=list)
    evidence_sufficient: bool = True          # False when corpus was too thin
    data_gaps: List[str] = Field(default_factory=list)  # ICP dimensions lacking evidence
    confidence: str = "high"                  # "high" | "low"
    grounding_coverage: float = 1.0           # grounded_claims / total_claims (triggers + buyers)


# ---------------------------------------------------------------------------
# Mode 2 outputs: fit + signals + emails + claim map
# ---------------------------------------------------------------------------
class Signal(BaseModel):
    """A time-sensitive buying signal found in the target's public footprint."""
    type: str                  # funding | hiring | leadership | launch | expansion | tech | other
    summary: str
    recency: str = ""          # free text e.g. "May 2026" if found
    evidence_ids: List[str] = Field(default_factory=list)


class DimensionJudgment(BaseModel):
    """Per-ICP-dimension grounded assessment."""
    judgment: str = "absent"        # "present" | "partial" | "absent"
    evidence_ids: List[str] = Field(default_factory=list)
    note: str = ""                  # one-line explanation


class QuestionAnswer(BaseModel):
    """Answer to one ICP qualifying question, grounded in evidence."""
    question: str
    answer: str = "unknown"         # "yes" | "partial" | "no" | "unknown"
    rationale: str = ""
    evidence_ids: List[str] = Field(default_factory=list)


class AntiPatternHit(BaseModel):
    name: str
    evidence_ids: List[str] = Field(default_factory=list)


class FitScore(BaseModel):
    # LLM-judged headline (§4) — these are the primary verdict fields
    pain_fit: int = 0                          # LLM 0-100, holistic judgment
    pain_fit_rationale: str = ""
    worth_reaching_out: str = ""               # "yes" | "maybe" | "no"
    worth_rationale: str = ""
    outreach_angle: str = ""                   # best realistic angle even for low-fit
    qualification: List[QuestionAnswer] = Field(default_factory=list)
    # Deterministic transparency check — sits alongside as an auditable sanity light
    score: int = 0             # 0-100, computed deterministically from dimensions
    tier: str = "Weak"         # Strong / Moderate / Weak
    rationale: str = ""
    dimensions: Dict[str, DimensionJudgment] = Field(default_factory=dict)
    anti_patterns_hit: List[AntiPatternHit] = Field(default_factory=list)
    dimension_notes: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)


class AnglePlan(BaseModel):
    """Strategist output: the two email angles, briefed before drafting."""
    chosen_pain: str = ""           # the #1 pain to lead the pain-led email
    chosen_trigger: str = ""        # the strongest signal to lead the trigger-led email
    pain_angle_brief: str = ""      # 2-3 sentence brief for the drafter
    trigger_angle_brief: str = ""   # 2-3 sentence brief for the drafter


class CriticVerdict(BaseModel):
    """Critic output: quality gate on the two drafted emails."""
    passed: bool = True
    issues: List[str] = Field(default_factory=list)
    emails: List[dict] = Field(default_factory=list)  # revised emails if rewritten


class Claim(BaseModel):
    """A single factual assertion used in an email, mapped to its sources."""
    text: str
    evidence_ids: List[str] = Field(default_factory=list)
    grounded: bool = True      # False => model could not cite => flagged


class Email(BaseModel):
    angle: str                 # "pain-led" | "trigger-led"
    angle_label: str           # human label
    subject: str
    body: str
    claims: List[Claim] = Field(default_factory=list)


class TargetReport(BaseModel):
    target_company: str
    persona_role: str
    persona_seniority: str
    summary: str
    signals: List[Signal] = Field(default_factory=list)
    fit: FitScore
    emails: List[Email] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------
class SenderRequest(BaseModel):
    url: str
    max_pages: int = 6   # hard cap; early-stop fires at coverage=100% before this


class TargetRequest(BaseModel):
    url: str
    persona_role: str
    persona_seniority: str = ""
    sender: SenderProfile          # the ICP/value-prop from mode 1
    max_pages: int = 5
