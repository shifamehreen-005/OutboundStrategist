"""Mode 1 smoke tests.

All tests run in MOCK_MODE so no API key or network is needed.
fetch_site is patched inline with realistic content (no fixture files).
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from backend.agent import Agent
from backend.schemas import SenderProfile
from backend.fetcher import Page


# ---------------------------------------------------------------------------
# Shared realistic fake pages — enough content for BM25 to find snippets
# ---------------------------------------------------------------------------

FAKE_PAGES = [
    Page(
        url="https://acme.co",
        title="Acme — AI Sales Platform",
        text=(
            "## Product\n"
            "Acme is an AI-powered B2B sales platform for outbound teams. "
            "We help sales development representatives automate lead research, "
            "email personalization, and meeting booking at scale.\n\n"
            "## Trusted by\n"
            "Fast-growing B2B SaaS companies trust Acme to power their outbound motion. "
            "Customers include TechCorp, FinStart, and GrowthCo."
        ),
    ),
    Page(
        url="https://acme.co/pricing",
        title="Acme Pricing",
        text=(
            "## Pricing\n"
            "Startup plan: $99/month for teams of 1-10. "
            "Growth plan: $499/month for teams of 10-50. "
            "Enterprise plan: custom pricing for companies over 200 employees.\n\n"
            "Trusted by over 500 B2B SaaS companies worldwide."
        ),
    ),
    Page(
        url="https://acme.co/customers",
        title="Acme Customers",
        text=(
            "## Customer Stories\n"
            "TechCorp raised a Series B and used Acme to hire their first BDR team. "
            "FinStart, a fintech startup, tripled their outbound pipeline in 90 days. "
            "VP of Sales at GrowthCo: 'Acme replaced our need to hire 5 SDRs.' "
            "Results: 3x reply rates, 2x meetings booked per rep per week."
        ),
    ),
    Page(
        url="https://acme.co/about",
        title="About Acme",
        text=(
            "## About\n"
            "Acme was founded in 2022 and has raised $12M in Series A funding. "
            "Our team of 45 employees is based in San Francisco and New York. "
            "We serve B2B SaaS companies, fintech startups, and tech-enabled services "
            "with 10-500 employees that run an outbound sales motion."
        ),
    ),
]


def _fake_fetch(url, max_pages=6, **kwargs):
    return FAKE_PAGES[:max_pages]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drain(gen) -> List[Dict[str, Any]]:
    return list(gen)


def _find_profile(events: List[Dict]) -> SenderProfile | None:
    for ev in reversed(events):
        if ev.get("step") == "synthesize" and ev.get("status") == "done":
            data = ev.get("data") or {}
            return SenderProfile(**data["profile"])
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sender_happy_path(monkeypatch):
    """Full pipeline produces a valid profile with non-empty ICP."""
    monkeypatch.setattr("backend.agent.fetch_site", _fake_fetch)
    monkeypatch.setattr("backend.agent.find_customer_links", lambda *a, **k: [])
    monkeypatch.setattr("backend.agent.fetch_enrichment", lambda d: [])

    events = _drain(Agent().run_sender("acme.co", 6))
    profile = _find_profile(events)

    assert profile is not None, "synthesize-done event was never emitted"
    assert profile.value_proposition, "value_proposition is empty"
    assert len(profile.icp.industries) >= 1, "ICP has no industries"
    assert len(profile.icp.triggers) >= 1, "ICP has no triggers"
    assert len(profile.evidence) >= 1, "evidence list is empty"


def test_sender_thin_site(monkeypatch):
    """A site with almost no text must complete without raising."""
    monkeypatch.setattr(
        "backend.agent.fetch_site",
        lambda url, max_pages=6, **_: [
            Page(url="https://thin.example", title="Thin", text="We do stuff.")
        ],
    )
    monkeypatch.setattr("backend.agent.find_customer_links", lambda *a, **k: [])
    monkeypatch.setattr("backend.agent.fetch_enrichment", lambda d: [])

    events = _drain(Agent().run_sender("thin.example", 6))
    steps = {ev.get("step") for ev in events}
    assert "synthesize" in steps, "synthesize step never emitted"


def test_customers_extracted_and_grounded(monkeypatch):
    """Mode 1 must extract named customers, each cited."""
    monkeypatch.setattr("backend.agent.fetch_site", _fake_fetch)
    monkeypatch.setattr("backend.agent.find_customer_links", lambda *a, **k: [])
    monkeypatch.setattr("backend.agent.fetch_enrichment", lambda d: [])

    events = _drain(Agent().run_sender("acme.co", 6))
    ev = next(
        (e for e in events if e.get("step") == "customers" and e.get("status") == "done"),
        None,
    )
    assert ev is not None, "customers-done event never emitted"

    profile = _find_profile(events)
    assert profile is not None
    assert len(profile.named_customers) >= 1, "no named customers extracted"
    for c in profile.named_customers:
        assert c.name, "named customer has empty name"
        assert c.evidence_ids or c.source_url, (
            f"customer '{c.name}' has neither evidence_ids nor a source_url"
        )


def test_icp_has_qualifying_questions(monkeypatch):
    """The ICP must expose qualifying questions."""
    monkeypatch.setattr("backend.agent.fetch_site", _fake_fetch)
    monkeypatch.setattr("backend.agent.find_customer_links", lambda *a, **k: [])
    monkeypatch.setattr("backend.agent.fetch_enrichment", lambda d: [])

    events = _drain(Agent().run_sender("acme.co", 6))
    profile = _find_profile(events)
    assert profile is not None
    assert len(profile.icp.qualifying_questions) >= 1, "ICP has no qualifying_questions"


def test_grounding_coverage(monkeypatch):
    """grounding_coverage must be in [0, 1] and grounded claims must have evidence_ids."""
    monkeypatch.setattr("backend.agent.fetch_site", _fake_fetch)
    monkeypatch.setattr("backend.agent.find_customer_links", lambda *a, **k: [])
    monkeypatch.setattr("backend.agent.fetch_enrichment", lambda d: [])

    events = _drain(Agent().run_sender("acme.co", 6))
    profile = _find_profile(events)

    assert profile is not None
    assert 0.0 <= profile.grounding_coverage <= 1.0

    for trigger in profile.icp.triggers:
        if trigger.grounded:
            assert trigger.evidence_ids, f"Trigger '{trigger.name}' grounded but no evidence_ids"

    for buyer in profile.icp.buyers:
        if buyer.grounded:
            assert buyer.evidence_ids, f"Buyer '{buyer.role}' grounded but no evidence_ids"


def test_sender_thin_fixture_confidence(monkeypatch):
    """Thin corpus must produce confidence='low' and non-empty data_gaps."""
    monkeypatch.setattr(
        "backend.agent.fetch_site",
        lambda url, max_pages=6, **_: [
            Page(url="https://sparse.example", title="Sparse Co", text="We do stuff.")
        ],
    )
    monkeypatch.setattr("backend.agent.find_customer_links", lambda *a, **k: [])
    monkeypatch.setattr("backend.agent.fetch_enrichment", lambda d: [])

    events = _drain(Agent().run_sender("sparse.example", 6))
    profile = _find_profile(events)

    assert profile is not None
    assert profile.confidence == "low"
    assert len(profile.data_gaps) > 0
    assert profile.evidence_sufficient is False
