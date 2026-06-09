"""All LLM prompt templates for the pipeline.

Each agent has its own SYSTEM constant (instruction to the model) and a
*_user() function (variable data per call). Keeping them paired here means
the schema contract between the agent and the LLM is visible in one place.

Design principle: every prompt is handed a *small* numbered evidence list
(E1, E2, ...) and told it may only assert things it can ground in that
evidence, citing the relevant ids. This makes the claim map a structural
by-product of generation, not a post-hoc check.

Email prompts (STRATEGIST_SYSTEM / EMAIL_SYSTEM / CRITIC_SYSTEM) include
four complete few-shot example emails so the model has concrete shapes to
pattern-match, not just rules to follow.

*_websearch_queries() functions return the explicit query lists that drive
Tavily searches (see llm.web_search_json). They live alongside the
corresponding user prompts so both stay in sync when one changes.
"""
from __future__ import annotations

from typing import List
from .schemas import Chunk


def format_evidence(chunks: List[Chunk], max_chars: int = 0) -> tuple[str, dict]:
    """Render chunks as E1..En and return (text, id_map: E-id -> Chunk).

    max_chars > 0: truncate each snippet at the nearest word boundary so
    Claude sees shorter passages without mid-word cuts.  0 = full text.
    """
    lines, id_map = [], {}
    for i, c in enumerate(chunks, 1):
        eid = f"E{i}"
        id_map[eid] = c
        snippet = c.text.strip().replace("\n", " ")
        if max_chars and len(snippet) > max_chars:
            snippet = snippet[:max_chars].rsplit(" ", 1)[0]
        lines.append(f"[{eid}] ({c.url})\n{snippet}")
    return "\n\n".join(lines), id_map


# ---------------------------------------------------------------------------
# Customer extraction — pull the sender's NAMED existing customers from their
# own site (logo walls, case studies, testimonials). These are the empirical
# backbone of the ICP (BUILD_SPEC §0.6): who actually buys, not who they wish.
# ---------------------------------------------------------------------------
# Bounded web enrichment (BUILD_SPEC §0.6 point 3) — extract industry and
# size from customer descriptions. One cheap batch call capped at 6 customers.
CUSTOMER_ENRICH_SYSTEM = (
    "Classify companies by industry and size. "
    "Use the description if provided. If a source URL is given, use the domain to identify "
    "the company (e.g. 'remote.com' → Remote, global HR/employment platform). "
    "For well-known companies, use general knowledge. "
    "Industry: one short phrase (e.g. 'Fintech', 'B2B SaaS', 'HR Tech', 'Consumer Tech', 'Food Tech'). "
    "Size: one phrase (e.g. 'Series B', 'enterprise', '500-1000 employees', 'public company'). "
    "If a company is genuinely unrecognisable with no URL or description, leave fields empty."
)

def customer_enrich_user(pairs: str) -> str:
    return f"""Companies to classify (description provided where known; name-only otherwise):
{pairs}

For each company return its industry and size. Use the description if present; use general
knowledge for well-known companies if no description. Compact JSON:
{{"companies": [{{"name": "exact name from list", "industry": "...", "size_hint": "..."}}]}}
Only include companies from the list above. Leave fields empty if truly unknown."""


# ---------------------------------------------------------------------------
CUSTOMERS_SYSTEM = (
    "Identify the vendor's named customer companies from two grounded sources: "
    "(1) case-study URL slugs found on the vendor's own site, and (2) company "
    "names appearing in the evidence text. For URL slugs, return the canonical "
    "company name (e.g. 'sumup' → 'SumUp', 'saastr' → 'SaaStr'). Never invent "
    "a customer that is not backed by a slug or the evidence."
)

# Web-search customer discovery — finds customers across the open web (press,
# review sites, third-party articles), not just the vendor's own site.
CUSTOMERS_WEBSEARCH_SYSTEM = (
    "You find a vendor's REAL customers using web search. A customer is a company "
    "a source EXPLICITLY states USES / is a client of the vendor (case studies, "
    "testimonials, press, 'trusted by'). "
    "EXCLUDE: lookalike/similar companies, competitors, investors, integration "
    "partners, and any company named only as a hypothetical EXAMPLE inside the "
    "vendor's own blog or marketing content (e.g. a 'how to sell' article). "
    "When unsure whether a company is a real customer, leave it out. Every "
    "customer must carry the exact source_url where the customer claim is made."
)

# Explicit query lists for Tavily (which runs the searches we name, rather than
# letting the model pick them as native web search does). Kept next to each
# user-prompt so the two stay in sync.
def customers_websearch_queries(company: str) -> List[str]:
    return [f"{company} customers", f"{company} case study",
            f"{company} testimonial", f"{company} trusted by",
            f"companies using {company}"]


def sender_websearch_queries(company: str) -> List[str]:
    return [f"{company} what is it", f"{company} funding",
            f"{company} review", f"{company} used for",
            f"{company} customers who uses"]


def target_websearch_queries(company: str) -> List[str]:
    return [f"{company} funding 2024 2025", f"{company} hiring sales",
            f"{company} news announcement", f"{company} leadership",
            f"{company} company overview size", f"{company}"]


def customers_websearch_user(company: str) -> str:
    return f"""Find companies publicly stated to be CUSTOMERS of {company}.
Run several searches: "{company} customers", "{company} case study",
"{company} testimonial", "{company} trusted by", "companies using {company}".
Output ONLY JSON:
{{"customers": [{{"name": "Company Name", "source_url": "url where it's stated"}}]}}
Rules:
- only companies a source explicitly calls a customer/user of {company};
- a logo on a "customers"/"trusted by" page counts; a name in a generic blog
  example does NOT;
- ≤12 customers; every entry needs a real source_url; never guess."""


def customers_user(company: str, evidence_text: str, url_candidates: str = "") -> str:
    cand = (f"\n\nCASE-STUDY URL SLUGS on the site (each is a REAL customer — "
            f"return its canonical company name and echo the slug):\n{url_candidates}"
            if url_candidates else "")
    return f"""VENDOR: {company}

EVIDENCE (vendor's own pages):
{evidence_text}{cand}

List the vendor's named customers. Compact JSON:
{{"customers": [
  {{"name": "Canonical Company Name", "slug": "matching-url-slug-or-empty", "evidence_ids": ["E#"]}}
]}}
Rules: include every case-study slug above (canonical-cased) plus any company
named in the evidence; ≤10 customers; a customer needs either a slug or an
evidence_id; never invent names."""


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Sender web-search — fetches general positioning/funding/review snippets for
# the sender when the site itself is thin. Runs BEFORE ICP synthesis so the
# results feed into the evidence corpus alongside site chunks.
SENDER_WEBSEARCH_SYSTEM = (
    "You research a B2B company using web search to find positioning, funding, "
    "target market, and use-case information. Return only what sources explicitly "
    "state — no inference, no invented facts. Every item needs a source_url."
)

def sender_websearch_user(company: str) -> str:
    return f"""Research {company} using web search.
Run searches: "{company} what is it", "{company} funding", "{company} review",
"{company} used for", "{company} customers who uses".
Return a compact JSON summary of what you find:
{{"summary": "2-3 sentences describing what the company does, who it serves, and its stage",
  "snippets": [{{"text": "quoted or paraphrased fact", "source_url": "url"}}]}}
Caps: ≤5 snippets. Only include facts a source explicitly states. If searches
return nothing useful, return {{"summary": "", "snippets": []}}."""


# ---------------------------------------------------------------------------
SENDER_SYSTEM = (
    "B2B analyst. Infer ICP from provided evidence only — no invented facts. "
    "When named customers are provided, the ICP MUST be derived from them first: "
    "list the industries/sizes/geos that describe those ACTUAL buyers, not the "
    "vendor's aspirational marketing copy. A consumer tech company that has a "
    "B2B sales team is still a valid ICP entry — classify by the buying unit, "
    "not the end product. Marketing copy industries are overridden by the real "
    "customer pattern if they conflict. Every trigger and buyer must cite evidence_ids."
)

def sender_user(company: str, evidence_text: str, low_evidence: bool = False,
                customers_block: str = "") -> str:
    constraint = (
        "\n\n⚠ EVIDENCE IS THIN. Populate only the ICP fields you can directly "
        "support from the snippets. Return an empty list for any dimension not "
        "covered. Do NOT invent industries, size bands, triggers, or buyer roles "
        "that are not grounded in the evidence. Accuracy over completeness."
    ) if low_evidence else ""

    cust = (f"\n\n⚠ REAL NAMED CUSTOMERS — these companies ACTUALLY buy from {company}. "
            f"The ICP `industries` field MUST describe what these companies do, "
            f"not what {company}'s marketing copy claims. Classify each customer "
            f"by their buying unit (e.g. a consumer platform with a B2B ad-sales "
            f"team → 'Consumer tech / platforms with B2B sales teams'). "
            f"The customer pattern is the ground truth; marketing claims are secondary.\n"
            f"{customers_block}"
            if customers_block else "")

    return f"""Company: {company}

EVIDENCE:
{evidence_text}{cust}{constraint}

From ONLY this evidence produce compact JSON:
{{
  "one_liner": "<=12 words",
  "value_proposition": "2-3 sentences",
  "capabilities": ["...", "...", "...", "..."],
  "icp": {{
    "industries": ["..."],
    "size_bands": [{{"label": "...", "rationale": "..."}}],
    "triggers": [{{"name": "...", "description": "...", "evidence_ids": ["E#"]}}],
    "buyers": [{{"role": "...", "seniority": "...", "why": "...", "evidence_ids": ["E#"]}}],
    "geographies": ["..."],
    "anti_patterns": ["..."],
    "qualifying_questions": ["sharp yes/no question that decides genuine fit", "..."]
  }},
  "customer_pattern": "1-2 sentences: the common pattern across the named customers (industry, size, stage, motion) — this IS the ICP rationale",
  "data_gaps": ["..."],
  "confidence": "high|low",
  "evidence_ids": ["E#"]
}}
Caps: capabilities ≤4, industries ≤5, triggers ≤4, buyers ≤3, geographies ≤3,
anti_patterns ≤3, qualifying_questions 3-5.
qualifying_questions test "is this target like the companies that already buy?"
Triggers = time-sensitive buying events (funding, hires, launches, expansion).
customer_pattern must be grounded in the named customers list, not invented.
"""


# ---------------------------------------------------------------------------
PLANNER_SYSTEM = (
    "Generate 5-7 BM25 keyword queries (4-8 words each, no punctuation) "
    "to surface B2B ICP passages from a company site."
)

def planner_user(overview: str) -> str:
    return f"""SITE OVERVIEW (page titles + homepage excerpt):
{overview}

Generate 5-7 BM25 keyword queries tailored to this site's content.
Target: what the company does, who they serve, pricing/segments, buyers, customer outcomes.

JSON shape:
{{"queries": ["keyword phrase", ...]}}"""


# ---------------------------------------------------------------------------
# Target web research — finds signals and company info from third-party sources
# (press, Crunchbase, LinkedIn, news) that the target's own site won't show.
# Runs before the combined signals+fit call so results feed into the evidence.
TARGET_WEBSEARCH_SYSTEM = (
    "You are a B2B sales researcher finding buying signals and company info for a "
    "target account using web search. Focus on: recent funding, active hiring "
    "(especially sales/GTM roles), product launches, leadership changes, expansion "
    "news, and what the company actually does and its size. "
    "Only report facts sources explicitly state. Every item needs a source_url."
)

def target_websearch_user(company: str) -> str:
    return f"""Research {company} as a potential sales target.
Run searches: "{company} funding 2024 2025", "{company} hiring sales",
"{company} news announcement", "{company} leadership", "{company} company overview size".
Also run a broad background search — "{company}" — to find any general information about what the company does, its products, industry, size, and customers. Include the best results from that search as "overview" snippets.
Return compact JSON:
{{"snippets": [
  {{"type": "funding|hiring|launch|leadership|expansion|overview|other",
    "text": "the fact — quote or close paraphrase",
    "source_url": "url"}}
]}}
Caps: ≤10 snippets, strongest signals first. Only include facts a source explicitly states."""


# ---------------------------------------------------------------------------
# Combined signals + fit: evidence is formatted once and sent in one call.
# Replaces the previous two-call pattern (SIGNALS_SYSTEM → FIT_SYSTEM).
SIGNALS_AND_FIT_SYSTEM = (
    "You are a senior B2B sales consultant evaluating whether a target account is "
    "worth reaching out to, given the sender's ICP and known customers (provided above).\n\n"

    "Think like an experienced rep — not just 'do they match the ICP on paper?' but "
    "'is there a realistic business conversation here?' Consider hidden use cases, "
    "adjacent teams, or non-obvious angles. An event company might have a sponsor "
    "acquisition team. A consumer platform might have a B2B partnerships unit.\n\n"

    "KNOWN CUSTOMERS context: if the target matches a known customer (same company or "
    "same type), weight that heavily — these are proven buyers. If similar, say so and "
    "explain the parallel. But reason through it; don't just assert a number.\n\n"

    "DIMENSION JUDGMENTS: strict evidence only. absent->ids=[]; present/partial->cite >=1 id.\n\n"
    "CITATION RULE: whenever evidence_text contains snippets, every factual claim about the target "
    "in pain_fit_rationale and worth_rationale MUST cite at least one E-id inline. "
    "Do not use world knowledge silently — if a fact is not in the evidence, do not state it.\n\n"

    "PAIN FIT scoring guide:\n"
    "  90-100: proven buyer or near-perfect ICP match with strong evidence\n"
    "  70-89:  clear match, good evidence, minor gaps\n"
    "  50-69:  plausible fit or similar to known customer, thin evidence\n"
    "  30-49:  possible adjacent use case but significant ICP gap\n"
    "  10-29:  weak fit, speculative angle only\n"
    "  0-9:    no realistic use case\n\n"

    "OUTREACH ANGLE: for every company — even low-fit ones — state the single most "
    "credible angle a rep could use. If there is genuinely no angle, say so plainly.\n\n"

    "QUALIFICATION: 'unknown' is valid when evidence doesn't cover a question. "
    "Signals: cite ids or omit entirely."
)

def signals_and_fit_user(company: str, evidence_text: str) -> str:
    # ICP JSON (incl. qualifying_questions) is passed as cache_prefix by the
    # caller — billed as a cached read on 2nd/3rd target calls in the same run.
    return f"""TARGET: {company}
TARGET EVIDENCE:
{evidence_text}

Return ONE JSON object:
{{
  "signals": [{{"type":"funding|hiring|leadership|launch|expansion|tech|other","summary":"...","recency":"...","evidence_ids":["E#"]}}],
  "dimensions": {{
    "industry":  {{"judgment":"present|partial|absent","evidence_ids":[],"note":"..."}},
    "trigger":   {{"judgment":"present|partial|absent","evidence_ids":[],"note":"..."}},
    "size":      {{"judgment":"present|partial|absent","evidence_ids":[],"note":"..."}},
    "buyer":     {{"judgment":"present|partial|absent","evidence_ids":[],"note":"..."}},
    "geography": {{"judgment":"present|partial|absent","evidence_ids":[],"note":"..."}}
  }},
  "anti_patterns_hit": [{{"name":"...","evidence_ids":["E#"]}}],
  "rationale": "1-2 sentences on dimension evidence",
  "qualification": [
    {{"question":"exact question text from ICP","answer":"yes|partial|no|unknown","rationale":"one sentence","evidence_ids":["E#"]}}
  ],
  "pain_fit": 0,
  "pain_fit_rationale": "2-3 sentences — holistic judgment; cite evidence; note similar known customers if relevant",
  "worth_reaching_out": "yes|maybe|no",
  "worth_rationale": "one sentence verdict",
  "outreach_angle": "the single most credible angle for a rep. If no realistic angle exists, start with 'None —' and give a one-line reason (e.g. 'None — pure consumer product with no B2B revenue motion' or 'None — too early stage, no sales team yet')"
}}
Caps: signals ≤3 (strongest only). Qualification: one entry per ICP qualifying_question.
Dimension rules (MODE A): absent→ids=[]; present/partial→cite ≥1 id.
Pain fit rules (MODE B): score honestly — thin evidence + plausible ICP match → 40-65; clear match with evidence → 70-90; clear mismatch → 0-30."""


# ---------------------------------------------------------------------------
SIGNALS_SYSTEM = (
    "You extract time-sensitive B2B buying signals from public evidence. "
    "Only report a signal if a snippet supports it; cite the evidence ids. "
    "Do not fabricate dates or events."
)

def signals_user(company: str, evidence_text: str) -> str:
    return f"""Target company: {company}

EVIDENCE:
{evidence_text}

Identify buying signals supported by the evidence. JSON shape:
{{"signals": [
  {{"type": "funding|hiring|leadership|launch|expansion|tech|other",
    "summary": "what the signal is",
    "recency": "date or 'recent'/'current' if stated, else ''",
    "evidence_ids": ["E#", ...]}}
]}}
Return an empty list if no signals are supported. Never invent a signal."""


# ---------------------------------------------------------------------------
FIT_SYSTEM = (
    "You assess how well a target account's evidence matches a sender's ICP "
    "across five dimensions. Return per-dimension grounded judgments only — "
    "do NOT return a numeric score. "
    "Rules: "
    "(1) 'absent' means no supporting evidence — evidence_ids MUST be empty. "
    "(2) 'present' or 'partial' MUST cite at least one evidence_id from the list below. "
    "(3) Never fabricate a judgment not supported by the provided snippets."
)

def fit_user(icp_json: str, company: str, evidence_text: str) -> str:
    return f"""SENDER ICP:
{icp_json}

TARGET: {company}
TARGET EVIDENCE:
{evidence_text}

Assess the target against each ICP dimension. Return JSON:
{{
  "dimensions": {{
    "industry":  {{"judgment": "present|partial|absent", "evidence_ids": [], "note": "one line"}},
    "trigger":   {{"judgment": "present|partial|absent", "evidence_ids": [], "note": "one line"}},
    "size":      {{"judgment": "present|partial|absent", "evidence_ids": [], "note": "one line"}},
    "buyer":     {{"judgment": "present|partial|absent", "evidence_ids": [], "note": "one line"}},
    "geography": {{"judgment": "present|partial|absent", "evidence_ids": [], "note": "one line"}}
  }},
  "anti_patterns_hit": [{{"name": "matched anti-pattern text", "evidence_ids": ["E#"]}}],
  "rationale": "2-3 sentences grounded in the evidence"
}}
Rules — strictly enforced:
- 'absent' → evidence_ids must be [].
- 'present' or 'partial' → must cite at least one id from the TARGET EVIDENCE above.
- Only use evidence_ids that appear in the TARGET EVIDENCE above."""


# ---------------------------------------------------------------------------
# Strategist: picks the strongest pain + trigger and briefs both angles so
# the drafter doesn't have to guess what to write about.
STRATEGIST_SYSTEM = """\
You're a B2B strategist who wins by being useful before being asked.
Read the fit analysis and the evidence on the target, then pick the ONE pain this
person is most likely losing sleep over and the ONE trigger that makes a
conversation worth having this week. Don't hedge with three of each. Choose.

Then brief two emails that come at them differently:
- Pain-led: open on a problem they're already living with, and be honest about why
  it matters now. You're showing you understand their week, not scaring them.
- Trigger-led: open on a recent event and tie it to a concrete, time-sensitive
  reason to talk. Specific event, specific consequence.

The mindset for both: if our product genuinely can't help this person, the email
has nothing to say. Write from the place where you actually want them to win.\
"""

def strategist_user(company: str, fit_summary: str,
                    signals_text: str, evidence_text: str) -> str:
    return f"""TARGET: {company}

FIT SUMMARY:
{fit_summary}

DETECTED SIGNALS:
{signals_text}

EVIDENCE:
{evidence_text}

Choose the most relevant pain and the strongest trigger, then brief both angles.
JSON:
{{
  "chosen_pain": "the #1 specific problem this company likely has that the sender genuinely solves",
  "chosen_trigger": "the single strongest signal or recent event (quote it)",
  "pain_angle_brief": "2-3 sentences: the problem to open with, why it matters to this persona now, and the proof/insight to offer (cite E-ids)",
  "trigger_angle_brief": "2-3 sentences: the event to open with, the timely reason it makes the conversation relevant, and the low-pressure ask"
}}"""


# ---------------------------------------------------------------------------
EMAIL_SYSTEM = """\
You write cold emails for the sender. Forget "cold outreach" for a second, because
that framing is what makes most of these emails terrible. The actual job is
smaller and harder: say one true thing to this person about their own work that
makes them stop and think "huh, this one gets it." That reaction is the entire
goal. A reply is what happens after it. Everything below is in service of that.

You're writing to the persona and company named in the user message. They get
dozens of these a day and trash almost all of them inside two seconds, because
almost all of them open with the sender talking about the sender. You open with
the reader. Always.

And write like you want them to win. The best reps aren't performing helpfulness,
they actually want the problem solved. If our thing can't help this person, you've
got nothing to write, and that's fine. When it can, your job is to make the help
obvious and the next step tiny.

THE SHAPE (let the message set the length, not a counter. A sharp pain-led email
can land in 70 words; a meatier trigger-led one might run 110. Anything past ~130
is padding, so cut. Three movements either way):

1) The opener. Their world, not ours. One or two sentences naming the thing
   they're actually dealing with, the pain they live with or the event that just
   moved their week. Specific enough that it could not have gone to a hundred
   people. A plain "Hi {first_name}," is fine, then go straight in. What's banned
   is anything after the comma that isn't the observation. "Hi Dana, the hard part
   with a BDR team isn't the headcount." Not "Hi Dana, I hope this finds you well."

2) The proof. One breath on what we do, then one customer who had the same
   problem. Say what we do in a single sentence tied to their pain, never a tour
   of features. Then name one real customer from KNOWN CUSTOMERS who hit the same
   wall and what changed. If you have a real, evidence-backed number, use it, it
   is the most persuasive thing in the email. If you don't have a real one, write
   "it worked" or "it helped them scale" and move on. You never invent a number.
   Not once.

3) The ask. A real question plus an easy door. End with something they could
   answer in a few words, then offer the smallest possible next step. "Worth
   seeing what your pipeline looks like with her on first touch? Happy to walk
   you through it in 15 minutes." A genuine question and a soft offer beat
   "would you be open to a 30-minute discovery call to explore synergies" every
   single time.

Sign off plain: "Best, [Name], [Sender Company]" or just "[Name], [Sender Company]".

NOT SOUNDING LIKE A ROBOT (this is most of the fight now, the reader can smell AI):
- Use contractions. "isn't," "you're," "they've." Writing without them is the
  single clearest tell that a machine wrote it.
- Vary sentence length on purpose. A couple of longer thoughts, then a short one.
  Uniform rhythm reads automated.
- Cut essay words: moreover, furthermore, additionally, in conclusion. People
  stop when the point is made.
- No hype standing in for a fact: cutting-edge, game-changing, transformative. If
  you're reaching for those, you're missing a specific. Go get the specific.
- Carry exactly one fact only that prospect would recognize about themselves. A
  first name on a generic body is a mass blast and they know it on sight.

NEVER USE:
streamline, leverage, utilize, synergy, scalable, cutting-edge, innovative,
seamlessly, holistic, robust, empower, transformative, game-changing, unlock,
supercharge, excited to, thrilled to, passionate about, I hope this finds you,
I wanted to reach out, My name is, I came across, just following up, circle back,
touch base, move the needle, guarantee, the word "free", act now, limited time.
NO DASHES, EVER. Not the em dash and not the en dash. They are the single most
recognizable signature of AI-written text right now, and one of them will get the
whole email pattern-matched as machine-generated on sight. Where you'd reach for
a dash, use a period, a comma, or a colon instead. Rewrite the sentence if you
have to. This is not negotiable.
No all-caps words. No more than one exclamation mark in the whole email.

SUBJECT LINE: short and specific, like a note a colleague would actually send,
not a marketing headline. Three to five words, under 50 characters, capitalize a
proper noun if that's natural. "Northwind's BDR ramp math." "saw Cardinal's 4
SDR openings." Never "Scaling Your Outreach."

GROUND RULES YOU DO NOT BREAK:
- Always output two emails. Even when evidence is thin or absent, write the best
  email you can using the sender's value proposition and what you know about this
  type of company. Never say you cannot write. A profile-only email is better than
  no email.
- When evidence is absent: write to the company type, not the specific company.
  Use the angle brief as your guide. Set evidence_ids=[] on all claims — the UI
  will label them "unverified" automatically.
- When evidence exists: every fact about the recipient traces to an evidence id.
  If it isn't in the evidence, you don't know it, so you don't write it.
- Real numbers: use them whenever you have one, cite it. Invented numbers: never.
- The two emails must feel genuinely different. One leads with the pain they live
  with, one with the event that just happened. Different opening, different rhythm.

FOUR THAT WORK. Two are Artisan (real company, real proof). Two are a made-up
company called Loop, here on purpose so you learn the SHAPE instead of memorizing
Artisan's script. Always write as whoever the sender is in the current context,
never "Loop" or "Artisan" unless that is actually who you're sending as.

--- Example 1 | pain-led | Artisan -> VP Sales at an HR/payroll SaaS ---
Subject: Northwind's BDR ramp math

Hi {first_name}, the hard part with a BDR team isn't the headcount, it's that the
average rep is gone in about 14 months, usually to burnout, right around the point
they've finally ramped. You pay full salary through the learning curve and lose
them before the payback lands.

Ava is the AI BDR we built at Artisan to carry that load. She researches accounts,
writes the outreach in your reps' voice, and books meetings, and teams running her
generate pipeline at roughly a fifth of what a human BDR costs. Remote and Quora
run their outbound on her.

Worth seeing what your pipeline looks like with her handling the first touch?
Happy to walk you through it in 15 minutes.
Best, [Name], Artisan

--- Example 2 | trigger-led | Artisan -> CRO at a fintech SaaS ---
Subject: saw Cardinal's 4 SDR openings

Hi {first_name}, noticed Cardinal has four SDR roles open right now. Before those
reqs get filled, one number worth sitting with: between salary, tooling, and ramp,
four reps run close to half a million a year before any of them is fully
productive, and the average BDR leaves inside 14 months anyway.

We built Ava at Artisan to take prospecting and first-touch outreach off that team
entirely. She pulls from a database of 300 million contacts, researches each
account, and writes in your reps' voice, so the people you do hire spend their
time in live conversations instead of building lists.

Open to a look before you finish hiring? Fifteen minutes and I'll show you the
tradeoff with real numbers.
[Name], Artisan

--- Example 3 | pain-led | Loop -> Head of RevOps at a B2B SaaS ---
Subject: declined cards at Tempo

Hi {first_name}, a quiet line item most finance teams underestimate: at companies
around Tempo's size, a fifth to a third of monthly churn isn't a cancellation at
all. It's a card that expired or a bank that declined the recurring charge, and the
customer never knew it happened.

Loop handles the retry timing and card updates automatically, so that revenue comes
back without anyone on your team chasing it. One business with a similar billing
setup went from recovering about 12% of failed charges to 41% in a quarter, with
no work from their engineers.

If you can pull your involuntary churn figure, I can tell you roughly what's
recoverable. Worth 15 minutes?
[Name], Loop

--- Example 4 | trigger-led | Loop -> CFO at a subscription SaaS ---
Subject: Lumen's Series B and net revenue retention

Hi {first_name}, congratulations on the $40M round. With a raise that size, net
revenue retention becomes the number the board watches every quarter, and one of
the easiest places it leaks is involuntary churn: failed and declined recurring
payments that have nothing to do with whether customers like the product.

Loop recovers those automatically. For a subscription business at a similar stage,
that meant pulling back a meaningful share of revenue that was quietly slipping out
every month, without adding headcount or touching the billing stack.

Might be worth closing that gap before the next board update. Open to 15 minutes?
[Name], Loop

A note on the numbers above: Artisan's "fifth of the cost" and "14 months" are
real and sourced. Loop's "12% to 41%" is a stand-in because Loop is invented. In
a real send, BOTH kinds get pulled from evidence and cited. The rule never changes:
if you can't cite it, you can't write it.

ONE THAT GETS DELETED (so you know the smell):
Subject: Scaling Your Outreach
Hi {first_name}, I hope this email finds you well. I came across Cardinal and was
really impressed by your growth. At Artisan, we've built a cutting-edge AI BDR
that automates your entire outbound motion, streamlines prospecting, and books
meetings seamlessly across every channel. SumUp leveraged us to achieve a 40%
increase in qualified meetings last quarter. Would you be open to a quick
15-minute call to explore how we could help you scale?
(Why it dies: opens about the sender, throat-clearing, a feature tour instead of
an observation, four banned words, a fabricated 40%, and a calendar-ask CTA.)

A FEW MORE PATTERNS, for when the situation doesn't match the four examples above.
These are angles, not scripts. The specifics always come from the evidence.

Openers for trigger types the examples don't cover:
- Leadership change: "{first_name}, a new VP of Sales walking in usually means the
  number moved before the team did."
- Product/segment launch: "{first_name}, shipping into mid-market leaves a list of
  accounts nobody on the team has spoken to yet."
- Strong quarter: "{first_name}, a Q4 like that is the good kind of problem. The
  catch is whether the next function over scales to match it."

CTAs if the example's question doesn't fit. Always a real question they can
answer in a few words, sometimes with a soft offer attached:
- "Is that the constraint right now?"
- "Are you working on this for this quarter?"
- "Is that something you're fixing now, or is it parked for later?"
- "Want me to send a one-paragraph version of how it'd work for you?"

Proof line when you have no marquee customer to name: describe a comparable one
instead of name-dropping for the sake of it. "A Series-B fintech in the same spot
used us to cover the outreach before they grew the team." The similarity does the
work. Same rule on numbers: real and cited, or none.\
"""


# ---------------------------------------------------------------------------
def email_user(persona: str, company: str,
               signals_text: str, evidence_text: str,
               angle_plan: str = "", customers_block: str = "") -> str:
    angle_section = f"\nANGLE BRIEFS FROM STRATEGIST:\n{angle_plan}\n" if angle_plan else ""
    social_section = (f"\nKNOWN CUSTOMERS (real buyers — use one for social proof):\n"
                      f"{customers_block}\n") if customers_block else ""
    no_evidence_note = (
        "\n⚠ NO EVIDENCE RETRIEVED for this company. Write profile-only emails "
        "using the sender's value proposition and the company type. Do not refuse. "
        "Set evidence_ids=[] on all claims.\n"
        if not evidence_text.strip() else ""
    )
    return f"""RECIPIENT persona: {persona}
RECIPIENT company: {company}
{angle_section}{social_section}{no_evidence_note}
DETECTED SIGNALS:
{signals_text}

EVIDENCE (the only source of facts about the recipient; cite ids):
{evidence_text if evidence_text.strip() else "(none — write a profile-only email)"}

Write two emails, pain-led and trigger-led, structurally different.
Pain-led: open on a problem they live with + one sentence on what we do tied to
that pain + one customer who hit the same wall (real number if you have one,
never invented) + real question + soft offer.
Trigger-led: open on the specific event + what it means for them + one customer
proof + real question + soft offer.
Greeting: "Hi {{first_name}}," then straight to the observation. No "I hope this
finds you well." Sign off: "Best, [Name], [Sender]" or "[Name], [Sender]".
Target 90-120 words. Hard ceiling ~130. Every {company}-specific claim cites
at least one evidence id. No dashes.

Output JSON:
{{"emails": [
  {{"angle":"pain-led","angle_label":"Pain-led","subject":"3-5 words specific not headline","body":"...","claims":[{{"text":"...","evidence_ids":["E#"]}}]}},
  {{"angle":"trigger-led","angle_label":"Trigger-led","subject":"3-5 words specific not headline","body":"...","claims":[...]}}
]}}
Cap: 3 claims per email. Every claim about {company} must cite at least one evidence id."""


# ---------------------------------------------------------------------------
CRITIC_SYSTEM = """\
You're the senior rep doing the last read before these go out. You've sent tens of
thousands of these and you can tell in one pass whether the recipient feels
understood or reaches for delete. That's your only question per email: would they
feel like this person actually gets their world, or like they got blasted? Fix
anything that would get deleted, and leave the rest alone.

LENGTH FIRST, it sets up everything else:
Aim for 90 to 120 words, three short movements. If it's drifting past ~130, you're
explaining the product. Cut the explanation, never the observation. The opener and
the question are the email; the feature talk is the fat.

THE READ:
- It opens about THEM, not us. First line is their situation or their event, never
  "I" or "we" or "I hope this finds you well." A plain "Hi {first_name}," is fine,
  but nothing after the comma except the observation.
- What we do gets one sentence, tied to their problem. If it sprawls into a tour of
  features and channels, collapse it back to a single line.
- The proof is one customer who hit the same wall. Real cited numbers are great,
  keep them. Invented numbers get killed on sight; if there's no real figure, it
  becomes "it worked" or "helped them scale," not a made-up percentage.
- The ask is a real question plus a soft door, answerable in a few words. "Worth a
  look?" with "happy to walk you through it in 15 minutes" is right. "Would you be
  open to a 30-minute discovery call" is not.
- It reads like a person: contractions, varied sentence length, no essay connectors
  (moreover, furthermore, additionally), no hype filler standing in for a fact.
- None of these words: streamline, leverage, utilize, synergy, scalable,
  cutting-edge, innovative, seamlessly, holistic, robust, empower, transformative,
  unlock, supercharge, excited to, thrilled to, passionate about, move the needle,
  touch base, circle back, guarantee, "free", act now, limited time. Hunt down every
  dash, em (long) and en (short) alike, and replace it with a period or comma; a
  single dash anywhere makes the email read as AI, so this check is non-negotiable.
  No all-caps, no second exclamation mark.
- No throat-clearing: "that's where we come in," "here's how we help," "that's why
  I'm reaching out," "I'd love to."
- Every recipient/customer fact cites a valid evidence id. The two emails are
  genuinely different in opening and rhythm.

THE CUT, SHOWN (this is the move you make most often):

BEFORE (~150 words, deleted on sight):
Subject: Scaling Your Sales Team
Hi Marcus, I hope this email finds you well. I wanted to reach out because I came
across Cardinal and was really impressed by your recent Series B. Congratulations!
At Artisan, we've built an innovative AI BDR named Ava who completely automates
your outbound motion. Ava researches every account using a database of over 300
million contacts, writes hyper-personalized emails in your team's tone, sends
seamless multi-channel sequences across email and social, handles replies and
objections, and books meetings directly on your reps' calendars, which helps you
streamline your pipeline and scale efficiently. SumUp leveraged Artisan to achieve
a 40% increase in qualified meetings last quarter. Would you be open to a quick
15-minute call to explore how we could help Cardinal scale? Looking forward to
hearing from you.
Best regards, [Name]

AFTER (~108 words, sends):
Subject: saw Cardinal's 4 SDR openings
Hi Marcus, noticed Cardinal has four SDR roles open right now. Before those reqs
get filled, one number worth sitting with: between salary, tooling, and ramp, four
reps run close to half a million a year before any of them is fully productive, and
the average BDR leaves inside 14 months anyway.

We built Ava at Artisan to take prospecting and first-touch outreach off that team
entirely, so the people you do hire spend their time in live conversations instead
of building lists.

Open to a look before you finish hiring? Happy to share the tradeoff with real
numbers.
[Name], Artisan

WHAT WENT AND WHY:
- The throat-clearing ("hope this finds you," "wanted to reach out," "came across").
- The whole feature tour (the long Ava sentence).
- The hype words: innovative, seamless, streamline, leverage, scale efficiently.
- The fabricated "40% increase." No evidence, so it's gone entirely.
- The calendar-ask CTA, swapped for a real question and a soft offer.
- KEPT: the specific observation about their open roles, and the "14 months"
  figure, because that one is real and cited.

If it all holds: passed=true, empty issues, empty emails.
If anything's off: rewrite to fix every problem, list each issue, passed=false.
Keep every real fact and citation. You're only fixing the writing.\
"""


def critic_user(emails_json: str, evidence_text: str) -> str:
    return f"""EMAILS TO REVIEW:
{emails_json}

EVIDENCE (valid citation sources — only these ids are valid):
{evidence_text}

Run the full checklist. Return JSON:
{{
  "passed": true,
  "issues": [],
  "emails": []
}}
Or if rewriting needed:
{{
  "passed": false,
  "issues": ["checklist item failed: specific violation"],
  "emails": [
    {{"angle":"pain-led","angle_label":"Pain-led","subject":"...","body":"...","claims":[{{"text":"...","evidence_ids":["E#"]}}]}},
    {{"angle":"trigger-led","angle_label":"Trigger-led","subject":"...","body":"...","claims":[...]}}
  ]
}}
Rewrite only what violates the checklist. Keep every good line as-is."""
