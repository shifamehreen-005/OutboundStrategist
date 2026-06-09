# Eval Plan

The goal is to check three things: does the tool score fit correctly, do the emails cite real sources, and can a human tell the two emails apart?

---

## What you need first — the gold set

Create a file called `gold.jsonl` in this folder. Each line is one test case — a sender, a target company, the buyer persona, and your gut-feel label for how good a fit the target is.

Example line:
```
{"sender":"artisan.co","target":"supabase.com","persona":"VP of Sales","fit_label":"strong"}
{"sender":"artisan.co","target":"doordash.com","persona":"VP of Sales","fit_label":"weak"}
{"sender":"artisan.co","target":"revolut.com","persona":"VP of Sales","fit_label":"partial"}
```

Aim for 6 cases: 2 strong, 2 partial, 2 weak. Label by your own judgment — that's the ground truth.

---

## What to check per run

For each target in your gold set, run Mode 2 in the app. Then check three things:

**1. Ungrounded claims (hallucination)**
Look at the email claim map. Any claim in red = ungrounded (Claude cited a source that doesn't exist). Count how many red claims you see. Target is zero.

**2. Fit score**
Note the pain_fit number on screen. Compare to your label:
- Strong fit → should score 65 or above
- Partial fit → should score 40–65
- Weak fit → should score 40 or below

**3. Email quality**
Read both emails. Ask yourself:
- Can you tell which is pain-led and which is trigger-led without reading the label?
- Does the trigger-led email open with a specific event or news item about that company?
- Is at least one claim specific to that company (not just their name)?

---

## Results

Fill in after running all 6 cases.

### Fit calibration

| Target | Your label | pain_fit score | Match? |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

Correctly matched: __ / 6

### Hallucination

Total email claims checked: __
Ungrounded (red) claims found: __
Hallucination rate: __%

### Email quality

| Target | Pain vs trigger distinguishable? | Trigger opens with named event? | Company-specific claim? |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

Emails structurally different: __ / 6

---

## Summary table (for eval_report.md)

| Metric | Result |
|---|---|
| Fit accuracy (within-band) | __ / 6 |
| Hallucination rate | __% |
| Emails structurally different | __ / 6 |
| Avg pain_fit — strong-fit targets | __ |
| Avg pain_fit — weak-fit targets | __ |
