// ---------- helpers ----------
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};
const esc = (s) => (s || "").replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const stripEmDash = s => (s || "").replace(/—/g, " - ");
const safe = s => esc(stripEmDash(s));

let SENDER_PROFILE = null;
let _lastQueries = [];

// ---------- health ----------
fetch("/api/health").then(r => r.json()).then(h => {
  const pill = $("#status-pill");
  if (h.mock_mode) { pill.textContent = "mock mode"; pill.classList.add("mock"); }
  else { pill.textContent = `live · ${h.model}`; pill.classList.add("live"); }
}).catch(() => { $("#status-pill").textContent = "offline"; });

// ---------- tabs ----------
$$(".tab").forEach(t => t.addEventListener("click", () => {
  $$(".tab").forEach(x => x.classList.remove("active"));
  $$(".panel").forEach(x => x.classList.remove("active"));
  t.classList.add("active");
  $("#panel-" + t.dataset.tab).classList.add("active");
}));

// ---------- NDJSON stream reader ----------
async function streamPost(url, body, onEvent) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
  if (buf.trim()) onEvent(JSON.parse(buf.trim()));
}

// ---------- timeline rendering ----------
function tlStep(container, ev) {
  let node = $(`[data-step="${ev.step}"]`, container);
  if (!node) {
    node = el("div", "tl-step");
    node.dataset.step = ev.step;
    node.appendChild(el("div", "tl-icon"));
    const body = el("div", "tl-body");
    body.appendChild(el("div", "tl-name", esc(ev.step)));
    body.appendChild(el("div", "tl-detail"));
    node.appendChild(body);
    container.appendChild(node);
  }
  node.className = "tl-step " + ev.status;
  const detailEl = $(".tl-detail", node);
  if (detailEl) detailEl.textContent = ev.detail || "";
}

function _shortModel(id) {
  const m = (id || "").match(/sonnet|haiku|opus/i);
  return m ? m[0].toLowerCase() : id;
}

// Per-million-token pricing by model ID prefix.
const _MODEL_PRICING = [
  { prefix: "claude-haiku",   in: 0.80, out: 4.00  },
  { prefix: "claude-sonnet",  in: 3.00, out: 15.00 },
  { prefix: "claude-opus",    in: 15.0, out: 75.00 },
];

function _modelPrice(modelId) {
  const id = (modelId || "").toLowerCase();
  return _MODEL_PRICING.find(p => id.startsWith(p.prefix))
      || { in: 3.00, out: 15.00 };
}

function _taskCost(t) {
  if (!t || (t.model || "") === "mock") return 0;
  const p = _modelPrice(t.model);
  const regular = Math.max(0, (t.input || 0) - (t.cache_write || 0) - (t.cache_read || 0));
  return (regular            * p.in
        + (t.cache_write||0) * p.in  * 1.25
        + (t.cache_read ||0) * p.in  * 0.10
        + (t.output      ||0) * p.out) / 1_000_000;
}

function usage(u) {
  if (!u) return;

  // Cost
  let cost = 0;
  if (u.by_task && Object.keys(u.by_task).length) {
    cost = Object.values(u.by_task).reduce((s, t) => s + _taskCost(t), 0);
  }
  const costStr = cost >= 0.001 ? `~$${cost.toFixed(3)}` : "";

  // Tokens + calls
  const parts = [`tokens · ${u.input_tokens} in / ${u.output_tokens} out · ${u.calls} calls`];

  if (u.by_model) {
    const models = Object.entries(u.by_model)
      .filter(([m]) => m !== "mock")
      .map(([m, n]) => `${_shortModel(m)}×${n}`)
      .join(" ");
    if (models) parts.push(models);
  }

  if (u.elapsed_ms != null) {
    const ms = u.elapsed_ms;
    parts.push(ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`);
  }

  if (costStr) parts.push(costStr);

  const badge = $("#usage-badge");
  if (badge) badge.textContent = parts.join(" · ");

  const usageDetail = $("#usage-detail");
  if (usageDetail) usageDetail.textContent = "";
}

// ============================================================
// MODE 1 — SENDER
// ============================================================
$("#run-sender").addEventListener("click", async () => {
  const url = $("#sender-url").value.trim();
  if (!url) return;
  const tl = $("#sender-timeline"); tl.innerHTML = "";
  _lastQueries = [];
  const out = $("#sender-output");
  out.innerHTML = `<div class="empty"><span class="spin">◆</span> agent working…</div>`;
  $("#run-sender").disabled = true;

  try {
    await streamPost("/api/sender",
      { url, max_pages: parseInt($("#sender-pages").value) || 6 },
      ev => {
        tlStep(tl, ev);
        if (ev.step === "retrieve" && ev.status === "done" && ev.data?.queries) {
          _lastQueries = ev.data.queries;
          renderResearchPlan(ev.data.queries, ev.data.customer_queries || []);
        }
        if (ev.step === "synthesize" && ev.status === "done") {
          SENDER_PROFILE = ev.data.profile;
          renderProfile(SENDER_PROFILE);
          usage(ev.data.usage);
          markICPReady();
        }
        // Customer box updates on EVERY customers-done event (site customers
        // arrive before synthesis; web-enriched customers arrive after).
        if (ev.step === "customers" && ev.status === "done" && ev.data?.named_customers?.length) {
          if (SENDER_PROFILE) SENDER_PROFILE.named_customers = ev.data.named_customers;
          renderCustomerBox(ev.data.named_customers, SENDER_PROFILE?.customer_pattern);
        }
        if (ev.step === "error") {
          out.innerHTML = `<div class="empty" style="color:var(--rust)">⚠ ${esc(ev.detail)}</div>`;
        }
      });
    $$(".tl-step.running", tl).forEach(n => { n.className = "tl-step done"; });
    tlStep(tl, { step: "done", status: "done", detail: "Agent finished" });
  } catch (e) {
    out.innerHTML = `<div class="empty" style="color:var(--rust)">⚠ ${esc(e.message)}</div>`;
  } finally {
    $("#run-sender").disabled = false;
  }
});

function renderProfile(p) {
  const out = $("#sender-output");
  // Preserve the research plan card across the re-render.
  const planCard = out.querySelector(".research-plan-card");
  out.innerHTML = "";
  if (planCard) {
    out.appendChild(planCard);
    const ph = planCard.querySelector("h3");
    if (ph && !ph.querySelector(".rp-done")) {
      const badge = el("span", "conf-badge conf-high rp-done", "done");
      ph.appendChild(badge);
    }
  }
  const icp = p.icp;

  // ── Low-confidence banner ─────────────────────────────────────────────────
  if (p.confidence === "low" && p.data_gaps?.length) {
    const banner = el("div", "conf-banner");
    banner.innerHTML = `<span class="conf-icon">!</span>
      <div><b>Low confidence</b> — limited evidence on:
        <span>${p.data_gaps.map(esc).join(", ")}</span>
      </div>`;
    out.appendChild(banner);
  }

  // ── 1. Value proposition ──────────────────────────────────────────────────
  const confCls = p.confidence === "low" ? "conf-low" : "conf-high";
  const vp = el("div", "card");
  vp.innerHTML = `
    <div class="report-header">
      <span class="report-company">${esc(p.company)}</span>
      <span class="conf-badge ${confCls}">${esc(p.confidence)} confidence</span>
      <span class="report-grounding">grounding ${Math.round((p.grounding_coverage||1)*100)}%</span>
    </div>
    <div class="vp">${safe(p.value_proposition)||'<em class="empty-state">Insufficient evidence.</em>'}</div>
    <div class="oneliner">${safe(p.one_liner)}</div>
    ${p.capabilities?.length
      ? `<ul class="caps rpt-caps">${p.capabilities.map(x=>`<li>${esc(x)}</li>`).join("")}</ul>`
      : ""}`;
  out.appendChild(vp);

  // ── 2. Customer evidence — WHO ACTUALLY BUYS ─────────────────────────────
  // renderCustomerBox is also called by the customers-done event handler so
  // the box updates live. Only call here if customers are already present.
  if (p.named_customers?.length) {
    renderCustomerBox(p.named_customers, p.customer_pattern);
  }

  // ── 3. ICP — derived from customer pattern ────────────────────────────────
  const icpCard = el("div", "card");
  icpCard.innerHTML = `<h3>Ideal customer profile
    <span class="conf-badge conf-high">evidence-backed</span>
  </h3>`;

  // Industries + Geographies row
  const row1 = el("div", "icp-grid");
  row1.appendChild(block("Industries",
    icp.industries?.length
      ? `<div class="chips">${icp.industries.map(i=>`<span class="chip">${esc(i)}</span>`).join("")}</div>`
      : `<span class="empty-state">Not identified.</span>`));
  row1.appendChild(block("Geographies",
    icp.geographies?.length
      ? `<div class="chips">${icp.geographies.map(i=>`<span class="chip">${esc(i)}</span>`).join("")}</div>`
      : `<span class="empty-state">Not identified.</span>`));
  icpCard.appendChild(row1);

  // Size bands full-width
  if (icp.size_bands?.length) {
    const sb = el("div", "rpt-section");
    sb.innerHTML = `<h4 class="rpt-section-title">Company size</h4>
      <div class="stack">${icp.size_bands.map(s=>
        `<div class="stack-item"><b>${esc(s.label)}</b><span>${esc(s.rationale)}</span></div>`
      ).join("")}</div>`;
    icpCard.appendChild(sb);
  }

  // Triggers — each with inline citation
  const trigSec = el("div", "rpt-section");
  trigSec.innerHTML = `<h4 class="rpt-section-title">Buying triggers <span class="rpt-hint">a target should show at least one</span></h4>`;
  if (icp.triggers?.length) {
    const tlist = el("div", "trigger-list");
    icp.triggers.forEach(t => {
      const chips = (t.evidence_ids||[]).filter(Boolean).map(citeChip).join("");
      const row = el("div", "trigger-row");
      row.innerHTML = `<span class="trigger-dot"></span>
        <span class="trigger-name">${esc(t.name)}</span>
        ${t.description ? `<span class="trigger-desc">${safe(t.description)}</span>` : ""}
        ${chips ? `<span class="cites">${chips}</span>` : ""}`;
      tlist.appendChild(row);
    });
    trigSec.appendChild(tlist);
  } else {
    trigSec.innerHTML += `<span class="empty-state">No triggers identified.</span>`;
  }
  icpCard.appendChild(trigSec);

  // Buyers — each with evidence + seniority
  const buyerSec = el("div", "rpt-section");
  buyerSec.innerHTML = `<h4 class="rpt-section-title">Buyer personas <span class="rpt-hint">who approves the deal</span></h4>`;
  if (icp.buyers?.length) {
    const blist = el("div", "stack");
    icp.buyers.forEach(b => {
      const chips = (b.evidence_ids||[]).filter(Boolean).map(citeChip).join("");
      blist.innerHTML += `<div class="stack-item buyer-item">
        <b>${esc(b.role)}</b>
        <span class="buyer-seniority">${esc(b.seniority)}</span>
        <span>${safe(b.why)}${chips ? ` <span class="cites">${chips}</span>` : ""}</span>
      </div>`;
    });
    buyerSec.appendChild(blist);
  } else {
    buyerSec.innerHTML += `<span class="empty-state">Not identified.</span>`;
  }
  icpCard.appendChild(buyerSec);

  // Anti-patterns
  if (icp.anti_patterns?.length) {
    const apSec = el("div", "rpt-section");
    apSec.innerHTML = `<h4 class="rpt-section-title">Not a fit <span class="rpt-hint">disqualifiers</span></h4>
      <div class="stack">${icp.anti_patterns.map(a=>
        `<div class="stack-item ap-item"><span class="ap-x">✕</span><span>${esc(a)}</span></div>`
      ).join("")}</div>`;
    icpCard.appendChild(apSec);
  }

  // Qualifying questions — the go/no-go checklist
  if (icp.qualifying_questions?.length) {
    const qqSec = el("div", "rpt-section qq-section");
    qqSec.innerHTML = `<h4 class="rpt-section-title">Qualifying questions
      <span class="rpt-hint">Mode 2 answers each of these</span></h4>`;
    const ol = el("ol", "qq-list");
    icp.qualifying_questions.forEach(q => ol.appendChild(el("li", "", esc(q))));
    qqSec.appendChild(ol);
    icpCard.appendChild(qqSec);
  }

  out.appendChild(icpCard);

  // ── 4. Evidence sources ───────────────────────────────────────────────────
  if (p.evidence?.length) out.appendChild(evidenceCard(p.evidence, "Evidence sources"));

  wireCiteHover();
}

function renderCustomerBox(customers, pattern) {
  if (!customers?.length) return;
  const out = $("#sender-output");

  // Always remove and re-create so updates (web enrichment) refresh cleanly.
  const existing = out.querySelector(".customer-box");
  if (existing) existing.remove();

  const box = el("div", "card customer-box");
  box.innerHTML = `<h3>Real customers
    <span class="conf-badge conf-high">${customers.length} found</span>
  </h3>
  <div class="section-label">Found at runtime from the site and web search — the ICP is built from their shared pattern</div>`;

  const grid = el("div", "cust-grid");
  customers.forEach(c => {
    const item = el("div", "cust-card");
    // Name — linked to source if available
    const nameHtml = c.source_url
      ? `<a class="cust-name" href="${esc(c.source_url)}" target="_blank" rel="noopener">${esc(c.name)}</a>`
      : `<span class="cust-name">${esc(c.name)}</span>`;
    // Badges: industry, size, provenance
    const provLabel = c.provenance === "web" ? "web ↗"
                    : c.provenance === "case-study" ? "case study" : "site";
    const badges = [
      c.industry  ? `<span class="cust-badge">${esc(c.industry)}</span>` : "",
      c.size_hint ? `<span class="cust-badge cust-badge-size">${esc(c.size_hint)}</span>` : "",
      `<span class="cust-badge cust-badge-prov">${esc(provLabel)}</span>`,
    ].filter(Boolean).join("");
    const rawDesc = (c.description || "").replace(/^#+\s*/, "").trim();
    const desc  = rawDesc ? `<div class="cust-desc">${esc(rawDesc)}</div>` : "";
    const cites = (c.evidence_ids||[]).filter(Boolean).map(citeChip).join("");
    item.innerHTML = `${nameHtml}
      <div class="cust-badges">${badges}</div>
      ${desc}
      ${cites ? `<div class="cust-cites">${cites}</div>` : ""}`;
    grid.appendChild(item);
  });
  box.appendChild(grid);

  if (pattern) {
    const pat = el("div", "cust-pattern");
    pat.innerHTML = `<span class="pat-icon">Pattern</span> ${esc(pattern)}`;
    box.appendChild(pat);
  }

  // Insert right after the value-prop card so it's the second card always.
  const vpCard = out.querySelector(".card:not(.research-plan-card)");
  if (vpCard) {
    vpCard.insertAdjacentElement("afterend", box);
  } else {
    out.appendChild(box);
  }
  wireCiteHover();
}

function block(title, inner) {
  const b = el("div", "icp-block");
  b.innerHTML = `<h4>${esc(title)}</h4>${inner}`;
  return b;
}

function renderResearchPlan(queries, customerQueries) {
  if (!queries?.length) return;
  const out = $("#sender-output");
  const old = out.querySelector(".research-plan-card");
  if (!old) {
    const placeholder = out.querySelector(".empty");
    if (placeholder) placeholder.remove();
  } else {
    old.remove();
  }
  const card = el("div", "card research-plan-card");
  card.innerHTML = `<h3>Research plan</h3>`;

  // ICP queries
  const icpLabel = el("div", "section-label");
  icpLabel.textContent = `${queries.length} ICP queries`;
  card.appendChild(icpLabel);
  const ul = el("ul", "rp-queries");
  queries.forEach(q => {
    const li = document.createElement("li");
    li.className = "rp-query";
    li.textContent = q;
    ul.appendChild(li);
  });
  card.appendChild(ul);

  // Customer queries (separate pool)
  if (customerQueries?.length) {
    const cl = el("div", "section-label");
    cl.style.marginTop = "10px";
    cl.textContent = `${customerQueries.length} customer queries`;
    card.appendChild(cl);
    const cul = el("ul", "rp-queries");
    customerQueries.forEach(q => {
      const li = document.createElement("li");
      li.className = "rp-query rp-query-cust";
      li.textContent = q;
      cul.appendChild(li);
    });
    card.appendChild(cul);
  }

  out.appendChild(card);
  out.appendChild(el("div", "empty synth-hint", `<span class="spin">◆</span> synthesizing ICP…`));
}

function markICPReady() {
  const s = $("#icp-status");
  if (s) s.classList.add("ready");
  const statusText = $("#icp-status-text");
  if (statusText) statusText.textContent =
    `ICP loaded from ${SENDER_PROFILE.company} — ready to evaluate targets.`;
  const runTarget = $("#run-target");
  if (runTarget) runTarget.disabled = false;
}

// ============================================================
// MODE 2 — TARGET
// ============================================================
$("#run-target").addEventListener("click", async () => {
  if (!SENDER_PROFILE) return;
  const url = $("#target-url").value.trim();
  if (!url) { $("#target-url").focus(); return; }
  const tl = $("#target-timeline"); tl.innerHTML = "";
  const out = $("#target-output");
  out.innerHTML = `<div class="empty"><span class="spin">◆</span> agent working…</div>`;
  $("#run-target").disabled = true;

  try {
    await streamPost("/api/target", {
      url,
      persona_role: $("#persona-role").value.trim(),
      persona_seniority: $("#persona-seniority").value.trim(),
      sender: SENDER_PROFILE,
      max_pages: 6,
    }, ev => {
      tlStep(tl, ev);
      if (ev.step === "draft" && ev.status === "done") {
        renderReport(ev.data.report);
        usage(ev.data.usage);
      }
      if (ev.step === "error") {
        out.innerHTML = `<div class="empty" style="color:var(--rust)">⚠ ${esc(ev.detail)}</div>`;
      }
    });
    $$(".tl-step.running", tl).forEach(n => { n.className = "tl-step done"; });
    tlStep(tl, { step: "done", status: "done", detail: "Agent finished" });
  } catch (e) {
    out.innerHTML = `<div class="empty" style="color:var(--rust)">⚠ ${esc(e.message)}</div>`;
  } finally {
    $("#run-target").disabled = false;
  }
});

function renderDimensions(fit) {
  // Structured dimension grid when new-format data is present; fallback to plain list.
  if (!fit.dimensions || !Object.keys(fit.dimensions).length) {
    return `<ul class="dim-notes">${(fit.dimension_notes || []).map(d => `<li>${esc(d)}</li>`).join("")}</ul>`;
  }
  const ORDER = ["industry", "trigger", "buyer", "size", "geography"];
  const rows = ORDER.map(name => {
    const dim = fit.dimensions[name];
    if (!dim) return "";
    const cls  = `dim-badge dim-badge-${esc(dim.judgment)}`;
    const cites = (dim.evidence_ids || []).map(citeChip).join("");
    return `<div class="dim-row">
      <span class="dim-label">${esc(name)}</span>
      <span class="${cls}">${esc(dim.judgment)}</span>
      <span class="dim-note">${esc(dim.note)}${cites ? ` <span class="cites">${cites}</span>` : ""}</span>
    </div>`;
  }).join("");
  const aps = (fit.anti_patterns_hit || []).map(ap =>
    `<div class="dim-row">
      <span class="dim-label">anti-pattern</span>
      <span class="dim-badge dim-badge-absent">hit</span>
      <span class="dim-note" style="color:var(--rust)">${esc(ap.name)}</span>
    </div>`
  ).join("");
  return `<div class="dim-grid">${rows}${aps}</div>`;
}

function renderReport(r) {
  const out = $("#target-output");
  out.innerHTML = "";

  const fit = r.fit;

  // ── Headline verdict card (LLM-judged pain fit + worth reaching out) ──────
  const painFit = typeof fit.pain_fit === "number" ? fit.pain_fit : 0;
  const col = painFit >= 70 ? "var(--sage)" : painFit >= 40 ? "var(--amber)" : "var(--rust)";
  const deg = Math.round((painFit / 100) * 360);
  const worthLabel = fit.worth_reaching_out === "yes" ? "✓ Worth reaching out"
                   : fit.worth_reaching_out === "maybe" ? "~ Maybe"
                   : fit.worth_reaching_out === "no" ? "✕ Not recommended"
                   : "";
  const worthCls = fit.worth_reaching_out === "yes" ? "worth-yes"
                 : fit.worth_reaching_out === "maybe" ? "worth-maybe" : "worth-no";
  const allAbsent = fit.dimensions && Object.values(fit.dimensions).every(d => d.judgment === "absent");
  const fc = el("div", "card");
  fc.innerHTML = `<h3>ICP fit · ${esc(r.target_company)}</h3>
    ${allAbsent ? `<div class="evidence-warn">No evidence retrieved — verdict is profile-only (LLM world knowledge, not fetched snippets). Treat as exploratory.</div>` : ""}
    <div class="fit-head">
      <div class="gauge" style="background:conic-gradient(${col} ${deg}deg, var(--ink-3) 0)">
        <div style="position:absolute;inset:7px;border-radius:50%;background:var(--ink-2)"></div>
        <span style="color:${col};position:relative">${painFit}</span>
        <small>/ 100</small>
      </div>
      <div>
        ${worthLabel ? `<div class="worth-badge ${worthCls}">${esc(worthLabel)}</div>` : ""}
        <div style="color:var(--muted);font-size:13.5px;margin-top:6px;max-width:46ch">${safe(fit.pain_fit_rationale || fit.rationale)}</div>
        ${fit.worth_rationale ? `<div style="color:var(--muted);font-size:12.5px;margin-top:4px;font-style:italic">${safe(fit.worth_rationale)}</div>` : ""}
        ${fit.outreach_angle ? `
        <div class="outreach-angle">
          <span class="angle-label">Outreach angle</span>
          <span class="angle-text">${safe(fit.outreach_angle)}</span>
        </div>` : ""}
      </div>
    </div>

    ${fit.qualification?.length ? `
    <div class="rpt-section">
      <h4 class="rpt-section-title">Qualifying questions <span class="rpt-hint">from Mode 1 ICP</span></h4>
      <div class="qual-list">
        ${fit.qualification.map(q => {
          const ac = q.answer === "yes" ? "qual-yes"
                   : q.answer === "partial" ? "qual-partial"
                   : q.answer === "no" ? "qual-no" : "qual-unknown";
          const cites = (q.evidence_ids || []).map(citeChip).join("");
          return `<div class="qual-row">
            <span class="qual-badge ${ac}">${esc(q.answer)}</span>
            <div class="qual-body">
              <span class="qual-q">${esc(q.question)}</span>
              ${q.rationale ? `<span class="qual-rat">${esc(q.rationale)}${cites ? ` <span class="cites">${cites}</span>` : ""}</span>` : ""}
            </div>
          </div>`;
        }).join("")}
      </div>
    </div>` : ""}

    <details class="dim-details">
      <summary>Transparency check <span class="rpt-hint">secondary formula · not the headline</span><span class="dim-check-score">${fit.score}/100</span></summary>
      ${renderDimensions(fit)}
    </details>`;
  out.appendChild(fc);

  // Signals
  if (r.signals?.length) {
    const sc = el("div", "card");
    sc.innerHTML = `<h3>Buying signals</h3>`;
    r.signals.forEach(s => {
      const row = el("div", "signal");
      row.innerHTML = `<div class="sig-type">${esc(s.type)}</div>
        <div class="sig-body">${safe(s.summary)}
        ${s.recency ? `<span class="recency">${esc(s.recency)}</span>` : ""}
        <span class="cites" style="margin-left:6px">${(s.evidence_ids || []).map(citeChip).join("")}</span></div>`;
      sc.appendChild(row);
    });
    out.appendChild(sc);
  }

  // Emails
  const ec = el("div", "card");
  ec.innerHTML = `<h3>Outbound drafts · two angles</h3>`;
  const grid = el("div", "emails");
  r.emails.forEach(em => {
    const e = el("div", "email" + (em.angle.includes("trigger") ? " trigger" : ""));
    const claimsHtml = em.claims?.length
      ? em.claims.map(c => {
          if (!c.grounded || !c.evidence_ids.length)
            return `<div class="claim ungrounded">${safe(c.text)} <span class="flag">unverified</span></div>`;
          return `<div class="claim">${safe(c.text)}<span class="cites">${c.evidence_ids.map(citeChip).join("")}</span></div>`;
        }).join("")
      : `<div class="claim-no-evidence">No source retrieved — profile-only draft. Verify before sending.</div>`;
    e.innerHTML = `<div class="email-angle">${safe(em.angle_label)}</div>
      <div class="email-subject">${safe(em.subject)}</div>
      <div class="email-body">${safe(em.body)}</div>
      <div class="email-claims"><div class="lbl">Claim map</div>${claimsHtml}</div>`;
    grid.appendChild(e);
  });
  ec.appendChild(grid);
  out.appendChild(ec);

  // Evidence panel
  if (r.evidence?.length) out.appendChild(evidenceCard(r.evidence, "Evidence panel", "cited evidence only · IDs non-sequential by design"));

  wireCiteHover();
}

function citeChip(id) {
  return `<a class="cite" data-cite="${esc(id)}" href="#ev-${esc(id)}">${esc(id)}</a>`;
}

function evidenceCard(evidence, title, subtitle) {
  const card = el("div", "card");
  card.innerHTML = `<h3>${esc(title)}</h3>
    <div class="section-label">${subtitle || "every claim above traces to a source below"}</div>`;
  evidence.forEach(e => {
    const item = el("div", "evidence-item");
    item.id = "ev-" + e.id;
    item.dataset.ev = e.id;
    item.innerHTML = `<div class="ev-head">
        <span class="ev-id">${esc(e.id)}</span>
        <a class="ev-url" href="${esc(e.url)}" target="_blank" rel="noopener">${esc(e.url)}</a>
      </div>
      <div class="ev-snippet">"${esc(e.snippet)}"</div>`;
    card.appendChild(item);
  });
  return card;
}

// hover a citation → highlight its evidence card (and vice-versa)
function wireCiteHover() {
  $$(".cite").forEach(c => {
    const id = c.dataset.cite;
    const target = $("#ev-" + id);
    c.addEventListener("mouseenter", () => { target?.classList.add("hot"); c.classList.add("hot"); });
    c.addEventListener("mouseleave", () => { target?.classList.remove("hot"); c.classList.remove("hot"); });
  });
}
