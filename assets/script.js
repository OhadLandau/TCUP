/* TCUP front end: upload, a sample column that fills as samples are analyzed, and a detail view per sample. */
(() => {
  "use strict";

  const SRC = {
    "Primary tumor": { color: "var(--src-primary)", label: "Primary tumor" },
    "Metastatic": { color: "var(--src-met)", label: "Metastasis" },
    "Healthy": { color: "var(--src-healthy)", label: "Normal tissue" },
  };
  const STAGES = ["Mapping genes to the model", "Embedding with SNN and CAE", "Running the classifier ensemble", "Scoring 45 tissue classes", "Ranking explanatory genes"];
  const CONCURRENCY = 2;

  const state = { config: null, samples: [], selected: null, filter: "", showAll: false, running: 0, queue: [] };
  let uid = 0;

  const $ = (s, el = document) => el.querySelector(s);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (x, d = 1) => `${(100 * x).toFixed(d)}%`;
  const fmtP = (p) => (p < 0.001 ? p.toExponential(1) : p.toFixed(3));
  const stars = (q) => (q < 0.001 ? "***" : q < 0.01 ? "**" : q < 0.05 ? "*" : "");
  const sampleType = () => document.querySelector('input[name="stype"]:checked').value;

  // ---------------- config and landing ----------------
  async function loadConfig() {
    try {
      const r = await fetch("/api/config");
      state.config = await r.json();
    } catch (e) {
      return;
    }
    const m = state.config.test_metrics;
    const set = (k, v) => { const el = document.querySelector(`[data-stat="${k}"]`); if (el) el.textContent = v; };
    set("classes", state.config.classes.length);
    set("genes", state.config.n_features.toLocaleString());
    set("acc", pct(m.all_accuracy));
    set("met", pct(m.metastatic_accuracy));
    const cards = [
      ["All test samples", m.all_accuracy, m.all_n, `Top-3 ${pct(m.all_top3)}`, "var(--navy)"],
      ["Primary tumors (TCGA)", m.tcga_accuracy, m.tcga_n, `Top-3 ${pct(m.tcga_top3)}`, SRC["Primary tumor"].color],
      ["Normal tissue (GTEx)", m.gtex_accuracy, m.gtex_n, `Top-3 ${pct(m.gtex_top3)}`, SRC.Healthy.color],
      ["Metastases (BCGSC)", m.metastatic_accuracy, m.metastatic_n, `Top-3 ${pct(m.metastatic_top3)}`, SRC.Metastatic.color],
    ];
    $("#perf-grid").innerHTML = cards.map(([t, a, n, extra, c]) => `
      <div class="perf-card"><h3><span class="dot" style="background:${c}"></span>${t}</h3>
        <div class="perf-big">${pct(a)}</div><div class="perf-meta">accuracy on ${n.toLocaleString()} samples · ${extra}</div>
        <div class="perf-bar"><span style="width:${100 * a}%;background:${c}"></span></div></div>`).join("");
    $("#perf-sub").textContent = `${state.config.model}, on held-out samples never used for training or model selection. Metastatic accuracy rests on 120 samples, so it is less precise than the others.`;
  }

  function showView(name) {
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("is-active", v.id === name));
    document.body.classList.toggle("in-workspace", name === "workspace");
    window.scrollTo({ top: 0 });
  }

  // ---------------- upload ----------------
  function setMsg(text, isError = false) {
    const el = $("#upload-msg");
    el.textContent = text;
    el.classList.toggle("is-error", isError);
  }

  async function uploadBlob(blob, name) {
    setMsg(`Reading ${name}…`);
    let data;
    try {
      const r = await fetch("/api/upload", { method: "POST", body: blob, headers: { "X-Filename": name } });
      data = await r.json();
      if (!r.ok) throw new Error(data.error || "Upload failed.");
    } catch (e) {
      setMsg(e.message, true);
      if (state.samples.length) toast(e.message);
      return;
    }
    setMsg("");
    const type = sampleType();
    const added = data.samples.map((id, i) => ({ key: ++uid, id, batch: data.batch, index: i, file: name, type, status: "queued", result: null, error: null, started: 0 }));
    state.samples.push(...added);
    state.queue.push(...added);
    if (state.selected === null) state.selected = added[0].key;
    showView("workspace");
    renderList();
    renderDetail();
    if (data.genes_found < data.n_features) {
      toast(`${data.genes_found.toLocaleString()} of ${data.n_features.toLocaleString()} model genes found in ${name}. The rest are filled with training medians.`);
    }
    pump();
  }

  function pump() {
    while (state.running < CONCURRENCY && state.queue.length) {
      const s = state.queue.shift();
      state.running++;
      analyze(s).finally(() => { state.running--; pump(); });
    }
    renderProgress();
  }

  async function analyze(s) {
    s.status = "running";
    s.started = performance.now();
    renderList();
    if (s.key === state.selected) renderDetail();
    try {
      const r = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ batch: s.batch, index: s.index, sample_type: s.type }) });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || "Analysis failed.");
      // keep the annotating state on screen briefly, so fast results do not flash
      const wait = 650 - (performance.now() - s.started);
      if (wait > 0) await new Promise((res) => setTimeout(res, wait));
      s.result = data;
      s.status = "done";
    } catch (e) {
      s.status = "error";
      s.error = e.message;
    }
    renderList();
    renderProgress();
    if (s.key === state.selected) renderDetail();
  }

  // ---------------- sidebar ----------------
  function renderProgress() {
    const n = state.samples.length;
    const done = state.samples.filter((s) => s.status === "done" || s.status === "error").length;
    $("#sample-count").textContent = n;
    $("#progress-bar").style.width = n ? `${(100 * done) / n}%` : "0";
    $("#side-progress").textContent = n === 0 ? "" : done < n ? `Annotating ${done + 1} of ${n}…` : `All ${n} analyzed`;
    $("#export-csv").disabled = !state.samples.some((s) => s.status === "done");
  }

  function sampleRow(s) {
    let sub;
    if (s.status === "queued") sub = `<span class="dot dot-queued"></span>Queued`;
    else if (s.status === "running") sub = `<span class="eq" aria-hidden="true"><i></i><i></i><i></i><i></i></span>Annotating…`;
    else if (s.status === "error") sub = `<span class="dot dot-error"></span>${esc(s.error)}`;
    else {
      const p = s.result.prediction;
      sub = `<span class="dot" style="background:${SRC[p.source].color}"></span><span class="code">${esc(p.code)}</span> ${esc(SRC[p.source].label)}`;
    }
    const pp = s.status === "done" ? s.result.prediction.probability : 0;
    const prob = s.status === "done" ? `<span class="sample-prob">${pp >= 0.9995 ? "&gt;99.9%" : pct(pp)}</span>` : "";
    return `<li class="sample${s.key === state.selected ? " is-selected" : ""}" role="option" aria-selected="${s.key === state.selected}" data-key="${s.key}" tabindex="-1">
      <span class="sample-id" title="${esc(s.id)}">${esc(s.id)}</span>${prob}<span class="sample-sub">${sub}</span></li>`;
  }

  function visibleSamples() {
    const f = state.filter.toLowerCase();
    return state.samples.filter((s) => !f || s.id.toLowerCase().includes(f) || (s.result && `${s.result.prediction.code} ${s.result.prediction.name}`.toLowerCase().includes(f)));
  }

  function renderList() {
    $("#sample-list").innerHTML = visibleSamples().map(sampleRow).join("");
    renderProgress();
  }

  function select(key) {
    state.selected = key;
    state.showAll = false;
    renderList();
    renderDetail();
    const el = document.querySelector(`.sample[data-key="${key}"]`);
    if (el) el.scrollIntoView({ block: "nearest", inline: "nearest" });
  }

  // ---------------- detail ----------------
  let stageTimer = null;

  function renderDetail() {
    clearInterval(stageTimer);
    const s = state.samples.find((x) => x.key === state.selected);
    const el = $("#detail");
    if (!s) { el.innerHTML = `<div class="empty">Select a sample.</div>`; return; }
    if (s.status === "queued" || s.status === "running") { el.innerHTML = annotating(s); animateStages(s); return; }
    if (s.status === "error") {
      el.innerHTML = `<div class="detail-id">${esc(s.id)}</div><div class="notice notice-crit">${icon("x")}<div><b>This sample could not be analyzed.</b><br>${esc(s.error)}</div></div>`;
      return;
    }
    el.innerHTML = resultView(s);
    el.firstElementChild.classList.add("fade-in");
    bindDetail(s);
  }

  function annotating(s) {
    const dots = Array.from({ length: 11 }, (_, i) => {
      const x = 12 + i * 19.5, y1 = 45 + 28 * Math.sin(i * 0.75), y2 = 45 - 28 * Math.sin(i * 0.75);
      return `<line x1="${x}" y1="${y1}" x2="${x}" y2="${y2}" stroke="var(--line-strong)" stroke-width="1.5"/>
        <circle cx="${x}" cy="${y1}" r="5" fill="var(--navy)" style="animation-delay:${i * 0.12}s"/>
        <circle cx="${x}" cy="${y2}" r="5" fill="var(--src-met)" style="animation-delay:${i * 0.12 + 0.6}s"/>`;
    }).join("");
    return `<div class="annot"><div class="annot-inner">
      <svg class="helix" viewBox="0 0 220 90" aria-hidden="true">${dots}</svg>
      <h3>${s.status === "queued" ? "Waiting in the queue" : "Annotating sample"}</h3>
      <div class="detail-id">${esc(s.id)}</div>
      <ul class="stage-list" id="stage-list">${STAGES.map((t) => `<li><i></i>${t}</li>`).join("")}</ul>
    </div></div>`;
  }

  function animateStages(s) {
    const items = [...document.querySelectorAll("#stage-list li")];
    const tick = () => {
      if (s.status !== "running") return;
      const k = Math.min(items.length - 1, Math.floor((performance.now() - s.started) / 160));
      items.forEach((li, i) => {
        li.className = i < k ? "done" : i === k ? "now" : "";
        li.firstElementChild.textContent = i < k ? "✓" : "";
      });
    };
    tick();
    stageTimer = setInterval(tick, 120);
  }

  function gauge(p, color) {
    const r = 54, c = 2 * Math.PI * r, off = c * (1 - p);
    return `<svg class="gauge" viewBox="0 0 132 132" role="img" aria-label="Probability ${pct(p)}">
      <circle cx="66" cy="66" r="${r}" fill="none" stroke="var(--surface-3)" stroke-width="10"/>
      <circle cx="66" cy="66" r="${r}" fill="none" stroke="${color}" stroke-width="10" stroke-linecap="round"
        stroke-dasharray="${c}" stroke-dashoffset="${off}" transform="rotate(-90 66 66)"/>
      <text x="66" y="68" text-anchor="middle" font-size="${p >= 0.9995 ? 24 : 28}" font-weight="600" fill="var(--ink)">${p >= 0.9995 ? "&gt;99.9" : (100 * p).toFixed(1)}%</text>
      <text x="66" y="88" text-anchor="middle" font-size="11" fill="var(--ink-3)" font-family="Inter">probability</text></svg>`;
  }

  function confidence(p) {
    if (p >= 0.9) return ["notice-ok", "check", "High-confidence call."];
    if (p >= 0.6) return ["notice-warn", "warn", "Moderate confidence. Consider the runner-up classes."];
    return ["notice-crit", "warn", "Low confidence. The profile does not match one class well; interpret with care."];
  }

  function resultView(s) {
    const R = s.result, P = R.prediction, src = SRC[P.source];
    const agreeN = Object.values(R.base_classifiers).filter((l) => l === P.label).length;
    const nBase = Object.keys(R.base_classifiers).length;
    const [cls, ic, ctext] = confidence(P.probability);
    const acc = P.test_accuracy != null ? `${pct(P.test_accuracy)} <span class="perf-meta">(95% CI ${pct(P.ci_low, 0)}–${pct(P.ci_high, 0)}, n = ${P.test_n})</span>` : "not in test set";
    const M = R.missing;
    let missNote;
    if (M.count === 0) missNote = `<div class="notice notice-ok">${icon("check")}<div>All ${M.of.toLocaleString()} model genes were present. No imputation.</div></div>`;
    else if (M.significant_missing.length) missNote = `<div class="notice notice-warn">${icon("warn")}<div><b>${M.count.toLocaleString()} of ${M.of.toLocaleString()} model genes were missing</b> and filled with the ${R.imputation_reference === "healthy" ? "normal-tissue" : "tumor"} training median. ${M.significant_missing.length} of them are significant genes for this class: <i>${M.significant_missing.map(esc).join(", ")}</i>.</div></div>`;
    else missNote = `<div class="notice notice-ok">${icon("check")}<div>${M.count.toLocaleString()} of ${M.of.toLocaleString()} model genes were missing and filled with training medians. None are significant genes for this class.</div></div>`;
    const warns = R.warnings.map((w) => `<div class="notice notice-crit">${icon("warn")}<div>${esc(w)}</div></div>`).join("");

    return `<div>
      <div class="detail-head">
        <div>
          <div class="detail-id">${esc(s.id)} · ${esc(s.file)}</div>
          <h2 class="detail-title">${esc(P.tissue)}</h2>
          <div class="chip-row">
            <span class="chip"><span class="dot" style="background:${src.color}"></span>${src.label}</span>
            <span class="chip code">${esc(P.label.replace(".csv", ""))}</span>
          </div>
        </div>
        <button class="btn btn-ghost btn-sm" id="copy-summary" type="button">Copy summary</button>
      </div>
      ${warns}
      <div class="grid-2">
        <div class="card">
          <div class="card-h"><h3>Prediction</h3><span class="hint">meta-learner output</span></div>
          <div class="verdict">
            ${gauge(P.probability, src.color)}
            <dl class="kv">
              <dt>Class</dt><dd>${esc(P.name)}</dd>
              <dt>Accuracy for this class</dt><dd>${acc}</dd>
              <dt>Base classifiers agreeing</dt><dd><span class="agree" title="${esc(Object.entries(R.base_classifiers).map(([k, v]) => `${k}: ${v}`).join("\n"))}">${Array.from({ length: nBase }, (_, i) => `<i class="${i < agreeN ? "on" : ""}"></i>`).join("")}<span style="margin-left:6px">${agreeN} of ${nBase}</span></span></dd>
            </dl>
          </div>
          <div class="notice ${cls}">${icon(ic)}<div>${ctext}</div></div>
        </div>
        <div class="card">
          <div class="card-h"><h3>Next most likely</h3><span class="hint">probability</span></div>
          <ol class="runners">${R.probabilities.slice(1, 5).map((c, i) => `<li><span class="rank">${i + 2}</span><span><span class="dot" style="display:inline-block;margin-right:8px;background:${SRC[c.source].color}"></span>${esc(c.name)}</span><span class="p">${c.p < 0.001 ? "&lt;0.1%" : pct(c.p)}</span></li>`).join("")}</ol>
          ${missNote}
        </div>
      </div>

      <div class="card">
        <div class="card-h"><h3>Class probabilities</h3>
          <div class="legend">${Object.values(SRC).map((v) => `<span><i style="background:${v.color}"></i>${v.label}</span>`).join("")}</div></div>
        <div class="bars" id="bars">${barsHTML(R)}</div>
        <button class="btn btn-link btn-sm toggle-all" id="toggle-all" type="button">${state.showAll ? "Show top 10" : `Show all ${R.probabilities.length} classes`}</button>
      </div>

      <div class="card">
        <div class="card-h"><h3>Top 20 genes for ${esc(P.code)} ${esc(src.label.toLowerCase())}</h3>
          <span class="hint">${geneSetHint(R.genes)}</span></div>
        <div class="scroll-x">${genesHTML(R)}</div>
        <p class="table-note">Accuracy drop: fall in correct calls when the gene was replaced with noise (Monte Carlo ablation, 10 cycles). Significant: drop stands out from all ${state.config ? state.config.n_features.toLocaleString() : ""} genes after Benjamini-Hochberg correction. This sample vs reference: σ from the ${R.genes.reference === "healthy" ? "normal-tissue" : "tumor"} training median; stars are two-sided z-tests, BH-corrected across these 20 genes (* q < 0.05, ** q < 0.01, *** q < 0.001).</p>
      </div>
    </div>`;
  }

  function geneSetHint(G) {
    if (G.set === "overall") return `ranked on all ${G.set_samples.toLocaleString()} test samples (too few for this class)`;
    if (G.set === "metastatic") return `ranked on all ${G.set_samples} metastatic test samples`;
    return `ranked on ${G.set_samples.toLocaleString()} test samples of this class · ${G.n_significant} significant`;
  }

  function barsHTML(R) {
    const list = state.showAll ? R.probabilities : R.probabilities.slice(0, 10);
    return list.map((c, i) => {
      const info = state.config ? state.config.classes.find((k) => k.label === c.label) : null;
      const acc = info && info.test_accuracy != null ? `Test accuracy ${pct(info.test_accuracy)} (n = ${info.test_n})` : "";
      return `<div class="bar-row${i === 0 ? " is-top" : ""}" data-tip="<b>${esc(c.name)}</b>${esc(SRC[c.source].label)} · ${pct(c.p, 2)}${acc ? "<br>" + esc(acc) : ""}">
        <span class="bar-label">${c.source === "Healthy" ? "" : `<span class="code">${esc(c.code)}</span>`}${esc(c.name.replace(/ \((metastasis|normal tissue)\)$/, ""))}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${Math.max(0.3, 100 * c.p)}%;background:${SRC[c.source].color}"></span></span>
        <span class="bar-val">${c.p < 0.001 ? "&lt;0.1%" : c.p >= 0.9995 ? "&gt;99.9%" : pct(c.p)}</span></div>`;
    }).join("");
  }

  function genesHTML(R) {
    const rows = R.genes.rows;
    const maxDrop = Math.max(...rows.map((r) => r.drop), 1e-6);
    const lim = 4;
    return `<table class="gene-table"><thead><tr><th>#</th><th>Gene</th><th>Accuracy drop when ablated</th><th>Model significance</th><th>This sample vs reference</th></tr></thead><tbody>${rows.map((g, i) => {
      const z = Math.max(-lim, Math.min(lim, g.sigma));
      const w = (50 * Math.abs(z)) / lim;
      const left = z >= 0 ? 50 : 50 - w;
      const dir = g.sigma >= 0 ? "above" : "below";
      const sym = encodeURIComponent(g.gene);
      return `<tr>
        <td class="g-rank">${i + 1}</td>
        <td><div class="g-name">${esc(g.gene)}</div>
          <div class="g-links">
            <a href="https://www.genecards.org/cgi-bin/carddisp.pl?gene=${sym}" target="_blank" rel="noopener" title="Open ${esc(g.gene)} on GeneCards">Description ${icon("ext")}</a>
            <a href="https://pubmed.ncbi.nlm.nih.gov/?term=${sym}%20AND%20cancer&sort=date" target="_blank" rel="noopener" title="PubMed: ${esc(g.gene)} AND cancer, newest first">In cancer ${icon("ext")}</a>
          </div></td>
        <td><div class="mini"><span class="mini-track"><span class="mini-fill" style="width:${(100 * g.drop) / maxDrop}%"></span></span><span class="mini-num">${pct(g.drop)}</span></div></td>
        <td><span class="sig-tag${g.significant_in_model ? "" : " no"}" title="BH FDR ${fmtP(g.fdr)}">${g.significant_in_model ? "Significant" : "Not significant"}</span></td>
        <td><div class="mini" data-tip="<b>${esc(g.gene)}</b>log2(TPM+1) ${g.expr_log2.toFixed(2)} vs median ${g.ref_median_log2.toFixed(2)}<br>${Math.abs(g.sigma).toFixed(1)}σ ${dir} · q = ${fmtP(g.q)}${g.imputed ? "<br>Missing in file: median used" : ""}">
          <span class="div-track"><span class="div-fill" style="left:${left}%;width:${w}%;background:${z >= 0 ? "var(--div-high)" : "var(--div-low)"}"></span></span>
          <span class="mini-num">${g.sigma >= 0 ? "+" : "−"}${Math.abs(g.sigma).toFixed(1)}σ</span><span class="stars">${stars(g.q)}</span>${g.imputed ? `<span class="imp-tag">imputed</span>` : ""}</div></td>
      </tr>`;
    }).join("")}</tbody></table>`;
  }

  function bindDetail(s) {
    const t = $("#toggle-all");
    if (t) t.addEventListener("click", () => { state.showAll = !state.showAll; $("#bars").innerHTML = barsHTML(s.result); t.textContent = state.showAll ? "Show top 10" : `Show all ${s.result.probabilities.length} classes`; });
    const c = $("#copy-summary");
    if (c) c.addEventListener("click", async () => {
      const R = s.result, P = R.prediction;
      const text = `${s.id}: ${P.name} (${P.label}), probability ${pct(P.probability)}. Next: ${R.probabilities.slice(1, 3).map((x) => `${x.name} ${pct(x.p)}`).join("; ")}. Missing model genes: ${R.missing.count}/${R.missing.of}. Model: ${state.config ? state.config.model : "TCUP"}.`;
      try { await navigator.clipboard.writeText(text); c.textContent = "Copied"; } catch (e) { c.textContent = "Copy failed"; }
      setTimeout(() => (c.textContent = "Copy summary"), 1600);
    });
  }

  // ---------------- small helpers ----------------
  function icon(name) {
    const p = {
      check: '<path d="M5 12.5l4.2 4.2L19 7" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
      warn: '<path d="M12 4 2.8 20h18.4L12 4z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 10v4.5M12 17.4v.1" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>',
      x: '<path d="M6 6l12 12M18 6 6 18" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>',
      ext: '<path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    }[name];
    return `<svg viewBox="0 0 24 24" aria-hidden="true">${p}</svg>`;
  }

  let toastTimer;
  function toast(text) {
    let el = $("#toast");
    if (!el) {
      el = document.createElement("div");
      el.id = "toast";
      el.className = "tooltip";
      el.style.cssText = "left:50%;bottom:24px;top:auto;transform:translateX(-50%);max-width:520px;pointer-events:auto";
      document.body.appendChild(el);
    }
    el.textContent = text;
    el.classList.add("is-on");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove("is-on"), 5200);
  }

  function exportCSV() {
    const head = ["SampleID", "File", "PredictedLabel", "PredictedClass", "Source", "Probability", "Second", "SecondProbability", "Third", "ThirdProbability", "MissingModelGenes", "SignificantGenesImputed"];
    const rows = state.samples.filter((s) => s.status === "done").map((s) => {
      const R = s.result, P = R.prediction, pr = R.probabilities;
      return [s.id, s.file, P.label, P.name, P.source, P.probability.toFixed(4), pr[1].label, pr[1].p.toFixed(4), pr[2].label, pr[2].p.toFixed(4), R.missing.count, R.missing.significant_missing.join(" ")];
    });
    const csv = [head, ...rows].map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "TCUP_results.csv";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  // ---------------- events ----------------
  const input = $("#file-input");
  input.addEventListener("change", () => { if (input.files[0]) uploadBlob(input.files[0], input.files[0].name); input.value = ""; });
  $("#dropzone").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  $("#add-file").addEventListener("click", () => input.click());
  $("#try-example").addEventListener("click", async () => {
    setMsg("Loading example samples…");
    const r = await fetch("/api/example");
    uploadBlob(await r.blob(), "ExampleTranscriptomics.csv");
  });
  $("#export-csv").addEventListener("click", exportCSV);
  $("#new-analysis").addEventListener("click", () => {
    if (state.running || state.queue.length) return toast("Wait for the current samples to finish.");
    state.samples = []; state.selected = null; state.queue = [];
    renderList(); showView("landing");
  });
  document.querySelectorAll('[data-nav="home"]').forEach((a) => a.addEventListener("click", (e) => { if (!state.samples.length) return; e.preventDefault(); showView("landing"); }));
  document.querySelectorAll('.topnav a[href^="#"]:not([data-nav])').forEach((a) => a.addEventListener("click", () => showView("landing")));

  $("#sample-list").addEventListener("click", (e) => { const li = e.target.closest(".sample"); if (li) select(Number(li.dataset.key)); });
  $("#side-search").addEventListener("input", (e) => { state.filter = e.target.value; renderList(); });
  document.addEventListener("keydown", (e) => {
    if (!document.body.classList.contains("in-workspace") || e.target.matches("input, textarea")) return;
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    const list = visibleSamples();
    const i = list.findIndex((s) => s.key === state.selected);
    const next = list[Math.max(0, Math.min(list.length - 1, i + (e.key === "ArrowDown" ? 1 : -1)))];
    if (next) { e.preventDefault(); select(next.key); }
  });

  // drag and drop anywhere on the page
  let depth = 0;
  const overlay = $("#drop-overlay"), dz = $("#dropzone");
  window.addEventListener("dragenter", (e) => { if (![...e.dataTransfer.types].includes("Files")) return; depth++; dz.classList.add("is-over"); if (document.body.classList.contains("in-workspace")) overlay.classList.add("is-on"); });
  window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) { dz.classList.remove("is-over"); overlay.classList.remove("is-on"); } });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault(); depth = 0; dz.classList.remove("is-over"); overlay.classList.remove("is-on");
    const f = e.dataTransfer.files[0];
    if (f) uploadBlob(f, f.name);
  });

  // tooltips for anything with data-tip
  const tip = $("#tooltip");
  document.addEventListener("mouseover", (e) => { const t = e.target.closest("[data-tip]"); if (!t) return; tip.innerHTML = t.dataset.tip; tip.classList.add("is-on"); });
  document.addEventListener("mousemove", (e) => {
    if (!tip.classList.contains("is-on")) return;
    const x = Math.min(window.innerWidth - tip.offsetWidth - 12, e.clientX + 14), y = Math.min(window.innerHeight - tip.offsetHeight - 12, e.clientY + 14);
    tip.style.left = `${x}px`; tip.style.top = `${y}px`;
  });
  document.addEventListener("mouseout", (e) => { if (e.target.closest("[data-tip]") && !e.relatedTarget?.closest?.("[data-tip]")) tip.classList.remove("is-on"); });

  loadConfig();
  // a link ending in #demo opens straight onto the example samples
  if (location.hash === "#demo") $("#try-example").click();
})();
