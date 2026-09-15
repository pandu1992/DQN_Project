/* ============================================================
 * results_view.js — renders the Q1 analysis artifacts into
 * results.html. Pure JS, no dependencies. Fetches the committed
 * report (.md), tables (.csv), and figures (.png) which are served
 * same-origin from GitHub Pages.
 * ============================================================ */
(function () {
  "use strict";

  // ---------- math (KaTeX) handling ----------
  // We extract $$...$$ (display) and $...$ (inline) math into placeholders
  // BEFORE markdown processing so the markdown/escaping steps can't mangle
  // the LaTeX, then render them with KaTeX and swap the HTML back in.
  const mathStore = [];
  const MATH_TOKEN = (i) => `\u0000MATH${i}\u0000`; // null-delimited, survives markdown

  function protectMath(md) {
    // display math first: $$ ... $$ (may span lines)
    md = md.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => {
      const idx = mathStore.push({ tex: tex.trim(), display: true }) - 1;
      return MATH_TOKEN(idx);
    });
    // inline math: $ ... $  (avoid matching $$ leftovers and currency like $5)
    md = md.replace(/\$([^\$\n]+?)\$/g, (m, tex) => {
      if (/^\s*\d/.test(tex) && !/[\\^_{}]/.test(tex)) return m; // likely currency, leave as-is
      const idx = mathStore.push({ tex: tex.trim(), display: false }) - 1;
      return MATH_TOKEN(idx);
    });
    return md;
  }

  function renderMathToken(idx) {
    const entry = mathStore[idx];
    if (!entry) return "";
    if (window.katex) {
      try {
        return window.katex.renderToString(entry.tex, {
          displayMode: entry.display, throwOnError: false, output: "html",
        });
      } catch (e) { /* fall through to raw */ }
    }
    // KaTeX not available: show the LaTeX in a code block rather than raw $$
    const cls = entry.display ? "math-fallback-block" : "math-fallback-inline";
    return `<code class="${cls}">${esc(entry.tex)}</code>`;
  }

  function restoreMath(html) {
    return html.replace(/\u0000MATH(\d+)\u0000/g, (_, i) => renderMathToken(+i));
  }

  // ---------- tiny, safe-ish Markdown -> HTML ----------
  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function inlineMd(s) {
    // escape first, then apply inline formatting
    s = esc(s);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
    s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    return s;
  }
  let eqCounter = 0;
  function renderMarkdown(md) {
    eqCounter = 0; // equation numbers restart per document
    md = protectMath(md.replace(/\r/g, ""));
    const lines = md.split("\n");
    let html = "", i = 0;
    let inCode = false, codeBuf = [];
    let listType = null, listBuf = [];
    let tableBuf = [];

    function flushList() {
      if (listType) {
        html += `<${listType}>` + listBuf.map((li) => `<li>${inlineMd(li)}</li>`).join("") + `</${listType}>`;
        listType = null; listBuf = [];
      }
    }
    function flushTable() {
      if (tableBuf.length >= 2) {
        const rows = tableBuf.filter((r) => !/^\s*\|?[\s:|-]+\|?\s*$/.test(r) || tableBuf.indexOf(r) === 0);
        const cells = (r) => r.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        const head = cells(tableBuf[0]);
        let t = "<table><thead><tr>" + head.map((h) => `<th>${inlineMd(h)}</th>`).join("") + "</tr></thead><tbody>";
        for (let r = 2; r < tableBuf.length; r++) {
          const cs = cells(tableBuf[r]);
          t += "<tr>" + cs.map((c) => `<td>${inlineMd(c)}</td>`).join("") + "</tr>";
        }
        t += "</tbody></table>";
        html += t;
      } else if (tableBuf.length) {
        tableBuf.forEach((r) => (html += `<p>${inlineMd(r)}</p>`));
      }
      tableBuf = [];
    }

    for (i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (/^```/.test(line)) {
        if (inCode) { html += `<pre><code>${esc(codeBuf.join("\n"))}</code></pre>`; inCode = false; codeBuf = []; }
        else { flushList(); flushTable(); inCode = true; }
        continue;
      }
      if (inCode) { codeBuf.push(line); continue; }

      // a line that is ONLY a display-math token -> emit as its own block
      // (KaTeX display math is block-level and must not sit inside <p>)
      const soleMath = line.trim().match(/^\u0000MATH(\d+)\u0000$/);
      if (soleMath && mathStore[+soleMath[1]] && mathStore[+soleMath[1]].display) {
        flushList();
        eqCounter++;
        html += `<div class="math-display" id="eq-${eqCounter}">` +
                `<span class="math-eq">${line.trim()}</span>` +
                `<span class="eq-num">(${eqCounter})</span></div>`;
        continue;
      }

      // table rows
      if (/^\s*\|.*\|\s*$/.test(line)) { flushList(); tableBuf.push(line.trim()); continue; }
      else if (tableBuf.length) flushTable();

      if (/^#{1,6}\s/.test(line)) {
        flushList();
        const lvl = line.match(/^#+/)[0].length;
        html += `<h${lvl}>${inlineMd(line.replace(/^#+\s/, ""))}</h${lvl}>`;
      } else if (/^\s*>\s?/.test(line)) {
        flushList();
        // merge consecutive blockquote lines into a single block
        const buf = [line.replace(/^\s*>\s?/, "")];
        while (i + 1 < lines.length && /^\s*>\s?/.test(lines[i + 1])) {
          buf.push(lines[++i].replace(/^\s*>\s?/, ""));
        }
        // GitHub-style admonition: first line "[!INSIGHT]" -> highlighted callout
        const adm = buf[0].trim().match(/^\[!(\w+)\]\s*$/);
        if (adm) {
          const kind = adm[1].toLowerCase();
          const body = buf.slice(1).join(" ");
          const label = kind === "insight" ? "💡 Key insight" :
                        kind === "note" ? "📝 Note" :
                        kind === "warning" ? "⚠ Caution" : kind;
          html += `<div class="admonition admonition-${kind}">` +
                  `<div class="admonition-title">${label}</div>` +
                  `<div class="admonition-body">${inlineMd(body)}</div></div>`;
        } else {
          html += `<blockquote>${inlineMd(buf.join(" "))}</blockquote>`;
        }
      } else if (/^\s*[-*]\s+/.test(line)) {
        if (listType !== "ul") { flushList(); listType = "ul"; }
        listBuf.push(line.replace(/^\s*[-*]\s+/, ""));
      } else if (/^\s*\d+\.\s+/.test(line)) {
        if (listType !== "ol") { flushList(); listType = "ol"; }
        listBuf.push(line.replace(/^\s*\d+\.\s+/, ""));
      } else if (/^\s*---\s*$/.test(line)) {
        flushList(); html += "<hr/>";
      } else if (line.trim() === "") {
        flushList();
      } else {
        flushList();
        html += `<p>${inlineMd(line)}</p>`;
      }
    }
    flushList(); flushTable();
    return restoreMath(html);
  }

  // ---------- CSV parsing (handles quoted fields w/ commas) ----------
  function parseCSV(text) {
    const rows = [];
    let row = [], field = "", inQ = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (inQ) {
        if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else inQ = false; }
        else field += c;
      } else {
        if (c === '"') inQ = true;
        else if (c === ",") { row.push(field); field = ""; }
        else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
        else if (c === "\r") { /* skip */ }
        else field += c;
      }
    }
    if (field.length || row.length) { row.push(field); rows.push(row); }
    return rows.filter((r) => r.length && !(r.length === 1 && r[0] === ""));
  }
  function csvToTable(rows) {
    if (!rows.length) return "<p class='loading'>(empty)</p>";
    const head = rows[0];
    let t = "<div class='csv-scroll'><table class='csv'><thead><tr>" +
      head.map((h) => `<th>${esc(h)}</th>`).join("") + "</tr></thead><tbody>";
    for (let r = 1; r < rows.length; r++) {
      t += "<tr>" + rows[r].map((c) => `<td>${esc(c)}</td>`).join("") + "</tr>";
    }
    return t + "</tbody></table></div>";
  }

  async function fetchText(url) {
    const res = await fetch(url + "?cb=" + Date.now());
    if (!res.ok) throw new Error(res.status + " " + url);
    return res.text();
  }

  // ---------- content definitions ----------
  const TABLES = [
    ["table1_configuration", "Table 1 — Algorithm & Experimental Configuration"],
    ["table2_overall_performance", "Table 2 — Overall Performance (mean ± 95% CI)"],
    ["table3_omnibus_algorithm", "Table 3 — Omnibus Algorithm Effect"],
    ["table4_factorial_effects", "Table 4 — Factorial Main Effects & Interactions"],
    ["table5_full_matrix", "Table 5 — Full 16-Configuration Matrix"],
    ["table6_statistical_comparison", "Table 6 — Pairwise Statistical Comparison"],
    ["table7_robustness", "Table 7 — Enhancement / Robustness"],
    ["table8_overall_findings", "Table 8 — Overall Findings"],
    ["table_methods", "Table — Mathematical & Statistical Methods"],
  ];
  const FIGURES = [
    "fig1_overall_performance", "fig2_success_across_configs", "fig3_reward_across_configs",
    "fig4_per_seed_distributions", "fig5_ci_comparison", "fig6_significance_matrix",
    "fig7_enhancement_tradeoff", "fig8_variance_dominance",
  ];

  // ---------- KaTeX readiness ----------
  // KaTeX is loaded with `defer`, so it may not be ready when a panel first
  // renders. We keep the raw markdown for math-bearing panels and re-render
  // once KaTeX is available (or after a short timeout for the fallback).
  const mdCache = {}; // panelId -> markdown string
  let katexReady = false;

  function whenKatexReady(cb) {
    if (window.katex) { katexReady = true; return cb(); }
    let waited = 0;
    const iv = setInterval(() => {
      waited += 100;
      if (window.katex) { katexReady = true; clearInterval(iv); cb(); }
      else if (waited >= 5000) { clearInterval(iv); cb(); } // give up -> fallback rendering
    }, 100);
  }

  function renderMdInto(elId, md) {
    document.getElementById(elId).innerHTML = renderMarkdown(md);
  }

  // ---------- loaders ----------
  async function loadReport() {
    try {
      const md = await fetchText("results/reports/Q1_experimental_report.md");
      mdCache.report = md;
      renderMdInto("reportBody", md);
      if (!katexReady) whenKatexReady(() => renderMdInto("reportBody", mdCache.report));
    } catch (e) {
      document.getElementById("reportBody").innerHTML =
        `<p class='loading'>Could not load report (${e.message}).</p>`;
    }
  }

  async function loadMethods() {
    try {
      const [repro, finding] = await Promise.all([
        fetchText("results/reports/REPRODUCIBILITY.md"),
        fetchText("results/reports/SATURATED_BENCHMARK_FINDING.md"),
      ]);
      const combine = () => renderMarkdown(mdCache.methodsFinding) + "<hr/>" + renderMarkdown(mdCache.methodsRepro);
      mdCache.methodsFinding = finding; mdCache.methodsRepro = repro;
      document.getElementById("methodsBody").innerHTML = combine();
      if (!katexReady) whenKatexReady(() => { document.getElementById("methodsBody").innerHTML = combine(); });
    } catch (e) {
      document.getElementById("methodsBody").innerHTML =
        `<p class='loading'>Could not load (${e.message}).</p>`;
    }
  }

  async function loadTables() {
    const host = document.getElementById("tablesBody");
    host.innerHTML = "";
    for (const [file, title] of TABLES) {
      const block = document.createElement("div");
      block.className = "csv-block";
      block.innerHTML = `<div class="csv-title">${title}</div>
        <div class="csv-desc"><a class="rp-dl" href="results/tables/${file}.csv" download>⬇ CSV</a>
        <a class="rp-dl" href="results/tables/${file}.tex" download>⬇ LaTeX</a></div>
        <div class="loading">Loading…</div>`;
      host.appendChild(block);
      try {
        const txt = await fetchText("results/tables/" + file + ".csv");
        block.querySelector(".loading").outerHTML = csvToTable(parseCSV(txt));
      } catch (e) {
        block.querySelector(".loading").textContent = "Could not load (" + e.message + ")";
      }
    }
  }

  async function loadFigures() {
    const host = document.getElementById("figuresBody");
    host.innerHTML = "";
    let captions = {};
    try {
      const capMd = await fetchText("results/figures/figures_captions.md");
      // parse "## name\n\ncaption"
      const re = /##\s+(\S+)\s*\n+([^#]+)/g; let m;
      while ((m = re.exec(capMd))) captions[m[1].trim()] = m[2].trim();
    } catch (e) { /* captions optional */ }
    for (const fig of FIGURES) {
      const card = document.createElement("div");
      card.className = "fig-card";
      const cap = captions[fig] || "";
      card.innerHTML =
        `<img loading="lazy" src="results/figures/${fig}.png" alt="${fig}" />
         <div class="fig-cap">${esc(cap)}</div>
         <div class="fig-links">
           <a class="rp-dl" href="results/figures/${fig}.png" download>⬇ PNG (320 dpi)</a>
           <a class="rp-dl" href="results/figures/${fig}.svg" download>⬇ SVG</a>
         </div>`;
      host.appendChild(card);
    }
  }

  // ---------- PDF export (print-to-PDF of the full report) ----------
  // Builds a clean, self-contained print document (report + all tables +
  // figures, with KaTeX math and justified text) and invokes the browser's
  // print dialog, where the user chooses "Save as PDF".
  async function downloadPdf(btn) {
    const origText = btn.textContent;
    btn.textContent = "Preparing PDF…";
    btn.disabled = true;
    try {
      const base = location.href.replace(/[^/]*$/, ""); // dir of results.html
      const [reportMd, findingMd, reproMd, capMd] = await Promise.all([
        mdCache.report ? Promise.resolve(mdCache.report) : fetchText("results/reports/Q1_experimental_report.md"),
        fetchText("results/reports/SATURATED_BENCHMARK_FINDING.md"),
        fetchText("results/reports/REPRODUCIBILITY.md"),
        fetchText("results/figures/figures_captions.md").catch(() => ""),
      ]);
      mdCache.report = reportMd;

      const reportHtml = renderMarkdown(reportMd);
      const findingHtml = renderMarkdown(findingMd);
      const reproHtml = renderMarkdown(reproMd);

      // tables section
      let tablesHtml = "";
      for (const [file, title] of TABLES) {
        try {
          const rows = parseCSV(await fetchText("results/tables/" + file + ".csv"));
          tablesHtml += `<h3 class="pdf-tbl-title">${esc(title)}</h3>` + csvToTable(rows);
        } catch (e) { /* skip */ }
      }

      // figures section (absolute URLs so the print window can load them)
      const captions = {};
      const re = /##\s+(\S+)\s*\n+([^#]+)/g; let m;
      while ((m = re.exec(capMd))) captions[m[1].trim()] = m[2].trim();
      let figsHtml = "";
      for (const fig of FIGURES) {
        figsHtml += `<figure class="pdf-fig"><img src="${base}results/figures/${fig}.png"/>` +
                    `<figcaption>${esc(captions[fig] || fig)}</figcaption></figure>`;
      }

      const katexCss = 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css';
      const now = new Date().toISOString().slice(0, 10);
      const doc = `<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Bintulu Port DQN — Q1 Experimental Report</title>
<link rel="stylesheet" href="${katexCss}">
<style>
  @page { size: A4; margin: 18mm 16mm; }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  body { font-family: 'Times New Roman', Georgia, serif; color:#111; line-height:1.5;
    font-size:11pt; max-width:800px; margin:0 auto; }
  h1 { font-size:20pt; margin:0 0 2pt; }
  h2 { font-size:14pt; border-bottom:1px solid #999; padding-bottom:2pt; margin:18pt 0 6pt;
    page-break-after:avoid; }
  h3 { font-size:12pt; margin:12pt 0 4pt; page-break-after:avoid; }
  p, li { text-align:justify; }
  code { font-family:'Courier New',monospace; background:#f2f2f2; padding:0 3px; font-size:9.5pt; }
  pre { background:#f6f6f6; border:1px solid #ddd; padding:8px; overflow:auto; font-size:9pt; }
  blockquote { border-left:3px solid #888; margin:8pt 0; padding:2pt 10pt; color:#333; }
  table { border-collapse:collapse; width:100%; margin:8pt 0; font-size:8.5pt; page-break-inside:avoid; }
  th,td { border:1px solid #bbb; padding:3px 5px; text-align:left; vertical-align:top; }
  th { background:#eee; }
  .admonition { border:1px solid #c9a227; background:#fff9e6; border-left:4px solid #d4a017;
    border-radius:4px; padding:6pt 10pt; margin:10pt 0; page-break-inside:avoid; }
  .admonition-title { font-weight:bold; color:#8a6d00; margin-bottom:3pt; font-size:10.5pt; }
  .math-display { text-align:center; margin:8pt 0; position:relative; }
  .eq-num { position:absolute; right:0; top:50%; transform:translateY(-50%); color:#333; }
  .katex { font-size:1em; }
  .csv-scroll { overflow:visible; border:none; }
  .pdf-tbl-title { font-size:10.5pt; color:#333; margin-top:10pt; }
  .pdf-fig { margin:10pt 0; page-break-inside:avoid; text-align:center; }
  .pdf-fig img { max-width:100%; height:auto; border:1px solid #ddd; }
  .pdf-fig figcaption { font-size:8.5pt; color:#333; text-align:justify; margin-top:3pt; }
  .pdf-cover { border-bottom:2px solid #333; padding-bottom:8pt; margin-bottom:10pt; }
  .pdf-cover .sub { color:#444; font-size:10pt; }
  .pagebreak { page-break-before:always; }
  a { color:#0b5; text-decoration:none; }
</style></head><body>
<div class="pdf-cover">
  <h1>Bintulu Port — Autonomous Vessel Navigation</h1>
  <div class="sub">Q1-grade comparative experimental evaluation of value-based DRL agents · generated ${now}</div>
  <div class="sub">Source: https://pandu1992.github.io/DQN_Project/</div>
</div>
${reportHtml}
<div class="pagebreak"></div><h2>Appendix A — All Tables</h2>${tablesHtml}
<div class="pagebreak"></div><h2>Appendix B — Figures</h2>${figsHtml}
<div class="pagebreak"></div><h2>Appendix C — Experimental Design Note</h2>${findingHtml}
<h2>Appendix D — Reproducibility</h2>${reproHtml}
</body></html>`;

      const w = window.open("", "_blank");
      if (!w) { alert("Please allow pop-ups to generate the PDF."); return; }
      w.document.open(); w.document.write(doc); w.document.close();
      // wait for KaTeX + images, then print
      const doPrint = () => { try { w.focus(); w.print(); } catch (e) {} };
      w.onload = () => setTimeout(doPrint, 900);
      // fallback if onload already passed
      setTimeout(() => { if (w && !w.closed) doPrint(); }, 2500);
    } catch (e) {
      alert("Could not build the PDF: " + e.message);
    } finally {
      btn.textContent = origText;
      btn.disabled = false;
    }
  }
  const pdfBtn = document.getElementById("btnDownloadPdf");
  if (pdfBtn) pdfBtn.addEventListener("click", () => downloadPdf(pdfBtn));

  // ---------- tabs ----------
  const loaded = {};
  function activate(tab) {
    document.querySelectorAll(".rp-tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
    document.querySelectorAll(".rp-panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + tab));
    if (!loaded[tab]) {
      loaded[tab] = true;
      if (tab === "report") loadReport();
      else if (tab === "tables") loadTables();
      else if (tab === "figures") loadFigures();
      else if (tab === "methods") loadMethods();
    }
  }
  document.getElementById("rpTabs").addEventListener("click", (e) => {
    const b = e.target.closest(".rp-tab");
    if (b) activate(b.dataset.tab);
  });

  // initial
  activate("report");
})();
