/* ============================================================
 * results_view.js — renders the Q1 analysis artifacts into
 * results.html. Pure JS, no dependencies. Fetches the committed
 * report (.md), tables (.csv), and figures (.png) which are served
 * same-origin from GitHub Pages.
 * ============================================================ */
(function () {
  "use strict";

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
  function renderMarkdown(md) {
    const lines = md.replace(/\r/g, "").split("\n");
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
        html += `<blockquote>${inlineMd(buf.join(" "))}</blockquote>`;
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
    return html;
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

  // ---------- loaders ----------
  async function loadReport() {
    try {
      const md = await fetchText("results/reports/Q1_experimental_report.md");
      document.getElementById("reportBody").innerHTML = renderMarkdown(md);
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
      document.getElementById("methodsBody").innerHTML =
        renderMarkdown(finding) + "<hr/>" + renderMarkdown(repro);
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
