/**
 * dashboard.js
 * =============
 * Lógica del dashboard financiero principal.
 *
 * On load:
 *  1. Carga KPIs desde /api/dashboard-summary
 *  2. Renderiza mini-heatmap con /api/heatmap?compact=true
 *  3. Rellena tabla de ranking con /api/risk-ranking
 */

"use strict";

document.addEventListener("DOMContentLoaded", () => {
  loadDashboard();
});

async function loadDashboard() {
  // Lanzar las tres peticiones en paralelo
  await Promise.allSettled([
    loadKPIs(),
    loadMiniHeatmap(),
    loadRanking(),
  ]);
}

/* ── KPIs ─────────────────────────────────────────────────────── */

async function loadKPIs() {
  try {
    const data = await apiFetch("/api/dashboard-summary");
    populateKPIs(data);
  } catch (err) {
    console.error("Error al cargar KPIs:", err);
  }
}

function populateKPIs(d) {
  // Activos y registros
  setText("kpi-assets",  d.n_assets  ?? "—");
  setText("kpi-records", d.n_records != null ? fmtInt(d.n_records) + " registros" : "—");

  // Período del dataset
  if (d.date_start && d.date_end) {
    const start = d.date_start.slice(0, 4);
    const end   = d.date_end.slice(0, 4);
    setText("dash-period", `${start} – ${end}`);
  }

  // Volatilidad promedio
  setText("kpi-avg-vol",
    d.avg_volatility_pct != null ? fmtPct(d.avg_volatility_pct / 100) : "—"
  );

  // Activo más riesgoso
  if (d.most_aggressive) {
    setText("kpi-most-aggressive",  d.most_aggressive.ticker ?? "—");
    setText("kpi-aggressive-vol",
      d.most_aggressive.annualized_vol_pct != null
        ? fmtPct(d.most_aggressive.annualized_vol_pct / 100)
        : "—"
    );
  }

  // Activo más conservador
  if (d.most_conservative) {
    setText("kpi-most-conservative", d.most_conservative.ticker ?? "—");
    setText("kpi-conservative-vol",
      d.most_conservative.annualized_vol_pct != null
        ? fmtPct(d.most_conservative.annualized_vol_pct / 100)
        : "—"
    );
  }

  // Par más correlacionado
  if (d.top_correlated_pair) {
    const p = d.top_correlated_pair;
    setText("kpi-top-corr-pair",  `${p.ticker_a} / ${p.ticker_b}`);
    setText("kpi-top-corr-value", `Pearson r = ${fmtNum(p.value)}`);
  }

  // Categorías de riesgo
  if (d.risk_categories) {
    setText("kpi-conservative-count", d.risk_categories["Conservador"] ?? 0);
    setText("kpi-moderate-count",     d.risk_categories["Moderado"]    ?? 0);
    setText("kpi-aggressive-count",   d.risk_categories["Agresivo"]    ?? 0);
  }
}

/* ── Mini Heatmap ─────────────────────────────────────────────── */

async function loadMiniHeatmap() {
  const loadingEl = document.getElementById("dash-heatmap-loading");
  const errorEl   = document.getElementById("dash-heatmap-error");
  const divEl     = document.getElementById("dash-heatmap");

  try {
    const data = await apiFetch("/api/heatmap?compact=true");
    if (loadingEl) loadingEl.style.display = "none";

    if (data.figure) {
      Plotly.newPlot(divEl, data.figure.data, data.figure.layout, {
        responsive: true,
        displayModeBar: false,
      });
    }
  } catch (err) {
    if (loadingEl) loadingEl.style.display = "none";
    if (errorEl) {
      errorEl.textContent = "Error al cargar heatmap: " + err.message;
      errorEl.style.display = "block";
    }
  }
}

/* ── Ranking mini ─────────────────────────────────────────────── */

async function loadRanking() {
  const tbody     = document.getElementById("dash-ranking-tbody");
  const loadingEl = document.getElementById("dash-ranking-loading");
  const errorEl   = document.getElementById("dash-ranking-error");

  try {
    if (loadingEl) loadingEl.style.display = "flex";
    const data = await apiFetch("/api/risk-ranking");
    if (loadingEl) loadingEl.style.display = "none";

    const rows = Array.isArray(data) ? data : (data.ranking ?? []);
    const top5 = rows.slice(0, 10); // top 10 en el mini

    if (!tbody) return;
    tbody.innerHTML = top5.map((r, i) => {
      const vol  = r.annualized_vol_pct != null ? fmtPct(r.annualized_vol_pct / 100) : "—";
      const cat  = r.risk_category ?? "";
      const cls  = riskBadgeClass(cat);
      return `
        <tr>
          <td>${i + 1}</td>
          <td><strong>${esc(r.ticker)}</strong></td>
          <td class="text-right">${vol}</td>
          <td><span class="badge ${cls}">${esc(cat)}</span></td>
        </tr>`;
    }).join("");
  } catch (err) {
    if (loadingEl) loadingEl.style.display = "none";
    if (errorEl) {
      errorEl.textContent = "Error al cargar ranking: " + err.message;
      errorEl.style.display = "block";
    }
  }
}

/* ── Helpers ─────────────────────────────────────────────────── */

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function esc(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
