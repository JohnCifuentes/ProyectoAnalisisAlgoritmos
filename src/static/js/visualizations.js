/**
 * visualizations.js
 * ==================
 * Hub de visualizaciones.
 *
 * On load:
 *  1. Carga extremos del heatmap desde /api/heatmap
 *  2. Muestra el par más/menos correlacionado
 *  3. Rellena la tabla de top-5 correlaciones positivas
 */

"use strict";

document.addEventListener("DOMContentLoaded", loadVisualizationsHub);

async function loadVisualizationsHub() {
  const loadingEl = document.getElementById("viz-corr-loading");
  const errorEl   = document.getElementById("viz-corr-error");
  const summaryEl = document.getElementById("viz-corr-summary");

  try {
    const data = await apiFetch("/api/heatmap");

    if (loadingEl) loadingEl.style.display = "none";

    if (data.extremes) {
      const ex = data.extremes;

      // Par más correlacionado
      if (ex.top_pair) {
        setText("viz-top-pair", `${ex.top_pair.ticker_a} / ${ex.top_pair.ticker_b}`);
        setText("viz-top-val",  `r = ${fmtNum(ex.top_pair.value)}`);
      }

      // Par menos correlacionado
      if (ex.bottom_pair) {
        setText("viz-bottom-pair", `${ex.bottom_pair.ticker_a} / ${ex.bottom_pair.ticker_b}`);
        setText("viz-bottom-val",  `r = ${fmtNum(ex.bottom_pair.value)}`);
      }

      if (summaryEl) summaryEl.style.display = "block";

      // Tabla top-5
      renderTopTable(ex.most_correlated ?? []);
    }

  } catch (err) {
    if (loadingEl) loadingEl.style.display = "none";
    if (errorEl) {
      errorEl.textContent = "Error al cargar correlaciones: " + err.message;
      errorEl.style.display = "block";
    }
    console.error("Visualizations hub error:", err);
  }
}

function renderTopTable(pairs) {
  const tbody = document.getElementById("viz-top-tbody");
  if (!tbody) return;

  if (!pairs || pairs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="2" class="text-muted text-center">Sin datos</td></tr>`;
    return;
  }

  tbody.innerHTML = pairs.map(p => {
    const r = p.value != null ? fmtNum(p.value) : "—";
    return `
      <tr>
        <td><strong>${esc(p.ticker_a)}</strong> / <strong>${esc(p.ticker_b)}</strong></td>
        <td class="text-right"><span class="badge badge--success">${r}</span></td>
      </tr>`;
  }).join("");
}

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
