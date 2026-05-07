/**
 * heatmap.js
 * ===========
 * Carga y renderiza el heatmap de correlación de Pearson.
 * Rellena las tablas de pares extremos.
 */

"use strict";

document.addEventListener("DOMContentLoaded", loadHeatmap);

async function loadHeatmap() {
  const loadingEl  = document.getElementById("heatmap-loading");
  const errorEl    = document.getElementById("heatmap-error");
  const containerEl = document.getElementById("heatmap-container");

  try {
    const data = await apiFetch("/api/heatmap");

    if (loadingEl) loadingEl.style.display = "none";

    // Renderizar figura
    if (data.figure && containerEl) {
      Plotly.newPlot(containerEl, data.figure.data, data.figure.layout, {
        responsive:     true,
        displayModeBar: true,
        modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
        displaylogo:    false,
      });
    }

    // Rellenar tablas
    if (data.extremes) {
      renderExtremesTable("heatmap-top-tbody",    data.extremes.most_correlated  ?? []);
      renderExtremesTable("heatmap-bottom-tbody", data.extremes.least_correlated ?? []);
    }

  } catch (err) {
    if (loadingEl) loadingEl.style.display = "none";
    if (errorEl) {
      errorEl.textContent = "Error al cargar heatmap: " + err.message;
      errorEl.style.display = "block";
    }
    console.error("Heatmap error:", err);
  }
}

function renderExtremesTable(tbodyId, pairs) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;

  if (!pairs || pairs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="2" class="text-muted text-center">Sin datos</td></tr>`;
    return;
  }

  tbody.innerHTML = pairs.map(p => {
    const r   = p.value != null ? fmtNum(p.value) : "—";
    const cls = p.value != null
      ? (p.value >= 0.7 ? "badge--success" : p.value <= 0.0 ? "badge--error" : "badge--info")
      : "";
    return `
      <tr>
        <td><strong>${esc(p.ticker_a)}</strong> / <strong>${esc(p.ticker_b)}</strong></td>
        <td class="text-right"><span class="badge ${cls}">${r}</span></td>
      </tr>`;
  }).join("");
}

function esc(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
