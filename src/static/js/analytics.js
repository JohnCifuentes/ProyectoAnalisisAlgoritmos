/**
 * analytics.js — Página Analytics: ranking de riesgo y stat cards.
 */

"use strict";

(async function () {
  const loadingEl  = document.getElementById("analytics-loading");
  const errorEl    = document.getElementById("analytics-error");
  const tbody      = document.getElementById("ranking-tbody");

  const elConservative = document.getElementById("count-conservative");
  const elModerate     = document.getElementById("count-moderate");
  const elAggressive   = document.getElementById("count-aggressive");

  setLoading(loadingEl, true, "Cargando ranking…");
  hideError(errorEl);

  try {
    const data    = await apiFetch("/api/risk-ranking");
    const ranking = data.ranking  || [];
    const summary = data.summary  || {};

    // Stat cards
    if (elConservative) elConservative.textContent = summary["Conservador"] || 0;
    if (elModerate)     elModerate.textContent     = summary["Moderado"]    || 0;
    if (elAggressive)   elAggressive.textContent   = summary["Agresivo"]    || 0;

    // Tabla
    if (!ranking.length) {
      tbody.innerHTML = '<tr><td colspan="9" class="text-muted" style="text-align:center;padding:24px">Sin datos</td></tr>';
      return;
    }

    const rows = ranking.map(r => {
      const rank     = r.rank            != null ? r.rank : "—";
      const annVol   = r.annualized_vol_pct != null
        ? fmtNum(r.annualized_vol_pct, 2) + "%"
        : "—";
      const dailyVol = r.daily_vol_pct != null
        ? fmtNum(r.daily_vol_pct, 2) + "%"
        : "—";
      const badgeCls = riskBadgeClass(r.risk_category);
      return `<tr>
        <td><strong>${rank}</strong></td>
        <td><strong>${r.ticker || "—"}</strong></td>
        <td>${r.instrument_type || "—"}</td>
        <td class="text-right">${annVol}</td>
        <td class="text-right">${dailyVol}</td>
        <td><span class="${badgeCls}">${r.risk_category || "—"}</span></td>
        <td class="text-right">${fmtInt(r.n_observations)}</td>
        <td>${r.date_start || "—"}</td>
        <td>${r.date_end   || "—"}</td>
      </tr>`;
    }).join("");

    tbody.innerHTML = rows;
  } catch (err) {
    showError(errorEl, err.message);
  } finally {
    setLoading(loadingEl, false);
  }
})();
