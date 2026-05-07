/**
 * patterns.js — Página Patrones: tabla de patrones de precio.
 */

"use strict";

(function () {
  const tickerSelect = document.getElementById("pattern-ticker-select");
  const btnFilter    = document.getElementById("btn-pattern-filter");
  const tbody        = document.getElementById("patterns-tbody");
  const loadingEl    = document.getElementById("patterns-loading");
  const errorEl      = document.getElementById("patterns-error");

  const PATTERN_LABELS = {
    "consecutive_gains_3d": "3 días consecutivos al alza",
    "drop_and_recovery":     "Caída y recuperación",
  };

  // Carga los tickers para el filtro
  async function loadTickers() {
    try {
      const data = await apiFetch("/api/tickers");
      (data.tickers || []).forEach(t => {
        const opt = document.createElement("option");
        opt.value = t;
        opt.textContent = t;
        tickerSelect.appendChild(opt);
      });
    } catch (err) {
      // No crítico: la tabla aún funciona sin filtro
      console.warn("No se pudo cargar la lista de activos:", err.message);
    }
  }

  async function loadPatterns() {
    hideError(errorEl);
    setLoading(loadingEl, true, "Cargando patrones…");
    tbody.innerHTML = "";

    const ticker = tickerSelect.value;
    const params = ticker ? `?ticker=${encodeURIComponent(ticker)}` : "";

    try {
      const data     = await apiFetch(`/api/patterns${params}`);
      const patterns = data.patterns || [];

      if (!patterns.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-muted" style="text-align:center;padding:24px">Sin patrones encontrados</td></tr>';
        return;
      }

      const rows = patterns.map(p => {
        const freq = p.frequency_pct != null
          ? fmtNum(p.frequency_pct, 2) + "%"
          : (p.frequency != null ? fmtNum(p.frequency * 100, 2) + "%" : "—");

        const patternLabel = PATTERN_LABELS[p.pattern] || p.pattern || "—";

        return `<tr>
          <td><strong>${p.ticker || "—"}</strong></td>
          <td>${p.instrument_type || "—"}</td>
          <td><code>${p.pattern || "—"}</code></td>
          <td>${patternLabel}</td>
          <td class="text-right">${fmtInt(p.n_occurrences)}</td>
          <td class="text-right">${fmtInt(p.total_windows)}</td>
          <td class="text-right">${freq}</td>
        </tr>`;
      }).join("");

      tbody.innerHTML = rows;
    } catch (err) {
      showError(errorEl, err.message);
    } finally {
      setLoading(loadingEl, false);
    }
  }

  btnFilter.addEventListener("click", loadPatterns);

  // Inicialización
  (async function () {
    await loadTickers();
    await loadPatterns();
  })();
})();
