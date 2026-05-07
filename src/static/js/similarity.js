/**
 * similarity.js — Página Similitud: comparación de pares de activos.
 */

"use strict";

(function () {
  // ── DOM ─────────────────────────────────────────────────────────
  const form           = document.getElementById("similarity-form");
  const tickerASelect  = document.getElementById("ticker-a");
  const tickerBSelect  = document.getElementById("ticker-b");
  const metricSelect   = document.getElementById("metric-select");
  const onSelect       = document.getElementById("on-select");
  const btnCompare     = document.getElementById("btn-compare");
  const btnCompareAll  = document.getElementById("btn-compare-all");
  const loadingEl      = document.getElementById("similarity-loading");
  const errorEl        = document.getElementById("similarity-error");

  const resultCard     = document.getElementById("result-card");
  const resultPair     = document.getElementById("result-pair");
  const resultBadge    = document.getElementById("result-metric-badge");
  const resultValue    = document.getElementById("result-value");
  const resultInterp   = document.getElementById("result-interpretation");
  const resultMeta     = document.getElementById("result-meta");

  const allCard        = document.getElementById("all-results-card");
  const allTitle       = document.getElementById("all-results-title");
  const allTbody       = document.getElementById("all-results-tbody");

  // ── Inicialización ───────────────────────────────────────────────
  async function init() {
    try {
      const data = await apiFetch("/api/tickers");
      const tickers = data.tickers || [];
      tickers.forEach(t => {
        [tickerASelect, tickerBSelect].forEach(sel => {
          const opt = document.createElement("option");
          opt.value = t;
          opt.textContent = t;
          sel.appendChild(opt);
        });
      });
    } catch (err) {
      showError(errorEl, "No se pudo cargar la lista de activos: " + err.message);
    }
  }

  // ── Comparación simple ───────────────────────────────────────────
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    await doCompare();
  });

  async function doCompare() {
    const payload = {
      ticker_a: tickerASelect.value,
      ticker_b: tickerBSelect.value,
      metric:   metricSelect.value,
      on:       onSelect.value,
    };

    if (!payload.ticker_a || !payload.ticker_b) {
      showError(errorEl, "Selecciona ambos activos.");
      return;
    }
    if (payload.ticker_a === payload.ticker_b) {
      showError(errorEl, "Selecciona activos distintos.");
      return;
    }

    hideError(errorEl);
    allCard.style.display    = "none";
    resultCard.style.display = "none";
    setLoading(loadingEl, true, "Calculando similitud…");
    btnCompare.disabled    = true;
    btnCompareAll.disabled = true;

    try {
      const data = await apiFetch("/api/similarity", {
        method: "POST",
        body:   JSON.stringify(payload),
      });
      renderSingleResult(data);
    } catch (err) {
      showError(errorEl, err.message);
    } finally {
      setLoading(loadingEl, false);
      btnCompare.disabled    = false;
      btnCompareAll.disabled = false;
    }
  }

  // ── Comparación con todos los algoritmos ────────────────────────
  btnCompareAll.addEventListener("click", async () => {
    const payload = {
      ticker_a: tickerASelect.value,
      ticker_b: tickerBSelect.value,
      on:       onSelect.value,
    };

    if (!payload.ticker_a || !payload.ticker_b) {
      showError(errorEl, "Selecciona ambos activos.");
      return;
    }
    if (payload.ticker_a === payload.ticker_b) {
      showError(errorEl, "Selecciona activos distintos.");
      return;
    }

    hideError(errorEl);
    resultCard.style.display = "none";
    allCard.style.display    = "none";
    setLoading(loadingEl, true, "Calculando todos los algoritmos…");
    btnCompare.disabled    = true;
    btnCompareAll.disabled = true;

    try {
      const data = await apiFetch("/api/similarity/all", {
        method: "POST",
        body:   JSON.stringify(payload),
      });
      renderAllResults(data);
    } catch (err) {
      showError(errorEl, err.message);
    } finally {
      setLoading(loadingEl, false);
      btnCompare.disabled    = false;
      btnCompareAll.disabled = false;
    }
  });

  // ── Renderizado ─────────────────────────────────────────────────
  function renderSingleResult(data) {
    resultPair.textContent  = `${data.ticker_a} vs ${data.ticker_b}`;
    resultBadge.className   = "badge badge--info";
    resultBadge.textContent = (data.metric || "").toUpperCase();
    resultValue.textContent = data.value != null ? fmtNum(data.value, 6) : "—";
    resultInterp.textContent = data.interpretation || "";

    const meta = [];
    if (data.n_points) meta.push(`${fmtInt(data.n_points)} puntos`);
    if (data.date_start) meta.push(`Inicio: ${data.date_start}`);
    if (data.date_end)   meta.push(`Fin: ${data.date_end}`);
    if (data.on)         meta.push(`Sobre: ${data.on}`);
    if (data.elapsed_ms != null) meta.push(`${data.elapsed_ms.toFixed(1)} ms`);

    resultMeta.innerHTML = meta.map(m => `<span>${m}</span>`).join("");

    // Color del valor según la magnitud (solo para métricas -1..1)
    const v = data.value;
    if (v != null && (data.metric === "pearson" || data.metric === "cosine")) {
      if (Math.abs(v) >= 0.7)      resultValue.style.color = "var(--color-conservative)";
      else if (Math.abs(v) >= 0.4) resultValue.style.color = "var(--color-moderate)";
      else                          resultValue.style.color = "var(--color-aggressive)";
    } else {
      resultValue.style.color = "var(--color-accent)";
    }

    resultCard.style.display = "block";
  }

  function renderAllResults(data) {
    const results = data.results || {};
    allTitle.textContent = `${data.ticker_a} vs ${data.ticker_b} · todos los algoritmos`;

    const METRIC_LABELS = {
      euclidean: "Euclidiana",
      pearson:   "Pearson",
      cosine:    "Coseno",
      dtw:       "DTW",
    };

    const rows = Object.entries(results).map(([metric, res]) => {
      const val = res.value != null ? fmtNum(res.value, 6) : "—";
      const interp = res.interpretation || "—";
      return `<tr>
        <td><span class="badge badge--info">${METRIC_LABELS[metric] || metric}</span></td>
        <td class="text-right" style="font-family:var(--font-mono)">${val}</td>
        <td>${interp}</td>
      </tr>`;
    }).join("");

    allTbody.innerHTML = rows || '<tr><td colspan="3" class="text-muted">Sin resultados</td></tr>';
    allCard.style.display = "block";
  }

  init();
})();
