/**
 * candlestick.js
 * ===============
 * Gestiona la vista del gráfico candlestick OHLC.
 *
 * Flujo:
 *  1. Al cargar: obtiene la lista de tickers y popula el dropdown.
 *  2. Llama loadChart() con valores por defecto (AAPL, 1y, SMA 20 + 50).
 *  3. Al cambiar ticker/período → llama loadChart() de nuevo.
 *  4. Al cambiar checkbox de SMA → toggle visibilidad de traza con Plotly.restyle().
 */

"use strict";

const CHART_DIV = "candlestick-container";

// Estado global de la vista
const state = {
  currentTicker:  "AAPL",
  currentPeriod:  "1y",
  smaTraceNames:  [], // nombres de trazas SMA en el gráfico actual
};

document.addEventListener("DOMContentLoaded", async () => {
  await populateTickers();
  await loadChart();
  bindPeriodButtons();
  bindSMACheckboxes();
});

/* ── Tickers ───────────────────────────────────────────────────── */

async function populateTickers() {
  const select = document.getElementById("ticker-select");
  if (!select) return;

  try {
    const tickers = await apiFetch("/api/tickers");
    const list    = Array.isArray(tickers) ? tickers : (tickers.tickers ?? []);

    select.innerHTML = list.map(t =>
      `<option value="${esc(t)}" ${t === "AAPL" ? "selected" : ""}>${esc(t)}</option>`
    ).join("");

    select.addEventListener("change", () => {
      state.currentTicker = select.value;
    });
  } catch (err) {
    select.innerHTML = `<option value="AAPL">AAPL</option>`;
  }
}

/* ── Período ───────────────────────────────────────────────────── */

function bindPeriodButtons() {
  document.querySelectorAll(".period-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".period-btn").forEach(b => b.classList.remove("period-btn--active"));
      btn.classList.add("period-btn--active");
      state.currentPeriod = btn.dataset.period;
    });
  });
}

/* ── SMA checkboxes ────────────────────────────────────────────── */

function bindSMACheckboxes() {
  document.querySelectorAll("input[type=checkbox][id^='sma-']").forEach(cb => {
    cb.addEventListener("change", toggleSMATrace);
  });
}

function toggleSMATrace(evt) {
  const period = parseInt(evt.target.value, 10);
  const name   = `SMA ${period}`;
  const divEl  = document.getElementById(CHART_DIV);
  if (!divEl || !divEl.data) return;

  const idx = divEl.data.findIndex(t => t.name === name);
  if (idx === -1) return;

  Plotly.restyle(divEl, { visible: evt.target.checked ? true : "legendonly" }, [idx]);
}

/* ── Carga del chart ───────────────────────────────────────────── */

async function loadChart() {
  const select   = document.getElementById("ticker-select");
  const ticker   = select ? select.value || "AAPL" : state.currentTicker;
  const period   = state.currentPeriod;
  const smaParts = getActiveSMAPeriods();

  const loadingEl = document.getElementById("candlestick-loading");
  const errorEl   = document.getElementById("candlestick-error");
  const divEl     = document.getElementById(CHART_DIV);
  const statsRow  = document.getElementById("stats-row");

  setLoading(loadingEl, true);
  hideError(errorEl);

  try {
    const url  = `/api/candlestick?ticker=${encodeURIComponent(ticker)}&period=${period}&sma=${smaParts.join(",")}`;
    const data = await apiFetch(url);

    setLoading(loadingEl, false);

    // Renderizar figura
    if (data.figure && divEl) {
      if (divEl.data && divEl.data.length > 0) {
        Plotly.react(divEl, data.figure.data, data.figure.layout, { responsive: true });
      } else {
        Plotly.newPlot(divEl, data.figure.data, data.figure.layout, {
          responsive:     true,
          displayModeBar: true,
          modeBarButtonsToRemove: ["select2d", "lasso2d"],
          displaylogo:    false,
        });
      }
    }

    // Mostrar estadísticas
    if (data.stats && statsRow) {
      statsRow.style.display = "grid";
      populateStats(data.stats, data.sma_current ?? {});
    }

    state.currentTicker = ticker;

  } catch (err) {
    setLoading(loadingEl, false);
    showError(errorEl, `Error al cargar ${ticker}: ${err.message}`);
  }
}

/* ── Helpers de UI ─────────────────────────────────────────────── */

function getActiveSMAPeriods() {
  const active = [];
  document.querySelectorAll("input[type=checkbox][id^='sma-']").forEach(cb => {
    if (cb.checked) active.push(parseInt(cb.value, 10));
  });
  return active.length > 0 ? active : [20, 50];
}

function populateStats(stats, smaCurrent) {
  setText("stat-name",    `${stats.name ?? stats.ticker} (${stats.ticker})`);
  setText("stat-close",   stats.current_close != null ? `$${fmtNum(stats.current_close)}` : "—");
  setText("stat-candles", stats.n_candles != null ? fmtInt(stats.n_candles) + " velas" : "—");
  setText("stat-max",     stats.max_high != null ? `$${fmtNum(stats.max_high)}` : "—");
  setText("stat-min",     stats.min_low  != null ? `$${fmtNum(stats.min_low)}`  : "—");

  // Variación con color
  const changeEl = document.getElementById("stat-change");
  if (changeEl && stats.change_pct != null) {
    const sign = stats.change_pct >= 0 ? "+" : "";
    changeEl.textContent  = `${sign}${fmtPct(stats.change_pct / 100)}`;
    changeEl.style.color  = stats.change_pct >= 0 ? "#3fb950" : "#f85149";
  }

  // Fechas
  if (stats.date_start && stats.date_end) {
    setText("stat-dates", `${stats.date_start} → ${stats.date_end}`);
  }

  // SMA values
  const smaContainer = document.getElementById("sma-values");
  if (smaContainer) {
    const entries = Object.entries(smaCurrent);
    if (entries.length === 0) {
      smaContainer.innerHTML = `<p class="text-muted">Sin SMAs seleccionadas</p>`;
    } else {
      smaContainer.innerHTML = entries.map(([p, v]) => `
        <div class="sma-item sma-item--${p}">
          <span class="sma-item__label">SMA ${p}</span>
          <span class="sma-item__value">${v != null ? "$" + fmtNum(v) : "Insuf. datos"}</span>
        </div>
      `).join("");
    }
  }
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
