/**
 * dataset.js — Página Dataset: tabla paginada de registros OHLCV.
 */

"use strict";

(function () {
  // ── Estado ──────────────────────────────────────────────────────
  let currentPage = 1;
  let totalPages  = 1;
  let currentTicker  = "";
  let currentPerPage = 50;

  // ── DOM ─────────────────────────────────────────────────────────
  const tickerSelect  = document.getElementById("ticker-select");
  const perPageSelect = document.getElementById("per-page-select");
  const btnFilter     = document.getElementById("btn-filter");
  const tbody         = document.getElementById("dataset-tbody");
  const statusEl      = document.getElementById("dataset-status");
  const errorEl       = document.getElementById("dataset-error");
  const btnPrev       = document.getElementById("btn-prev");
  const btnNext       = document.getElementById("btn-next");
  const pageInfo      = document.getElementById("page-info");

  // ── Inicialización ───────────────────────────────────────────────
  async function init() {
    await loadTickers();
    await loadData();
  }

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
      showError(errorEl, "No se pudo cargar la lista de activos: " + err.message);
    }
  }

  async function loadData() {
    hideError(errorEl);
    setLoading(statusEl, true);
    tbody.innerHTML = "";

    const params = new URLSearchParams({
      page:     currentPage,
      per_page: currentPerPage,
    });
    if (currentTicker) params.set("ticker", currentTicker);

    try {
      const data = await apiFetch(`/api/dataset?${params}`);
      totalPages = data.total_pages || 1;
      renderTable(data.data || []);
      renderPagination(data);
    } catch (err) {
      showError(errorEl, err.message);
    } finally {
      setLoading(statusEl, false);
    }
  }

  function renderTable(records) {
    if (!records.length) {
      tbody.innerHTML = '<tr><td colspan="8" class="text-muted" style="text-align:center;padding:24px">Sin resultados</td></tr>';
      return;
    }

    const rows = records.map(r => `
      <tr>
        <td>${r.date || "—"}</td>
        <td><strong>${r.ticker}</strong></td>
        <td>${r.instrument_type || "—"}</td>
        <td class="text-right">${fmtNum(r.open, 2)}</td>
        <td class="text-right">${fmtNum(r.high, 2)}</td>
        <td class="text-right">${fmtNum(r.low, 2)}</td>
        <td class="text-right">${fmtNum(r.close, 2)}</td>
        <td class="text-right">${fmtInt(r.volume)}</td>
      </tr>`
    ).join("");
    tbody.innerHTML = rows;
  }

  function renderPagination(data) {
    pageInfo.textContent = `Página ${data.page} de ${data.total_pages} · ${fmtInt(data.total)} registros`;
    btnPrev.disabled = data.page <= 1;
    btnNext.disabled = data.page >= data.total_pages;
  }

  // ── Event listeners ──────────────────────────────────────────────
  btnFilter.addEventListener("click", () => {
    currentPage   = 1;
    currentTicker = tickerSelect.value;
    currentPerPage = parseInt(perPageSelect.value, 10);
    loadData();
  });

  btnPrev.addEventListener("click", () => {
    if (currentPage > 1) { currentPage--; loadData(); }
  });

  btnNext.addEventListener("click", () => {
    if (currentPage < totalPages) { currentPage++; loadData(); }
  });

  init();
})();
