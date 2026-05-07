/**
 * home.js — Página de inicio: carga las estadísticas del dataset.
 */

"use strict";

(async function () {
  try {
    const data = await apiFetch("/api/dataset/summary");
    const summary = data.summary || [];

    const n_assets  = summary.length;
    const n_records = summary.reduce((acc, r) => acc + (r.n_records || 0), 0);

    const elAssets  = document.getElementById("stat-assets");
    const elRecords = document.getElementById("stat-records");

    if (elAssets)  elAssets.textContent  = n_assets;
    if (elRecords) elRecords.textContent = fmtInt(n_records);
  } catch (err) {
    // No crítico — la página sigue funcionando sin estas estadísticas
    console.warn("No se pudieron cargar las estadísticas:", err.message);
  }
})();
