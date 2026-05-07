/**
 * app.js — Utilidades globales compartidas por todas las páginas.
 * Cargado en base.html antes de los scripts específicos de cada vista.
 */

"use strict";

/**
 * Wrapper sobre fetch() que devuelve un objeto JSON o lanza un Error
 * con el mensaje de la API si la respuesta no es 2xx.
 *
 * @param {string} url
 * @param {RequestInit} [options]
 * @returns {Promise<any>}
 */
async function apiFetch(url, options = {}) {
  const defaults = {
    headers: { "Content-Type": "application/json" },
  };
  const config = Object.assign({}, defaults, options);
  if (options.headers) {
    config.headers = Object.assign({}, defaults.headers, options.headers);
  }

  const response = await fetch(url, config);
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(`Respuesta no-JSON del servidor (${response.status}): ${text.slice(0, 200)}`);
  }

  if (!response.ok) {
    const msg = (data && data.error) ? data.error : `Error HTTP ${response.status}`;
    throw new Error(msg);
  }

  return data;
}

/**
 * Activa o desactiva el estado de carga en un elemento.
 *
 * @param {HTMLElement} el
 * @param {boolean} loading
 * @param {string} [loadingText]
 */
function setLoading(el, loading, loadingText = "Cargando…") {
  if (!el) return;
  if (loading) {
    el.style.display = "flex";
    el.innerHTML = `<span class="spinner"></span> ${loadingText}`;
  } else {
    el.style.display = "none";
    el.innerHTML = "";
  }
}

/**
 * Muestra un mensaje de error en un contenedor de alerta.
 *
 * @param {HTMLElement} container
 * @param {string} message
 */
function showError(container, message) {
  if (!container) return;
  container.style.display = "block";
  container.textContent = message;
}

/**
 * Oculta el contenedor de error.
 *
 * @param {HTMLElement} container
 */
function hideError(container) {
  if (!container) return;
  container.style.display = "none";
  container.textContent = "";
}

/**
 * Formatea un número como porcentaje con 2 decimales.
 * @param {number|null} v
 * @returns {string}
 */
function fmtPct(v) {
  if (v == null) return "—";
  return (v * 100).toFixed(2) + "%";
}

/**
 * Formatea un número decimal con n dígitos.
 * @param {number|null} v
 * @param {number} digits
 * @returns {string}
 */
function fmtNum(v, digits = 4) {
  if (v == null) return "—";
  return Number(v).toFixed(digits);
}

/**
 * Formatea un número grande con separadores de miles.
 * @param {number|null} v
 * @returns {string}
 */
function fmtInt(v) {
  if (v == null) return "—";
  return Number(v).toLocaleString("es-CO");
}

/**
 * Devuelve la clase CSS del badge según la categoría de riesgo.
 * @param {string} category
 * @returns {string}
 */
function riskBadgeClass(category) {
  const c = (category || "").toLowerCase();
  if (c.includes("conserv")) return "badge badge--conservative";
  if (c.includes("moder"))   return "badge badge--moderate";
  if (c.includes("agres"))   return "badge badge--aggressive";
  return "badge badge--info";
}
