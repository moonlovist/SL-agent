/* ============================================================
   LLM-Spectro-Agent — theme toggle
   ============================================================ */

(function () {
  'use strict';

  var STORAGE_KEY = 'llm-sa-theme';
  var LIGHT = 'light';

  /* ---- resolve initial theme ---- */
  function resolveTheme() {
    var stored = localStorage.getItem(STORAGE_KEY);
    if (stored === LIGHT) return LIGHT;
    if (stored === 'dark') return 'dark';
    // first visit: follow OS preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
      return LIGHT;
    }
    return LIGHT;
  }

  /* ---- apply ---- */
  function apply(theme) {
    if (theme === LIGHT) {
      document.documentElement.setAttribute('data-theme', LIGHT);
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
    updateIcon(theme);
  }

  /* ---- toggle ---- */
  function toggle() {
    var current = document.documentElement.getAttribute('data-theme') === LIGHT ? LIGHT : 'dark';
    var next = current === LIGHT ? 'dark' : LIGHT;
    apply(next);
    localStorage.setItem(STORAGE_KEY, next);
  }

  /* ---- icon: text-only, no emoji ---- */
  function updateIcon(theme) {
    var btn = document.getElementById('theme-toggle-btn');
    if (!btn) return;
    // ☀ → switch to light  /  ☾ → switch to dark
    btn.textContent = theme === LIGHT ? '☽' : '☀';
    btn.setAttribute('aria-label', theme === LIGHT ? 'Switch to dark theme' : 'Switch to light theme');
  }

  /* ---- build button ---- */
  function buildButton() {
    var btn = document.createElement('button');
    btn.id = 'theme-toggle-btn';
    btn.className = 'theme-toggle';
    btn.setAttribute('aria-label', 'Toggle theme');
    btn.addEventListener('click', toggle);
    document.body.appendChild(btn);
  }

  /* ---- init ---- */
  function init() {
    var theme = resolveTheme();
    buildButton();
    apply(theme);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
