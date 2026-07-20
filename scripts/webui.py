"""
FORMA WebUI — Gradio-based web interface for spectral analysis.

Three-tab layout:
  1. Config  — LLM keys, Redrock toggle, run mode
  2. Run     — FITS upload, live log, start button
  3. Results — folder-structured file browser + batch download

Architecture: wraps the existing CLI pipeline via subprocess.
No refactoring of agent / orchestrator code needed.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import zipfile
import io as _io
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ═══════════════════════════════════════════════════════════════════════════
# Config persistence — saved to volume-mounted output dir, survives restarts
# ═══════════════════════════════════════════════════════════════════════════

_CONFIG_DIR = str(PROJECT_ROOT / "scripts")
_CONFIG_PATH = os.path.join(_CONFIG_DIR, ".forma_webui_config.json")
_CONFIG_SENSITIVE = {"LLM_API_KEY"}

_saved = {}
try:
    if os.path.isfile(_CONFIG_PATH):
        with open(_CONFIG_PATH, "r") as f:
            _saved = json.load(f)
except Exception:
    pass


def _cfg(key: str, fallback: str = "") -> str:
    """Saved config > env var > hardcoded fallback."""
    if key in _saved and _saved[key] != "":
        return str(_saved[key])
    val = os.environ.get(key, "")
    if val:
        return val
    return fallback


def _save_cfg(data: dict) -> None:
    """Write non-sensitive fields to the persisted JSON file."""
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    safe = {k: v for k, v in data.items() if k not in _CONFIG_SENSITIVE}
    with open(_CONFIG_PATH, "w") as f:
        json.dump(safe, f, indent=2)


def _default_env(key: str, fallback: str = "") -> str:
    """Read a value from the real .env / environment, returning *fallback* if unset or empty."""
    val = os.environ.get(key, "")
    return val if val else fallback


def _collect_outputs_tree(output_dir: str) -> dict[str, list[str]]:
    """Return {relative_folder: [file_paths]} sorted newest-first."""
    if not os.path.isdir(output_dir):
        return {}
    tree: dict[str, list[str]] = defaultdict(list)
    for root, _, names in os.walk(output_dir):
        for name in names:
            if name.startswith("."):
                continue
            full = os.path.join(root, name)
            rel_dir = os.path.relpath(root, output_dir)
            if rel_dir == ".":
                rel_dir = "/ (root)"
            tree[rel_dir].append(full)
    # sort files within each folder by mtime descending
    for k in tree:
        tree[k].sort(key=lambda p: os.path.getmtime(p), reverse=True)
    # sort folders: root first, then alphabetical
    return dict(sorted(tree.items(), key=lambda x: (x[0] != "/ (root)", x[0])))


def _build_results_markdown(tree: dict[str, list[str]]) -> str:
    """Render a folder-structured file listing as Markdown."""
    if not tree:
        return "*No output files found.*"

    lines = []
    total = sum(len(v) for v in tree.values())
    lines.append(f"**{total} files** in {len(tree)} folder(s):\n")

    for folder, files in tree.items():
        lines.append(f"### {folder}")
        for f in files:
            name = os.path.basename(f)
            size = os.path.getsize(f)
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            lines.append(f"- `{name}` ({size_str})")
        lines.append("")

    return "\n".join(lines)


def _zip_output_dir(output_dir: str) -> str | None:
    """Create a zip of all output files. Returns path to the zip, or None."""
    if not os.path.isdir(output_dir):
        return None
    zip_path = os.path.join(output_dir, "..", "FORMA_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, names in os.walk(output_dir):
            for name in names:
                if name.startswith("."):
                    continue
                full = os.path.join(root, name)
                arcname = os.path.relpath(full, output_dir)
                zf.write(full, arcname)
    return zip_path


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline runner (generator -> streaming log)
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    fits_files,
    llm_api_key: str, llm_base_url: str, llm_model: str,
    llm_temperature: float, llm_max_tokens: str,
    use_archetypes: bool,
    rr_template_dir: str, archetype_dir: str,
    rr_nminima: int, rr_nnearest: int, omp_threads: int,
    run_mode: str,
    arm_name: str, arm_wavelength_range: str,
    cwt_snr_thresh: float, cwt_min_ridge: int,
    cwt_min_width: float, cwt_max_width: float, cwt_n_scales: int,
):
    """Generator: yields (log_line_str, None, None) while running,
       (log_line_str, result_md, zip_path) on completion."""

    # ── Build session temp directory ──
    session_dir = tempfile.mkdtemp(prefix="forma_session_")
    input_dir = os.path.join(session_dir, "input")
    output_dir = os.path.join(session_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ── Copy uploaded FITS files ──
    if fits_files is None:
        yield "ERROR: No FITS files uploaded.", None, None
        return

    if isinstance(fits_files, str):
        fits_files = [fits_files]

    file_names = []
    for f in fits_files:
        if f is None:
            continue
        fname = os.path.basename(f)
        shutil.copy2(f, os.path.join(input_dir, fname))
        stem = os.path.splitext(fname)[0]
        file_names.append(stem)

    if not file_names:
        yield "ERROR: Could not read any uploaded files.", None, None
        return

    # ── Build env overrides ──
    env_override = {
        "LLM_API_KEY": llm_api_key or "",
        "LLM_BASE_URL": llm_base_url or "",
        "LLM_MODEL": llm_model or "",
        "LLM_TEMPERATURE": str(llm_temperature),
        "LLM_MAX_TOKENS": str(llm_max_tokens) if llm_max_tokens else "",
        "LLM_THINKING": "disabled",
        "REDROCK": "true",
        "RR_TEMPLATE_DIR": rr_template_dir or os.environ.get("RR_TEMPLATE_DIR", "/opt/redrock/py/redrock/templates"),
        "ARCHETYPE_DIR": archetype_dir or os.environ.get("ARCHETYPE_DIR", "/opt/redrock/redrock-archetypes"),
        "USE_ARCHETYPES": "true" if use_archetypes else "false",
        "NMINIMA": str(rr_nminima),
        "NNEAREST": str(rr_nnearest),
        "OMP_NUM_THREADS": str(omp_threads),
        "INPUT_DIR": input_dir,
        "INPUT_FORMAT": "fits",
        "OUTPUT_DIR": output_dir,
        "RUN_MODE": run_mode,
        "FILE_NAME": file_names[0] if run_mode == "s" else "",
        "ARM_NAME": arm_name,
        "ARM_WAVELENGTH_RANGE": arm_wavelength_range,
        "CWT_SNR_THRESH": str(cwt_snr_thresh),
        "CWT_MIN_RIDGE_LENGTH": str(cwt_min_ridge),
        "CWT_MIN_WIDTH": str(cwt_min_width),
        "CWT_MAX_WIDTH": str(cwt_max_width),
        "CWT_N_SCALES": str(cwt_n_scales),
        "SELF_EVOLVE": "false",
    }

    yield f"[{datetime.now().strftime('%H:%M:%S')}] Session: {session_dir}\n", None, None
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Files: {', '.join(file_names)}\n", None, None
    yield f"[{datetime.now().strftime('%H:%M:%S')}] Starting pipeline...\n", None, None

    # ── Run pipeline as subprocess ──
    proc_env = {**os.environ, **env_override}

    try:
        proc = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "main.py")],
            cwd=str(PROJECT_ROOT),
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in iter(proc.stdout.readline, ""):
            yield line, None, None

        proc.stdout.close()
        returncode = proc.wait()

        if returncode != 0:
            yield f"\n[ERROR] Pipeline exited with code {returncode}\n", None, None

    except Exception as exc:
        yield f"\n[ERROR] Failed to launch pipeline: {exc}\n", None, None
        return

    # ── Collect results ──
    yield f"\n[{datetime.now().strftime('%H:%M:%S')}] Pipeline finished.\n", None, None

    tree = _collect_outputs_tree(output_dir)
    md = _build_results_markdown(tree)
    zip_path = _zip_output_dir(output_dir)

    # Also include in_brief.csv content preview if it exists
    for folder, files in tree.items():
        for f in files:
            if os.path.basename(f) == "in_brief.csv":
                try:
                    with open(f, "r") as fh:
                        lines = fh.readlines()
                    if len(lines) > 1:
                        md += "\n---\n### in_brief.csv preview\n```csv\n"
                        md += "".join(lines[:20])  # first 20 lines
                        if len(lines) > 20:
                            md += f"... ({len(lines) - 20} more rows)\n"
                        md += "```\n"
                    else:
                        md += "\n---\n### in_brief.csv\n```csv\n" + "".join(lines) + "```\n*(header only)*\n"
                except Exception:
                    pass
                break

    yield "\nDone.\n", md, zip_path


# ═══════════════════════════════════════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════════════════════════════════════

CSS = r"""
/* ── Imports ── */
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,500;0,600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

/* ── Tokens (light theme from .repo_info/common.css) ── */
:root {
  --space:    #f2f1f5;
  --void:     #e8e7ed;
  --surface:  #dcdbe3;
  --surface2: #d0cfd8;
  --amber:    #9c0c13;
  --amber-dim:#7a4848;
  --emission: #3a7ec8;
  --emission-dim: #2a5a90;
  --absorption: #3465a8;
  --blue-dim: #5078a0;
  --text:     #181a20;
  --text-dim: #3d3f47;
  --text-faint: #6a6c74;
  --rule:     #c6c5ce;
  --green:    #3a8050;
  --font-serif: 'EB Garamond', Georgia, serif;
  --font-mono:  'IBM Plex Mono', monospace;
  --font-sans:  'IBM Plex Sans', system-ui, -apple-system, sans-serif;
}

/* ── Reset & base ── */
body, .gradio-container {
  background: var(--space) !important;
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 22px;
  line-height: 1.7;
}
.gradio-container { max-width: 1300px !important; margin: 0 auto 60px; }
.gradio-container > .contain { padding: 0 44px; }
* { box-sizing: border-box; }

/* ── Spectral edge — the signature left vertical bar ── */
.gradio-container::before {
  content: '';
  position: fixed;
  left: 0; top: 0; bottom: 0;
  width: 5px;
  z-index: 100;
  background: linear-gradient(
    to bottom,
    #1a3a6a 0%, #2a5a9a 8%, #3a7acc 16%, #4a9eff 24%,
    #5ac0c0 32%, #6ad88a 40%, #8ad860 48%, #c0c040 56%,
    #e8a830 64%, #e87030 72%, #e85028 80%,
    #c03020 88%, #8a1a1a 96%, #4a0a0a 100%
  );
}

/* ── Header ── */
h1 {
  font-family: var(--font-serif) !important;
  font-size: 2.8em !important;
  font-weight: 600 !important;
  color: var(--amber) !important;
  letter-spacing: 0.01em !important;
  line-height: 1.2 !important;
  margin: 32px 0 4px !important;
  padding: 0 !important;
  border: none !important;
}

/* Section labels (mono, tiny, uppercase — like "MODULE 1, STEP 1" or "LLM BACKEND") */
.section-label {
  display: block;
  font-family: var(--font-mono);
  font-size: 0.7em;
  color: var(--text-faint);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 16px 0 8px;
  border-bottom: 1px solid var(--rule);
  margin-bottom: 12px;
}

/* ── Overview box ── */
.overview-box {
  background: var(--void);
  border: 1px solid var(--rule);
  border-left: 3px solid var(--absorption);
  padding: 16px 20px;
  margin: 0 0 24px;
  font-size: 0.9em;
  color: var(--text-dim);
  line-height: 1.6;
}
.overview-box strong { color: var(--text); }
.overview-box code { font-family: var(--font-mono); font-size: 0.85em; color: var(--amber-dim); }

/* ── Tags ── */
.tag-row { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; }
.tag {
  font-family: var(--font-mono);
  font-size: 0.68em;
  padding: 3px 8px;
  border-radius: 2px;
  letter-spacing: 0.03em;
  background: rgba(52,101,168,0.08);
  color: var(--absorption);
  border: 1px solid rgba(52,101,168,0.16);
}
.tag.primary { background: rgba(58,126,200,0.14); color: var(--emission); border: 1px solid rgba(58,126,200,0.24); }
.tag.secondary { background: rgba(52,101,168,0.08); color: var(--absorption); border: 1px solid rgba(52,101,168,0.16); }
.tag.muted { background: rgba(0,0,0,0.03); color: var(--text-faint); border: 1px solid var(--rule); }

/* ── Tab bar ── */
.tabs { border: none !important; margin-top: 24px; }
.tab-nav {
  border-bottom: 1px solid var(--rule) !important;
  gap: 0 !important;
  margin-bottom: 28px !important;
}
.tab-nav button {
  font-family: var(--font-mono) !important;
  font-size: 0.78em !important;
  font-weight: 500 !important;
  letter-spacing: 0.05em !important;
  text-transform: uppercase;
  color: var(--text-faint) !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  padding: 10px 22px !important;
  margin: 0 !important;
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s;
}
.tab-nav button.selected {
  color: var(--text) !important;
  border-bottom-color: var(--emission) !important;
}
.tab-nav button:hover { color: var(--text-dim) !important; }

/* ── Card-like group for config sections ── */
.config-group {
  background: var(--void);
  border: 1px solid var(--rule);
  padding: 18px 22px;
  margin-bottom: 16px;
}

/* ── Form fields ── */
input:not([type="checkbox"]):not([type="radio"]):not([type="range"]),
textarea, select {
  background: var(--space) !important;
  border: 1px solid var(--rule) !important;
  color: var(--text) !important;
  border-radius: 2px !important;
  font-family: var(--font-sans) !important;
  font-size: 0.95em !important;
  padding: 10px 12px !important;
  transition: border-color 0.15s;
}
input:focus, textarea:focus, select:focus {
  border-color: var(--absorption) !important;
  box-shadow: 0 0 0 2px rgba(52,101,168,0.1) !important;
  outline: none !important;
}

label {
  font-family: var(--font-sans) !important;
  font-size: 0.75em !important;
  font-weight: 600 !important;
  color: var(--text-dim) !important;
  text-transform: none;
  letter-spacing: 0.01em;
  margin-bottom: 4px !important;
}

input[type="checkbox"], input[type="radio"] { accent-color: var(--emission); }
input[type="range"] { accent-color: var(--emission); }

/* Radio — override Gradio's blue background on selected items */
.wrap-inner label.selected,
fieldset label.selected,
.radio-group label.selected {
  background: var(--void) !important;
  color: var(--text) !important;
}
/* Disabled checkbox — grey it out */
input[type="checkbox"]:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Radio row */
.radio-row {
  display: flex;
  gap: 24px;
  margin: 6px 0;
}
.radio-row label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: var(--font-sans) !important;
  font-size: 0.85em !important;
  color: var(--text-dim) !important;
  text-transform: none;
  letter-spacing: 0;
  cursor: pointer;
}

/* ── Buttons ── */
button.primary, button[variant="primary"], #start_btn {
  background: var(--emission) !important;
  color: #fff !important;
  border: none !important;
  font-family: var(--font-sans) !important;
  font-size: 0.85em !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em;
  text-transform: none;
  padding: 12px 32px !important;
  border-radius: 2px !important;
  cursor: pointer;
  transition: background 0.15s;
}
button.primary:hover, button[variant="primary"]:hover {
  background: #4a8ed8 !important;
}

button.secondary, button[variant="secondary"] {
  background: var(--surface) !important;
  color: var(--text-dim) !important;
  border: 1px solid var(--rule) !important;
  font-family: var(--font-sans) !important;
  font-size: 0.85em !important;
  letter-spacing: 0.02em;
  text-transform: none;
  padding: 10px 22px !important;
  border-radius: 2px !important;
  cursor: pointer;
  transition: border-color 0.15s, color 0.15s;
}
button.secondary:hover, button[variant="secondary"]:hover {
  border-color: var(--amber-dim) !important;
  color: var(--amber) !important;
}

/* ── Log output ── */
#log_output textarea, #log_output > div {
  background: var(--space) !important;
  border: 1px solid var(--rule) !important;
  color: var(--text-dim) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.72em !important;
  line-height: 1.6 !important;
  padding: 14px !important;
  border-radius: 2px !important;
  white-space: pre-wrap !important;
  word-break: break-all;
}

/* ── File upload area ── */
.file-preview, .upload-container, [data-testid="file-upload"] {
  background: var(--void) !important;
  border: 1px dashed var(--rule) !important;
  border-radius: 2px !important;
}

/* ── Results markdown ── */
#results_tree_md .prose {
  font-family: var(--font-sans) !important;
  color: var(--text-dim);
  font-size: 0.85em;
}
#results_tree_md .prose h3 {
  font-family: var(--font-mono) !important;
  font-size: 0.72em !important;
  font-weight: 500 !important;
  color: var(--amber) !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  margin: 20px 0 6px !important;
  padding-bottom: 4px;
  border-bottom: 1px solid var(--rule);
}
#results_tree_md .prose code {
  font-family: var(--font-mono) !important;
  color: var(--amber-dim) !important;
  background: var(--void);
  padding: 2px 5px;
  border-radius: 2px;
  font-size: 0.9em;
}
#results_tree_md .prose pre {
  background: var(--void);
  border: 1px solid var(--rule);
  border-radius: 2px;
  padding: 12px 14px;
  font-size: 0.78em;
}

/* ── Accordion ── */
.accordion {
  background: var(--void) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 2px !important;
}
.accordion > .label-wrap {
  font-family: var(--font-mono) !important;
  font-size: 0.68em !important;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* ── Misc ── */
footer { display: none !important; }
a { color: var(--absorption); text-decoration: none; transition: color 0.15s; }
a:hover { color: var(--amber); }
:focus-visible { outline: 2px solid var(--emission); outline-offset: 2px; }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }

@media (max-width: 768px) {
  .gradio-container::before { width: 3px; }
  .gradio-container > .contain { padding: 0 16px; }
  h1 { font-size: 1.6em !important; }
}
"""

HEAD_HTML = """
<script>
(function(){
  var done = false;
  var obs = new MutationObserver(function(){
    if (done) return;
    var el = document.querySelector('[data-testid="file-upload"]');
    if (!el) return;
    var spans = el.querySelectorAll('span');
    spans.forEach(function(s){
      var t = s.textContent;
      if (t.includes('拖放') || t.includes('Drop')) s.textContent = 'Drag and drop .fits files here, or click to browse';
      if (t.includes('点击') || t.includes('Click')) s.textContent = '';
      if (t.includes('或') || t.includes('or')) s.textContent = '— or —';
    });
    done = true;
    obs.disconnect();
  });
  obs.observe(document.body, {childList:true, subtree:true});
})();
</script>
"""

with gr.Blocks(title="FORMA — Spectral Analysis", head=HEAD_HTML) as demo:

    gr.HTML("""
    <h1>
      <span style="display:block;font-family:'IBM Plex Mono',monospace;font-size:0.25em;
                   font-weight:400;color:var(--text-faint);letter-spacing:0.1em;
                   text-transform:uppercase;margin-bottom:8px;">
        Multi-Agent Spectroscopy
      </span>
      FORMA
    </h1>
    <div class="overview-box">
      <p>
        <strong>An LLM-powered multi-agent system that reads astronomical
        spectra the way a human astronomer does.</strong> From a spectral
        FITS file, it detects features, generates redshift hypotheses
        through <code>redrock</code>, evaluates each with independent LLM
        agents, cross-verifies them through adversarial review, and delivers
        a structured final report with calibrated confidence.
      </p>
      <p style="margin-top:6px;">
        Scientific targets: <strong>source classification</strong>
        (QSO / GALAXY) &amp; <strong>redshift measurement</strong>.
      </p>
      <div class="tag-row">
        <span class="tag primary">Python 3.12</span>
        <span class="tag primary">LangGraph</span>
        <span class="tag primary">LangChain</span>
        <span class="tag secondary">DeepSeek</span>
        <span class="tag muted">Redrock</span>
        <span class="tag muted">CWT</span>
      </div>
    </div>
    """)

    # ── Session state ──
    last_output_dir = gr.State("")

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════
        # Tab 1: Config
        # ══════════════════════════════════════════════════════════════
        with gr.TabItem("Config"):
            gr.HTML('<span class="section-label">LLM Backend</span>')
            with gr.Row():
                llm_key = gr.Textbox(
                    label="API Key", type="password",
                    value=_default_env("LLM_API_KEY"),
                    placeholder="sk-...",
                )
                llm_model = gr.Textbox(
                    label="Model",
                    value=_cfg("LLM_MODEL", "deepseek-v4-pro"),
                    placeholder="deepseek-v4-pro",
                )
            llm_url = gr.Textbox(
                label="Base URL",
                value=_cfg("LLM_BASE_URL"),
                placeholder="https://api.deepseek.com/v1",
            )
            with gr.Row():
                llm_temp = gr.Slider(
                    label="Temperature", minimum=0.0, maximum=1.0, step=0.05,
                    value=float(_cfg("LLM_TEMPERATURE", "0.1")),
                )
                llm_max_tok = gr.Textbox(
                    label="Max Tokens",
                    value=_cfg("LLM_MAX_TOKENS", ""),
                    placeholder="API default",
                )

            gr.HTML('<span class="section-label">Redshift Fitter — Redrock</span>')
            with gr.Row():
                use_redrock = gr.Checkbox(
                    label="Use Redrock",
                    value=True, interactive=False,
                )
                use_archetypes = gr.Checkbox(
                    label="Use Archetypes",
                    value=_default_env("USE_ARCHETYPES", "true").lower() == "true",
                )
            with gr.Row():
                rr_template_dir = gr.Textbox(
                    label="Template Directory",
                    value=_default_env("RR_TEMPLATE_DIR", "/opt/redrock/py/redrock/templates"),
                    placeholder="/opt/redrock/py/redrock/templates",
                )
                archetype_dir = gr.Textbox(
                    label="Archetype Directory",
                    value=_default_env("ARCHETYPE_DIR", "/opt/redrock/redrock-archetypes"),
                    placeholder="/opt/redrock/redrock-archetypes",
                )
            with gr.Row():
                rr_nminima = gr.Number(
                    label="N Minima", value=int(_default_env("NMINIMA", "9")), precision=0,
                    info="Number of redshift minima to explore. Default: 9",
                )
                rr_nnearest = gr.Number(
                    label="N Nearest", value=int(_default_env("NNEAREST", "2")), precision=0,
                    info="Number of nearest archetypes. Default: 2",
                )
                omp_threads = gr.Number(
                    label="OMP Threads", value=int(_default_env("OMP_NUM_THREADS", "1")), precision=0,
                    info="OpenMP threads for Redrock. Default: 1",
                )

            gr.HTML('<span class="section-label">Pipeline</span>')
            run_mode = gr.Radio(
                label="Run Mode", choices=[("Single file", "s"), ("Batch", "b")], value="s",
            )
            with gr.Row():
                arm_name = gr.Textbox(
                    label="Arm Name",
                    value=_cfg("ARM_NAME", "B,R,Z"),
                    placeholder="B,R,Z",
                )
                arm_range = gr.Textbox(
                    label="Wavelength Range per Arm",
                    value=_cfg("ARM_WAVELENGTH_RANGE", "3600-5800,5760-7620,7520-9824"),
                    placeholder="3600-5800,5760-7620,7520-9824",
                )

            gr.HTML('<span class="section-label">CWT Feature Detection</span>')
            cwt_snr = gr.Slider(
                label="SNR Threshold", minimum=1.0, maximum=20.0, step=0.5,
                value=float(_cfg("CWT_SNR_THRESH", "8.0")),
                info="CWT coefficient / local noise lower bound. Larger = stricter. Default: 5.0",
            )
            cwt_min_ridge = gr.Slider(
                label="Min Ridge Length", minimum=1, maximum=10, step=1,
                value=int(_cfg("CWT_MIN_RIDGE_LENGTH", "4")),
                info="Feature must be detected on at least this many scales to be valid. Default: 4",
            )
            with gr.Row():
                cwt_min_width = gr.Number(
                    label="Min Line Width", value=float(_cfg("CWT_MIN_WIDTH", "1.0")), precision=1,
                    info="Narrowest spectral line the CWT detects (FWHM ≈ value × 2.355 px). Default: 1.0",
                )
                cwt_max_width = gr.Number(
                    label="Max Line Width", value=float(_cfg("CWT_MAX_WIDTH", "80.0")), precision=1,
                    info="Widest spectral line the CWT detects (FWHM ≈ value × 2.355 px). Default: 80.0",
                )
            cwt_n_scales = gr.Slider(
                label="Number of Scales", minimum=8, maximum=48, step=4,
                value=int(_cfg("CWT_N_SCALES", "24")),
                info="Wavelet scales, logarithmically spaced. Default: 24",
            )

            gr.HTML('<span class="section-label" style="border-bottom:none;padding-bottom:0;"></span>')
            save_btn = gr.Button("Save Config", variant="primary", elem_id="save_btn")
            save_msg = gr.Markdown("", elem_id="save_msg")

        # ══════════════════════════════════════════════════════════════
        # Tab 2: Run
        # ══════════════════════════════════════════════════════════════
        with gr.TabItem("Run"):
            gr.HTML('<span class="section-label">Input</span>')
            fits_upload = gr.File(
                label="FITS Files",
                file_count="multiple",
                file_types=[".fits", ".fit"],
            )

            start_btn = gr.Button("Start Analysis", variant="primary", elem_id="start_btn")
            gr.HTML('<span class="section-label">Log</span>')
            log_output = gr.Textbox(
                show_label=False,
                lines=20,
                max_lines=80,
                autoscroll=True,
                interactive=False,
                elem_id="log_output",
            )

        # ══════════════════════════════════════════════════════════════
        # Tab 3: Results
        # ══════════════════════════════════════════════════════════════
        with gr.TabItem("Results"):
            gr.HTML('<span class="section-label">Output Files</span>')
            results_tree_md = gr.Markdown("*Run the pipeline to see results.*",
                elem_id="results_tree_md",
            )
            with gr.Row():
                download_all = gr.DownloadButton(
                    "Download All (ZIP)", variant="secondary",
                )
                refresh_btn = gr.Button("Refresh", variant="secondary")

    # ══════════════════════════════════════════════════════════════════
    # Event wiring
    # ══════════════════════════════════════════════════════════════════

    def _start_pipeline(fits, key, url, model, temp, max_tok,
                        use_arch, rr_tmp, arch_dir, nmin, nnear, omp,
                        mode, arm, arm_rng,
                        snr, min_ridge, min_width, max_width, n_scales):
        """Consume the generator and accumulate log + results."""
        log_lines = []
        result_md = ""
        zip_file = None
        for line, md, zpath in run_pipeline(
            fits, key, url, model, temp, max_tok,
            use_arch, rr_tmp, arch_dir, nmin, nnear, omp,
            mode, arm, arm_rng,
            snr, min_ridge, min_width, max_width, n_scales,
        ):
            log_lines.append(line)
            if md is not None:
                result_md = md
            if zpath is not None:
                zip_file = zpath
            yield (
                "".join(log_lines),
                result_md or "*Running...*",
                zip_file,
            )

    start_btn.click(
        fn=_start_pipeline,
        inputs=[
            fits_upload, llm_key, llm_url, llm_model, llm_temp, llm_max_tok,
            use_archetypes, rr_template_dir, archetype_dir,
            rr_nminima, rr_nnearest, omp_threads,
            run_mode, arm_name, arm_range, cwt_snr,
            cwt_min_ridge, cwt_min_width, cwt_max_width, cwt_n_scales,
        ],
        outputs=[log_output, results_tree_md, download_all],
    )

    def _refresh_results():
        """Re-scan the most recent session output."""
        tmp_root = tempfile.gettempdir()
        candidates = sorted(
            [d for d in os.listdir(tmp_root) if d.startswith("forma_session_")],
            reverse=True,
        )
        if not candidates:
            return "*No session found.*", None
        output_dir = os.path.join(tmp_root, candidates[0], "output")
        tree = _collect_outputs_tree(output_dir)
        md = _build_results_markdown(tree)
        zip_path = _zip_output_dir(output_dir)
        return md, zip_path

    def _do_save(key, model, url, temp, max_tok,
                  use_arch, rr_tmp, arch_dir, nmin, nnear, omp,
                  mode, arm, arm_rng,
                  snr, min_ridge, min_width, max_width, n_scales):
        _save_cfg({
            "LLM_MODEL": model,
            "LLM_BASE_URL": url,
            "LLM_TEMPERATURE": str(temp),
            "LLM_MAX_TOKENS": max_tok,
            "USE_ARCHETYPES": "true" if use_arch else "false",
            "RR_TEMPLATE_DIR": rr_tmp,
            "ARCHETYPE_DIR": arch_dir,
            "NMINIMA": str(nmin),
            "NNEAREST": str(nnear),
            "OMP_NUM_THREADS": str(omp),
            "RUN_MODE": mode,
            "ARM_NAME": arm,
            "ARM_WAVELENGTH_RANGE": arm_rng,
            "CWT_SNR_THRESH": str(snr),
            "CWT_MIN_RIDGE_LENGTH": str(min_ridge),
            "CWT_MIN_WIDTH": str(min_width),
            "CWT_MAX_WIDTH": str(max_width),
            "CWT_N_SCALES": str(n_scales),
        })
        return f"Saved to `{_CONFIG_PATH}`. Restart the container to reload."

    save_btn.click(
        fn=_do_save,
        inputs=[
            llm_key, llm_model, llm_url, llm_temp, llm_max_tok,
            use_archetypes, rr_template_dir, archetype_dir,
            rr_nminima, rr_nnearest, omp_threads,
            run_mode, arm_name, arm_range, cwt_snr,
            cwt_min_ridge, cwt_min_width, cwt_max_width, cwt_n_scales,
        ],
        outputs=[save_msg],
    )

    refresh_btn.click(
        fn=_refresh_results,
        inputs=[],
        outputs=[results_tree_md, download_all],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    demo.queue(max_size=3).launch(
        server_name=host,
        server_port=port,
        share=False,
        css=CSS,
        theme=gr.themes.Soft(),
    )
