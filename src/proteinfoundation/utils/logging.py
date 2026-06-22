"""Project-local config logging: Rich console tree + HTML dashboard."""

import html
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.syntax import Syntax
from rich.tree import Tree

# ---------------------------------------------------------------------------
# pruning helpers (private; previously lived in foundry.utils.logging)
# ---------------------------------------------------------------------------

_EMPTY_VALUES: tuple[Any, ...] = (0, 0.0, False, None, "", [], {})


def _prune_dead_values(
    obj: Any,
    custom_prune: Callable[[dict], tuple[dict, int] | None] | None = None,
) -> tuple[Any, int]:
    """Recursively drop zero/None/False/empty values from a plain container.

    ``custom_prune`` is a hook called on every dict before generic pruning; it
    returns ``(transformed_dict, hidden_count)`` to pre-transform the dict, or
    ``None`` to fall through. ``hidden_count`` is summed into the returned total.
    """
    if isinstance(obj, dict):
        container: dict = obj
        local_hidden = 0
        if custom_prune is not None:
            result = custom_prune(obj)
            if result is not None:
                container, local_hidden = result
        out: dict[str, Any] = {}
        descendant_hidden = 0
        for key, value in container.items():
            pruned, child_hidden = _prune_dead_values(value, custom_prune=custom_prune)
            descendant_hidden += child_hidden
            if pruned in _EMPTY_VALUES:
                continue
            out[key] = pruned
        return out, descendant_hidden + local_hidden
    if isinstance(obj, list):
        pruned_list, hidden = [], 0
        for item in obj:
            pruned_item, child_hidden = _prune_dead_values(item, custom_prune=custom_prune)
            hidden += child_hidden
            pruned_list.append(pruned_item)
        return pruned_list, hidden
    return obj, 0


def _render_pruned_yaml(
    container: Any,
    comment_keys: dict[str, Callable[[str], str]] | None = None,
) -> str:
    """YAML-serialise a pruned container, rewriting sentinel keys as comments."""
    if not container:
        return "{}\n"
    raw = OmegaConf.to_yaml(OmegaConf.create(container))
    if not comment_keys:
        return raw
    out_lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.lstrip(" ")
        indent = line[: len(line) - len(stripped)]
        for key, formatter in comment_keys.items():
            prefix = f"{key}:"
            if stripped.startswith(prefix):
                out_lines.append(f"{indent}{formatter(stripped[len(prefix):].strip())}")
                break
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def _prune_design_tasks(obj: dict) -> tuple[dict, int] | None:
    """Collapse ``design_tasks`` entries with ``frequency: 0.0`` into a sentinel.

    A dict is recognised as a ``design_tasks`` mapping if every value is itself a
    dict containing a ``"frequency"`` key. Zero-frequency entries are dropped and
    counted under ``__hidden_tasks__``. Returns ``None`` when *obj* is not a
    design-tasks mapping (fall through to generic pruning).
    """
    if not obj or not all(isinstance(v, dict) and "frequency" in v for v in obj.values()):
        return None
    kept: dict[str, Any] = {}
    hidden = 0
    for key, value in obj.items():
        if value.get("frequency") in (0, 0.0):
            hidden += 1
        else:
            kept[key] = value
    if hidden > 0:
        kept["__hidden_tasks__"] = hidden
    return kept, hidden


_DESIGN_TASK_COMMENT_KEYS = {
    "__hidden_tasks__": lambda v: f"# hidden: {v} tasks @ frequency 0.0",
}


# ---------------------------------------------------------------------------
# Public entry point (console + optional HTML dashboard)
# ---------------------------------------------------------------------------


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    resolve: bool = False,
    console_width: int = 100,
    title: str = "Config",
    hide_defaults: bool = True,
    save_html: str | Path | None = None,
) -> None:
    """Print a DictConfig as a Rich tree; optionally save an HTML dashboard.

    Zero-frequency ``design_tasks`` entries are collapsed into a single summary
    comment in both the console output and the HTML dashboard.
    """
    console = Console(width=console_width)
    style = "dim"
    tree = Tree(title, style=style, guide_style=style)

    for field in cfg:
        branch = tree.add(field, style=style, guide_style=style)
        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            if hide_defaults:
                container = OmegaConf.to_container(config_group, resolve=resolve)
                pruned, _ = _prune_dead_values(container, custom_prune=_prune_design_tasks)
                branch_content = _render_pruned_yaml(pruned, comment_keys=_DESIGN_TASK_COMMENT_KEYS)
            else:
                branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)
        branch.add(Syntax(branch_content, "yaml", word_wrap=True))

    console.print(tree)

    if save_html is not None:
        save_config_html(cfg, save_html, title=title)


# ---------------------------------------------------------------------------
# HTML dashboard — palette + stylesheet
# ---------------------------------------------------------------------------

# Tableau-style qualitative palette extended for experiments with many datasets.
# Works on both light and dark bg.
_PALETTE: list[str] = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#EECA3B",
    "#FF9DA6",
    "#9D755D",
    "#17BECF",
    "#BCBD22",
    "#AEC7E8",
]

# Self-contained stylesheet. Compact, but kept under the 120-col line limit;
# retains the light/dark themes and layout, drops animation and responsive
# breakpoints.
_CSS = r"""
:root {
  --bg: #f7f8fa; --bg-grad-a: #f4f5f8; --bg-grad-b: #eef0f5; --surface: #ffffff;
  --surface-2: #f3f4f7; --surface-3: #eceef3; --border: #e4e6eb; --border-strong: #d4d7df;
  --text: #1a1d24; --text-muted: #626774; --text-subtle: #8f95a3; --accent: #5b6cff;
  --accent-soft: rgba(91, 108, 255, 0.10); --accent-border: rgba(91, 108, 255, 0.30);
  --shadow-sm: 0 1px 2px rgba(18, 22, 35, 0.04);
  --shadow: 0 2px 4px rgba(18, 22, 35, 0.05), 0 1px 2px rgba(18, 22, 35, 0.04);
  --shadow-lg: 0 8px 24px rgba(18, 22, 35, 0.07), 0 2px 4px rgba(18, 22, 35, 0.04);
  --radius: 12px; --radius-sm: 8px; --radius-xs: 5px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0c0e13; --bg-grad-a: #0f1218; --bg-grad-b: #0b0d12; --surface: #16191f;
    --surface-2: #1b1e26; --surface-3: #22252e; --border: #262a33; --border-strong: #353945;
    --text: #e9ebef; --text-muted: #a0a5b2; --text-subtle: #6c7280; --accent: #8b95ff;
    --accent-soft: rgba(139, 149, 255, 0.14); --accent-border: rgba(139, 149, 255, 0.35);
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.30);
    --shadow: 0 2px 4px rgba(0, 0, 0, 0.30), 0 1px 2px rgba(0, 0, 0, 0.35);
    --shadow-lg: 0 12px 32px rgba(0, 0, 0, 0.45), 0 2px 4px rgba(0, 0, 0, 0.30);
  }
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  min-height: 100vh; padding: 10px 12px 24px; font-size: 12.5px; line-height: 1.45;
  color: var(--text); -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue",
               Arial, "Inter", sans-serif;
  background: radial-gradient(circle at 15% 0%, var(--bg-grad-a) 0%, transparent 40%),
              radial-gradient(circle at 85% 100%, var(--bg-grad-b) 0%, transparent 50%), var(--bg);
}
code, .mono, pre {
  font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Monaco, Consolas,
               "Liberation Mono", monospace;
  font-variant-ligatures: none;
}
.container { max-width: 1600px; margin: 0 auto; }

/* ---- Header ---- */
.top-header {
  display: flex; align-items: flex-end; justify-content: space-between; gap: 12px;
  flex-wrap: wrap; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border);
}
.top-header .title-group { display: flex; align-items: baseline; flex-wrap: wrap; gap: 6px 10px; }
.top-header .title-group h1 {
  margin: 0; font-size: 18px; font-weight: 700; letter-spacing: -0.015em;
  background: linear-gradient(135deg, var(--text) 0%, var(--text-muted) 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.top-header .subtitle { color: var(--text-muted); font-size: 12px; margin-top: 3px; }
.top-header .badge-row { display: flex; flex-wrap: wrap; gap: 5px; align-items: center; }

/* ---- Main grid + cards ---- */
.grid { display: grid; grid-template-columns: repeat(12, minmax(0, 1fr)); gap: 10px; }
.card {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  box-shadow: var(--shadow); padding: 12px 14px; grid-column: span 6; min-width: 0;
}
.card:hover { box-shadow: var(--shadow-lg); }
.card.span-12 { grid-column: span 12; }
.card.span-8 { grid-column: span 8; }
.card.span-6 { grid-column: span 6; }
.card.span-4 { grid-column: span 4; }
.card.span-3 { grid-column: span 3; }
.card-title {
  font-size: 10.5px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--text-muted); margin: 0 0 8px; display: flex; align-items: center; gap: 8px;
}
.card-title::before {
  content: ""; width: 3px; height: 12px; border-radius: 2px; background: var(--accent);
  display: inline-block;
}
.card-title .count {
  margin-left: auto; font-size: 10.5px; color: var(--text-subtle); font-weight: 500;
  letter-spacing: 0.04em;
}
.section-label {
  font-size: 10.5px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--text-subtle); margin: 18px 0 8px;
}

/* ---- Chips / pills ---- */
.chip {
  display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 999px;
  background: var(--surface-2); color: var(--text); font-size: 11.5px; font-weight: 500;
  border: 1px solid var(--border); white-space: nowrap;
}
.chip.accent { background: var(--accent-soft); color: var(--accent); border-color: var(--accent-border); }
.chip.muted { color: var(--text-muted); }
.chip .k { color: var(--text-muted); font-weight: 400; }

/* ---- KV tables ---- */
dl.kv {
  display: grid; grid-template-columns: max-content 1fr; gap: 6px 20px; margin: 0; font-size: 13px;
}
dl.kv dt { color: var(--text-muted); font-weight: 500; font-size: 12.5px; }
dl.kv dd {
  margin: 0; color: var(--text); font-family: ui-monospace, monospace; font-size: 12.5px;
  overflow-wrap: anywhere;
}

/* ---- Stats (hardware) ---- */
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(95px, 1fr)); gap: 8px; }
.stat {
  padding: 8px 10px 7px; border-radius: var(--radius-sm); background: var(--surface-2);
  border: 1px solid var(--border);
}
.stat .value {
  font-size: 20px; font-weight: 700; color: var(--text); font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em; line-height: 1.1;
}
.stat .value .unit { font-size: 11px; color: var(--text-muted); font-weight: 500; margin-left: 3px; }
.stat .label {
  font-size: 9.5px; color: var(--text-subtle); margin-top: 3px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.06em;
}
.stat.accent { border-color: var(--accent-border); }
.stat.accent .value { color: var(--accent); }

/* ---- Train mix bar ---- */
.mix-bar {
  display: flex; width: 100%; height: 34px; border-radius: var(--radius-xs); overflow: hidden;
  background: var(--surface-2); border: 1px solid var(--border);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.04);
}
.mix-bar > .seg {
  display: flex; align-items: center; padding: 0 10px; color: #ffffff; font-size: 11.5px;
  font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.12);
}
.mix-bar > .seg + .seg { border-left: 1px solid rgba(255, 255, 255, 0.18); }
.mix-bar > .seg:hover { filter: brightness(1.08); }
.mix-legend { display: flex; flex-wrap: wrap; gap: 8px 16px; margin-top: 10px; font-size: 12px; }
.mix-legend .item { display: inline-flex; align-items: center; gap: 7px; color: var(--text-muted); }
.mix-legend .item .swatch { width: 10px; height: 10px; border-radius: 2px; }
.mix-legend .item .val { color: var(--text); font-variant-numeric: tabular-nums; font-weight: 500; }

/* ---- Shared-preset banners ---- */
.preset-banners { margin: 14px 0 4px; display: flex; flex-direction: column; gap: 8px; }
.preset-banner {
  padding: 10px 12px; background: var(--surface-2); border: 1px solid var(--border);
  border-radius: var(--radius-sm); font-size: 12.5px; display: flex; flex-wrap: wrap;
  align-items: center; gap: 8px 10px; border-left: 3px solid var(--accent);
}
.preset-banner .pb-label {
  font-size: 10.5px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--text-subtle);
}
.preset-banner .pb-name {
  font-family: ui-monospace, monospace; font-size: 12.5px; color: var(--text); font-weight: 600;
}
.preset-banner .pb-arrow { color: var(--text-subtle); }
.preset-banner .pb-datasets { display: flex; flex-wrap: wrap; gap: 5px; }

/* ---- Dataset panels ---- */
.dataset-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 10px;
  margin-top: 10px;
}
.dataset-panel {
  background: var(--surface-2); border: 1px solid var(--border);
  border-left: 3px solid var(--border-strong); border-radius: var(--radius-sm);
  padding: 9px 10px 8px; display: flex; flex-direction: column; gap: 6px;
}
.dataset-panel:hover { box-shadow: var(--shadow); }
.dataset-panel .panel-head {
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
}
.dataset-panel .panel-name-wrap { min-width: 0; flex: 1; }
.dataset-panel .panel-name {
  font-weight: 600; font-size: 14px; color: var(--text); letter-spacing: -0.005em;
}
.dataset-panel .panel-sub {
  color: var(--text-muted); font-size: 11px; margin-left: 6px; font-weight: 400;
}
.dataset-panel .prob-pill {
  padding: 3px 10px; border-radius: 999px; font-size: 11.5px; font-weight: 700; color: #ffffff;
  font-variant-numeric: tabular-nums; letter-spacing: -0.01em;
  text-shadow: 0 1px 1px rgba(0, 0, 0, 0.10);
}
.dataset-panel .panel-preset {
  display: inline-flex; align-items: center; gap: 6px; font-size: 11px; color: var(--text-muted);
}
.dataset-panel .panel-preset .swatch { width: 8px; height: 8px; border-radius: 2px; }
.dataset-panel .panel-preset .name { font-family: ui-monospace, monospace; color: var(--text); }
.dataset-panel .panel-facts { display: flex; flex-wrap: wrap; gap: 4px; font-size: 11.5px; }
.dataset-panel .path {
  font-family: ui-monospace, monospace; font-size: 10.5px; color: var(--text-muted);
  word-break: break-all; line-height: 1.4;
}

/* ---- Task distribution (chips + mini bar) ---- */
.task-block { margin-top: 2px; }
.task-block .tb-heading {
  display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 5px;
  font-size: 10.5px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--text-subtle);
}
.task-bar {
  display: flex; width: 100%; height: 5px; border-radius: 3px; overflow: hidden;
  margin-bottom: 7px; background: var(--surface-3);
}
.task-bar > span:hover { filter: brightness(1.15); }
.task-chips { display: flex; flex-wrap: wrap; gap: 5px; }
.task-chip {
  padding: 4px 9px; border-radius: 5px; font-size: 11.5px; background: var(--surface);
  border: 1px solid var(--border); display: inline-flex; align-items: baseline; gap: 7px;
  min-width: 0;
}
.task-chip:hover { border-color: var(--border-strong); }
.task-chip .name { font-weight: 500; color: var(--text); }
.task-chip .freq {
  color: var(--text-subtle); font-family: ui-monospace, monospace; font-size: 10px;
}
.task-chip .pct {
  color: var(--text); font-size: 12.5px; font-weight: 700; font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em;
}
.hidden-note { margin-top: 6px; color: var(--text-subtle); font-size: 11px; font-style: italic; }

/* ---- Task modifiers + per-task kwargs ---- */
.modifiers-strip {
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 8px;
  padding: 6px 8px; background: var(--surface); border: 1px dashed var(--border-strong);
  border-radius: var(--radius-xs);
}
.modifiers-strip .mod-label {
  font-size: 9.5px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--text-subtle); margin-right: 2px;
}
.mod-pill {
  display: inline-flex; align-items: baseline; gap: 5px; padding: 2px 8px; border-radius: 999px;
  background: var(--accent-soft); border: 1px solid var(--accent-border); font-size: 11px;
}
.mod-pill .mod-name {
  color: var(--text-muted); font-family: ui-monospace, monospace; font-size: 10.5px;
}
.mod-pill .mod-val { color: var(--accent); font-weight: 700; font-variant-numeric: tabular-nums; }
/* ---- Per-task modifiers list (inside dataset panel) ---- */
.task-mods-list { margin-top: 6px; display: flex; flex-direction: column; gap: 3px; }
.mod-row {
  display: flex; align-items: center; gap: 8px; padding: 2px 0; min-width: 0;
}
.mod-row .mod-row-name {
  font-family: ui-monospace, monospace; font-size: 11px; color: var(--text);
  flex: 0 0 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  max-width: 50%;
}
.mod-row .mod-row-pills { display: flex; flex-wrap: wrap; gap: 4px; min-width: 0; }
.task-kwargs-expander { margin-top: 8px; }
.task-details-body {
  display: flex; flex-direction: column; gap: 8px; background: var(--bg); padding: 10px 12px;
  min-width: 0; overflow-x: hidden;
}
.task-detail {
  min-width: 0; display: flex; flex-direction: column; gap: 3px; padding-bottom: 7px;
  border-bottom: 1px dashed var(--border);
}
.task-detail:last-child { border-bottom: none; padding-bottom: 0; }
.td-head {
  display: flex; align-items: baseline; flex-wrap: wrap; gap: 4px 8px; min-width: 0;
}
.td-name {
  font-family: ui-monospace, monospace; font-weight: 600; color: var(--text); font-size: 12px;
  overflow-wrap: anywhere; min-width: 0;
}
.td-pct { font-variant-numeric: tabular-nums; font-weight: 600; color: var(--accent); font-size: 11px; }
.td-kwargs { display: flex; flex-wrap: wrap; gap: 4px 7px; padding-left: 8px; min-width: 0; }
.td-kwargs .kw {
  display: inline-flex; align-items: baseline; gap: 4px; font-size: 11px; padding: 1px 6px;
  background: var(--surface-2); border-radius: 3px; max-width: 100%; overflow-wrap: anywhere;
  min-width: 0;
}
.td-kwargs .kw .k {
  color: var(--text-muted); font-family: ui-monospace, monospace; overflow-wrap: anywhere;
}
.td-kwargs .kw .v {
  color: var(--text); font-family: ui-monospace, monospace; font-weight: 500;
  overflow-wrap: anywhere;
}
.td-kwargs .kw-none {
  font-size: 11px; color: var(--text-subtle); font-style: italic; padding-left: 8px;
}

/* ---- Header run-name label ---- */
.run-badge {
  display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 999px;
  border: 1px solid var(--accent-border);
  background: linear-gradient(135deg, rgba(91, 108, 255, 0.12) 0%, rgba(91, 108, 255, 0.05) 100%);
}
.run-badge, .run-badge * { -webkit-text-fill-color: initial; }
.run-badge::before {
  content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}
.run-badge .run-label {
  font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted);
  font-weight: 700;
}
.run-badge .run-name {
  font-family: ui-monospace, "SF Mono", monospace; font-weight: 600; color: var(--text);
  font-size: 13px; letter-spacing: -0.005em;
}
@media (prefers-color-scheme: dark) {
  .run-badge {
    background: linear-gradient(135deg, rgba(139, 149, 255, 0.14) 0%, rgba(139, 149, 255, 0.06) 100%);
  }
}

/* ---- Expand toggles ---- */
details.expander { margin-top: 2px; }
details.expander > summary {
  cursor: pointer; font-size: 11.5px; color: var(--text-muted); padding: 4px 0; list-style: none;
  user-select: none; display: inline-flex; align-items: center; gap: 4px;
}
details.expander > summary::-webkit-details-marker { display: none; }
details.expander > summary::before {
  content: "\25b8"; display: inline-block; font-size: 10px; color: var(--text-subtle);
}
details.expander[open] > summary::before { transform: rotate(90deg); }
details.expander > summary:hover { color: var(--text); }
details.expander > .body {
  margin-top: 8px; padding: 12px 14px; background: var(--bg); border-radius: var(--radius-sm);
  border: 1px solid var(--border);
}
pre.yaml {
  margin: 0; font-family: ui-monospace, "SF Mono", monospace; font-size: 12px; line-height: 1.55;
  white-space: pre; overflow-x: auto; color: var(--text); max-height: 440px; overflow-y: auto;
}
pre.yaml::-webkit-scrollbar { width: 8px; height: 8px; }
pre.yaml::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }
pre.yaml::-webkit-scrollbar-track { background: transparent; }

/* ---- Val datasets ---- */
.val-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px;
  margin-top: 10px;
}
.val-item {
  padding: 10px 12px; background: var(--surface-2); border: 1px solid var(--border);
  border-radius: var(--radius-sm); font-size: 12.5px; display: flex; flex-direction: column;
  gap: 6px;
}
.val-item .val-name { font-weight: 600; color: var(--text); font-size: 13px; }
.val-item .val-row {
  display: flex; flex-wrap: wrap; gap: 5px; color: var(--text-muted); font-size: 11px;
}
.val-item .metric-tag {
  padding: 1px 7px; border-radius: 4px; background: var(--surface-3); color: var(--text-subtle);
  font-size: 10.5px; font-weight: 500;
}
.val-item .val-metrics { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 2px; }
.val-item .val-metrics .metrics-label {
  font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-subtle);
  font-weight: 600; align-self: center; margin-right: 4px;
}
.val-item .val-metrics .metric-chip {
  padding: 2px 8px; border-radius: 4px; background: var(--accent-soft); color: var(--accent);
  border: 1px solid var(--accent-border); font-size: 10.5px; font-weight: 500;
  font-family: ui-monospace, monospace;
}
.val-item .val-metrics .metric-chip.universal { opacity: 0.75; }
.val-item .no-metrics { font-size: 10.5px; color: var(--text-subtle); font-style: italic; }

/* ---- Feature pills + system card ---- */
.feat-pills { display: flex; flex-wrap: wrap; gap: 5px; }
.feat-pill {
  padding: 3px 9px; border-radius: 5px; background: var(--surface-2); border: 1px solid var(--border);
  font-size: 11.5px; font-family: ui-monospace, monospace; color: var(--text);
}
.system-card .stats { gap: 10px; }
.system-card .stat { padding: 14px 14px 12px; }
.system-card .stat .sub {
  font-size: 10.5px; color: var(--text-subtle); margin-top: 2px; font-family: ui-monospace, monospace;
}
.feature-row {
  display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; padding-top: 14px;
  border-top: 1px solid var(--border);
}
.feature-chip {
  display: inline-flex; align-items: baseline; gap: 7px; padding: 5px 11px;
  background: var(--surface-2); border-radius: 6px; font-size: 12px; border: 1px solid var(--border);
}
.feature-chip .label {
  color: var(--text-muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em;
  font-weight: 600;
}
.feature-chip .value { font-family: ui-monospace, monospace; font-weight: 500; color: var(--text); }
.feature-chip .value.on { color: #54a24b; }
.feature-chip .value.off { color: var(--text-subtle); }
.ckpt-row {
  display: flex; align-items: center; gap: 14px; margin-top: 14px; padding: 12px 14px;
  background: var(--surface-2); border-radius: var(--radius-sm); border: 1px solid var(--border);
  border-left: 3px solid var(--accent); flex-wrap: wrap;
}
.ckpt-row .ckpt-label {
  font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--accent);
}
.ckpt-row .ckpt-file {
  font-family: ui-monospace, monospace; font-weight: 600; color: var(--text); font-size: 13px;
}
.ckpt-row .ckpt-dir {
  font-family: ui-monospace, monospace; color: var(--text-subtle); font-size: 11px;
  overflow-wrap: anywhere;
}
"""


# ---------------------------------------------------------------------------
# small rendering helpers
# ---------------------------------------------------------------------------


def _safe(s: Any) -> str:
    """HTML-escape a value (including quotes)."""
    return html.escape(str(s), quote=True)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a ``#rrggbb`` color to an ``rgba(...)`` string with the given alpha."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(140, 140, 140, {alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _num(x: Any, default: float = 0.0) -> float:
    """Coerce ``x`` to ``float``, returning ``default`` when it is not numeric."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _pretty_yaml(value: Any) -> str:
    """Render a value as HTML-escaped pruned YAML."""
    if not isinstance(value, dict | list):
        return _safe(value)
    pruned, _ = _prune_dead_values(value, custom_prune=_prune_design_tasks)
    return _safe(_render_pruned_yaml(pruned, comment_keys=_DESIGN_TASK_COMMENT_KEYS))


def _fmt_percent(x: float) -> str:
    """Format a fraction in ``[0, 1]`` as a percentage string."""
    return f"{x * 100:.1f}%" if x < 0.995 else f"{x * 100:.0f}%"


def _fmt_num(x: Any) -> str:
    """Format a number compactly, dropping a trailing ``.0`` from whole floats."""
    if isinstance(x, float):
        return str(int(x)) if x == int(x) else f"{x:g}"
    return str(x)


def _stat_tile(value: str, label: str, sub: str = "", accent: bool = False) -> str:
    """Render one stat tile for the system card. ``value`` is treated as raw HTML."""
    cls = "stat accent" if accent else "stat"
    sub_html = f'<div class="sub">{_safe(sub)}</div>' if sub else ""
    return (
        f'<div class="{cls}"><div class="value">{value}</div>'
        f'{sub_html}<div class="label">{_safe(label)}</div></div>'
    )


def _card(title: str, body: str, span: int = 6, count: str = "", extra_cls: str = "") -> str:
    """Wrap *body* in a titled ``<section class="card">``. ``count`` is raw HTML."""
    cls = f"card span-{span}" + (f" {extra_cls}" if extra_cls else "")
    count_html = f' <span class="count">{count}</span>' if count else ""
    return f'<section class="{cls}"><h2 class="card-title">{_safe(title)}{count_html}</h2>' f"{body}</section>"


# ---------------------------------------------------------------------------
# dataset / preset extraction
# ---------------------------------------------------------------------------


def _iter_leaf_datasets(raw_train: dict, resolved_train: dict) -> list[tuple[str, dict, dict | None]]:
    """Flatten train datasets to leaves, expanding the ``sub_datasets`` pattern.

    A top-level entry with a ``sub_datasets`` dict is expanded into leaves named
    ``<parent>/<child>``; child probabilities are normalised to sum to the
    parent's share (equal split when no child declares one).

    Returns a list of ``(leaf_name, resolved_leaf, raw_leaf_or_None)``.
    """
    leaves: list[tuple[str, dict, dict | None]] = []
    for name, resolved_entry in (resolved_train or {}).items():
        if not isinstance(resolved_entry, dict):
            continue
        raw_entry = raw_train.get(name) if isinstance(raw_train, dict) else None
        sub = resolved_entry.get("sub_datasets")
        if not (isinstance(sub, dict) and sub):
            leaves.append((name, resolved_entry, raw_entry))
            continue

        parent_prob = _num(resolved_entry.get("probability"))
        raw_sub = raw_entry.get("sub_datasets") if isinstance(raw_entry, dict) else None
        child_probs = {sn: _num(se.get("probability")) for sn, se in sub.items() if isinstance(se, dict)}
        total_child = sum(child_probs.values())
        if total_child <= 0:
            eq = parent_prob / max(len(child_probs), 1)
            child_probs = {k: eq for k in child_probs}
        else:
            child_probs = {k: parent_prob * v / total_child for k, v in child_probs.items()}

        for sub_name, sub_entry in sub.items():
            if not isinstance(sub_entry, dict):
                continue
            leaf = dict(sub_entry)
            leaf["probability"] = child_probs.get(sub_name, 0.0)
            raw_leaf = raw_sub.get(sub_name) if isinstance(raw_sub, dict) else None
            leaves.append((f"{name}/{sub_name}", leaf, raw_leaf))
    return leaves


def _leaf_raw_design_tasks(raw_leaf: Any) -> Any:
    """Return the raw (possibly ``${...}``) design_tasks value for a leaf, or None."""
    if not isinstance(raw_leaf, dict):
        return None
    dataset = raw_leaf.get("dataset")
    transform = dataset.get("transform") if isinstance(dataset, dict) else None
    return transform.get("design_tasks") if isinstance(transform, dict) else None


def _preset_key(raw_design_tasks: Any, fallback_path: str) -> tuple[str, str]:
    """Bucket a design_tasks value by preset reference; inline values get per-leaf keys."""
    if isinstance(raw_design_tasks, str):
        s = raw_design_tasks.strip()
        if s.startswith("${") and s.endswith("}"):
            return ("preset", s[2:-1].lstrip("_"))
    return ("inline", fallback_path)


def _build_preset_index(
    leaves: list[tuple[str, dict, dict | None]],
) -> tuple[dict[tuple[str, str], list[str]], dict[str, tuple[str, str]]]:
    """Return ``({preset_key: [leaf_name, ...]}, {leaf_name: preset_key})``."""
    index: dict[tuple[str, str], list[str]] = {}
    per_leaf: dict[str, tuple[str, str]] = {}
    for leaf_name, _resolved, raw_leaf in leaves:
        key = _preset_key(_leaf_raw_design_tasks(raw_leaf), leaf_name)
        index.setdefault(key, []).append(leaf_name)
        per_leaf[leaf_name] = key
    return index, per_leaf


def _assign_colors(preset_index: dict[tuple[str, str], list[str]]) -> dict[tuple[str, str], str]:
    """Assign a unique palette color to every preset / inline entry."""
    colors: dict[tuple[str, str], str] = {}
    for i, key in enumerate(preset_index):
        colors[key] = _PALETTE[i % len(_PALETTE)]
    return colors


# ---------------------------------------------------------------------------
# card renderers
# ---------------------------------------------------------------------------


def _applicable_metrics(val_entry: dict, metrics: dict | None) -> list[tuple[str, dict, bool]]:
    """Return metrics applicable to a val entry as ``(name, cfg, is_universal)`` tuples.

    A metric applies if its ``required_tags_any`` overlaps the val's
    ``transform.metrics_tags``. A missing/empty ``required_tags_any`` means the
    metric is universal and applies to every val dataset.
    """
    if not isinstance(metrics, dict) or not metrics:
        return []
    dataset = val_entry.get("dataset") or {}
    transform = dataset.get("transform") or {}
    val_tags = transform.get("metrics_tags") if isinstance(transform, dict) else None
    val_tag_set = set(val_tags) if isinstance(val_tags, list) else set()

    out: list[tuple[str, dict, bool]] = []
    for name, m in metrics.items():
        if not isinstance(m, dict):
            continue
        req = m.get("required_tags_any")
        if not req:
            out.append((name, m, True))
        elif isinstance(req, list) and (set(req) & val_tag_set):
            out.append((name, m, False))
    return out


def _render_datasets_card(
    datasets: dict,
    raw_datasets: dict | None,
    metrics: dict | None = None,
    unpruned_datasets: dict | None = None,
) -> str:
    """Centerpiece card: train mix bar + shared-preset banners + dataset grid + val list."""
    if not isinstance(datasets, dict):
        return ""
    train = datasets.get("train") or {}
    val = datasets.get("val") or {}
    raw_train = (raw_datasets or {}).get("train") or {}

    leaves = _iter_leaf_datasets(raw_train, train)
    preset_index, per_leaf_key = _build_preset_index(leaves)
    colors = _assign_colors(preset_index)

    # Unpruned leaves let us recover 0.0 modifier values (pruned out of `resolved`)
    # so each task can show its full resolved global_design_task_kwargs.
    unpruned_train = (unpruned_datasets or {}).get("train") or {}
    unpruned_leaves = _iter_leaf_datasets({}, unpruned_train) if isinstance(unpruned_train, dict) else []
    unpruned_by_name: dict[str, dict] = {n: lf for n, lf, _ in unpruned_leaves}

    parts: list[str] = [_render_train_mix(leaves, per_leaf_key, colors)]

    shared = [(key, names) for key, names in preset_index.items() if key[0] == "preset" and len(names) >= 2]
    if shared:
        banners: list[str] = []
        for key, names in shared:
            color = colors[key]
            name_chips = "".join(
                f'<span class="chip" style="border-color:{_hex_to_rgba(color, 0.45)};'
                f'background:{_hex_to_rgba(color, 0.10)}">{_safe(n)}</span>'
                for n in names
            )
            banners.append(
                '<div class="preset-banner"><span class="pb-label">shared preset</span>'
                f'<span class="swatch" style="background:{color}"></span>'
                f'<span class="pb-name">{_safe(key[1])}</span>'
                '<span class="pb-arrow">&rarr;</span>'
                f'<span class="pb-datasets">{name_chips}</span></div>'
            )
        parts.append('<div class="preset-banners">' + "".join(banners) + "</div>")

    panels = "".join(
        _render_dataset_panel(
            name,
            resolved,
            raw,
            per_leaf_key[name],
            colors[per_leaf_key[name]],
            unpruned_leaf=unpruned_by_name.get(name),
        )
        for name, resolved, raw in leaves
    )
    parts.append(f'<div class="dataset-grid">{panels}</div>')

    if val:
        parts.append('<h3 class="section-label">Validation &mdash; metrics per dataset</h3>')
        parts.append(_render_val_grid(val, metrics))

    count = f"{len(leaves)} train &middot; {len(val)} val"
    return _card("Datasets", "".join(parts), span=12, count=count)


def _render_train_mix(
    leaves: list[tuple[str, dict, dict | None]],
    per_leaf_key: dict[str, tuple[str, str]],
    colors: dict[tuple[str, str], str],
) -> str:
    """Horizontal stacked bar + legend for train probabilities."""
    items = [(name, _num(resolved.get("probability"))) for name, resolved, _raw in leaves]
    total = sum(p for _, p in items)
    if total <= 0:
        return ""

    segs: list[str] = []
    legend: list[str] = []
    for name, prob in sorted(items, key=lambda x: -x[1]):
        color = colors[per_leaf_key[name]]
        frac = prob / total
        label = f"{_safe(name)}  {_fmt_percent(frac)}"
        segs.append(
            f'<div class="seg" style="background:{color};flex:{prob} 0 0" '
            f'title="{_safe(name)} &middot; {_fmt_percent(frac)} ({_fmt_num(prob)})">'
            f'{label if frac * 100 >= 6 else ""}</div>'
        )
        legend.append(
            '<span class="item">'
            f'<span class="swatch" style="background:{color}"></span>'
            f"<span>{_safe(name)}</span>"
            f'<span class="val">{_fmt_percent(frac)}</span></span>'
        )
    return f'<div class="mix-bar">{"".join(segs)}</div>' f'<div class="mix-legend">{"".join(legend)}</div>'


def _render_dataset_panel(
    name: str,
    resolved_leaf: dict,
    raw_leaf: dict | None,
    preset_key: tuple[str, str],
    color: str,
    unpruned_leaf: dict | None = None,
) -> str:
    """Render one box per leaf dataset."""
    prob = resolved_leaf.get("probability")
    dataset = resolved_leaf.get("dataset") or {}
    transform = dataset.get("transform") if isinstance(dataset, dict) else None

    unpruned_dataset = unpruned_leaf.get("dataset") if isinstance(unpruned_leaf, dict) else None
    unpruned_transform = unpruned_dataset.get("transform") if isinstance(unpruned_dataset, dict) else None
    unpruned_design_tasks = unpruned_transform.get("design_tasks") if isinstance(unpruned_transform, dict) else None

    head = f'<div class="panel-name-wrap"><span class="panel-name">{_safe(name)}</span></div>'
    if prob is not None:
        head += (
            f'<span class="prob-pill" style="background:{color}" title="probability">'
            f"{_fmt_percent(_num(prob))}</span>"
        )

    if preset_key[0] == "preset":
        preset_tag = (
            '<div class="panel-preset">'
            f'<span class="swatch" style="background:{color}"></span>'
            f'<span>preset</span><span class="name">{_safe(preset_key[1])}</span></div>'
        )
    else:
        preset_tag = (
            '<div class="panel-preset">'
            f'<span class="swatch" style="background:{color}"></span>'
            "<span>inline design_tasks</span></div>"
        )

    facts: list[str] = []
    data_path = None
    crop_size = None
    if isinstance(dataset, dict):
        data_path = dataset.get("data")
        if isinstance(transform, dict):
            crop_size = transform.get("crop_size")
            for k in ("crop_spatial_prob", "max_atoms_in_crop"):
                if k in transform and transform[k] not in (None, 0, 0.0):
                    facts.append(
                        f'<span class="chip"><span class="k">{_safe(k)}</span>' f"{_fmt_num(transform[k])}</span>"
                    )
    if crop_size is not None:
        facts.insert(0, f'<span class="chip"><span class="k">crop</span>{_fmt_num(crop_size)}</span>')
    facts_html = f'<div class="panel-facts">{"".join(facts)}</div>' if facts else ""
    path_html = f'<div class="path" title="{_safe(data_path)}">{_safe(data_path)}</div>' if data_path else ""

    design_tasks = transform.get("design_tasks") if isinstance(transform, dict) else None
    task_html = _render_task_block(design_tasks, color, unpruned_design_tasks=unpruned_design_tasks)

    expand_html = ""
    if isinstance(transform, dict):
        transform_for_yaml = {k: v for k, v in transform.items() if k != "design_tasks"}
        if transform_for_yaml:
            expand_html = (
                '<details class="expander"><summary>Show full transform</summary>'
                f'<div class="body"><pre class="yaml">{_pretty_yaml(transform_for_yaml)}</pre>'
                "</div></details>"
            )

    return (
        f'<div class="dataset-panel" style="border-left-color:{color}">'
        f'<div class="panel-head">{head}</div>'
        f"{preset_tag}{path_html}{facts_html}{task_html}{expand_html}</div>"
    )


# Friendly short labels for global conditioning knobs so the modifier strip
# stays compact.
_GLOBAL_KWARG_LABELS: dict[tuple[str, str], str] = {
    ("rasa_kwargs", "p_add_rasa_condition"): "rasa",
    ("hbond_kwargs", "p_add_hbond_condition"): "hbond",
    ("secondary_structure_kwargs", "p_add_secondary_structure_condition"): "ss",
    ("cath_kwargs", "p_add_cath_c_condition"): "cath-c",
    ("cath_kwargs", "p_add_cath_ca_condition"): "cath-ca",
    ("cath_kwargs", "p_add_cath_cat_condition"): "cath-cat",
}

_KWARG_SKIP = {"frequency", "_target_", "global_design_task_kwargs"}


def _nonzero_task_modifiers(unpruned_task: Any) -> list[tuple[str, float]]:
    """Return ``[(label, value), ...]`` of non-zero global modifiers for one task."""
    if not isinstance(unpruned_task, dict):
        return []
    transform = unpruned_task.get("transform")
    if not isinstance(transform, dict):
        return []
    gdk = transform.get("global_design_task_kwargs")
    if not isinstance(gdk, dict) or not gdk:
        return []
    out: list[tuple[str, float]] = []
    for (group, key), label in _GLOBAL_KWARG_LABELS.items():
        block = gdk.get(group)
        if not isinstance(block, dict):
            continue
        v_f = _num(block.get(key), default=-1.0)
        if v_f > 0:
            out.append((label, v_f))
    return out


def _render_per_task_modifiers_list(
    items: list[tuple[str, float, dict]],
    unpruned_by_name: dict,
) -> str:
    """One row per task showing the exact non-zero modifiers actually applied."""
    rows: list[str] = []
    for name, _freq, _td in items:
        mods = _nonzero_task_modifiers(unpruned_by_name.get(name))
        if not mods:
            continue
        pills = "".join(
            f'<span class="mod-pill"><span class="mod-name">{_safe(label)}</span>'
            f'<span class="mod-val">{_fmt_percent(val)}</span></span>'
            for label, val in mods
        )
        rows.append(
            f'<div class="mod-row"><span class="mod-row-name">{_safe(name)}</span>'
            f'<div class="mod-row-pills">{pills}</div></div>'
        )
    if not rows:
        return ""
    heading = '<div class="tb-heading"><span>Modifiers (per task)</span></div>'
    return f'<div class="task-mods-list">{heading}{"".join(rows)}</div>'


def _extract_task_kwargs(task_dict: dict) -> list[tuple[str, Any]]:
    """Return a task's specific kwargs, flattening one level of nesting for readability.

    Excludes ``frequency`` / ``_target_`` / global kwargs; ``foo: {x: 1}`` becomes
    ``foo.x: 1``.
    """
    transform = task_dict.get("transform") if isinstance(task_dict, dict) else None
    if not isinstance(transform, dict):
        return []
    out: list[tuple[str, Any]] = []
    for k, v in transform.items():
        if k in _KWARG_SKIP:
            continue
        if isinstance(v, dict):
            out.extend((f"{k}.{inner_k}", inner_v) for inner_k, inner_v in v.items())
        else:
            out.append((k, v))
    return out


def _render_task_block(
    design_tasks: Any,
    color: str,
    unpruned_design_tasks: Any = None,
) -> str:
    """Mini bar + chips + modifiers + per-task kwargs details for one dataset."""
    if not isinstance(design_tasks, dict) or not design_tasks:
        return ""

    items: list[tuple[str, float, dict]] = []
    hidden = 0
    for k, v in design_tasks.items():
        if k == "__hidden_tasks__":
            hidden = int(_num(v))
            continue
        if not isinstance(v, dict):
            continue
        freq_f = _num(v.get("frequency"))
        if freq_f > 0:
            items.append((k, freq_f, v))

    hidden_html = f'<div class="hidden-note">+ {hidden} tasks @ frequency 0.0</div>' if hidden else ""
    if not items:
        return hidden_html

    items.sort(key=lambda x: -x[1])
    total = sum(f for _, f, _ in items)
    n = len(items)

    bar = "".join(
        f'<span style="flex:{f} 0 0;background:'
        f'{_hex_to_rgba(color, 0.35 + 0.65 * (1 - i / max(n - 1, 1)))}"></span>'
        for i, (_, f, _t) in enumerate(items)
    )
    chips = "".join(
        f'<span class="task-chip" style="border-color:{_hex_to_rgba(color, 0.35)};'
        f'background:{_hex_to_rgba(color, 0.07)}" title="frequency: {_fmt_num(freq)}">'
        f'<span class="pct">{_fmt_percent(freq / total)}</span>'
        f'<span class="name">{_safe(name)}</span></span>'
        for name, freq, _t in items
    )

    unpruned_by_name = unpruned_design_tasks if isinstance(unpruned_design_tasks, dict) else {}
    modifiers_html = _render_per_task_modifiers_list(items, unpruned_by_name)

    detail_rows: list[str] = []
    for name, freq, task_dict in items:
        kw = _extract_task_kwargs(task_dict)
        kwargs_html = (
            "".join(
                f'<span class="kw"><span class="k">{_safe(k)}</span>'
                f'<span class="v">{_safe(_fmt_kwarg_value(v))}</span></span>'
                for k, v in kw
            )
            if kw
            else '<span class="kw-none">no extra kwargs</span>'
        )
        detail_rows.append(
            '<div class="task-detail"><div class="td-head">'
            f'<span class="td-name">{_safe(name)}</span>'
            f'<span class="td-pct">{_fmt_percent(freq / total)}</span></div>'
            f'<div class="td-kwargs">{kwargs_html}</div></div>'
        )
    details_html = (
        '<details class="expander task-kwargs-expander">'
        f'<summary>Show per-task kwargs &middot; {n} task{"s" if n != 1 else ""}</summary>'
        f'<div class="body task-details-body">{"".join(detail_rows)}</div></details>'
    )

    heading = '<div class="tb-heading"><span>Design tasks</span>' f"<span>{n} active</span></div>"
    return (
        f'<div class="task-block">{heading}<div class="task-bar">{bar}</div>'
        f'<div class="task-chips">{chips}</div>'
        f"{modifiers_html}{details_html}{hidden_html}</div>"
    )


def _fmt_kwarg_value(v: Any) -> str:
    """Format a kwarg value compactly for inline display."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else f"{v:g}"
    if isinstance(v, list | tuple):
        if len(v) <= 4 and all(isinstance(x, int | float | str | bool) for x in v):
            return "[" + ", ".join(_fmt_kwarg_value(x) for x in v) + "]"
        return f"[{len(v)} items]"
    if isinstance(v, dict):
        if len(v) <= 2 and all(isinstance(val, int | float | str | bool) for val in v.values()):
            return "{" + ", ".join(f"{k}={_fmt_kwarg_value(val)}" for k, val in v.items()) + "}"
        return f"{{...{len(v)} keys}}"
    return str(v)


def _render_val_grid(val: dict, metrics: dict | None = None) -> str:
    """Render the validation datasets grid with the metrics that apply to each."""
    items: list[str] = []
    for name, entry in val.items():
        if not isinstance(entry, dict):
            continue
        dataset = entry.get("dataset") or {}
        transform = dataset.get("transform") or {}
        subset_keys = dataset.get("subset_to_keys")
        num_reps = dataset.get("num_replicates")
        metric_tags = transform.get("metrics_tags") if isinstance(transform, dict) else None

        row_bits: list[str] = []
        if isinstance(subset_keys, list):
            row_bits.append(f"{len(subset_keys)} keys")
        if num_reps:
            row_bits.append(f"x{_fmt_num(num_reps)}")
        if isinstance(metric_tags, list):
            row_bits += [f'<span class="metric-tag">{_safe(t)}</span>' for t in metric_tags]
        row_html = " &middot; ".join(row_bits)

        applicable = _applicable_metrics(entry, metrics)
        if applicable:
            chips = "".join(
                f'<span class="{"metric-chip universal" if universal else "metric-chip"}" '
                f'title="{_safe(str(m.get("_target_", "")) + ("  (universal)" if universal else ""))}">'
                f"{_safe(mname)}</span>"
                for mname, m, universal in applicable
            )
            metrics_html = '<div class="val-metrics"><span class="metrics-label">metrics</span>' f"{chips}</div>"
        elif isinstance(metrics, dict) and metrics:
            metrics_html = '<div class="no-metrics">no matching metrics</div>'
        else:
            metrics_html = ""

        transform_yaml = _pretty_yaml(transform) if isinstance(transform, dict) and transform else ""
        expander = (
            '<details class="expander"><summary>Show transform</summary>'
            f'<div class="body"><pre class="yaml">{transform_yaml}</pre></div></details>'
            if transform_yaml
            else ""
        )
        items.append(
            f'<div class="val-item"><div class="val-name">{_safe(name)}</div>'
            f'<div class="val-row">{row_html}</div>{metrics_html}{expander}</div>'
        )
    return '<div class="val-grid">' + "".join(items) + "</div>"


def _render_system_card(resolved: dict) -> str:
    """Top overview card: compute + dataloader + training cadence + autoencoder."""
    hw = resolved.get("hardware") or {}
    dl = resolved.get("dataloaders") or {}
    opt = resolved.get("opt") or {}
    nn_cfg = resolved.get("nn") or {}

    ngpus = int(_num(hw.get("ngpus_per_node_")))
    nnodes = int(_num(hw.get("nnodes_")))
    total_gpus = ngpus * nnodes if (ngpus and nnodes) else (ngpus or 0)
    ncpus = hw.get("ncpus_per_task_train_") or hw.get("ncpus_per_task_") if isinstance(hw, dict) else None

    batch_size = int(_num(dl.get("batch_size")))
    accum = int(_num(opt.get("accumulate_grad_batches"), 1.0))
    effective_bs = batch_size * accum * max(total_gpus, 1)
    steps_per_epoch = int(_num(opt.get("steps_per_epoch")))
    examples_per_epoch = steps_per_epoch * effective_bs if (steps_per_epoch and effective_bs) else 0
    val_every = opt.get("val_every_n_epochs")

    def _count(n: int) -> str:
        return f"{n:,}" if n else "—"

    tiles: list[str] = []
    if total_gpus:
        sub = f"{ngpus} × {nnodes} nodes" if (ngpus and nnodes) else ""  # noqa: RUF001
        tiles.append(_stat_tile(f'{total_gpus}<span class="unit">GPUs</span>', "Compute", sub, accent=True))
    if batch_size:
        tiles.append(_stat_tile(str(batch_size), "Per-GPU batch"))
    if effective_bs:
        sub_parts: list[str] = []
        if batch_size:
            sub_parts.append(str(batch_size))
        if accum and accum != 1:
            sub_parts.append(f"×{accum}")  # noqa: RUF001
        if total_gpus and total_gpus != 1:
            sub_parts.append(f"×{total_gpus}")  # noqa: RUF001
        tiles.append(_stat_tile(_count(effective_bs), "Effective batch", " ".join(sub_parts), accent=True))
    if accum and accum != 1:
        tiles.append(_stat_tile(str(accum), "Accum grad"))
    if steps_per_epoch:
        tiles.append(_stat_tile(_count(steps_per_epoch), "Steps / epoch"))
    if examples_per_epoch:
        tiles.append(_stat_tile(_count(examples_per_epoch), "Examples / epoch"))
    if val_every is not None:
        ve = int(_num(val_every, 1.0))
        txt = f"every {ve}" if ve != 1 else "every epoch"
        unit = '<span class="unit">epochs</span>' if ve != 1 else ""
        tiles.append(_stat_tile(f"{txt}{unit}", "Validation"))
    if ncpus is not None:
        tiles.append(_stat_tile(_fmt_num(ncpus), "CPUs / task"))
    stats_html = f'<div class="stats">{"".join(tiles)}</div>' if tiles else ""

    def _chip(label: str, value: str, cls: str = "") -> str:
        return (
            f'<span class="feature-chip"><span class="label">{_safe(label)}</span>'
            f'<span class="value {cls}">{_safe(value)}</span></span>'
        )

    chips: list[str] = []
    tc = nn_cfg.get("use_torch_compile")
    if tc is not None:
        chips.append(_chip("torch.compile", "on" if tc else "off", "on" if tc else "off"))
    attn = nn_cfg.get("attention_type")
    if attn:
        chips.append(_chip("attention", str(attn)))
    train_dl = dl.get("train") or {}
    params = (train_dl.get("dataloader_params") or {}) if isinstance(train_dl, dict) else {}
    collate = params.get("collate_fn") if isinstance(params, dict) else None
    fixed = collate.get("fixed_dimensions") if isinstance(collate, dict) else None
    if isinstance(fixed, dict) and fixed:
        chips.append(_chip("fixed dims", ", ".join(f"{k}={_fmt_num(v)}" for k, v in fixed.items())))
    nw = dl.get("num_workers")
    if nw is not None:
        chips.append(_chip("workers", _fmt_num(nw)))
    pf = dl.get("prefetch_factor")
    if pf is not None:
        chips.append(_chip("prefetch", _fmt_num(pf)))
    strat = opt.get("dist_strategy")
    if strat:
        chips.append(_chip("strategy", str(strat)))
    features_html = f'<div class="feature-row">{"".join(chips)}</div>' if chips else ""

    ae_ckpt = resolved.get("autoencoder_ckpt_path")
    ae_html = ""
    if isinstance(ae_ckpt, str) and ae_ckpt:
        split = ae_ckpt.rsplit("/", 1)
        parent, filename = split if len(split) == 2 else ("", ae_ckpt)
        ae_html = (
            '<div class="ckpt-row"><span class="ckpt-label">Autoencoder</span>'
            f'<span class="ckpt-file">{_safe(filename)}</span>'
            + (f'<span class="ckpt-dir">{_safe(parent)}/</span>' if parent else "")
            + "</div>"
        )

    return _card("System", f"{stats_html}{features_html}{ae_html}", span=12, extra_cls="system-card")


def _render_nn_card(value: Any) -> str:
    """Structured card for the ``nn`` config: scalars as a KV table, lists as pills."""
    if not isinstance(value, dict):
        return _render_generic_card("nn", value)
    scalar_rows: list[str] = []
    list_blocks: list[str] = []
    other: dict[str, Any] = {}
    for k, v in value.items():
        if isinstance(v, str | int | float | bool):
            scalar_rows.append(f"<dt>{_safe(k)}</dt><dd>{_safe(_fmt_num(v))}</dd>")
        elif isinstance(v, list) and all(isinstance(x, str | int | float | bool) for x in v):
            pills = "".join(f'<span class="feat-pill">{_safe(x)}</span>' for x in v)
            list_blocks.append(f'<h4 class="section-label">{_safe(k)}</h4>' f'<div class="feat-pills">{pills}</div>')
        else:
            other[k] = v

    parts: list[str] = []
    if scalar_rows:
        parts.append(f'<dl class="kv">{"".join(scalar_rows)}</dl>')
    parts.extend(list_blocks)
    if other:
        parts.append(
            '<details class="expander"><summary>Show nested config</summary>'
            f'<div class="body"><pre class="yaml">{_pretty_yaml(other)}</pre></div></details>'
        )
    return _card("Model (nn)", "".join(parts), span=6)


def _render_generic_card(key: str, value: Any) -> str:
    """Fallback card: title + pruned YAML in a collapsible body."""
    body = (
        '<details class="expander" open><summary>Show</summary>'
        f'<div class="body"><pre class="yaml">{_pretty_yaml(value)}</pre></div></details>'
    )
    return _card(key, body, span=6)


# ---------------------------------------------------------------------------
# top-level assembly
# ---------------------------------------------------------------------------


_SPECIAL_RENDERERS: dict[str, Callable[[Any], str]] = {"nn": _render_nn_card}

# Keys rolled into the System card or rendered inline elsewhere — no generic
# fallback card is emitted for them.
_ABSORBED_KEYS: set[str] = {
    "hardware",
    "dataloaders",
    "autoencoder_ckpt_path",  # System card
    "run_name_",
    "run_name",  # Used for doc_title
    "metrics",  # Inlined in per-val dataset chips
    # Intentionally hidden (user said they don't care)
    "seed",
    "precision",
    "force_precision_f32",
    "co_generate_targets",
    "pretrain_ckpt_path",
    "out_dir",
}


def save_config_html(cfg: DictConfig, path: str | Path, title: str = "Config") -> str:
    """Render a Hydra config as a self-contained HTML dashboard.

    Writes the dashboard to *path* and returns the HTML string, so callers can
    forward it to e.g. ``wandb.Html`` without re-reading the file. The document
    is fully standalone (inline CSS, no external assets).

    Returns:
      The rendered HTML document.
    """
    raw = OmegaConf.to_container(cfg, resolve=False)
    resolved_full = OmegaConf.to_container(cfg, resolve=True)
    resolved, _ = _prune_dead_values(resolved_full, custom_prune=_prune_design_tasks)
    if not isinstance(resolved, dict):
        resolved = {"value": resolved}
    if not isinstance(raw, dict):
        raw = {}

    run_name = resolved.get("run_name_") or resolved.get("run_name") or ""
    doc_title = f"{title} - {run_name}" if run_name else title

    body_parts: list[str] = [_render_system_card(resolved)]

    # Datasets (centerpiece) — receives metrics so val datasets can show which
    # metric(s) apply to them (matched via required_tags_any). The unpruned
    # datasets block is forwarded so the global-modifier strip can show 0.0
    # entries that pruning would otherwise drop.
    if "datasets" in resolved:
        unpruned_datasets = resolved_full.get("datasets") if isinstance(resolved_full, dict) else None
        body_parts.append(
            _render_datasets_card(
                resolved["datasets"],
                raw.get("datasets"),
                metrics=resolved.get("metrics"),
                unpruned_datasets=unpruned_datasets,
            )
        )

    for key, renderer in _SPECIAL_RENDERERS.items():
        if key in resolved:
            body_parts.append(renderer(resolved[key]))

    # Generic fallback cards for everything else. Top-level keys starting with
    # "_" are Hydra composition aliases (preset scratchpads) and duplicate
    # content already rendered in the dataset cards.
    skip_keys = {"datasets"} | set(_SPECIAL_RENDERERS) | _ABSORBED_KEYS
    for key, value in resolved.items():
        if key in skip_keys or (isinstance(key, str) and key.startswith("_")):
            continue
        body_parts.append(_render_generic_card(key, value))

    html_doc = (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        f"<title>{_safe(doc_title)}</title><style>{_CSS}</style></head>"
        '<body><div class="container">'
        f'<main class="grid">{"".join(body_parts)}</main>'
        "</div></body></html>\n"
    )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc)
    return html_doc
