"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds.pde

module: dgcv.eds.pde._panels

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import textwrap

from ..._aux._backends._display_engine import is_rich_displaying_available
from ..._aux._utilities._config import get_dgcv_settings_registry


def summary_options(theme, use_latex, plain_text, kwargs):
    settings = get_dgcv_settings_registry()
    if use_latex is None:
        use_latex = settings.get("use_latex", False)
    if plain_text is None:
        plain_text = not bool(use_latex)
    if not is_rich_displaying_available():
        plain_text = True
    style_key = (
        theme if isinstance(theme, str) else kwargs.get("style") or settings.get("theme", "dark")
    )
    extra_support = settings.get("extra_support_for_math_in_tables") is True
    return bool(plain_text), bool(use_latex), style_key, extra_support


singularity_sources = (
    ("distribution_rank", "the rank of the Pfaff system forms"),
    ("chart", "the chart solver"),
    ("integral_flag", "polar system ranks in the Cartan involutivity test (possibly spurious)"),
)


def _header_block(title, width):
    inner = f" {title} "
    pad = max(0, width - len(inner))
    left = pad // 2
    return "===" + "=" * left + inner + "=" * (pad - left) + "==="



class _HTMLWrapper:
    def __init__(self, html):
        self._html = html

    def to_html(self, *args, **kwargs):
        return self._html

    def _repr_html_(self):
        return self._html



def _stack(blocks, container_id, theme_vars, theme_data, wrapped=()):
    custom = theme_data.custom_css_vars
    if custom.get("--dgcv-special-background"):
        panel_bg, panel_hd, text = (
            "var(--dgcv-special-background)",
            "none",
            "var(--dgcv-special-text,var(--dgcv-text-heading))",
        )
    else:
        panel_bg, panel_hd, text = (
            "var(--dgcv-bg-primary)",
            "var(--dgcv-bg-surface)",
            "var(--dgcv-text-main)",
        )
    inner = "\n".join(
        f'<div class="section{" section-wrap" if i in wrapped else ""}">{b}</div>'
        for i, b in enumerate(blocks)
    )
    return textwrap.dedent(f"""
        <div id="{container_id}">
        <style>
        {theme_vars.replace(":root", f"#{container_id}")}
        #{container_id} .stack {{ display: flex; flex-direction: column; gap: 16px; align-items: stretch; width: 100%; margin: 0; }}
        #{container_id} .section {{ width: 100%; }}
        #{container_id} .dgcv-panel {{
            background: {panel_bg};
            box-shadow: var(--dgcv-table-shadow, none);
            border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main);
            border-image: var(--dgcv-border-image, none);
            color: {text};
            font-family: var(--dgcv-font-family, inherit);
            overflow: hidden;
            padding: 4px 4px;
            margin: 0;
        }}
        #{container_id} .dgcv-panel-head {{ padding: 0.75rem 1rem; background: {panel_hd}; }}
        #{container_id} .dgcv-panel-title {{ margin: 0; font-size: 1rem; font-weight: 600; color: var(--dgcv-special-text, var(--dgcv-text-heading)); text-shadow: var(--dgcv-text-shadow, none); }}
        #{container_id} .dgcv-panel-rule {{ border: 0; height: 2px; background: var(--dgcv-border-main); margin: 0; }}
        #{container_id} .dgcv-panel-body {{ padding: 0.75rem 1rem; overflow-x: auto; width: 100%; box-sizing: border-box; }}
        #{container_id} .dgcv-panel-list ul {{ margin: 0.25rem 0 0 1.25rem; color: {text}; }}
        #{container_id} .dgcv-panel-list li {{ overflow-x: auto; overflow-y: hidden; white-space: nowrap; list-style-position: inside; }}
        #{container_id} .section-wrap .dgcv-panel-body {{ overflow-x: visible; }}
        #{container_id} .section-wrap .dgcv-panel-primary, #{container_id} .section-wrap .dgcv-panel-list li, #{container_id} .section-wrap .dgcv-panel-footer {{ white-space: normal; overflow-x: visible; overflow-y: visible; overflow-wrap: break-word; }}
        #{container_id} .dgcv-panel-footer {{ padding: 0.5rem 1rem; background: var(--dgcv-bg-alt); color: var(--dgcv-text-alt); border-top: 1px solid var(--dgcv-border-alt); }}
        #{container_id} .dgcv-panel-divider {{ border: 0; border-top: 1px solid var(--dgcv-border-alt); margin: 0.75rem 0; }}
        </style>
        <div class="stack">
        {inner}
        </div>
        </div>
    """).strip()


def _corners(i, total, r):
    if total <= 1:
        return {"ul": r, "ur": r, "ll": r, "lr": r}
    if i == 0:
        return {"ul": r, "ur": r, "ll": 0, "lr": 0}
    if i == total - 1:
        return {"ul": 0, "ur": 0, "ll": r, "lr": r}
    return {"ul": 0, "ur": 0, "ll": 0, "lr": 0}


