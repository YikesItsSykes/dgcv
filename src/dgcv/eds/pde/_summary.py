"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds.pde

module: dgcv.eds.pde._summary

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import html
import uuid
from math import comb

from ..._aux._backends._display import latex as _routed_latex
from ..._aux._utilities._config import latex_in_html
from ..._aux._utilities._styles import get_style
from ..._aux.printing._tables import panel_view
from ..._aux.printing.printing._dgcv_display import LaTeX, show
from ._panels import (
    _corners,
    _header_block,
    _HTMLWrapper,
    _stack,
    singularity_sources,
    summary_options,
)

_container_id = "dgcv-pde-summary"


def equations_tex(equations):
    return r"\left\{" + r",\ ".join(f"{LaTeX(F)} = 0" for F in equations) + r"\right\}"


def equations_text(equations):
    return "{" + ", ".join(f"{F} = 0" for F in equations) + "}"


def _jet_dimension(n, m, k):
    return n + m * sum(comb(n + length - 1, length) for length in range(k + 1))


def _join(items, math):
    return ", ".join(math(item) for item in items) if items else "none"


def _sections(data, math, use_latex):
    n, m, k = data.signature
    derivatives = sorted(
        ((atom, key) for atom, key in data.derivatives.items() if key[1]),
        key=lambda item: (item[1][0], len(item[1][1]), item[1][1]),
    )
    pairs, lone, shown = [], [], set()
    for key, value in data.conjugates.items():
        if value is None:
            lone.append(key)
        elif value is not key and str(value) != str(key) and str(key) not in shown:
            pairs.append((key, value))
            shown.update((str(key), str(value)))
    first = [
        f"unknown functions: {_join(data.dependent_variables, math)}",
        "derivatives: " + _join([atom for atom, _ in derivatives], math),
    ]
    if lone:
        first.append("holomorphic directions: " + _join([key for key in lone if key in data.coframe.forms], math))
    if pairs:
        first.append("conjugate pairs: " + ", ".join(f"({math(a)}, {math(b)})" for a, b in pairs))
    first += [
        f"coframe: {math(data.coframe)}",
        f"parameters: {_join(data.parameters, math)}",
        f"order: {k}",
        (
            f"jet space: ${{J^{{{k}}}({n}, {m})}}$, dimension {_jet_dimension(n, m, k)}"
            if use_latex
            else f"jet space: J^{k}({n}, {m}), dimension {_jet_dimension(n, m, k)}"
        ),
    ]
    second = [
        f"from {_source_labels.get(source, source)}: "
        + "; ".join(" = ".join(math(c) for c in locus) + " = 0" for locus in loci)
        for source, loci in data.singularities().items()
    ]
    return first, second


_titles = ("System of differential equations", "Singularities")
_source_labels = dict(singularity_sources)


def _plain(data):
    width = max(len(f" {t} ") for t in _titles)
    first, second = _sections(data, str, False)
    lines = [_header_block(_titles[0], width), "• " + equations_text(data.equations)]
    lines += [f"• {item}" for item in first]
    if second:
        lines += ["", _header_block(_titles[1], width)]
        lines += [f"• {item}" for item in second]
    return "\n".join(lines)


def _rich(data, use_latex, style_key, extra_support):
    theme_vars, theme_data = get_style(style_key, return_theme_data=True)
    radius = int(theme_data.custom_css_vars.get("--dgcv-border-radius", "12px").replace("px", ""))
    plaque = {
        "plaque_fill": "--plaque-fill" in theme_data.custom_css_vars,
        "plaque_border": "--plaque-border" in theme_data.custom_css_vars,
        "plaque_content": True,
    }
    math = (lambda x: f"${_routed_latex(x)}$") if use_latex else (lambda x: html.escape(str(x)))
    first, second = _sections(data, math, use_latex)
    equations = (
        f"${equations_tex(data.equations)}$" if use_latex else html.escape(equations_text(data.equations))
    )
    specs = [{"header": _titles[0], "primary_text": equations, "itemized_text": first}]
    if second:
        specs.append({"header": _titles[1], "itemized_text": second})
    blocks = [
        panel_view(
            theme_css_vars="",
            extra_css="",
            slim=True,
            use_latex=use_latex,
            **plaque,
            **_corners(i, len(specs), radius),
            **spec,
        ).to_html()
        for i, spec in enumerate(specs)
    ]
    container_id = f"{_container_id}-{uuid.uuid4().hex[:8]}"
    return latex_in_html(
        _HTMLWrapper(_stack(blocks, container_id, theme_vars, theme_data)),
        extra_support_for_math_in_tables=extra_support,
    )


def pde_summary(
    data,
    *,
    theme=None,
    use_latex=None,
    plain_text=None,
    return_displayable=False,
    **kwargs,
):
    plain_text, use_latex, style_key, extra_support = summary_options(
        theme, use_latex, plain_text, kwargs
    )
    out = (
        _plain(data) if plain_text else _rich(data, use_latex, style_key, extra_support)
    )
    if return_displayable:
        return out
    if plain_text:
        print(out)
        return
    show(out)
