"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds.Pfaff_systems

module: dgcv.eds.Pfaff_systems._summary

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import html
import uuid

from ..._aux._utilities._config import get_variable_registry, latex_in_html
from ..._aux._utilities._progress import _timed_progress_call
from ..._aux._utilities._styles import get_style
from ..._aux.printing._tables import build_plain_table, panel_view
from ..._aux.printing.printing._dgcv_display import LaTeX, show
from ..pde._panels import (
    _corners,
    _header_block,
    _HTMLWrapper,
    _stack,
    singularity_sources,
    summary_options,
)

_title = "Integral flag of the Pfaff system"
_container_id = "dgcv-integral-flag-summary"

_extra_css = """
.dgcv-data-table th:nth-child(1), .dgcv-data-table td:nth-child(1),
.dgcv-data-table th:nth-child(3), .dgcv-data-table td:nth-child(3),
.dgcv-data-table th:nth-child(4), .dgcv-data-table td:nth-child(4) {
    width: 1%;
    white-space: nowrap;
}
.dgcv-side-panel {
    border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main);
    background: var(--dgcv-special-background, var(--dgcv-bg-surface));
    color: var(--dgcv-special-text, var(--dgcv-text-heading));
    padding: 12px;
    height: 100%;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
}
.dgcv-side-plaque {
    display: block;
    box-sizing: border-box;
    width: 100%;
    padding: 0.75rem 1rem;
    background: var(--plaque-fill);
    border-radius: inherit;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
    overflow-x: auto;
    overflow-y: hidden;
    border: var(--dgcv-border-width, 1px) solid var(--plaque-border, var(--dgcv-border-main));
}
.dgcv-side-divider {
    border-top: calc(var(--dgcv-border-width, 1px) * 2) solid var(--dgcv-border-main);
}
.dgcv-side-title {
    padding: 12px;
    border-bottom: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main);
    font-weight: bold;
}
.dgcv-side-list {
    margin: 0;
    padding: 0;
    list-style: none;
    overflow-y: visible;
    height: fit-content;
}
.dgcv-side-list li {
    overflow-x: auto;
    overflow-y: hidden;
    white-space: nowrap;
    padding: 8px;
    border-bottom: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main);
    display: list-item;
    list-style-type: disc;
    list-style-position: inside;
}
.dgcv-data-table tfoot td {
    text-align: left;
    font-weight: bold;
    background-color: var(--dgcv-bg-primary);
    color: var(--dgcv-text-heading);
    border-top: 2px solid var(--dgcv-border-main);
}
"""


def _math(value, use_latex):
    return f"${LaTeX(value)}$" if use_latex else html.escape(str(value))


def _retex(system, tex):
    table = getattr(system, "_coordinate_tex", None)
    if not table:
        return tex
    pairs = sorted(((LaTeX(x), target) for x, target in table.items()), key=lambda pair: -len(pair[0]))
    for default, target in pairs:
        if default and default != target:
            tex = tex.replace(default, target)
    return tex


def _point(at, use_latex):
    if not at:
        return ""
    if use_latex:
        return (
            " at $"
            + ",\\ ".join(f"{LaTeX(k)} = {LaTeX(v)}" for k, v in at.items())
            + "$"
        )
    return " at " + ", ".join(f"{k} = {v}" for k, v in at.items())


def _levels(data):
    vectors = data.flag[-1] if data.flag else ()
    return list(
        zip(range(1, len(vectors) + 1), vectors, data.polar_ranks, data.characters)
    )


def _numbers(system, data, at):
    genre = "none" if data.genre is None else data.genre
    return (
        f"independent Pfaff system forms: {system.Pfaff_distribution_codim()}, "
        f"distribution rank: {system.Pfaff_distribution_rank(at)}, genre: {genre}"
    )


def _plain(system, data, at):
    sections = [_title + _point(at, False), "Weights", "Singularities"]
    width = max(len(f" {name} ") for name in sections)
    lines = [_header_block(sections[0], width)]
    for p, vector, r, s in _levels(data):
        inside = "" if p == 1 else f"level {p - 1} + "
        lines.append(f"• level {p}: {inside}<{vector}>   polar rank {r}, character {s}")
    if not data.flag:
        lines.append("• the distribution is zero, so there is no line")
    lines.append(f"• {_numbers(system, data, at)}")
    lines.append("")
    lines.append(_header_block(sections[1], width))
    lines.append(
        "• [" + ", ".join(str(w) for w in data.weights) + "]"
        if data.weights
        else "• none"
    )
    lines.append("")
    lines.append(_header_block(sections[2], width))
    if data.singularities:
        lines.append("• [" + ", ".join(str(d) for d in data.singularities) + "]")
    else:
        lines.append("• none")
    return "\n".join(lines)


def _panel_section(title, items, include_divider):
    divider = "<div class='dgcv-side-divider'></div>" if include_divider else ""
    lis = "".join(f"<li>{item}</li>" for item in items)
    return f"{divider}<div class='dgcv-side-title'>{title}</div><ul class='dgcv-side-list'>{lis}</ul>"


def _rich(system, data, at, use_latex, theme_vars, extra_support):
    rows = []
    for p, vector, r, s in _levels(data):
        rows.append([str(p), _math(vector, use_latex), str(r), str(s)])
    if not rows:
        rows.append(["--", "the distribution is zero, so there is no line", "--", "--"])
    columns = ["Level", "Spanning vector added", "Polar rank", "Character"]
    footer = [[{"html": _numbers(system, data, at), "attrs": {"colspan": 4}}]]
    weights = ", ".join(_math(w, use_latex) for w in data.weights) or "none"
    singular = [_math(d, use_latex) for d in data.singularities] or ["none"]
    content = _panel_section("Weights", [weights], False) + _panel_section(
        "Singularities", singular, True
    )
    if "--plaque-fill" in (theme_vars or ""):
        content = f"<div class='dgcv-side-plaque'>{content}</div>"
    table = build_plain_table(
        columns=columns,
        rows=rows,
        caption=_title + _point(at, use_latex),
        theme_css_vars=theme_vars,
        extra_css=_extra_css,
        table_attrs='style="table-layout:auto;"',
        cell_align="center",
        escape_cells=False,
        escape_headers=True,
        secondary_panel_html=f"<div class='dgcv-side-panel'>{content}</div>",
        layout="row",
        gap_px=10,
        side_width="340px",
        container_id=_container_id,
        footer_rows=footer,
        ul=12,
        ur=12,
        ll=12,
        lr=12,
    )
    return latex_in_html(table, extra_support_for_math_in_tables=extra_support)


def integral_flag_summary(
    system,
    data,
    *,
    at=None,
    theme=None,
    use_latex=None,
    plain_text=None,
    return_displayable=False,
    **kwargs,
):
    plain_text, use_latex, style_key, extra_support = summary_options(
        theme, use_latex, plain_text, kwargs
    )
    if plain_text:
        out = _plain(system, data, at)
        if return_displayable:
            return out
        print(out)
        return
    out = _rich(system, data, at, use_latex, get_style(style_key, legacy=False), extra_support)
    if return_displayable:
        return out
    show(out)


_system_container_id = "dgcv-pfaff-summary"
_flag_note = "For the integral flag used to compute these invariants, access integral_flag_summary()."
_terminology_note = (
    "Terminology follows Alekseevskij, Vinogradov and Lychagin, Geometry I, Ch. 5."
)
_generality_note = (
    "These counts follow from discussion in "
    "Bryant, Chern, Gardner, Goldschmidt and Griffiths, Exterior Differential Systems, Ch. III §2."
)


def _coframe_name(coframe, math):
    for label, entry in get_variable_registry()["eds"]["coframes"].items():
        if entry.get("coframe") is coframe:
            return label
    return "unregistered, with forms " + ", ".join(math(f) for f in coframe.forms)


def _yes_no(value):
    return "yes" if value else "no"


def _warm_summary(system, at, threshold, on_update):
    key = None if at is None else frozenset(at.items())
    cache = system._summary_cache.setdefault(key, {})
    steps = (
        (
            "chain",
            "computing the polar system ranks, the characters and the genre",
            lambda: system._warm_caches(at),
        ),
        (
            "integrable",
            "testing complete integrability",
            system.is_completely_integrable,
        ),
        (
            "characteristic",
            "computing the characteristic distribution",
            lambda: system.characteristic_distribution(at),
        ),
    )
    if system.horizontal_forms:
        steps += (
            (
                "verdict",
                "deciding involutivity in Cartan's sense",
                lambda: system._Cartan_obstruction(at=at),
            ),
        )
    if system.defining_equations and at is None:
        steps += (
            (
                "random_regular",
                "testing the polar system ranks at a random point of the defining equations",
                lambda: _random_regularity(system),
            ),
        )
    for name, desc, fn in steps:
        if name in cache:
            continue
        cache[name] = _timed_progress_call(
            fn,
            timed=True,
            threshold_s=threshold,
            step_desc=desc,
            continue_desc="finish the summary",
            progress_message=None,
            _on_timed_update=on_update,
        )
    return cache


def _random_regularity(system):
    try:
        point, verdict, detail = system._sample_point(seed=0)
    except ValueError:
        return None, "inconclusive", None
    if point is None:
        return None, verdict, detail
    if not system.is_regular_point(point):
        return point, False, "the Pfaff system forms drop rank there"
    try:
        ranks = system.polar_ranks(at=point)
    except ValueError:
        return point, False, "the polar systems are singular there"
    generic = system.polar_ranks()
    if ranks != generic:
        return (
            point,
            False,
            f"the polar system ranks there are {ranks} against the generic {generic}",
        )
    return point, True, None


def _singularity_lines(system, math):
    lines = []
    loci_by_source = system.singularities()
    for key, label in singularity_sources:
        loci = loci_by_source.get(key, [])
        if loci:
            lines.append(
                f"from {label}: "
                + "; ".join(
                    " = ".join(math(c) for c in locus) + " = 0" for locus in loci
                )
            )
    return lines


def _sections(system, cache, at, math):
    data = cache.get("chain")
    source = system._source
    first = [f"manifold dimension: {system.dimension}"]
    if system.defining_equations:
        first.append(
            "defining equations: "
            + ", ".join(f"{math(F)} = 0" for F in system.defining_equations)
        )
    if source is None:
        first.append("coordinates: " + ", ".join(math(x) for x in system.coordinates))
    else:
        first.append(f"coframe: {_coframe_name(source, math)}")
    if at:
        first.append(
            "point: " + ", ".join(f"{math(k)} = {math(v)}" for k, v in at.items())
        )
    first.append(f"independent Pfaff system forms: {system.Pfaff_distribution_codim()}")
    dr = system.Pfaff_distribution_rank(at)
    first.append(f"distribution rank: {dr}")
    first.append(
        "complementary forms completing coframe: "
        + (", ".join(math(f) for f in system.complementary_forms) or "none")
    )
    if system._parameters:
        first.append("parameters: " + ", ".join(math(x) for x in system._parameters))
    if system.horizontal_forms:
        first.append(
            "horizontal forms independent modulo the Pfaff system: "
            + ", ".join(math(f) for f in system.horizontal_forms)
        )
    second = []
    second.append(
        "completely integrable: "
        + (
            "not computed"
            if "integrable" not in cache
            else _yes_no(cache["integrable"])
        )
    )
    if data is None:
        second.append("polar system ranks, characters, genre: not computed")
    else:
        second.append(f"polar system ranks: {data.polar_ranks}")
        second.append(f"characters: {data.characters}")
        second.append(
            "genre: "
            + (
                "none, the distribution is zero"
                if data.genre is None
                else str(data.genre)
            )
        )
    if "characteristic" not in cache:
        second.append("characteristic distribution: not computed")
    else:
        kernel = cache["characteristic"]
        line = f"characteristic distribution: dimension {len(kernel)}"
        if 0 < len(kernel) <= 3:
            line += ", spanned by " + ", ".join(math(v) for v in kernel)
        second.append(line)
    if system.defining_equations and "random_regular" in cache:
        point, flag, detail = cache["random_regular"]
        if point is None:
            if flag == "inconsistent":
                second.append("Sampling suggests that the PDE system is likely inconsistent")
            elif flag == "non_real":
                second.append(
                    "Sampling suggests that the PDE system is likely inconsistent, while solutions may exist "
                    "over C for variables currently assumed real (" + ", ".join(math(y) for y in detail) + ")"
                )
        elif flag:
            second.append("PDE contains regular points (confirmed by sampling)")
        else:
            second.append(
                f"WARNING: sampling suggests PDE does not contain regular points ({detail}). "
                "Suggestion: use PfaffSystem(..., restrict=True) instead"
            )
    if system.horizontal_forms:
        if "verdict" not in cache:
            second.append("Cartan involutive: not computed")
        else:
            obstruction = cache["verdict"]
            second.append(
                "Cartan involutive: yes"
                if obstruction is None
                else f"Cartan involutive: no ({obstruction})"
            )
    third = None
    if system.horizontal_forms and cache.get("verdict", True) is None:
        third = _generality(system, data, dr)
    return first, second, third


def _count(number, noun):
    return f"{number} {noun}" + ("" if number == 1 else "s")


def _count_lines(lead, counts):
    parts = []
    for k, s in enumerate(counts):
        if s:
            parts.append(
                _count(s, "constant")
                if k == 0
                else f"{_count(s, 'function')} of {_count(k, 'variable')}"
            )
    if not parts:
        return lead + " no free data", []
    return lead + ":", parts


def _generality(system, data, distribution_rank):
    n = len(system.horizontal_forms)
    earlier = list(data.characters[: n - 1])
    counts = [system.Pfaff_distribution_codim()] + earlier
    counts.append(distribution_rank - n - sum(earlier))
    lead = f"the horizontal {n}-dimensional integral submanifolds (i.e., nowhere-vertical submanifolds tangent to the distribution of the Pfaff system) depend on"
    corollary = None
    if system.defining_equations:
        codim = len(system.defining_equations)
        corollary = _count_lines(
            f"PDE corollary: since the PDE has codimension {codim} in the jet space, its solutions depend on",
            [counts[0] - codim] + counts[1:],
        )
    return _count_lines(lead, counts) + (corollary,)


def _generality_html(lead, items, corollary):
    def block(text, parts):
        lis = "".join(f"<li>{part}</li>" for part in parts)
        return text + (
            f'<div class="dgcv-panel-list"><ul>{lis}</ul></div>' if lis else ""
        )

    return block(lead, items) + '<hr class="dgcv-panel-divider"/>' + block(*corollary)


def _system_plain(system, cache, at):
    first, second, third = _sections(system, cache, at, str)
    titles = ("Pfaff system", "Invariants", "Integral submanifold space")
    width = max(len(f" {t} ") for t in titles)
    lines = [_header_block(titles[0], width), "• " + _equations(system, False)]
    lines += [f"• {item}" for item in first]
    lines += ["", _header_block("Structure equations of the Pfaff system forms", width)]
    lines += [f"• {item}" for item in _structure_lines(system, False)]
    lines += ["", _header_block(titles[1], width)]
    lines += [f"• {item}" for item in second]
    lines += ["", f"• {_flag_note}", f"• {_terminology_note}"]
    singular = _singularity_lines(system, str)
    if singular:
        lines += ["", _header_block("Singularities", width)]
        lines += [f"• {item}" for item in singular]
    if third:
        lead, items, corollary = third
        lines += ["", _header_block(titles[2], width), f"• {lead}"]
        lines += [f"    - {item}" for item in items]
        if corollary:
            lines += ["", "-" * width, f"• {corollary[0]}"]
            lines += [f"    - {item}" for item in corollary[1]]
        lines += ["", f"• {_generality_note}"]
    return "\n".join(lines)


_structure_equation_limit = 15


def _structure_lines(system, use_latex):
    equations = system.coframe.structure_equations
    forms = list(system.Pfaffian_forms)
    shown = forms[:_structure_equation_limit]
    lines = []
    for form in shown:
        if use_latex:
            lines.append(
                rf"$\operatorname{{d}}{LaTeX(form)} = {_retex(system, LaTeX(equations[form]))}$"
            )
        else:
            lines.append(f"d {form} = {equations[form]}")
    rest = len(forms) - len(shown)
    if rest:
        lines.append(f"\u2026 and {rest} more, see coframe.structure_equations")
    return lines


def _equations(system, use_latex):
    if use_latex:
        return (
            "$" + r",\, ".join(f"{LaTeX(f)} = 0" for f in system.Pfaffian_forms) + "$"
        )
    text = ", ".join(f"{f} = 0" for f in system.Pfaffian_forms)
    return text


def _system_rich(system, cache, at, use_latex, style_key, extra_support):
    theme_vars, theme_data = get_style(style_key, return_theme_data=True)
    radius = int(
        theme_data.custom_css_vars.get("--dgcv-border-radius", "12px").replace("px", "")
    )
    plaque = {
        "plaque_fill": "--plaque-fill" in theme_data.custom_css_vars,
        "plaque_border": "--plaque-border" in theme_data.custom_css_vars,
        "plaque_content": True,
    }
    math = (lambda x: f"${_retex(system, LaTeX(x))}$") if use_latex else (lambda x: html.escape(str(x)))
    first, second, third = _sections(system, cache, at, math)
    equations = (
        _equations(system, use_latex)
        if use_latex
        else html.escape(_equations(system, False))
    )
    structure = _structure_lines(system, use_latex)
    if not use_latex:
        structure = [html.escape(line) for line in structure]
    specs = [
        {"header": "Pfaff system", "primary_text": equations, "itemized_text": first},
        {
            "header": "Structure equations of the Pfaff system forms",
            "itemized_text": structure,
        },
        {
            "header": "Invariants",
            "itemized_text": second,
            "footer": _HTMLWrapper(f"{_flag_note}<br>{_terminology_note}"),
        },
    ]
    singular = _singularity_lines(system, math)
    if singular:
        specs.append({"header": "Singularities", "itemized_text": singular})
    wrapped = ()
    if third:
        lead, items, corollary = third
        spec = {"header": "Integral submanifold space", "footer": _generality_note}
        if corollary:
            spec["primary_text"] = _HTMLWrapper(
                _generality_html(lead, items, corollary)
            )
        else:
            spec["primary_text"], spec["itemized_text"] = lead, items
        specs.append(spec)
        wrapped = (len(specs) - 1,)
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
    container_id = f"{_system_container_id}-{uuid.uuid4().hex[:8]}"
    return latex_in_html(
        _HTMLWrapper(_stack(blocks, container_id, theme_vars, theme_data, wrapped)),
        extra_support_for_math_in_tables=extra_support,
    )


def system_summary(
    system,
    *,
    at=None,
    theme=None,
    use_latex=None,
    plain_text=None,
    return_displayable=False,
    interrupt_to_partial_report=True,
    reporting_threshold_s=7.0,
    **kwargs,
):
    plain_text, use_latex, style_key, extra_support = summary_options(
        theme, use_latex, plain_text, kwargs
    )
    threshold = float(reporting_threshold_s)
    updates = []
    try:
        cache = _warm_summary(system, at, threshold, updates.append)
    except KeyboardInterrupt:
        if not interrupt_to_partial_report:
            raise
        updates.append(None)
        print(
            "\nInterrupted. Rendering the summary from the results computed so far; "
            "they are kept, so re-running summary resumes from here."
        )
        cache = system._summary_cache.get(
            None if at is None else frozenset(at.items()), {}
        )
    if plain_text:
        out = _system_plain(system, cache, at)
    else:
        out = _timed_progress_call(
            lambda: _system_rich(system, cache, at, use_latex, style_key, extra_support),
            timed=True,
            threshold_s=threshold,
            step_desc="rendering the summary",
            continue_desc=None,
            progress_message=None,
            _on_timed_update=updates.append,
        )
    if updates:
        print()
    if return_displayable:
        return out
    if plain_text:
        print(out)
        return
    show(out)
