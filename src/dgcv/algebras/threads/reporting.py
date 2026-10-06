from __future__ import annotations

import textwrap
import uuid
from html import escape as _esc

from ..._aux._backends._display import fast_printable
from ..._aux._backends._polynomials import expr_union_primitives
from ..._aux._backends._symbolic_router import _scalar_is_zero, get_free_symbols
from ..._aux._utilities._config import get_dgcv_settings_registry, latex_in_html
from ..._aux._utilities._styles import get_style
from ..._aux._vmf.vmf import order_coordinates
from ..._aux.printing._tables import build_matrix_table, panel_view
from .heating import _ideal_iso_label, _ideal_over_complexification
from .report_components import (
    _alg_name_plain,
    _basic_items_plain,
    _dimension_item,
    _fmt_angle_list,
    _fmt_grading_plain,
    _is_trivial_level,
    _level_dim,
    _levi_terse,
    _radical_kind,
    _singularity_sources,
)


_MAX_BASIS_COLUMNS = 24
_MAX_LIST_ITEMS = 12
_LIST_NODE_BUDGET = 250
_ELISION_NOTE = (
    "Some large output was elided from this report. "
    "Pass `elide_large_output=False` to `summary` to show everything."
)


def _element_size(elem, max_nodes):
    coeffs = getattr(elem, "coeff_dict", None)
    if not isinstance(coeffs, dict):
        return fast_printable(elem, max_nodes=max_nodes, return_count=True)
    total = 0
    for c in coeffs.values():
        total += fast_printable(
            c, max_nodes=max(max_nodes - total, 0), return_count=True
        )
        if total > max_nodes:
            break
    return total


def _elide_items(items, sized=False):
    items = list(items)
    budget = _LIST_NODE_BUDGET
    shown = []
    for item in items[:_MAX_LIST_ITEMS]:
        if sized:
            budget -= _element_size(item, budget)
            if budget < 0:
                break
        shown.append(item)
    return shown, len(items) - len(shown)


def _summary_render_plain(
    parentAlg,
    refAlg,
    *,
    subAlg: bool,
    algebra_name: str,
    algebra_name_cap: str,
    show_singularities: bool | None = None,
    elide_large_output: bool = True,
) -> str:
    max_items = 12 if elide_large_output else 10**9
    nm = _alg_name_plain(parentAlg)
    alg_dim = getattr(refAlg, "dimension", None)

    lines = [f"=== Algebra Summary: {nm} ({alg_dim} dimensional) ==="]

    if getattr(refAlg, "dimension", None) == 0:
        if subAlg:
            lines.append(
                f"  - This is the trivial 0-dimensional subalgebra in {algebra_name}."
            )
        else:
            lines.append("  - This is the trivial 0-dimensional algebra.")
        return "\n".join(lines).rstrip()

    lines.append("Basic properties:")
    for it in _basic_items_plain(refAlg, subAlg=subAlg, algebra_name=algebra_name):
        lines.append(f"  - {it}")

    basis = getattr(refAlg, "basis", ()) or ()
    lines.append("Basis and grading:")
    lines.append(f"  - basis: {_fmt_angle_list(basis, max_items=max_items)}")
    grad = getattr(refAlg, "grading", None)
    lines.append(f"  - grading: {_fmt_grading_plain(grad, max_items=max_items)}")
    if isinstance(grad, (list, tuple)):
        for gi, g in enumerate(grad, start=1):
            if not isinstance(g, (list, tuple)) or len(g) != len(basis):
                lines.append(f"  - warning: grading {gi} invalid or length mismatch")

    center = getattr(refAlg, "_center_cache", None)
    if center:
        cbasis = getattr(center, "basis", ()) or ()
        lines.append("Center:")
        lines.append(f"  - dimension: {getattr(center, 'dimension', None)}")
        lines.append(f"  - basis: {_fmt_angle_list(cbasis, max_items=max_items)}")

    ld = getattr(refAlg, "_Levi_deco_cache", None)
    if getattr(refAlg, "_lie_algebra_cache", None) is True and isinstance(ld, dict):
        comps = ld.get("LD_components", None)
        Levi_component = (
            comps[0] if isinstance(comps, (list, tuple)) and len(comps) > 0 else None
        )
        rad = comps[1] if isinstance(comps, (list, tuple)) and len(comps) > 1 else None
        simples = ld.get("simple_ideals", None)

        def _ideal_lines(indent, index, alg):
            adim = getattr(alg, "dimension", None)
            try:
                rank = alg.approximate_rank(_use_cache=True, assume_semisimple=True)
            except Exception:
                rank = "?"
            text, scope = _ideal_iso_label(
                alg, use_latex=False, rank=rank, refAlg=refAlg, return_scope=True
            )
            kind = "complexified type" if scope == "complexification" else "type"
            head = f"{indent}- Ideal {index}: dim={adim}, rank={rank}"
            if text is not None:
                head += f", {kind} {text}"
            out = [head]
            ibasis = getattr(alg, "basis", ()) or ()
            if ibasis:
                out.append(
                    f"{indent}    basis: {_fmt_angle_list(ibasis, max_items=max_items)}"
                )
            return out

        lines.append("Levi decomposition:")
        lines.append(f"  - {_levi_terse(refAlg)}")
        if refAlg.is_solvable():
            lines.append("  - The algebra equals its own maximal solvable ideal.")
        elif refAlg.is_semisimple():
            if simples is None:
                lines.append(
                    "  - The algebra is semisimple; the simple-ideal decomposition is not yet evaluated."
                )
            elif len(simples) == 1:
                only = next(iter(simples))
                complexified = _ideal_over_complexification(only, refAlg)
                iso, scope = _ideal_iso_label(
                    only, use_latex=False, refAlg=refAlg, return_scope=True
                )
                head = (
                    "  - The complexified algebra is simple"
                    if complexified
                    else "  - The algebra is simple"
                )
                if iso is None:
                    lines.append(f"{head}.")
                elif complexified or scope != "complexification":
                    lines.append(f"{head}, isomorphic to {iso}.")
                else:
                    lines.append(f"{head}, with complexification isomorphic to {iso}.")
                try:
                    only_rank = only.approximate_rank(
                        _use_cache=True, assume_semisimple=True
                    )
                except Exception:
                    only_rank = "?"
                lines.append(f"      - rank: {only_rank}")
            else:
                subject = (
                    "The complexified algebra"
                    if any(_ideal_over_complexification(a, refAlg) for a in simples)
                    else "The algebra"
                )
                lines.append(
                    f"  - {subject} is a direct sum of the following simple ideals:"
                )
                for idx, alg in enumerate(simples, start=1):
                    lines.extend(_ideal_lines("      ", idx, alg))
        else:
            ss_dim = (
                getattr(Levi_component, "dimension", "?") if Levi_component else "?"
            )
            rad_dim = getattr(rad, "dimension", "?") if rad else "?"
            lines.append("  - Semidirect sum of semisimple and solvable components:")
            lines.append(f"      - semisimple part: {ss_dim} dimensional")
            if Levi_component is not None:
                lines.append(
                    "          basis: "
                    f"{_fmt_angle_list(getattr(Levi_component, 'basis', ()) or (), max_items=max_items)}"
                )
            lines.append(
                f"      - max. solvable ideal: {rad_dim} dimensional, {_radical_kind(rad)}"
                if rad
                else f"      - max. solvable ideal: {rad_dim} dimensional"
            )
            if rad is not None:
                lines.append(
                    "          basis: "
                    f"{_fmt_angle_list(getattr(rad, 'basis', ()) or (), max_items=max_items)}"
                )

        if (
            Levi_component is not None
            and getattr(Levi_component, "dimension", 0) != 0
            and simples is not None
            and len(simples) >= 2
        ):
            lines.append("Simple ideals in semisimple complement:")
            for idx, alg in enumerate(simples, start=1):
                lines.extend(_ideal_lines("  ", idx, alg))

        if rad is not None and getattr(rad, "dimension", 0) != 0:
            for cache_attr, title in (
                (
                    "_derived_series_cache",
                    "Derived series of the maximal solvable ideal:",
                ),
                (
                    "_lower_central_series_cache",
                    "Lower central series of the maximal solvable ideal:",
                ),
            ):
                cache = getattr(rad, cache_attr, None)
                if (
                    not isinstance(cache, (list, tuple))
                    or not cache
                    or cache[0] is None
                ):
                    continue
                lines.append(title)
                for idx, level in enumerate(cache[0], start=1):
                    elems = getattr(level, "basis", level)
                    if _is_trivial_level(elems):
                        lines.append(f"  - Level {idx}: dimension 0, empty")
                    else:
                        lines.append(
                            f"  - Level {idx}: dimension {_level_dim(elems)}, "
                            f"{_fmt_angle_list(elems, max_items=max_items)}"
                        )

    if show_singularities is not False and getattr(refAlg, "_singularities", False):
        items_sing = []
        max_operands = 1000
        for key, source in _singularity_sources:
            terms = list(refAlg._singularities.get(key, []))
            if not terms:
                continue
            if show_singularities is not True:
                printable = True
                for term in terms:
                    max_operands -= fast_printable(
                        term, max_nodes=max_operands, return_count=True
                    )
                    if max_operands < 0:
                        printable = False
                        break
                if not printable:
                    numb = len(terms)
                    plur = ("y", "it") if numb == 1 else ("ies", "them")
                    items_sing.append(
                        f"From {source}: {numb} singularit{plur[0]} omitted from "
                        f"report. (set `show_singularities=True` to display "
                        f"{plur[1]}. Warning: typically very long.)"
                    )
                    continue
            items_sing.append(
                f"From {source}: {', '.join(repr(term) for term in terms)}"
            )
        if items_sing:
            lines.append("Parameter space singularities:")
            for item in items_sing:
                lines.append(f"  - {item}")

    return "\n".join(lines).rstrip()


def _summary_render_rich(
    *,
    refAlg,
    subAlg: bool,
    algebra_name: str,
    algebra_name_cap: str,
    style,
    use_latex: bool,
    extra_support_for_math_in_tables: bool,
    show_singularities: bool | None = None,
    full=False,
    elide_large_output: bool = True,
):
    theme_vars, theme_data = get_style(style, return_theme_data=True)
    elided = False
    border_radius = int(
        theme_data.custom_css_vars.get("--dgcv-border-radius", "12px").replace("px", "")
    )
    container_id = f"dgcv-alg-summary-{uuid.uuid4().hex[:8]}"
    scoped_theme = theme_vars.replace(":root", f"#{container_id}")

    class _HTMLWrapper:
        def __init__(self, html):
            self._html = html

        def to_html(self, *args, **kwargs):
            return self._html

        def _repr_html_(self):
            return self._html

    uses_plaque = "--plaque-fill" in theme_data.custom_css_vars
    uses_plaque_border = "--plaque-border" in theme_data.custom_css_vars

    # for _stack_many
    if theme_data.custom_css_vars.get("--dgcv-special-background", None):
        panel_bg = "var(--dgcv-special-background)"
        panel_hd = "none"
        text_bg = "var(--dgcv-special-text,var(--dgcv-text-heading))"
    else:
        panel_bg = "var(--dgcv-bg-primary)"
        panel_hd = "var(--dgcv-bg-surface)"
        text_bg = "var(--dgcv-text-main)"

    def _stack_many(blocks) -> str:
        inner = "\n".join(f'<div class="section">{b}</div>' for b in blocks)
        return textwrap.dedent(f"""
            <div id="{container_id}">
            <style>
            {scoped_theme}
            #{container_id} .stack {{ display: flex; flex-direction: column; gap: 16px; align-items: stretch; width: 100%; margin: 0; }}
            #{container_id} .section {{ width: 100%; }}
            #{container_id} .dgcv-data-table th.row_heading {{ position: sticky; left: 0; z-index: 1; white-space: nowrap; }}
            #{container_id} .dgcv-panel {{
                background: {panel_bg};
                box-shadow: var(--dgcv-table-shadow, none);
                border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main);
                border-image: var(--dgcv-border-image, none);
                color: {text_bg};
                font-family: var(--dgcv-font-family, inherit);
                overflow: hidden;
                padding: 4px 4px;
                margin: 0;
            }}
            #{container_id} .dgcv-panel-head {{ padding: 0.75rem 1rem; background: {panel_hd}; }}
            #{container_id} .dgcv-panel-title {{ margin: 0; font-size: 1rem; font-weight: 600; color: var(--dgcv-special-text, var(--dgcv-text-heading)); text-shadow: var(--dgcv-text-shadow, none); }}
            #{container_id} .dgcv-panel-rule {{ border: 0; height: 2px; background: var(--dgcv-border-main); margin: 0; }}
            #{container_id} .dgcv-panel-body {{ padding: 0.75rem 1rem; overflow-x: auto; width: 100%; box-sizing: border-box; }}
            #{container_id} .dgcv-panel-list ul {{ margin: 0.25rem 0 0 1.25rem; color: {text_bg}; }}
            #{container_id} .dgcv-panel-footer {{ padding: 0.5rem 1rem; background: var(--dgcv-bg-alt); color: var(--dgcv-text-alt); border-top: 1px solid var(--dgcv-border-alt); }}
            #{container_id} .dgcv-data-table {{ width: 100%; border-collapse: collapse; background: var(--dgcv-bg-primary); color: var(--dgcv-text-main); margin: 0; }}
            #{container_id} .dgcv-data-table td, #{container_id} .dgcv-data-table th {{ border-right: 1px solid var(--dgcv-border-main); padding: 8px 12px; }}
            #{container_id} .dgcv-data-table thead th {{ background-color: var(--dgcv-bg-surface); color: var(--dgcv-text-heading); border-bottom: 3px solid var(--dgcv-border-main); }}
            #{container_id} .dgcv-data-table th.row_heading {{ background-color: var(--dgcv-bg-surface) !important; color: var(--dgcv-text-heading) !important; font-weight: bold; }}
            #{container_id} .dgcv-data-table tr:nth-child(even) {{ background-color: var(--dgcv-bg-alt); color: var(--dgcv-text-alt); }}
            #{container_id} .dgcv-table-wrap {{ overflow-x: auto; max-width: 100%; box-sizing: border-box; padding: 0; margin: 0; }}
            #{container_id} .dgcv-table-wrap > table.dgcv-data-table {{ min-width: 40rem; width: 100%; table-layout: fixed; }}
            #{container_id} .dgcv-data-table tbody tr {{
                transition: var(--dgcv-hover-transition, transform 0.2s, box-shadow 0.2s, background-color 0.2s);
            }}
            #{container_id} .dgcv-data-table tbody tr:hover {{
                background-color: var(--dgcv-bg-hover) !important;
                color: var(--dgcv-text-hover) !important;
                transform: var(--dgcv-hover-transform, none);
            }}
            #{container_id} .dgcv-data-table tbody tr:hover th.row_heading {{
                background-color: var(--dgcv-bg-surface) !important;
                color: var(--dgcv-text-heading) !important;
                transform: none !important;
            }}
            </style>
            <div class="stack">
            {inner}
            </div>
            </div>
        """).strip()

    def _corners_for(i: int, total: int):
        r = border_radius
        if total <= 1:
            return {"ul": r, "ur": r, "ll": r, "lr": r}
        if i == 0:
            return {"ul": r, "ur": r, "ll": 0, "lr": 0}
        if i == total - 1:
            return {"ul": 0, "ur": 0, "ll": r, "lr": r}
        return {"ul": 0, "ur": 0, "ll": 0, "lr": 0}

    def _fmt_bool_cache(v):
        return "true" if v is True else ("false" if v is False else "not yet evaluated")

    empty_tok = r"$\varnothing$" if use_latex else "empty"

    def _is_trivial_level(level) -> bool:
        if not level:
            return True
        if isinstance(level, (list, tuple)) and len(level) == 1:
            z = level[0]
            return _scalar_is_zero(z)
        return False

    def _omitted_tok(omitted, any_shown=True):
        nonlocal elided
        elided = True
        if any_shown:
            return f"… (+{omitted} more)"
        return f"{omitted} large element{'s' if omitted != 1 else ''} omitted"

    def _fmt_basis_list(elems):
        if _is_trivial_level(elems):
            return empty_tok
        omitted = 0
        if elide_large_output:
            elems, omitted = _elide_items(elems, sized=True)
        if use_latex:
            out = []
            for elem in elems:
                try:
                    out.append(f"${elem._repr_latex_(raw=True)}$")
                except Exception:
                    out.append(repr(elem))
        else:
            out = [repr(elem) for elem in elems]
        if omitted:
            out.append(_omitted_tok(omitted, bool(out)))
        return ", ".join(out)

    params_check = list(getattr(refAlg, "_parameters", []))
    params_omitted = 0
    if elide_large_output:
        params_shown, params_omitted = _elide_items(params_check)
    else:
        params_shown = params_check
    if use_latex:
        try:
            from ..._aux.printing.printing._dgcv_display import LaTeX_list

            params = LaTeX_list(params_shown, math_mode="$")
        except Exception:
            params = [repr(b) for b in params_shown]
    else:
        params = [repr(b) for b in params_shown]
    if params_omitted:
        params = f"{params} {_omitted_tok(params_omitted)}"

    dimension_item = _dimension_item(refAlg, use_latex)
    if params:
        items = (
            [
                f"Subalgebra family contained in {algebra_name}",
                dimension_item,
                f"Parameters: {params}",
            ]
            if subAlg
            else [dimension_item, f"Parameters: {params}"]
        )
    else:
        items = (
            [
                f"Subalgebra contained in {algebra_name}",
                dimension_item,
            ]
            if subAlg
            else [dimension_item]
        )

    lie = getattr(refAlg, "_lie_algebra_cache", None)
    if lie is True:
        items.append("Lie algebra: true")
        special_property = getattr(refAlg, "_educed_properties", dict()).get(
            "special_type", None
        )
        if special_property is not None:
            items.append(f"special properties: {special_property}")
        elif (
            getattr(refAlg, "_is_semisimple_cache", None) is False
            and getattr(refAlg, "_is_solvable_cache", None) is False
        ):
            items.append("special properties: neither solvable nor semisimple")
        else:
            items.append("special properties: not yet evaluated")
    elif lie is False:
        items.append("Lie algebra: false")
        items.append(
            f"Skew symmetric: {_fmt_bool_cache(getattr(refAlg, '_skew_symmetric_cache', None))}"
        )
        items.append(
            f"Jacobi identity satisfied: {_fmt_bool_cache(getattr(refAlg, '_jacobi_identity_cache', None))}"
        )
    else:
        items.append("Lie algebra: not yet evaluated")

    if refAlg.dimension == 0:
        pv0 = panel_view(
            header="Basic properties of the subalgebra"
            if subAlg
            else f"Basic properties of {algebra_name}",
            itemized_text=[
                f"This is the trivial 0-dimensional subalgebra in {algebra_name}."
            ]
            if subAlg
            else ["This is the trivial 0-dimensional algebra."],
            theme_css_vars=theme_vars,
            extra_css="",
            slim=True,
        ).to_html()
        return latex_in_html(
            _HTMLWrapper(_stack_many([pv0])),
            extra_support_for_math_in_tables=extra_support_for_math_in_tables,
        )

    basis_elems = getattr(refAlg, "basis", ()) or ()
    n_basis = len(basis_elems)
    n_columns = min(n_basis, _MAX_BASIS_COLUMNS) if elide_large_output else n_basis
    basis_elems = list(basis_elems)[:n_columns]
    if use_latex:
        try:
            basis_labels = [f"${b._repr_latex_(raw=True)}$" for b in basis_elems]
        except Exception:
            basis_labels = [repr(b) for b in basis_elems]
    else:
        basis_labels = [repr(b) for b in basis_elems]

    rows = [list(basis_labels)]
    grad_index_labels = ["Basis"]
    warn_msgs = []
    grad = getattr(refAlg, "grading", None)

    def _fmt_weight(x):
        if use_latex and hasattr(x, "_repr_latex_"):
            try:
                s = str(x._repr_latex_())
                if s.startswith("$") and s.endswith("$"):
                    s = s[1:-1]
                reduced = (
                    s.replace(r"\\displaystyle", "")
                    .replace(r"\displaystyle", "")
                    .strip()
                )
                return f"${reduced}$"
            except Exception:
                pass
        return str(x)

    if isinstance(grad, (list, tuple)) and grad:
        for gi, g in enumerate(grad, start=1):
            if isinstance(g, (list, tuple)) and len(g) == n_basis:
                rows.append([_fmt_weight(x) for x in g[:n_columns]])
                grad_index_labels.append(f"Grading {gi}")
            else:
                warn_msgs.append(f"grading {gi} invalid or length mismatch")

    if n_columns < n_basis:
        elided = True
        for row in rows:
            row.append("⋯")
        warn_msgs.append(f"showing the first {n_columns} of {n_basis} basis elements")
    footer_rows = (
        [
            [
                {
                    "html": f"<em>{_esc(' | '.join(warn_msgs))}</em>",
                    "attrs": {"colspan": len(rows[0])},
                }
            ]
        ]
        if warn_msgs
        else None
    )
    sections = []

    def _build_basic_panel(corner_kwargs):
        label = (
            ("the subalgebra family" if subAlg else "the algebra family")
            if params
            else ("the subalgebra" if subAlg else algebra_name)
        )
        return panel_view(
            header=f"Basic properties of {label}",
            itemized_text=items,
            theme_css_vars="",
            extra_css="",
            slim=True,
            plaque_fill=uses_plaque,
            plaque_border=uses_plaque_border,
            plaque_content=True,
            **corner_kwargs,
        ).to_html()

    sections.append(("panel", _build_basic_panel))

    def _build_basis_panel(corner_kwargs):
        table_obj = build_matrix_table(
            show_headers=False,
            index_labels=grad_index_labels,
            columns=[],
            rows=rows,
            caption="",
            theme_css_vars="",
            extra_css="",
            footer_rows=footer_rows,
            table_attrs='style="table-layout:auto;"',
            cell_align=None,
            escape_cells=False,
            escape_headers=True,
            nowrap=False,
            dashed_corner=False,
            slim=True,
            panel_content=True,
        )
        return panel_view(
            header="Basis and assigned grading(s)",
            primary_text=table_obj,
            theme_css_vars="",
            extra_css="",
            slim=True,
            plaque_fill=uses_plaque,
            plaque_border=uses_plaque_border,
            plaque_content=False,
            **corner_kwargs,
        ).to_html()

    sections.append(("panel", _build_basis_panel))

    if (
        getattr(refAlg, "_center_cache", None)
        and getattr(refAlg, "_is_semisimple_cache", None) is not True
    ):

        def _center_panel(corner_kwargs):
            IT = []
            center = getattr(refAlg, "_center_cache", None)
            center_basis = list(getattr(center, "basis", ()) or ())
            if elide_large_output and _elide_items(center_basis, sized=True)[1]:
                PT = f"Dimension {len(center_basis)}, spanned by {_fmt_basis_list(center_basis)}"
            else:
                PT = center._repr_latex_(raw=False, verbose=True)
            return panel_view(
                header="Center",
                primary_text=PT,
                itemized_text=IT,
                theme_css_vars="",
                extra_css="",
                slim=True,
                plaque_fill=uses_plaque,
                plaque_border=uses_plaque_border,
                plaque_content=True,
                **corner_kwargs,
            ).to_html()

        sections.append(("panel", _center_panel))

    ld = getattr(refAlg, "_Levi_deco_cache", None)
    if getattr(refAlg, "_lie_algebra_cache", None) is True and isinstance(ld, dict):
        simples = ld.get("simple_ideals", None)
        Levi_component, rad = ld.get("LD_components", (None, None))

        def _LD_panel(corner_kwargs):
            IT = []
            solv, semi = (
                getattr(refAlg, "_is_solvable_cache", None),
                getattr(refAlg, "_is_semisimple_cache", None),
            )
            if solv is True:
                PT = (
                    "The subalgebra equals its own maximal solvable ideal."
                    if subAlg
                    else f"{algebra_name_cap} equals its own maximal solvable ideal."
                )
            elif semi is True:
                if simples is None:
                    PT = (
                        "The subalgebra is semisimple"
                        if subAlg
                        else f"{algebra_name_cap} is semisimple"
                    )
                elif len(simples) == 1:
                    alg = next(iter(simples))
                    complexified = _ideal_over_complexification(alg, refAlg)
                    if complexified:
                        PT = (
                            "The complexified subalgebra is simple"
                            if subAlg
                            else f"The complexification of {algebra_name} is simple"
                        )
                    else:
                        PT = (
                            "The subalgebra is simple"
                            if subAlg
                            else f"{algebra_name_cap} is simple"
                        )
                    if full is False:
                        PT += "."
                    else:
                        rank = "?"
                        try:
                            ss = True if len(params) > 0 else False
                            rankout = alg.approximate_rank(
                                _use_cache=True,
                                assume_semisimple=True,
                                surface_singularities=ss,
                            )
                            if ss:
                                rank, divisors = rankout
                            else:
                                rank = rankout
                            if divisors:
                                new_sing = list(
                                    alg._singularities.get("subalgebra_ranks", [])
                                ) + [v for v in divisors if get_free_symbols(v)]
                                if get_dgcv_settings_registry().get(
                                    "simplify_singularity_ideals_by_default", True
                                ):
                                    new_sing = expr_union_primitives(
                                        new_sing,
                                        order_coordinates(alg._parameters),
                                        process_rationals=True,
                                        fail_quietly=True,
                                    )
                                alg._singularities["subalgebra_ranks"] = new_sing
                                if alg != refAlg:
                                    new_sing = list(
                                        refAlg._singularities.get(
                                            "subalgebra_ranks", []
                                        )
                                    ) + [v for v in divisors if get_free_symbols(v)]
                                    if get_dgcv_settings_registry().get(
                                        "simplify_singularity_ideals_by_default", True
                                    ):
                                        new_sing = expr_union_primitives(
                                            new_sing,
                                            order_coordinates(refAlg._parameters),
                                            process_rationals=True,
                                            fail_quietly=True,
                                        )
                                    refAlg._singularities["subalgebra_ranks"] = new_sing
                        except Exception:
                            pass
                        IC, scope = _ideal_iso_label(
                            alg,
                            use_latex=use_latex,
                            rank=rank,
                            refAlg=refAlg,
                            return_scope=True,
                        )
                        if IC is None:
                            PT += "."
                        elif complexified or scope != "complexification":
                            PT += f" and isomorphic to {IC}."
                        else:
                            PT += f", with complexification isomorphic to {IC}."
                else:
                    if any(_ideal_over_complexification(a, refAlg) for a in simples):
                        PT = (
                            "The complexified subalgebra is a direct sum"
                            if subAlg
                            else f"The complexification of {algebra_name} is a direct sum"
                        )
                    else:
                        PT = (
                            "The subalgebra is a direct sum"
                            if subAlg
                            else f"{algebra_name_cap} is a direct sum"
                        )
                    shown, omitted = (
                        _elide_items(simples)
                        if elide_large_output
                        else (list(simples), 0)
                    )
                    for a in shown:
                        IT.append(
                            f"${a._repr_latex_(raw=True, abbrev=True)}$"
                            if use_latex
                            else repr(a)
                        )
                    if omitted:
                        IT.append(_omitted_tok(omitted))
            else:
                PT = (
                    "The subalgebra is a semidirect sum"
                    if subAlg
                    else f"{algebra_name_cap} is a semidirect sum"
                )
                if isinstance(ld.get("LD_components", None), (list, tuple)):
                    for a in ld["LD_components"]:
                        IT.append(
                            f"${a._repr_latex_(raw=True, abbrev=True)}$"
                            if use_latex
                            else repr(a)
                        )
            return panel_view(
                header="Levi decomposition",
                primary_text=PT,
                itemized_text=IT,
                theme_css_vars="",
                extra_css="",
                slim=True,
                plaque_fill=uses_plaque,
                plaque_border=uses_plaque_border,
                plaque_content=True,
                **corner_kwargs,
            ).to_html()

        sections.append(("panel", _LD_panel))

        if getattr(refAlg, "_is_simple_cache", None) is not True and (
            Levi_component is not None
            and getattr(Levi_component, "dimension", 0) != 0
            and simples is not None
        ):

            def _ss_compl_panel(corner_kwargs):
                rows2 = []
                scopes = []
                for idx, a in enumerate(simples):
                    rank = "?"
                    try:
                        ss = True if len(params) > 0 else False
                        rankout = a.approximate_rank(
                            _use_cache=True,
                            assume_semisimple=True,
                            surface_singularities=ss,
                        )
                        if ss:
                            rank, divisors = rankout
                        else:
                            rank = rankout
                        if divisors:
                            new_sing = list(
                                refAlg._singularities.get("subalgebra_ranks", [])
                            ) + [v for v in divisors if get_free_symbols(v)]
                            if get_dgcv_settings_registry().get(
                                "simplify_singularity_ideals_by_default", True
                            ):
                                new_sing = expr_union_primitives(
                                    new_sing,
                                    order_coordinates(refAlg._parameters),
                                    process_rationals=True,
                                    fail_quietly=True,
                                )
                            refAlg._singularities["subalgebra_ranks"] = new_sing
                            if refAlg != a:
                                new_sing = list(
                                    a._singularities.get("subalgebra_ranks", [])
                                ) + [v for v in divisors if get_free_symbols(v)]
                                if get_dgcv_settings_registry().get(
                                    "simplify_singularity_ideals_by_default", True
                                ):
                                    new_sing = expr_union_primitives(
                                        new_sing,
                                        order_coordinates(a._parameters),
                                        process_rationals=True,
                                        fail_quietly=True,
                                    )
                                a._singularities["subalgebra_ranks"] = new_sing
                    except Exception:
                        pass
                    iso, scope = _ideal_iso_label(
                        a,
                        use_latex=use_latex,
                        rank=rank,
                        refAlg=refAlg,
                        return_scope=True,
                    )
                    scopes.append(scope)
                    rows2.append(
                        [
                            f"subalgebra {idx + 1}",
                            f"{getattr(a, 'dimension', None)}",
                            f"{rank}",
                            iso or "?",
                            _fmt_basis_list(getattr(a, "basis", ()) or ()),
                        ]
                    )
                table_obj = build_matrix_table(
                    index_labels=None,
                    columns=[
                        "Ideal #",
                        "Dimension",
                        "Rank",
                        "Complexified Iso. Class"
                        if "complexification" in scopes
                        else "Iso. Class",
                        "Basis",
                    ],
                    rows=rows2,
                    theme_css_vars="",
                    extra_css="",
                    table_attrs='style="table-layout:auto;"',
                    escape_cells=False,
                    escape_headers=True,
                    dashed_corner=False,
                    slim=True,
                )
                return panel_view(
                    header="Simple ideals in semisimple complement",
                    primary_text=table_obj,
                    theme_css_vars="",
                    extra_css="",
                    slim=True,
                    plaque_fill=uses_plaque,
                    plaque_border=uses_plaque_border,
                    plaque_content=False,
                    **corner_kwargs,
                ).to_html()

            sections.append(("panel", _ss_compl_panel))

        if rad is not None and getattr(rad, "dimension", 0) != 0:
            for cache_attr, title in [
                ("_lower_central_series_cache", "Lower central series of radical"),
                ("_derived_series_cache", "Derived series of radical"),
            ]:
                cache = getattr(rad, cache_attr, None)
                if isinstance(cache, (list, tuple)) and cache and cache[0] is not None:

                    def _series_panel(corner_kwargs, c=cache[0], t=title):
                        rows2 = [
                            [
                                f"Level {idx + 1}",
                                f"{_level_dim(getattr(lvl, 'basis', lvl))}",
                                _fmt_basis_list(getattr(lvl, "basis", lvl)),
                            ]
                            for idx, lvl in enumerate(c)
                        ]
                        table_obj = build_matrix_table(
                            index_labels=None,
                            columns=["Filtration Level", "Dimension", "Basis"],
                            rows=rows2,
                            theme_css_vars="",
                            extra_css="",
                            table_attrs='style="table-layout:auto;"',
                            dashed_corner=False,
                            escape_cells=False,
                            escape_headers=True,
                            slim=True,
                        )
                        return panel_view(
                            header=t,
                            primary_text=table_obj,
                            theme_css_vars="",
                            extra_css="",
                            slim=True,
                            plaque_fill=uses_plaque,
                            plaque_border=uses_plaque_border,
                            plaque_content=False,
                            **corner_kwargs,
                        ).to_html()

                    sections.append(("panel", _series_panel))

    if show_singularities is not False and getattr(refAlg, "_singularities", False):

        def singularities_panel(corner_kwargs):
            items_sing = []
            max_operands = 1000
            for key, label in _singularity_sources:
                if show_singularities is not True:
                    sings = refAlg._singularities.get(key, set())
                    printable = True
                    for sing in sings:
                        node_count = fast_printable(
                            sing, max_nodes=max_operands, return_count=True
                        )
                        max_operands -= node_count
                        if max_operands < 0:
                            printable = False
                            break
                    if not printable:
                        numb = len(sings)
                        plur = ("y", "it") if numb == 1 else ("ies", "them")
                        items_sing.append(
                            f"From {label}: {numb} singularit{plur[0]} omitted from report. (set `show_singularities=True` to display {plur[1]}. Warning: typically very long.)"
                        )
                        continue
                terms = list(refAlg._singularities.get(key, []))
                if terms:
                    formatted = (
                        LaTeX_list(terms, math_mode="$")
                        if use_latex
                        else ", ".join([repr(x) for x in terms])
                    )
                    items_sing.append(f"From {label}: {formatted}")
            return panel_view(
                header="Parameter space singularities",
                itemized_text=items_sing,
                theme_css_vars="",
                extra_css="",
                slim=True,
                plaque_fill=uses_plaque,
                plaque_border=uses_plaque_border,
                plaque_content=True,
                **corner_kwargs,
            ).to_html()

        sections.append(("singularities", singularities_panel))

    built_blocks = [None] + [
        builder(_corners_for(i, len(sections)))
        for i, (_, builder) in enumerate(sections)
        if i > 0
    ]
    if elided:
        items.append(_ELISION_NOTE)
    built_blocks[0] = sections[0][1](_corners_for(0, len(sections)))
    return latex_in_html(
        _HTMLWrapper(_stack_many(built_blocks)),
        extra_support_for_math_in_tables=extra_support_for_math_in_tables,
    )
