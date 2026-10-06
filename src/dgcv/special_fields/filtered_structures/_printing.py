from __future__ import annotations

from ..._aux._backends._display_engine import is_rich_displaying_available
from ..._aux._utilities._config import get_dgcv_settings_registry, latex_in_html
from ..._aux._utilities._styles import get_style
from ..._aux._vmf.vmf import order_coordinates
from ..._aux.printing._tables import build_plain_table
from ..._aux.printing.printing._dgcv_display import LaTeX, show


_summary_level_limit = 15
_summary_low_rows = 8


def _elide_levels(items, verbose):
    if verbose or len(items) <= _summary_level_limit:
        return list(items), 0
    high = _summary_level_limit - _summary_low_rows
    return (
        list(items[:_summary_low_rows]) + [None] + list(items[-high:]),
        len(items) - _summary_level_limit,
    )


class _symbol_printing:
    def summary(
        self,
        theme=None,
        use_latex=None,
        display_length=500,
        table_scroll=False,
        cell_scroll=False,
        plain_text: bool | None = None,
        condense_tensor_labels: bool = True,
        return_displayable: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        dgcvSR = get_dgcv_settings_registry()
        extra_support_for_math_in_tables = bool(
            dgcvSR.get("extra_support_for_math_in_tables") is True
        )
        params = len(self._parameters) > 0
        sing = len(self._singularities.get("prolongation", set())) > 0

        if use_latex is None:
            use_latex = dgcvSR.get("use_latex", False)
        if plain_text is None:
            plain_text = not bool(use_latex)
        if not is_rich_displaying_available():
            plain_text = True

        levels = self.levels or {}
        level_items = sorted(levels.items(), key=lambda kv: kv[0])
        shown_items, omitted = _elide_levels(level_items, verbose)
        omitted_note = (
            f"{omitted} graded level{'s' if omitted != 1 else ''} omitted; pass verbose=True to display all"
        )
        have_prolongations = any(w >= 0 for w in levels.keys())
        if all(len(v) == 0 for w, v in self.nonneg_levels.items() if w >= 1):
            condense_tensor_labels = False
        paramessage = "Parametric " if params else ""
        main_title = paramessage + (
            "Tanaka Symbol (+ prolongation) Components"
            if have_prolongations
            else "Tanaka Symbol Components"
        )

        if plain_text:

            def _header_block(title: str, inner_width: int) -> str:
                inner = f" {title} "
                pad = max(0, inner_width - len(inner))
                left = pad // 2
                right = pad - left
                return "===" + ("=" * left) + inner + ("=" * right) + "==="

            dsubs = getattr(self, "distinguished_subspaces", []) or []
            unsupported_DS = len(getattr(self, "_inadmissible_DS", [])) > 0
            render_panel = any(dsubs)
            sub_title = "Distinguished Subspaces"
            paratitle = "Parameters"
            singtitle = "Singularities in Parameter Space"

            inner_width = max(
                len(f" {main_title} "),
                len(f" {sub_title} ") if render_panel else 0,
                len(f" {paratitle} ") if params else 0,
                len(f" {singtitle} ") if sing else 0,
            )
            lines = [_header_block(main_title, inner_width)]

            for item in shown_items:
                if item is None:
                    lines.append(f"• ... ({omitted_note})")
                    continue
                w, basis = item
                dim_here = len(basis or [])
                lines.append(f"• graded level {w} ({dim_here} dimensional)")
                basis_str = ", ".join(str(b) for b in (basis or []))
                if display_length is not None and len(basis_str) > display_length:
                    basis_str = "output too long to display; raise `display_length` to a higher bound if needed."
                lines.append(f"  ◦ [{basis_str}]")

            if render_panel:
                lines.append("")
                lines.append(_header_block(sub_title, inner_width))
                for sub in dsubs:
                    inner = ", ".join(str(e) for e in (sub or [])) if sub else "∅"
                    lines.append(f"• [{inner}]")
                if unsupported_DS:
                    lines.append(
                        "  ◦ at least one listed subspace was found inadmissible during preprocessing and is being disregarded; see the warnings raised when this symbol was constructed"
                    )
            if params:
                lines.append("")
                lines.append(_header_block(paratitle, inner_width))
                inner = ", ".join(str(e) for e in order_coordinates(self._parameters))
                lines.append(f"• [{inner}]")
            if sing:
                lines.append("")
                lines.append(_header_block(singtitle, inner_width))
                inner = ", ".join(
                    str(e) for e in self._singularities.get("prolongation", [])
                )
                lines.append(f"• [{inner}]")
            out = "\n".join(lines)
            if return_displayable:
                return out
            print(out)
            return

        # HTML path
        if not isinstance(theme, str):
            style_key = kwargs.get("style", None) or dgcvSR.get("theme", "dark")
        else:
            style_key = theme
        theme_vars = get_style(style_key, legacy=False)

        def _to_string(e, ul=False):
            if ul:
                s = e._repr_latex_(verbose=False, alias=condense_tensor_labels)
                if s.startswith("$") and s.endswith("$"):
                    s = s[1:-1]
                s = (
                    s.replace(r"\\displaystyle", "")
                    .replace(r"\displaystyle", "")
                    .strip()
                )
                return f"${s}$"
            return str(e)

        rows = []
        sum_computed_dimensions = sum(len(basis) for _, basis in level_items)
        for item in shown_items:
            if item is None:
                rows.append(["...", "...", omitted_note])
                continue
            w, basis = item
            basis_str = ", ".join(_to_string(b, ul=use_latex) for b in basis)
            if display_length is not None and len(basis_str) > display_length:
                basis_str = "output too long to display; raise `display_length` to a higher bound if needed."
            dim_here = len(basis)
            rows.append([str(w), str(dim_here), basis_str])

        footer = [
            [
                {
                    "html": f"Total dimension: {sum_computed_dimensions}",
                    "attrs": {"colspan": 3},
                }
            ]
        ]

        dsubs = getattr(self, "distinguished_subspaces", []) or []
        unsupported_DS = len(getattr(self, "_inadmissible_DS", [])) > 0
        render_panel = any(dsubs) or params or sing
        secondary_panel_html = None
        if render_panel:

            def _to_math(e):
                if use_latex:
                    return f"${LaTeX(e)}$"
                return str(e)

            def _panel_section(title, items_html, include_divider):
                # Clean, semantic layout nodes with no presentation styling properties
                divider = (
                    "<div class='dgcv-side-divider'></div>" if include_divider else ""
                )

                lis = "".join(f"<li>{item}</li>" for item in items_html)
                return (
                    f"{divider}"
                    f"<div class='dgcv-side-title'>{title}</div>"
                    f"<ul class='dgcv-side-list'>{lis}</ul>"
                )

            sections = []

            if any(dsubs):
                if use_latex:
                    dsub_items = []
                    for sub in dsubs:
                        if sub:
                            inner = ", ".join(
                                (e._repr_latex_(raw=True) or "").strip() for e in sub
                            )
                        else:
                            inner = r"\varnothing"
                        dsub_items.append(f"$\\left\\langle {inner} \\right\\rangle$")
                else:
                    import html

                    dsub_items = [
                        f"[{', '.join(html.escape(str(e)) for e in sub) if sub else '∅'}]"
                        for sub in dsubs
                    ]
                if unsupported_DS:
                    dsub_items.append(
                        "at least one listed subspace was found inadmissible during preprocessing and is being disregarded; see the warnings raised when this symbol was constructed"
                    )
                sections.append(
                    _panel_section("Distinguished Subspaces", dsub_items, False)
                )

            if params:
                param_items = [_to_math(e) for e in order_coordinates(self._parameters)]
                sections.append(
                    _panel_section(
                        "Parameters", [", ".join(param_items)], len(sections) > 0
                    )
                )

            if sing:
                sing_items = [
                    _to_math(e) for e in self._singularities.get("prolongation", [])
                ]
                sections.append(
                    _panel_section(
                        "Singularities in Parameter Space",
                        [", ".join(sing_items)],
                        len(sections) > 0,
                    )
                )

            raw_content = "".join(sections)

            if "--plaque-fill" in (theme_vars or ""):
                content_html = f"<div class='dgcv-side-plaque'>{raw_content}</div>"
            else:
                content_html = raw_content

            secondary_panel_html = f"<div class='dgcv-side-panel'>{content_html}</div>"

        extra_css = """
.dgcv-data-table th:nth-child(1), .dgcv-data-table td:nth-child(1),
.dgcv-data-table th:nth-child(2), .dgcv-data-table td:nth-child(2) {
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
.dgcv-side-scroll-area { overflow-x: auto; width: 100%; }
.dgcv-side-panel h3 { margin: 0; font-weight: bold; font-size: 1.1em; }
.dgcv-side-panel hr { border: 0; border-top: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main); margin: 8px 0; }
.dgcv-data-table tfoot td {
    text-align: left;
    font-weight: bold;
    background-color: var(--dgcv-bg-primary);
    color: var(--dgcv-text-heading);
    border-top: 2px solid var(--dgcv-border-main);
}
"""

        table = build_plain_table(
            columns=["Weight", "Dimension", "Basis"],
            rows=rows,
            caption=main_title,
            theme_css_vars=theme_vars,
            extra_css=extra_css,
            table_attrs='style="table-layout:auto;"',
            cell_align="center",
            escape_cells=False,
            escape_headers=True,
            secondary_panel_html=secondary_panel_html,
            layout="row",
            gap_px=10,
            side_width="340px",
            container_id="tanaka-summary",
            footer_rows=footer,
            table_scroll=table_scroll,
            cell_scroll=cell_scroll,
            ul=12,
            ur=12,
            ll=12,
            lr=12,
        )

        out = latex_in_html(
            table, extra_support_for_math_in_tables=extra_support_for_math_in_tables
        )
        if return_displayable:
            return out
        show(out)

    def __str__(self):
        levels = self.levels
        total_dim = sum(len(basis) for basis in levels.values())

        all_weights = list(levels.keys()) + ["total"]
        all_dims = [len(basis) for basis in levels.values()] + [total_dim]

        max_len = max(
            max(len(str(w)) for w in all_weights),
            max(len(str(d)) for d in all_dims),
        )

        weights_row = " │ ".join(str(w).ljust(max_len) for w in all_weights)
        dims_row = " │ ".join(str(d).ljust(max_len) for d in all_dims)

        weights_line = f"Weights    │ {weights_row}"
        dims_line = f"Dimensions │ {dims_row}"
        line_len = max(len(weights_line), len(dims_line)) + 1
        header_len = len("Weights    │ ")

        top = f"┌{'─' * (header_len - 1)}┬{'─' * (1 + line_len - header_len)}┐"
        middle = f"├{'─' * (header_len - 1)}┼{'─' * (1 + line_len - header_len)}┤"
        bottom = f"└{'─' * (header_len - 1)}┴{'─' * (1 + line_len - header_len)}┘"

        result = [
            "Tanaka Symbol:",
            top,
            f"│ {weights_line} │",
            middle,
            f"│ {dims_line} │",
            bottom,
        ]
        return "\n".join(result)

    def _repr_latex_(self):
        levels = self.levels
        weights = list(levels.keys())
        dims = [len(basis) for basis in levels.values()]
        total_dim = sum(dims)

        weights_row = " & ".join(map(str, weights)) + r" & \text{total} \\"
        dims_row = " & ".join(map(str, dims)) + rf" & {total_dim} \\"

        lines = [
            r"\textbf{Tanaka Symbol}\\[0.5em]",
            r"\begin{array}{|c||" + "c" * (len(weights)) + r"|c|}",
            r"\hline",
            r"\text{Weights} & " + weights_row,
            r"\hline",
            r"\text{Dimensions} & " + dims_row,
            r"\hline",
            r"\end{array}",
        ]
        return "$" + "\n".join(lines) + "$"

    def _sympystr(self, printer):
        result = ["Tanaka Symbol:"]
        result.append("Weights and Dimensions:")
        for weight, basis in self.levels.items():
            dim = len(basis)
            basis_str = ", ".join(printer.doprint(b) for b in basis)
            result.append(f"  {weight}: Dimension {dim}, Basis: [{basis_str}]")
        return "\n".join(result)
