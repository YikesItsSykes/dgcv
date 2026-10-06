"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv._aux.printing.printing._dgcv_display

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from __future__ import annotations

import numbers
from collections.abc import Mapping
from typing import Any

from ....core.base import dgcv_class
from ....core.conversions.conversions import symToHol, symToReal
from ..._backends._display import latex as _backend_latex
from ..._backends._display_engine import is_rich_displaying_available
from ..._backends._engine import _get_sympy_module, is_sympy_available
from ..._backends._notebooks import display_latex as _nb_display_latex
from ..._backends._types_and_constants import expr_types
from ..._utilities._config import (
    dgcv_warning,
    dgcvDeprecationWarning,
    get_dgcv_settings_registry,
)
from ..._utilities._settings import _toggle_or_set_verbosity
from ..._vmf._safeguards import check_dgcv_category, get_dgcv_category
from ._string_processing import _strip_display_dollars, _unwrap_math_delims

__all__ = ["LaTeX", "LaTeX_eqn_system", "LaTeX_list", "show"]


# -----------------------------------------------------------------------------
# body
# -----------------------------------------------------------------------------
def _is_engine_expr(x: Any) -> bool:
    return isinstance(x, expr_types())


def _try_symToHol(x: Any, removeBARs: bool) -> Any:
    if not removeBARs:
        return x
    try:
        return symToHol(x, convert_everything=False)
    except Exception:
        return x


def _latex_from_engine_expr(x: Any) -> str:
    s = _backend_latex(x)
    if not isinstance(s, str):
        return str(x)
    return _unwrap_math_delims(s)


def _is_tensorish_dgcv(x: Any) -> bool:
    return get_dgcv_category(x) == "tensor_field"


def _has_varSpace_type(x: Any) -> bool:
    return getattr(x, "_varSpace_type", None) is not None


def LaTeX(obj: Any, removeBARs: bool | None = None, verbose: bool = False) -> str:
    """
    Custom LaTeX function for dgcv. Attempts to produce a LaTeX-ish string for "mathy" objects.
    """
    if removeBARs is None:
        removeBARs = get_dgcv_settings_registry().get("compile_latex_conjugation", True)
    if verbose:
        _toggle_or_set_verbosity(setting=1)  # enable temp verbosity

    def _latex_of(x: Any) -> str:
        if x is None:
            return ""

        if isinstance(x, list):
            elems = [_strip_display_dollars(_latex_of(e)) for e in x]
            return r"\left[ " + ", ".join(elems) + r" \right]"
        if isinstance(x, tuple):
            elems = [_strip_display_dollars(_latex_of(e)) for e in x]
            return r"\left( " + ", ".join(elems) + r" \right)"
        if isinstance(x, set):
            elems = [_strip_display_dollars(_latex_of(e)) for e in x]
            return r"\left\{ " + ", ".join(elems) + r" \right\}"
        if isinstance(x, Mapping):
            elems = [
                f"{_strip_display_dollars(_latex_of(k))} : "
                f"{_strip_display_dollars(_latex_of(v))}"
                for k, v in x.items()
            ]
            return r"\left\{ " + ", ".join(elems) + r" \right\}"

        if _is_tensorish_dgcv(x):
            if removeBARs:
                x2 = symToHol(x, convert_everything=False)
            else:
                vft = getattr(x, "_varSpace_type", None)
                if vft == "real":
                    x2 = symToReal(x)
                elif vft == "complex":
                    x2 = symToHol(x)
                else:
                    x2 = x

            if _is_engine_expr(x2):
                try:
                    return _latex_from_engine_expr(x2)
                except Exception:
                    return str(x2)

            f = getattr(x2, "_repr_latex_", None)
            if callable(f):
                try:
                    s = f()
                    if isinstance(s, str):
                        return _unwrap_math_delims(s)
                except Exception:
                    pass
            if isinstance(x2, numbers.Number) and not isinstance(x2, bool):
                return _backend_latex(x2)
            return str(x2)

        if check_dgcv_category(x):
            if not _has_varSpace_type(x):
                if removeBARs:
                    x2 = symToHol(x, convert_everything=False)
                f = getattr(x, "_repr_latex_", None)
                if callable(f):
                    try:
                        s = f()
                        if isinstance(s, str):
                            return _unwrap_math_delims(s)
                    except Exception:
                        pass

            x2 = _try_symToHol(x, removeBARs)

            if _is_engine_expr(x2):
                try:
                    return _latex_from_engine_expr(x2)
                except Exception:
                    return str(x2)

            f = getattr(x2, "_repr_latex_", None)
            if callable(f):
                try:
                    s = f()
                    if isinstance(s, str):
                        return _unwrap_math_delims(s)
                except Exception:
                    pass
            if isinstance(x2, numbers.Number) and not isinstance(x2, bool):
                return _backend_latex(x2)
            return str(x2)

        x2 = _try_symToHol(x, removeBARs)

        if _is_engine_expr(x2):
            try:
                return _latex_from_engine_expr(x2)
            except Exception:
                return str(x2)

        f = getattr(x2, "_repr_latex_", None)
        if callable(f):
            try:
                s = f()
                if isinstance(s, str):
                    return _unwrap_math_delims(s)
            except Exception:
                pass

        if isinstance(x2, numbers.Number) and not isinstance(x2, bool):
            return _backend_latex(x2)
        return str(x2)

    out = _strip_display_dollars(_latex_of(obj)) or ""
    if verbose:
        _toggle_or_set_verbosity()  # enable temp verbosity
    return out


def LaTeX_eqn_system(
    eqn_dict,
    math_mode="$$",
    relation=" = ",
    left_prefix="",
    left_suffix="",
    right_prefix="",
    right_suffix="",
    one_line=False,
    bare_latex=False,
    punctuation=None,
    add_period=False,
    conjuction: str = None,
    verbose: bool = False,
    force_oxford_comma: bool = False,
):
    if verbose:
        _toggle_or_set_verbosity(setting=1)  # enable temp verbosity
    list_format = False
    if isinstance(eqn_dict, (list, tuple)):
        if all(isinstance(elem, (list, tuple)) and len(elem) == 2 for elem in eqn_dict):
            list_format = True
        else:
            rhs_only_format = True
    else:
        rhs_only_format = False

    if add_period is True:
        punct = "."
    elif isinstance(punctuation, str):
        punct = punctuation
    else:
        punct = ""
    if not isinstance(conjuction, str):
        conjuction = "and"

    if bare_latex is True:
        joiner = r", "
        boundary = ""
        penultim = (
            joiner
            if conjuction.strip() == ""
            else rf",\quad\text{{{conjuction}}}\quad "
        )
    elif math_mode == "$":
        joiner = "$, $"
        boundary = "$"
        penultim = joiner if conjuction.strip() == "" else f"$, {{{conjuction}}} $"
    elif one_line is True:
        joiner = r", \quad "
        boundary = "$$"
        penultim = (
            joiner
            if conjuction.strip() == ""
            else rf",\quad\text{{{conjuction}}}\quad "
        )
    else:
        joiner = r",$$ $$ "
        boundary = "$$"
        penultim = joiner if conjuction.strip() == "" else f",$$ {{{conjuction}}} $$"

    if list_format is True:
        kv_pairs = [
            f"{left_prefix}{LaTeX(k)}{left_suffix}{relation}{right_prefix}{LaTeX(v)}{right_suffix}"
            for k, v in eqn_dict
        ]
    elif rhs_only_format is True:
        kv_pairs = [
            f"0{relation}{right_prefix}{LaTeX(k)}{right_suffix}" for k in eqn_dict
        ]
    else:
        kv_pairs = [
            f"{left_prefix}{LaTeX(k)}{left_suffix}{relation}{right_prefix}{LaTeX(v)}{right_suffix}"
            for k, v in eqn_dict.items()
        ]
    oxc = ", " if force_oxford_comma else ""
    if verbose:
        _toggle_or_set_verbosity()  # disable temp verbosity
    if len(kv_pairs) == 0:
        return punct
    if len(kv_pairs) == 1:
        return boundary + kv_pairs[0] + punct + boundary
    if len(kv_pairs) == 2:
        if bare_latex is True:
            return (
                boundary
                + kv_pairs[0]
                + oxc
                + (r"\quad " if conjuction.strip() == "" else r"\quad\text{and}\quad ")
                + kv_pairs[1]
                + punct
                + boundary
            )
        if math_mode == "$":
            return (
                boundary
                + kv_pairs[0]
                + boundary
                + oxc
                + (" " if conjuction.strip() == "" else "and")
                + boundary
                + kv_pairs[1]
                + punct
                + boundary
            )
        if one_line is True:
            return (
                boundary
                + kv_pairs[0]
                + oxc
                + (
                    r"\quad "
                    if conjuction.strip() == ""
                    else r" \quad \text{ and }\quad "
                )
                + kv_pairs[1]
                + punct
                + boundary
            )
        return (
            boundary
            + kv_pairs[0]
            + boundary
            + oxc
            + (" " if conjuction.strip() == "" else "and")
            + boundary
            + kv_pairs[1]
            + punct
            + boundary
        )

    return (
        boundary
        + joiner.join(kv_pairs[:-1])
        + penultim
        + kv_pairs[-1]
        + punct
        + boundary
    )


def LaTeX_list(
    list_to_print,
    math_mode="$$",
    prefix="",
    suffix="",
    one_line=False,
    items_per_line=1,
    bare_latex=False,
    punctuation=None,
    item_labels=None,
    verbose: bool = False,
):
    if len(list_to_print) == 0:
        return ""
    if verbose:
        _toggle_or_set_verbosity(setting=1)  # enable temp verbosity
    if not isinstance(list_to_print, (list, tuple)):
        if bare_latex is not True and (math_mode == "$" or math_mode == "$$"):
            out = f"{math_mode}{LaTeX(list_to_print)}{math_mode}"
        else:
            out = LaTeX(list_to_print)
        if verbose:
            _toggle_or_set_verbosity()  # disable temp verbosity
        return out

    if (
        one_line is True
        or math_mode == "$"
        or not isinstance(items_per_line, numbers.Integral)
        or items_per_line < 1
    ):
        items_per_line = len(list_to_print)

    if not isinstance(item_labels, (list, tuple)):
        item_labels = []
    item_labels = [
        str(label) + " = "
        for label in item_labels[: min(len(item_labels), len(list_to_print))]
    ] + ([""]) * max(0, len(list_to_print) - len(item_labels))

    punct = punctuation if isinstance(punctuation, str) else ""

    if bare_latex is True:
        joiner = r", "
        boundary = ""
        penultim = r",\quad\text{and}\quad "
    elif math_mode == "$":
        joiner = "$, $"
        boundary = "$"
        penultim = "$, and $"
    elif items_per_line != 1:
        joiner = r", \quad "
        boundary = "$$"
        penultim = r",\quad\text{and}\quad "
    else:
        joiner = r",$$ $$ "
        boundary = "$$"
        penultim = r",$$ and $$"

    formatted_elems = [
        f"{j}{prefix}{LaTeX(k)}{suffix}" for j, k in zip(item_labels, list_to_print)
    ]
    formatted_chunks = [
        formatted_elems[j : j + items_per_line]
        for j in range(0, len(formatted_elems), items_per_line)
    ]

    def line_printer(formatted_items, conjunction=False, pun=","):
        if len(formatted_items) == 0:
            return pun
        if len(formatted_items) == 1:
            return boundary + formatted_items[0] + pun + boundary
        if len(formatted_items) == 2:
            if conjunction is False:
                insert = joiner
            else:
                insert = r"\quad\text{and}\quad "
            if bare_latex is True:
                return (
                    boundary
                    + formatted_items[0]
                    + insert
                    + formatted_items[1]
                    + pun
                    + boundary
                )
            if math_mode == "$":
                insert2 = ", " if conjunction is False else " and "
                return (
                    boundary
                    + formatted_items[0]
                    + boundary
                    + insert2
                    + boundary
                    + formatted_items[1]
                    + pun
                    + boundary
                )
            insert2 = joiner if conjunction is False else r" \quad \text{ and }\quad "
            return (
                boundary
                + formatted_items[0]
                + insert2
                + formatted_items[1]
                + pun
                + boundary
            )

        if conjunction is False:
            return boundary + joiner.join(formatted_items) + pun + boundary
        return (
            boundary
            + joiner.join(formatted_items[:-1])
            + penultim
            + formatted_items[-1]
            + pun
            + boundary
        )

    to_print = ""
    for fc in formatted_chunks[:-1]:
        to_print += line_printer(fc) + " "
    conjuction = (
        " and " if len(formatted_chunks) > 1 and len(formatted_chunks[-1]) == 1 else ""
    )
    if verbose:
        _toggle_or_set_verbosity()  # disable temp verbosity
    return (
        to_print
        + conjuction
        + line_printer(formatted_chunks[-1], conjunction=True, pun=punct)
    )


def display_dgcv(*args):
    dgcv_warning(
        "`display_dgcv` has been deprecated as part of a shift toward standardizing naming styles in the dgcv library.",
        dgcvDeprecationWarning,
        stacklevel=2,
        old_kw="display_dgcv",
        new_kw="show",
        sunset="2026",
    )
    return show(*args)


def display_DGCV(*args):
    dgcv_warning(
        "`display_DGCV` has been deprecated as part of a shift toward standardizing naming styles in the dgcv library.",
        dgcvDeprecationWarning,
        stacklevel=2,
        old_kw="display_DGCV",
        new_kw="show",
        sunset="2026",
    )
    return show(*args)


def show(*args, verbose: bool = False, plain_text=False, force_latex_compiling=False):
    """
    Display dgcv objects with IPython if available.

    If IPython rich display is not available/relevant, this function falls back to `print(str(obj))`
    """
    if verbose:
        _toggle_or_set_verbosity(setting=1)  # enable temp verbosity
    if plain_text is True or not is_rich_displaying_available():
        print(*args)
        if verbose:
            _toggle_or_set_verbosity()  # disable temp verbosity
        return
    for j in args:
        _display_dgcv_single(j, FLC=force_latex_compiling)
    if verbose:
        _toggle_or_set_verbosity()  # disable temp verbosity


def _display_dgcv_single(arg: Any, FLC=False) -> None:
    if isinstance(arg, str):
        try:
            ok = _nb_display_latex(arg)
            if ok:
                return
        except Exception:
            pass
        print(str(arg))
        return

    # If already an HTML-capable object, attempt IPython display
    try:
        if hasattr(arg, "_repr_html_") or hasattr(arg, "to_html"):
            from IPython.display import display  # type: ignore # local optional import

            if FLC:
                from ..._utilities._config import latex_in_html

                method = getattr(arg, "to_html", getattr(arg, "_repr_html_"))
                dtext = method() if callable(method) else method
                display(latex_in_html(dtext, extra_support_for_math_in_tables=True))
                return

            display(arg)
            return
    except Exception:
        pass

    if _is_engine_expr(arg) or isinstance(arg, dgcv_class) or check_dgcv_category(arg):
        _complexDisplay(arg)
        return

    try:
        from IPython.display import Math, display  # type: ignore

        display(Math(f"$\\displaystyle {LaTeX(arg)}$"))
        return
    except Exception:
        try:
            from IPython.display import display  # type: ignore

            display(arg)
            return
        except Exception:
            pass

    print(str(arg))


def _complexDisplay(*args):
    def _to_math_payload(x: Any) -> str:
        if getattr(x, "_varSpace_type", None) in ("real", "complex"):
            try:
                return (
                    f"$\\displaystyle {LaTeX(symToHol(x, convert_everything=False))}$"
                )
            except Exception:
                return f"$\\displaystyle {LaTeX(x)}$"
        return f"$\\displaystyle {LaTeX(x)}$"

    converted = [_to_math_payload(j) for j in args]

    try:
        from IPython.display import (  # type: ignore # local optional import
            Math,
            display,
        )

        for expr in converted:
            display(Math(expr))
        return
    except Exception:
        pass

    for j in converted:
        print(str(j))


def dgcv_init_printing(minimal_scope: bool = False, *args, **kwargs):
    """
    Initialize notebook display formatting for dgcv objects.

    With dgcv classes implementing `_repr_mimebundle_` via `dgcv_class`, there is
    nothing to register with IPython. This function optionally keeps SymPy's
    init_printing for users who want it.
    """
    if is_sympy_available() and minimal_scope is False:
        sp = _get_sympy_module()
        sp.init_printing(*args, **kwargs)
    return


def DGCV_init_printing(*args, **kwargs):
    dgcv_warning(
        "`DGCV_init_printing` has been deprecated, as its functionality has been consolidated into the `set_dgcv_settings` function.",
        dgcvDeprecationWarning,
        stacklevel=2,
        old_kw="DGCV_init_printing",
        new_kw="set_dgcv_settings(format_displays=True)",
        sunset="2026",
    )
    return dgcv_init_printing(*args, **kwargs)
