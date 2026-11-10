"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds.pde

module: dgcv.eds.pde._functions

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers

from ..._aux._backends._symbolic_router import function_occurrences, get_free_symbols
from ..._aux._vmf._safeguards import get_dgcv_category
from ..._aux.printing.printing._dgcv_display import LaTeX
from ..._aux.printing.printing._string_processing import _process_label
from .._atoms import zero_form_atom


def _same(a, b):
    if a is b:
        return True
    try:
        return hash(a) == hash(b) and str(a) == str(b)
    except TypeError:
        return False


def _position(value, items):
    for k, item in enumerate(items):
        if _same(value, item):
            return k
    return None


def _symbols_in(value):
    if isinstance(value, numbers.Number):
        return set()
    try:
        return set(get_free_symbols(value))
    except Exception:
        return set()


def _is_symbol(value):
    symbols = _symbols_in(value)
    return len(symbols) == 1 and str(next(iter(symbols))) == str(value)


def function_text(label, args):
    return f"{label}({', '.join(str(arg) for arg in args)})"


def function_tex(label, args):
    return _process_label(label) + r"\left(" + ", ".join(LaTeX(arg) for arg in args) + r"\right)"


def fixed_functions_line(functions, use_latex, text=str):
    shown = []
    for function in functions:
        label, args, _, _ = function_parts(function)
        if use_latex:
            shown.append(f"${function_tex(label, args)}$")
        else:
            shown.append(text(function_text(label, args)))
    return "fixed functions (in system parameters): " + ", ".join(shown)


def partial_text(label, args, derivs):
    indices = []
    for k, (arg, order) in enumerate(zip(args, derivs)):
        indices.extend([str(arg) if _is_symbol(arg) else str(k + 1)] * order)
    if not indices:
        return label
    return f"D_{'_'.join(indices)}({label})"


def listed_names(names):
    names = [str(name) for name in names]
    if len(names) < 2:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def split_locus(locus, independent):
    independent = set(independent)
    base = []
    if independent:
        for condition in locus:
            free = _symbols_in(condition)
            if free and free <= independent:
                base.append(condition)
    return base, [c for c in locus if not any(c is b for b in base)]


def locus_text(locus, independent, equation):
    base, rest = split_locus(locus, independent)
    if base and rest:
        return equation(*rest, 0) + " over the points where " + equation(*base, 0)
    return equation(*locus, 0)


def function_parts(value):
    if isinstance(value, (numbers.Number, zero_form_atom, str)):
        return None
    try:
        occurrences = function_occurrences(value)
    except Exception:
        return None
    for occurrence, label, args, derivs, function in occurrences:
        if _same(occurrence, value):
            return label, tuple(args), tuple(derivs), function
    return None


def function_name(value):
    if callable(value) and get_dgcv_category(value) == "function":
        return str(getattr(value, "label", None) or getattr(value, "__name__", value))
    parts = function_parts(value)
    if parts is None:
        return None
    label, args, derivs, _ = parts
    return partial_text(label, args, derivs) if any(derivs) else function_text(label, args)


def read_functions(exprs, plain, is_unknown):
    functions, partials = {}, {}
    for expr in exprs:
        if isinstance(expr, numbers.Number):
            continue
        for occurrence, label, args, derivs, function in function_occurrences(expr):
            key = (label, tuple(str(arg) for arg in args))
            if key not in functions:
                if any(is_unknown(s) for arg in args for s in _symbols_in(arg)):
                    functions[key] = None
                else:
                    supported = all(_position(arg, plain) is not None for arg in args)
                    functions[key] = (function, label, tuple(args), supported)
            if functions[key] is not None:
                partials.setdefault((key, tuple(derivs)), occurrence)
    listed = {key: functions[key] for key in sorted(functions) if functions[key] is not None}
    return listed, partials
