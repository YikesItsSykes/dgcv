"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds.Pfaff_systems

module: dgcv.eds.Pfaff_systems._util

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers

from ..._aux._backends._engine import engine_capability, engine_kind, engine_module
from ..._aux._backends._symbolic_router import (
    _fast_simplify,
    _scalar_is_zero,
    get_free_symbols,
    is_zero_knowing_zero_is_expected,
)
from ..._aux._backends._symbolic_router import (
    subs as _routed_subs,
)
from ..._aux._backends._types_and_constants import rational
from ..._aux._vmf._safeguards import query_dgcv_categories
from ...core.arrays._matrix_linalg import _certainly_nonzero, _pivot_is_zero
from .._atoms import zero_form_atom
from .._forms import (
    abstract_differential_form,
    abstract_differential_form_atom,
    abstract_differential_form_monomial,
)
from .._vectors import _constant_symbol
from .._zero_forms import zero_form_class


def _as_list(items):
    if items is None:
        return []
    if isinstance(items, (list, tuple)):
        return list(items)
    return [items]


def _free(value):
    if isinstance(value, numbers.Number):
        return set()
    return set(get_free_symbols(value))


def _subs(value, rules):
    if isinstance(value, numbers.Number) or not rules:
        return value
    try:
        if value in rules:
            return rules[value]
    except TypeError:
        pass
    if isinstance(value, (zero_form_class, zero_form_atom)):
        return value.subs(rules)
    if any(isinstance(v, (zero_form_atom, zero_form_class)) for v in rules.values()):
        return zero_form_class(value).subs(rules)
    return _routed_subs(value, rules)


def _translated(condition, rules):
    try:
        return _subs(condition, rules)
    except (ValueError, TypeError, NotImplementedError):
        return condition


def _dot(row, column):
    total = 0
    for x, y in zip(row, column):
        if _scalar_is_zero(x) or _scalar_is_zero(y):
            continue
        total = total + x * y
    return total


def _unit(f, k):
    return [1 if idx == k else 0 for idx in range(f)]


def _scale(vector, scalar):
    return [0 if _scalar_is_zero(entry) else scalar * entry for entry in vector]


def _quotient(a, b):
    if isinstance(a, numbers.Integral) and isinstance(b, numbers.Integral):
        return rational(a, b)
    return a / b


def _add(u, v):
    return [a + b for a, b in zip(u, v)]


def _symbols(prefix, count):
    return [_constant_symbol(f"{prefix}{idx + 1}") for idx in range(count)]


def _zero_by_simplifying(value):
    if isinstance(value, numbers.Number):
        return value == 0
    if _pivot_is_zero(value):
        return True
    if _certainly_nonzero(value):
        return False
    if is_zero_knowing_zero_is_expected(value):
        return True
    exponentialize = getattr(value, "exponentialize", None)
    return exponentialize is not None and is_zero_knowing_zero_is_expected(
        exponentialize()
    )


def _zero_test():
    by_exp = engine_capability("zero_by_exp")
    if by_exp is None:
        return _zero_by_simplifying

    def test(value):
        if isinstance(value, numbers.Number):
            return value == 0
        if _pivot_is_zero(value):
            return True
        try:
            return by_exp(value)
        except (ZeroDivisionError, NotImplementedError):
            return False

    return test


def _chain_simplifier():
    if engine_kind() != "sympy":
        return _fast_simplify
    sp = engine_module()

    def simplify_entry(value):
        try:
            masks = {}
            masked = value.replace(
                lambda e: isinstance(e, sp.Function) and e.is_number,
                lambda e: masks.setdefault(e, sp.Dummy()),
            )
            out = sp.cancel(masked)
            if masks:
                out = out.xreplace({d: e for e, d in masks.items()})
            return out
        except Exception:
            return _fast_simplify(value)

    return simplify_entry


def _rank(matrix):
    return matrix.rank(
        simplify_steps=True,
        zero_test=_zero_test(),
        simplifier=_chain_simplifier(),
    )


def _form_is_zero(form):
    if isinstance(form, numbers.Number):
        return form == 0
    return bool(getattr(form, "is_zero", False))


def _is_abstract(form):
    return isinstance(
        form,
        (
            abstract_differential_form_atom,
            abstract_differential_form_monomial,
            abstract_differential_form,
        ),
    )


def _is_vector(obj):
    return bool(query_dgcv_categories(obj, "eds_vector"))


def _tidy(vector):
    return [
        entry if isinstance(entry, numbers.Number) else _fast_simplify(entry)
        for entry in vector
    ]
