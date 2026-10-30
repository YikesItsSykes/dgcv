"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv._aux._backends

module: dgcv._aux._backends._groebner


Description: Groebner bases of polynomial systems built from dgcv objects, dispatched to the
active symbolic engine or, under the builtin engine, to the secondary engine.

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from ._engine import engine_capability, engine_kind, engine_module
from ._symbolic_router import (
    _builtin_unsupported,
    _scalar_is_zero,
    as_numer_denom,
    get_free_symbols,
)

__all__ = ["groebner"]

_ORDERS = {
    "lex": ("lex", "lex"),
    "grlex": ("grlex", "deglex"),
    "deglex": ("grlex", "deglex"),
    "grevlex": ("grevlex", "degrevlex"),
    "degrevlex": ("grevlex", "degrevlex"),
}

_ADJOINED_UNIT = "dgcv_imag_unit"


def _order_names(order):
    try:
        return _ORDERS[str(order).lower()]
    except KeyError:
        raise ValueError(
            f"unknown monomial order {order!r}; use 'lex', 'grlex' or 'grevlex'"
        ) from None


def _ordered(vars_):
    try:
        from .._vmf.vmf import order_coordinates

        return list(order_coordinates(vars_))
    except Exception:
        return sorted(vars_, key=str)


def groebner(eqns, vars_to_solve=None, *, order="grevlex", return_divisors=False):
    from ...core.solvers._normalization import (
        _equations_preprocessing,
        normalize_equations_and_vars,
    )

    sympy_order, sage_order = _order_names(order)
    if not isinstance(eqns, (list, tuple)):
        eqns = [eqns]
    eqns, vars_ = normalize_equations_and_vars(list(eqns), vars_to_solve)
    if vars_to_solve is None:
        vars_ = _ordered(vars_)
    lowered, gens, extra, bridge = _equations_preprocessing(eqns, vars_)
    if extra:
        raise ValueError(
            "`groebner_dgcv` solves for single variables; one of the given unknowns is a compound expression"
        )

    polys = []
    divisors = []
    seen = set()
    for e in lowered:
        num, den = as_numer_denom(e)
        if _scalar_is_zero(num):
            continue
        polys.append(num)
        if get_free_symbols(den) and str(den) not in seen:
            seen.add(str(den))
            divisors.append(den)

    if polys:
        kind = engine_kind()
        if kind == "builtin":
            fn = engine_capability("groebner")
            if fn is None:
                raise _builtin_unsupported("groebner")
            basis = fn(polys, gens, sympy_order)
        else:
            basis = _compute(kind, engine_module(), polys, gens, sympy_order)
    else:
        basis = []

    basis = [bridge.lift_expr(g) for g in basis]
    if return_divisors:
        return basis, [bridge.lift_expr(d) for d in divisors]
    return basis


def _compute(kind, mod, eqns, gens, order):
    sympy_order, sage_order = _order_names(order)
    if kind == "sympy":
        try:
            return list(mod.groebner(list(eqns), *gens, order=sympy_order))
        except Exception as exc:
            raise RuntimeError(
                f"the Groebner basis could not be computed: {exc}"
            ) from exc
    if kind == "sage":
        return _sage_groebner(mod, eqns, gens, sage_order)
    raise _builtin_unsupported("groebner")


def _sage_groebner(mod, eqns, gens, order):
    SR = mod.SR
    exprs = [SR(e) for e in eqns]
    names = [str(SR(g)) for g in gens]
    params = sorted({str(v) for e in exprs for v in e.variables()} - set(names))
    has_unit = any(e.has(mod.I) for e in exprs)

    base = mod.PolynomialRing(mod.QQ, params).fraction_field() if params else mod.QQ
    adjoin = has_unit and bool(params)
    if has_unit and not params:
        base = mod.QuadraticField(-1, "i")

    if adjoin:
        unit = SR.var(_ADJOINED_UNIT)
        exprs = [e.subs({mod.I: unit}) for e in exprs]
        ring = mod.PolynomialRing(
            base,
            len(names) + 1,
            names + [_ADJOINED_UNIT],
            order=mod.TermOrder(f"{order}({len(names)}),lex(1)"),
        )
    else:
        ring = mod.PolynomialRing(base, len(names), names, order=order)

    try:
        polys = [ring(e) for e in exprs]
    except (TypeError, ValueError, ArithmeticError, NotImplementedError) as exc:
        raise ValueError(
            "the Groebner basis could not be computed under Sage: coefficients must be rational numbers, `I` and parameters, and the unknowns must not appear in denominators"
        ) from exc

    relation = None
    if adjoin:
        relation = ring.gen(len(names)) ** 2 + 1
        polys.append(relation)

    try:
        basis = ring.ideal(polys).groebner_basis()
    except Exception as exc:
        raise RuntimeError(f"the Groebner basis could not be computed: {exc}") from exc

    out = []
    for g in basis:
        if relation is not None and g == relation:
            continue
        g = SR(g)
        if adjoin:
            g = g.subs({unit: mod.I})
        out.append(g)
    return out
