"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.core

module: dgcv.core.polynomials


Description: Polynomial construction utilities for dgcv.

This module provides creator functions that build (multi)graded polynomials in a given
variable space, using dgcv's variable management framework to generate coefficient
families. The returned objects are dgcv.polynomial_dgcv instances.

key functions
---------
- createPolynomial
- createBigradPolynomial
- createMultigradedPolynomial
- monomialWeight
- getWeightedTerms


---
Author (of this sub-package): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from __future__ import annotations

import functools
import numbers
import operator
from typing import Any, List, Literal, Optional, Sequence, Union

from ..._aux._backends._polynomials import (
    make_poly,
    poly_coeffs,
    poly_gens,
    poly_monoms,
)
from ..._aux._backends._symbolic_router import expand, simplify, subs
from ..._aux._backends._types_and_constants import symbol, zero
from ..._aux._utilities._config import dgcv_warning
from ..._aux._vmf.vmf import clearVar
from ..combinatorics.combinatorics import chooseOp
from ..dgcv_core import polynomial_dgcv, variableProcedure

ScalarLike = Union[int, float, complex]
WeightVec = Sequence[int]
WeightSystems = Sequence[Sequence[int]]


__all__ = [
    "createPolynomial",
    "createBigradPolynomial",
    "createMultigradedPolynomial",
    "monomialWeight",
    "getWeightedTerms",
]


# -----------------------------------------------------------------------------
# body
# -----------------------------------------------------------------------------
def _monomial_from_exponents(vars_: Sequence[Any], exps: Sequence[int]) -> Any:
    return functools.reduce(
        operator.mul, (v ** int(e) for v, e in zip(vars_, exps) if int(e) != 0), 1
    )


def createPolynomial(
    coeff_label: str,
    degree: int,
    variables: Sequence[Any],
    homogeneous: bool = False,
    weightedHomogeneity: Optional[Sequence[int]] = None,
    degreeCap: int = 0,
    initialIndex=1,
    register_coeffs_in_vmf=False,
    register_coeffs_with_VF=False,
    _tempVar: Optional[bool] = None,
    return_components: bool = False,
    assumptions: dict = None,
    remove_guardrails: bool = False,
    report_vmf_updates: bool = False,
    **kwargs,
) -> Union[polynomial_dgcv, List[Any]]:

    if kwargs.get("assumeReal", False) is True:  # deprecated kwarg support
        if not isinstance(assumptions, dict):
            assumptions = {"real": True}
        else:
            assumptions["real"] = True
    if register_coeffs_with_VF is True:
        register_coeffs_in_vmf = True

    vars_ = tuple(variables)
    n = len(vars_)
    deg = int(degree)

    if homogeneous:
        exponent_tuples = list(chooseOp(range(deg + 1), n, restrictHomogeneity=deg))

    elif isinstance(weightedHomogeneity, (list, tuple)):
        w = list(weightedHomogeneity)
        if len(w) != n:
            raise ValueError("weightedHomogeneity must match len(variables).")

        if 0 in w:
            zero_idx = [j for j, ww in enumerate(w) if ww == 0]
            nonzero_idx = [j for j, ww in enumerate(w) if ww != 0]

            vars_reordered = tuple(
                [vars_[j] for j in nonzero_idx] + [vars_[j] for j in zero_idx]
            )
            w_reordered = [w[j] for j in nonzero_idx] + [w[j] for j in zero_idx]

            nonzero_exps = [
                e
                for e in chooseOp(range(deg + 1), len(nonzero_idx))
                if sum(a * b for a, b in zip(e, w_reordered[: len(nonzero_idx)])) == deg
            ]
            zero_exps = list(chooseOp(range(int(degreeCap) + 1), len(zero_idx)))
            exponent_tuples = [e0 + e1 for e0 in nonzero_exps for e1 in zero_exps]
            vars_ = vars_reordered
        else:
            exponent_tuples = [
                e
                for e in chooseOp(range(deg + 1), n)
                if sum(a * b for a, b in zip(e, w)) == deg
            ]

    else:
        exponent_tuples = [e for e in chooseOp(range(deg + 1), n) if sum(e) <= deg]

    if register_coeffs_in_vmf:
        clearVar(coeff_label, report=report_vmf_updates)
        created = variableProcedure(
            coeff_label,
            len(exponent_tuples),
            _tempVar=_tempVar,
            withVF=register_coeffs_with_VF,
            initialIndex=initialIndex,
            assumptions=assumptions,
            remove_guardrails=remove_guardrails,
            return_created_object=True,
        )
        coeffs = created[0] if created else ()
    else:
        coeffs = [
            symbol(f"{coeff_label}{idx}", assumptions=assumptions)
            for idx in range(initialIndex, initialIndex + len(exponent_tuples))
        ]

    monomials: List[Any] = []
    for idx, exps in enumerate(exponent_tuples):
        monomials.append(coeffs[idx] * _monomial_from_exponents(vars_, exps))

    expr = sum(monomials) if monomials else zero()
    if return_components:
        return polynomial_dgcv(expr, varSpace=vars_), monomials, coeffs

    return polynomial_dgcv(expr, varSpace=vars_)


def createBigradPolynomial(
    coeff_label: str,
    degrees: Sequence[int],
    variables: Sequence[Any],
    weights_1: Sequence[int],
    weights_2: Sequence[int],
    register_coeffs_in_vmf: bool = False,
    initialIndex: int = 1,
    _tempVar: Optional[bool] = None,
    return_components: bool = False,
    remove_guardrails: bool = False,
    report_vmf_updates: bool = False,
    assumptions: dict = None,
    **kwargs,
) -> Union[polynomial_dgcv, List[Any]]:
    if kwargs.get("assumeReal", False) is True:  # deprecated kwarg support
        if not isinstance(assumptions, dict):
            assumptions = {"real": True}
        else:
            assumptions["real"] = True

    degs = tuple(degrees)
    if len(degs) != 2:
        raise ValueError("degrees must have length 2.")
    d0, d1 = int(degs[0]), int(degs[1])

    vars_ = tuple(variables)
    w0 = tuple(weights_1)
    w1 = tuple(weights_2)
    if not (len(vars_) == len(w0) == len(w1)):
        raise ValueError("variables, weights_1, weights_2 must have the same length.")

    max_deg = max(abs(d0), abs(d1))
    candidates = chooseOp(range(max_deg + 1), len(vars_))
    exponent_tuples = [
        exps
        for exps in candidates
        if sum(e * ww for e, ww in zip(exps, w0)) == d0
        and sum(e * ww for e, ww in zip(exps, w1)) == d1
    ]
    if register_coeffs_in_vmf:
        clearVar(coeff_label, report=report_vmf_updates)
        created = variableProcedure(
            coeff_label,
            len(exponent_tuples),
            assumptions=assumptions,
            _tempVar=_tempVar,
            remove_guardrails=remove_guardrails,
            return_created_object=True,
        )
        coeffs = created[0] if created else ()
    else:
        coeffs = [
            symbol(f"{coeff_label}{idx}", assumptions=assumptions)
            for idx in range(initialIndex, initialIndex + len(exponent_tuples))
        ]

    monomials: List[Any] = []
    for idx, exps in enumerate(exponent_tuples):
        monomials.append(coeffs[idx] * _monomial_from_exponents(vars_, exps))

    expr = sum(monomials) if monomials else zero()
    if return_components:
        return polynomial_dgcv(expr, varSpace=vars_), monomials, coeffs

    return polynomial_dgcv(expr, varSpace=vars_)


def _weighted_exponent_tuples(weights, degree):
    n = len(weights)
    if degree < 0:
        return []
    if n == 0:
        return [()] if degree == 0 else []
    out = []
    exps = [0] * n

    def rec(j, remaining):
        w = weights[j]
        if j == n - 1:
            if remaining % w == 0:
                exps[j] = remaining // w
                out.append(tuple(exps))
            return
        for e in range(remaining // w + 1):
            exps[j] = e
            rec(j + 1, remaining - e * w)

    rec(0, degree)
    return out


def createMultigradedPolynomial(
    coeff_label: str,
    degrees: Union[int, Sequence[int]],
    vars: Sequence[Any],
    weight_systems: Union[Sequence[int], Sequence[Sequence[int]]],
    _tempVar: Optional[bool] = None,
    return_components: bool = False,
    remove_guardrails: bool = False,
    assumptions: bool = None,
    report_vmf_updates: bool = False,
    degreeCap: int = 10,
    register_coeffs_in_vmf: bool = False,
    initialIndex: int = 1,
    **kwargs,
) -> Union[polynomial_dgcv, List[Any]]:
    if kwargs.get("assumeReal", False) is True:  # deprecated kwarg support
        if not isinstance(assumptions, dict):
            assumptions = {"real": True}
        else:
            assumptions["real"] = True
    degs = tuple(degrees) if isinstance(degrees, (list, tuple)) else (int(degrees),)
    ws = weight_systems
    if not isinstance(ws, (list, tuple)) or (
        ws and not isinstance(ws[0], (list, tuple))
    ):
        ws = [ws]
    ws = [tuple(w) for w in ws]

    vars_ = tuple(vars)

    if len(degs) != len(ws):
        raise KeyError("degrees and weight_systems must have the same length.")
    if any(len(w) != len(vars_) for w in ws):
        raise ValueError("Each weight vector must match len(vars).")

    max_deg = 0
    for d, w in zip(degs, ws):
        if all(j > 0 for j in w) or all(j < 0 for j in w):
            dd = int(abs(d))
            if max_deg == 0 or dd < max_deg:
                max_deg = dd

    if not isinstance(max_deg, numbers.Integral):
        max_deg = int(abs(max_deg))
    if max_deg == 0:
        max_deg = max(int(abs(d)) for d in degs)
        max_deg = max(max_deg, int(degreeCap))

    exponent_tuples = None
    for k0, w0 in enumerate(ws):
        sign = 1 if all(j > 0 for j in w0) else -1 if all(j < 0 for j in w0) else 0
        if sign == 0:
            continue
        d0 = sign * int(degs[k0])
        exponent_tuples = [
            exps
            for exps in _weighted_exponent_tuples([sign * j for j in w0], d0)
            if all(
                sum(e * wj for e, wj in zip(exps, ws[k])) == int(degs[k])
                for k in range(len(degs))
                if k != k0
            )
        ]
        break
    if exponent_tuples is None:
        candidates = chooseOp(range(max_deg + 1), len(vars_))
        exponent_tuples = [
            exps
            for exps in candidates
            if all(
                sum(e * wj for e, wj in zip(exps, ws[k])) == int(degs[k])
                for k in range(len(degs))
            )
        ]

    if register_coeffs_in_vmf:
        clearVar(coeff_label, report=report_vmf_updates)
        created = variableProcedure(
            coeff_label,
            len(exponent_tuples),
            _tempVar=_tempVar,
            assumptions=assumptions,
            remove_guardrails=remove_guardrails,
            return_created_object=True,
        )
        coeffs = created[0] if created else ()
    else:
        coeffs = [
            symbol(f"{coeff_label}{idx}", assumptions=assumptions)
            for idx in range(initialIndex, initialIndex + len(exponent_tuples))
        ]

    monomials: List[Any] = []
    for idx, exps in enumerate(exponent_tuples):
        monomials.append(coeffs[idx] * _monomial_from_exponents(vars_, exps))
    expr = sum(monomials) if monomials else zero()
    if return_components:
        return polynomial_dgcv(expr, varSpace=vars_), monomials, coeffs
    return polynomial_dgcv(expr, varSpace=vars_)


def monomialWeight(
    monomial: Any, variables: Sequence[Any], weights: Sequence[int]
) -> Any:
    """
    Return the weighted degree of a monomial in `variables` with weight vector `weights`.
    """
    if len(variables) != len(weights):
        raise ValueError("variables and weights must have the same length.")

    expr = simplify(expand(monomial))

    try:
        P = make_poly(expr, variables)
        terms = poly_monoms(P)
        if len(terms) > 1:
            dgcv_warning("Input appears to have multiple terms; proceeding anyway.")
    except Exception:
        dgcv_warning(
            "Input could not be interpreted as a polynomial; proceeding anyway."
        )

    coeff = simplify(subs(expr, {v: 1 for v in variables}))
    if coeff == 0:
        return zero()

    try:
        P = make_poly(expr, variables)
        monoms = poly_monoms(P)
        if not monoms:
            return zero()
        m = monoms[0]
        return sum(int(e) * int(w) for e, w in zip(m, weights))
    except Exception:
        return sum(int(w) for w in weights) * 0


def getWeightedTerms(
    poly: polynomial_dgcv,
    *,
    target_degrees: Sequence[int] = None,
    weight_systems: Sequence[dict] = None,
    formatting: Literal["unformatted", "complex", "real"] = None,
) -> polynomial_dgcv:
    """
    Filter terms of `polynomial_dgcv` instance whose monomials satisfy weighted degree constraints.

    Parameters:
    -----------
    target_degrees: list of weights (numeric)
        list of scalars representing a target multi-weight
    weight_systems: list of dict
        list of dictionaries mapping variables in the polynomial to their weights. Unlisted variables are assumed to have weight 0. must have same length as list of target_degrees
    """
    if target_degrees is None:
        multipl = len(weight_systems) if weight_systems else 1
        target_degrees = [0 for _ in range(multipl)]
    if weight_systems is None:
        weight_systems = [dict() for _ in range(len(target_degrees))]

    if formatting not in {"unformatted", "complex", "real"}:
        formatting = "unformatted"

    if formatting == "unformatted":
        P = poly.poly_obj_unformatted
    elif formatting == "complex":
        P = poly.poly_obj_complex
    elif formatting == "real":
        P = poly.poly_obj_real
    else:
        raise ValueError(
            "`formatting` must be one of: 'unformatted', 'complex', 'real'"
        )

    gens = poly_gens(P)
    if all(
        isinstance(w, (list, tuple)) for w in weight_systems
    ):  # supporting old syntax
        try:
            ws = []
            for w in weight_systems:
                assert len(w) == len(gens)
                ws.append({gens[idx]: weight for idx, weight in enumerate(w)})
            weight_systems = ws

        except Exception:
            raise TypeError(
                "Providing the `weight_systems` parameter as a list of lists is no longer supported in `dgcv`. The parameter must be a list of dicts mapping variables to weights."
            )
    if len(target_degrees) != len(weight_systems):
        raise ValueError("target_degrees and weight_systems must have the same length.")
    monoms = poly_monoms(P)
    coeffs = poly_coeffs(P)

    admissible = list(range(len(monoms)))
    for deg, w in zip(target_degrees, weight_systems):
        weights_vec = [int(w.get(g, 0)) for g in gens]
        admissible = [
            j
            for j in admissible
            if sum(int(monoms[j][k]) * weight for k, weight in enumerate(weights_vec))
            == int(deg)
        ]

    terms: List[Any] = []
    for j in admissible:
        term = coeffs[j]
        for g, e in zip(gens, monoms[j]):
            ee = int(e)
            if ee:
                term = term * (g**ee)
        terms.append(term)

    expr = sum(terms) if terms else zero()
    return polynomial_dgcv(expr, varSpace=poly.varSpace, parameters=poly.parameters)


def createRational(
    coeff_label: str,
    degree: int,
    variables: Sequence[Any],
    homogeneous: bool = False,
    weightedHomogeneity: Optional[Sequence[int]] = None,
    degreeCap: int = 0,
    initialIndex=1,
    register_coeffs_in_vmf=False,
    _tempVar: Optional[bool] = None,
    return_components: bool = False,
    assumptions: bool | None = None,
    remove_guardrails: bool = False,
    report_vmf_updates: bool = False,
) -> Union[polynomial_dgcv, List[Any]]:
    numerator, n_monoms, n_coeffs = createPolynomial(
        coeff_label,
        degree,
        variables,
        homogeneous=homogeneous,
        weightedHomogeneity=weightedHomogeneity,
        degreeCap=degreeCap,
        initialIndex=initialIndex,
        register_coeffs_in_vmf=False,
        return_components=True,
        assumptions=assumptions,
    )
    denominator, d_monoms, d_coeffs = createPolynomial(
        coeff_label,
        degree,
        variables,
        homogeneous=homogeneous,
        weightedHomogeneity=weightedHomogeneity,
        degreeCap=degreeCap,
        initialIndex=initialIndex + len(n_coeffs),
        register_coeffs_in_vmf=False,
        return_components=True,
        assumptions=assumptions,
    )
    if register_coeffs_in_vmf is True:
        variableProcedure(
            coeff_label,
            len(n_coeffs) + len(d_coeffs),
            initialIndex=initialIndex,
            assumptions=assumptions,
            remove_guardrails=remove_guardrails,
        )
    if return_components:
        return numerator / denominator, [n_monoms, d_monoms], n_coeffs + d_coeffs
    return numerator / denominator
