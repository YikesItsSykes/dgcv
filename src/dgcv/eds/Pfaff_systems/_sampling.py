"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds.Pfaff_systems

module: dgcv.eds.Pfaff_systems._sampling

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import itertools
import math
import numbers

from ..._aux._backends._calculus import diff
from ..._aux._backends._engine import engine_capability, engine_kind
from ..._aux._backends._polynomials import expr_union_primitives
from ..._aux._backends._symbolic_router import _fast_simplify, _scalar_is_zero
from ..._aux._backends._types_and_constants import imag_unit, rational
from ..._aux._utilities._config import get_dgcv_settings_registry
from ...core.arrays import matrix_dgcv
from .. import _bridges
from ..pde._systems import _equation_list
from ._chart import _linear_steps
from ._coordinates import _conjugated_scalar
from ._util import _free, _quotient, _subs, _zero_test


def _singular_factors(divisors):
    if get_dgcv_settings_registry().get("simplify_singularity_ideals_by_default", True):
        return expr_union_primitives(
            divisors, process_rationals=True, fail_quietly=True
        )
    return _distinct(divisors)


def _random_rational(rng, bound):
    denominator = rng.randint(1, 4)
    return rational(rng.randint(-bound * denominator, bound * denominator), denominator)


def _equation_list_or_none(values, is_zero):
    try:
        return _equation_list(values, is_zero)
    except ValueError:
        return None


def _designated_coordinates(equations, coordinates, reality=None):
    order = list(coordinates)[::-1]
    candidates = []
    for equation in equations:
        free = _free(equation)
        linear = [step[2] for step in sorted(_linear_steps(equation, order, ()), key=lambda s: s[0])]
        candidates.append(linear + [x for x in order if x in free and x not in linear])
    match = {}

    def assign(i, seen):
        for x in candidates[i]:
            if x in seen:
                continue
            seen.add(x)
            if x not in match or assign(match[x], seen):
                match[x] = i
                return True
        return False

    for i in range(len(equations)):
        if not assign(i, set()):
            raise ValueError(
                "the defining equations share too few coordinates to sample a point on them"
            )
    chosen = {i: x for x, i in match.items()}
    designated = [chosen[i] for i in range(len(equations))]
    if reality is not None and reality.closed and reality.pairs:
        is_zero = _zero_test()
        for i, equation in enumerate(equations):
            partner = reality.pairs.get(designated[i])
            if partner is None or partner in designated:
                continue
            conj = _conjugated_scalar(equation, reality)
            for j, other in enumerate(equations):
                if j > i and partner in _free(other) and is_zero(other - conj):
                    designated[j] = partner
                    break
    return designated


def _random_values(rng, bound, coordinates, pairs):
    values = {}
    for x in coordinates:
        if x in values:
            continue
        if x in pairs:
            re, im = _random_rational(rng, bound), _random_rational(rng, bound)
            values[x] = re + im * imag_unit()
            values[pairs[x]] = re - im * imag_unit()
        else:
            values[x] = _random_rational(rng, bound)
    return values


def _conjugate_consistent(point, reality):
    is_zero = _zero_test()
    return all(
        partner in point and is_zero(point[partner] - _conjugated_scalar(point[x], reality))
        for x, partner in reality.pairs.items()
        if x in point
    )


def _completed_point(rng, bound, fixed, solution, designated, parameters, pairs):
    values = {y: solution.get(y, y) for y in designated}
    stray = set()
    for value in values.values():
        stray |= {s for s in _free(value) if s not in parameters}
    if stray:
        extra = {s: fixed[s] for s in stray if s in fixed}
        extra |= _random_values(rng, bound, sorted((s for s in stray if s not in fixed), key=str), pairs)
        values = {y: _subs(value, extra) for y, value in values.items()}
        if any(s not in parameters for value in values.values() for s in _free(value)):
            return None
    return fixed | values


def _polynomial_in(expr, symbols, cap=32):
    is_zero = _zero_test()
    for y in symbols:
        current = expr
        for _ in range(cap + 1):
            if is_zero(current):
                break
            current = diff(current, y)
        else:
            return False
    return True


def _unavailable_message(exc):
    secondary = engine_capability("secondary_kind")
    if engine_kind() == "builtin" and secondary is not None and secondary() is None:
        return (
            "sampling a point of the defining equations needs a nonlinear solve, which the builtin engine "
            "hands to a secondary engine; none is available"
        )
    return f"sampling a point of the defining equations needs a solve the engine could not do: {exc}"


def _sampling_tail(verdict, detail):
    if verdict == "inconsistent":
        return "; the equations appear inconsistent"
    if verdict == "non_real":
        return "; the roots found were not real for the coordinates assumed real (" + ", ".join(
            str(y) for y in detail
        ) + ")"
    return "; sampling was inconclusive"


def _polynomial_coefficients(expr, symbols, is_zero):
    present = [t for t in symbols if t in _free(expr)]
    if not present:
        return [expr]
    t = present[0]
    out, current, k = [], expr, 0
    while not is_zero(current):
        out.append(_fast_simplify(_subs(current, {t: 0}) * rational(1, math.factorial(k))))
        current = diff(current, t)
        k += 1
        if k > 64:
            break
    return [
        coefficient
        for value in out
        if not is_zero(value)
        for coefficient in _polynomial_coefficients(value, present[1:], is_zero)
    ]


def _locus_key(locus):
    return frozenset(str(condition) for condition in locus)


def _pruned_loci(loci):
    keyed = {}
    for locus in loci:
        keyed.setdefault(_locus_key(locus), locus)
    keys = list(keyed)
    kept = [
        key for key in keys if not any(other < key for other in keys)
    ]
    return [list(keyed[key]) for key in sorted(kept, key=lambda key: (len(key), sorted(key)))]


_minor_limit = 300


def _rank_drop_loci(rows, contact_count, equation_count):
    contact = [list(row) for row in rows[:contact_count]]
    reduced = [list(row) for row in rows[contact_count : contact_count + equation_count]]
    if not reduced or not contact:
        return []
    is_zero = _zero_test()
    width = len(rows[0])
    pivots = []
    for i, row in enumerate(contact):
        pivot = next(
            (
                j
                for j in range(width)
                if _bridges._is_number(row[j])
                and not _scalar_is_zero(row[j])
                and all(_scalar_is_zero(other[j]) for k, other in enumerate(contact) if k != i)
            ),
            None,
        )
        if pivot is None:
            return []
        pivots.append(pivot)
    for i, row in enumerate(contact):
        j = pivots[i]
        for target in reduced:
            if not _scalar_is_zero(target[j]):
                scale = _quotient(target[j], row[j])
                for col in range(width):
                    target[col] = target[col] - scale * row[col]
    columns = [j for j in range(width) if j not in pivots]
    size = len(reduced)
    if len(columns) < size:
        return []
    choices = list(itertools.combinations(columns, size))
    if len(choices) > _minor_limit:
        return []
    conditions = []
    for choice in choices:
        if size == 1:
            minor = reduced[0][choice[0]]
        else:
            minor = matrix_dgcv([[row[j] for j in choice] for row in reduced]).det()
        minor = _fast_simplify(minor)
        if is_zero(minor):
            continue
        if _bridges._is_number(minor):
            return []
        for factor in _singular_factors([minor]) or [minor]:
            if not any(str(factor) == str(other) for other in conditions):
                conditions.append(factor)
    return [tuple(conditions)] if conditions else []


def _loci_from_divisors(divisors, flag_symbols):
    loci = []
    is_zero = _zero_test()
    for divisor in _singular_factors([d for d in divisors if not isinstance(d, numbers.Number)]):
        if _bridges._is_number(divisor):
            continue
        conditions = []
        empty = False
        try:
            coefficients = _polynomial_coefficients(divisor, flag_symbols, is_zero)
        except ZeroDivisionError:
            continue
        for coefficient in coefficients:
            if _bridges._is_number(coefficient):
                if not is_zero(coefficient):
                    empty = True
                    break
                continue
            conditions.append(coefficient)
        if empty or not conditions:
            continue
        factor_lists = [_singular_factors([condition]) or [condition] for condition in conditions]
        for combination in itertools.product(*factor_lists):
            locus = []
            for factor in combination:
                if not any(str(factor) == str(other) for other in locus):
                    locus.append(factor)
            loci.append(tuple(locus))
    return _pruned_loci(loci)


def _distinct(values):
    is_zero = _zero_test()
    out = []
    for value in values:
        if not any(is_zero(value - seen) for seen in out):
            out.append(value)
    return out
