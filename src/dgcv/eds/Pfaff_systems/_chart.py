"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds.Pfaff_systems

module: dgcv.eds.Pfaff_systems._chart

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers

from ..._aux._backends._calculus import diff
from ..._aux._backends._symbolic_router import _fast_simplify, _scalar_is_zero
from ..._aux._backends._types_and_constants import imag_unit, rational
from ..._aux._utilities._config import dgcv_warning
from ...core.solvers import solve_dgcv
from .. import _bridges
from ..pde._systems import _equation_list
from ._coordinates import _conjugated_scalar, _coordinate_reality
from ._util import _form_is_zero, _free, _subs, _zero_test


def _is_complex_formula(value, is_zero):
    if isinstance(value, numbers.Number):
        return False
    has = getattr(value, "has", None)
    if has is not None:
        return bool(has(imag_unit()))
    return not is_zero(_subs(value, {imag_unit(): 0}) - value)


def _linear_steps(equation, order, last):
    free = _free(equation)
    for position, y in enumerate(order):
        if y not in free:
            continue
        coefficient = diff(equation, y)
        if y in _free(coefficient) or _scalar_is_zero(coefficient):
            continue
        yield (
            (y in last, not _bridges._is_number(coefficient), position),
            equation,
            y,
            coefficient,
        )


def _linear_root(equation, y, coefficient):
    rest = _subs(equation, {y: 0})
    if _scalar_is_zero(rest):
        return 0
    if isinstance(coefficient, numbers.Integral):
        return rest * rational(-1, coefficient)
    return _fast_simplify(-rest / coefficient)


def _single_root(equation, order, reality, is_zero):
    free = _free(equation)
    y = next((coordinate for coordinate in order if coordinate in free), None)
    if y is None:
        raise ValueError(f"could not solve {equation} = 0: it contains no coordinate")
    try:
        roots = [
            root[y]
            for root in solve_dgcv(equation, [y])
            if y in root and y not in _free(root[y])
        ]
    except NotImplementedError:
        roots = []
    if not roots:
        raise ValueError(
            f"could not solve {equation} = 0 for {y}: enter the branch you want as an equation"
        )
    if reality.is_real(y):
        real = [root for root in roots if not _is_complex_formula(root, is_zero)]
    else:
        real = roots
    if not real:
        raise ValueError(f"no real root of {equation} = 0 for {y} was found")
    if len(real) > 1:
        listed = ", ".join(str(root) for root in real)
        raise ValueError(
            f"{equation} = 0 has several roots for {y}: {listed}. Enter the branch you want as an equation"
        )
    return y, real[0]


def _linear_step(step):
    _, equation, y, coefficient = step
    divisor = None if _bridges._is_number(coefficient) else coefficient
    return equation, y, _linear_root(equation, y, coefficient), divisor


def _next_step(pending, order, last, reality, is_zero):
    steps = [
        step for equation in pending for step in _linear_steps(equation, order, last)
    ]
    best = min(steps, key=lambda step: step[0], default=None)
    if best is not None and best[2] not in last:
        return _linear_step(best)
    inner = [y for y in order if y not in last]
    for equation in pending:
        if _free(equation).intersection(inner):
            return (equation, *_single_root(equation, inner, reality, is_zero), None)
    if best is not None:
        return _linear_step(best)
    return (pending[0], *_single_root(pending[0], order, reality, is_zero), None)


def _solve_for_coordinates(equations, coordinates, last=(), reality=None):
    if reality is None:
        reality = _coordinate_reality(coordinates)
    order = list(coordinates)[::-1]
    last = set(last)
    solved, divisors = {}, []
    pending = list(equations)
    is_zero = _zero_test()
    while pending:
        equation, y, value, divisor = _next_step(pending, order, last, reality, is_zero)
        if _is_complex_formula(value, is_zero) and reality.is_real(y):
            raise ValueError(f"no real root of {equation} = 0 for {y} was found")
        if divisor is not None:
            divisors.append(divisor)
        rules = {y: value}
        solved = {key: _subs(entry, rules) for key, entry in solved.items()}
        solved[y] = value
        pushed = []
        if y in reality.pairs:
            partner = reality.pairs[y]
            conj = _subs(_conjugated_scalar(value, reality), solved)
            if partner in solved:
                if not is_zero(solved[partner] - conj):
                    raise ValueError(
                        f"the equations fix {partner} and the conjugate of {y} differently"
                    )
            elif partner not in _free(conj):
                rules[partner] = conj
                solved = {key: _subs(entry, {partner: conj}) for key, entry in solved.items()}
                solved[partner] = conj
            else:
                pushed.append(partner - conj)
        pending = _equation_list(
            [_subs(other, rules) for other in pending if other is not equation] + pushed, is_zero
        )
    return solved, divisors


def _remaining_coordinates(coordinates, solved):
    chart = [x for x in coordinates if x not in solved]
    if not chart:
        raise ValueError("the equations fix every coordinate, so no manifold is left")
    return chart


def _warn_on_divisors(divisors):
    if divisors:
        where = ", ".join(f"{divisor} = 0" for divisor in divisors)
        dgcv_warning(
            f"the equations were solved by dividing, so the result may fail at {where}"
        )


def _pull_form(form, coordinates, solved, differentials, objects, is_zero):
    total = 0
    for x in coordinates:
        value = _bridges._scalar(form(objects.field(x)))
        if is_zero(value):
            continue
        value = _subs(value, solved)
        if is_zero(value):
            continue
        dx = differentials[x]
        if _form_is_zero(dx):
            continue
        total = total + value * dx
    return total


def _restrict_forms(equations, coordinates, objects, reality, *form_lists, last=()):
    solved, divisors = _solve_for_coordinates(equations, coordinates, last=last, reality=reality)
    chart = _remaining_coordinates(coordinates, solved)
    restricted = [_restricted(forms, coordinates, solved, chart, objects) for forms in form_lists]
    return solved, divisors, chart, restricted


def _restricted(forms, coordinates, solved, chart, objects):
    differentials = {x: objects.form(x) for x in chart}
    for y, value in solved.items():
        free = _free(value)
        differentials[y] = sum(
            (diff(value, z) * differentials[z] for z in chart if z in free), 0
        )
    is_zero = _zero_test()
    return [
        0
        if _form_is_zero(form)
        else _pull_form(form, coordinates, solved, differentials, objects, is_zero)
        for form in forms
    ]
