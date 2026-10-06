# backends/_numeric_router.py

from __future__ import annotations

from math import gcd
from random import getrandbits, randint
from typing import Any, Optional

from .._vmf._safeguards import create_key
from ._symbolic_router import _scalar_is_zero, subs
from ._types_and_constants import rational, symbol


def zeroish(
    x: Any,
    *,
    abs_tol: Optional[float] = None,
    rel_tol: Optional[float] = None,
    force_exact: bool = False,
    **kwargs,
) -> bool:
    dgcv_hook = getattr(x, "__dgcv_zero_obstr__", None)
    if dgcv_hook:
        obstructions, variables = dgcv_hook
        ev_point = {var: rational_sample() for var in variables}
        for expr in obstructions:
            if not _scalar_is_zero(subs(expr, ev_point)):
                return False
        return True
    return _scalar_is_zero(x)


def rational_sample(
    *,
    low: int = 1,
    high: int = 20,
    span: int = 20,
    avoid: tuple = (1, -1, 2, -2),
):
    while True:
        a = randint(low, high)
        b = randint(a + 1, a + span)

        g = gcd(a, b)
        if g != 1:
            continue

        if getrandbits(1):
            a, b = b, a

        q = rational(a, b)

        if getrandbits(1):
            q = -q

        if avoid and q in avoid:
            continue

        return q


def _extract_basis_over_number_field(
    objs,
    *,
    samples: int = 2,
    max_tries_per_obj: int = 4,
):
    from ...core.solvers import solve_dgcv

    objs = [o for o in objs if o is not None]
    if not objs:
        return []

    def _obstr(obj):
        ob = getattr(obj, "__dgcv_zero_obstr__", None)
        if ob is None:
            raise TypeError(f"{type(obj).__name__} lacks __dgcv_zero_obstr__")
        eqns, vars_ = ob
        return list(eqns or []), list(vars_ or [])

    def _solve_constant_dep(obj, basis):
        label = create_key(prefix="var")
        coef_vars = [symbol(f"{label}{i}") for i in range(len(basis))]

        residual = obj
        for u, b in zip(coef_vars, basis):
            residual = residual - u * b

        eqns, vars_ = _obstr(residual)
        if not eqns:
            return False

        sample_vars = [v for v in vars_ if v not in set(coef_vars)]
        if not sample_vars:
            sols = solve_dgcv(eqns, coef_vars, method="linsolve")
            if not sols:
                return False
            sol = sols[0]
            return all((u in sol) for u in coef_vars)

        zero_subs = {u: 0 for u in coef_vars}
        b_templates = [subs(e, zero_subs) for e in eqns]

        col_templates = []
        for uj in coef_vars:
            s = dict(zero_subs)
            s[uj] = 1
            e1 = [subs(e, s) for e in eqns]
            col_templates.append([e1i - bi for e1i, bi in zip(e1, b_templates)])

        stacked_eqns = []
        points_used = 0
        tries = 0

        while points_used < samples and tries < max_tries_per_obj:
            tries += 1
            approximation_point = {v: rational_sample() for v in sample_vars}

            b_num = [subs(e, approximation_point) for e in b_templates]
            A_cols = [
                [subs(e, approximation_point) for e in col] for col in col_templates
            ]
            A_rows = list(map(list, zip(*A_cols))) if A_cols else []

            rhs = [(-x) for x in b_num]
            lin_eqns = []
            for row, r in zip(A_rows, rhs):
                s = 0
                for aij, uj in zip(row, coef_vars):
                    s = s + aij * uj
                lin_eqns.append(s - r)

            stacked_eqns.extend(lin_eqns)

            sols = solve_dgcv(stacked_eqns, coef_vars, method="linsolve")
            if not sols:
                return False

            points_used += 1

        return points_used >= samples

    basis = []
    for obj in objs:
        eqns, _ = _obstr(obj)
        if eqns and all(_scalar_is_zero(e) for e in eqns):
            continue

        if not basis:
            basis.append(obj)
            continue

        if _solve_constant_dep(obj, basis):
            continue

        basis.append(obj)

    return basis
