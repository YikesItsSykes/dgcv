"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds.Pfaff_systems

module: dgcv.eds.Pfaff_systems._flag

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers
import random
from dataclasses import dataclass

from ..._aux._backends._engine import engine_capability, engine_kind
from ..._aux._backends._symbolic_router import _scalar_is_zero
from ..._aux._backends._types_and_constants import _disposable_symbols
from ..._aux._vmf._safeguards import create_key
from ...core.arrays import matrix_dgcv
from ...core.arrays._matrix_linalg import _eliminate
from ...core.solvers import solve_dgcv
from ._util import (
    _add,
    _cancel_generators,
    _chain_simplifier,
    _dot,
    _free,
    _rank,
    _scale,
    _subs,
    _symbols,
    _tidy,
    _unit,
    _zero_test,
)

_exact_chain_work_limit = 100000


_cas_characters_per_term = 12


_cas_cancel_generator_limit = 15


_sample_bound = 9


_sample_tries = 8


_sample_runs = 4


class _ExactChainTooLarge(Exception):
    pass


def _entry_size(value):
    nf = getattr(value, "_nf", None)
    if nf is not None:
        return len(nf.num) + len(nf.den)
    if isinstance(value, numbers.Number):
        return 1
    return 1 + len(str(value)) // _cas_characters_per_term


def _exact_polar_level(polar, limit):
    width = polar.ncols
    rows = [[polar[i, j] for j in range(width)] + [0] for i in range(polar.nrows)]
    inner = _zero_test()
    work = [0]

    def test(value):
        if limit is not None:
            work[0] += sum(_entry_size(entry) for row in rows for entry in row)
            if work[0] > limit:
                raise _ExactChainTooLarge
        return inner(value)

    simplifier = _chain_simplifier()
    if limit is not None and engine_kind() == "sympy":
        cancel = simplifier

        def simplifier(value):
            count = _cancel_generators(value)
            if count is not None and count >= _cas_cancel_generator_limit:
                raise _ExactChainTooLarge
            return cancel(value)

    rank, pivot_rows, _, divisors = _eliminate(
        rows,
        rhs_cols=1,
        record_divisors=True,
        simplify_steps=True,
        want_pivmap=True,
        zero_test=test,
        simplifier=simplifier,
    )
    return rank, [d for d in divisors if not isinstance(d, numbers.Number)], rows, pivot_rows


def _general_solution(rows, pivot_rows, params):
    width = len(params)
    free = [j for j in range(width) if j not in pivot_rows]
    general = [params[j] if j in free else 0 for j in range(width)]
    for col, row in pivot_rows.items():
        value = rows[row][width]
        for j in free:
            value = value - rows[row][j] * params[j]
        general[col] = value
    return _tidy(general)


def _projects_onto(projection, n, weights):
    if weights:
        rng = random.Random(0)
        for _ in range(2):
            sample = {w: rng.randint(-_sample_bound, _sample_bound) or 1 for w in weights}
            try:
                numeric = [[_subs(entry, sample) for entry in row] for row in projection]
            except ZeroDivisionError:
                continue
            if _rank(matrix_dgcv(numeric)) >= n:
                return True
    return _rank(matrix_dgcv(projection)) >= n


@dataclass(frozen=True)
class _chain_data:
    vectors: list
    ranks: list
    divisors: list
    samples: tuple
    exact_levels: int | None = None


def _same_symbol(entry, symbol):
    return entry is symbol or (not isinstance(entry, numbers.Number) and str(entry) == str(symbol))


def _polar_general_solution(polar, params):
    f = len(params)
    if engine_capability("linsolve") is not None:
        equations = []
        for i in range(polar.nrows):
            total = _dot(polar.row(i), params)
            if not _scalar_is_zero(total):
                equations.append(total)
        if not equations:
            return list(params), [], 0
        solutions, divisors = solve_dgcv(equations, list(params), return_divisors=True)
        if solutions:
            solution = solutions[0]
            general = _tidy([solution.get(x, x) for x in params])
            free = sum(1 for x, entry in zip(params, general) if _same_symbol(entry, x))
            return general, [d for d in divisors if not isinstance(d, numbers.Number)], f - free
    rank, divisors, rows, pivot_rows = _exact_polar_level(polar, None)
    return _general_solution(rows, pivot_rows, params), divisors, rank


def _polar_rank(polar):
    if engine_capability("linsolve") is None:
        return _rank(polar)
    return _polar_general_solution(polar, _symbols("s_", polar.ncols))[2]


def _flag_weights(flag, f):
    free = set().union(*(_free(entry) for v in flag for entry in v)) if flag else set()
    return [w for p in range(1, len(flag) + 1) for w in _symbols(f"t_{p}_", f) if w in free]


def _specialized_flag(matrices, f, flag, ranks, rng):
    weights = _flag_weights(flag, f)
    for _ in range(_sample_tries):
        sample = {w: rng.randint(-_sample_bound, _sample_bound) or 1 for w in weights}
        try:
            numeric = [_tidy([_subs(entry, sample) for entry in v]) for v in flag]
        except ZeroDivisionError:
            continue
        if all(
            _polar_rank(_polar(matrices, numeric[:p], f)) == ranks[p - 1]
            for p in range(1, len(flag) + 1)
        ):
            return numeric, sample
    raise ValueError(
        f"no generic choice of the integral flag's weights was found in {_sample_tries} random tries"
    )


def _sampled_level(matrices, f, flag, vector, rng):
    free = set().union(*(_free(entry) for entry in vector))
    weights = [w for w in _symbols(f"t_{len(flag) + 1}_", f) if w in free]
    best = None
    for _ in range(_sample_tries):
        sample = {w: rng.randint(-_sample_bound, _sample_bound) or 1 for w in weights}
        try:
            numeric = _tidy([_subs(entry, sample) for entry in vector])
        except ZeroDivisionError:
            continue
        rank = _polar_rank(_polar(matrices, flag + [numeric], f))
        if best is not None and rank == best[2]:
            return best
        if best is None or rank > best[2]:
            best = (numeric, sample, rank)
    if best is None:
        raise ValueError(
            f"no generic choice of the integral flag's weights was found in {_sample_tries} random tries"
        )
    return best


def _sampled_run(matrices, f, exact_flag, exact_ranks, rng):
    flag, ranks, samples = list(exact_flag), list(exact_ranks), []
    if flag:
        flag, sample = _specialized_flag(matrices, f, flag, ranks, rng)
        samples.append(sample)
    while True:
        p = len(flag) + 1
        if flag:
            vector, _, _ = _polar_general_solution(
                _polar(matrices, flag, f), _symbols(f"t_{p}_", f)
            )
        else:
            vector = _combination([_unit(f, k) for k in range(f)], "t_1_")
        numeric, sample, rank = _sampled_level(matrices, f, flag, vector, rng)
        ranks.append(rank)
        if rank >= f - p or p >= f:
            return _chain_data(flag + [vector], ranks, [], tuple(samples))
        flag = flag + [numeric]
        samples.append(sample)


def _combination(basis, prefix):
    total = [0] * len(basis[0])
    for weight, vector in zip(_symbols(prefix, len(basis)), basis):
        total = _add(total, _scale(vector, weight))
    return _tidy(total)


def _polar(matrices, vectors, f):
    rows = [
        [_dot(xi, matrix.col(j)) for j in range(f)]
        for matrix in matrices
        for xi in vectors
    ]
    return matrix_dgcv(rows or [[0] * f])


def _polar_solution(polar, params):
    return polar.solve_right(
        [[0] for _ in range(polar.nrows)],
        simplify_steps=True,
        parametric_vars=params,
        zero_test=_zero_test(),
    )


def _one_hot(general, params):
    free = [k for k, entry in enumerate(general) if entry is params[k]]
    out = []
    for k in free:
        rules = {params[j]: 1 if j == k else 0 for j in free}
        out.append(_tidy([_subs(entry, rules) for entry in general]))
    return out


def _polar_basis(polar, f):
    params = _disposable_symbols(create_key(prefix="ps"), f)
    return _one_hot(_polar_solution(polar, params), params)


@dataclass(frozen=True)
class _flag_data:
    chart: object
    vectors: tuple
    polar_ranks: tuple
    characters: tuple
    genre: int | None
    singularities: tuple
    weights: tuple
    samples: tuple = ()
    exact_levels: int | None = None

    @property
    def flag(self):
        elements = [self.chart.element(v) for v in self.vectors]
        return tuple(tuple(elements[:p]) for p in range(1, len(elements) + 1))
