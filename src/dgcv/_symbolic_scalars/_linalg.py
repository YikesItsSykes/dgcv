# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

import random
from fractions import Fraction

from .._aux._backends._symbolic_router import _scalar_is_zero, get_free_symbols, subs
from .._aux._backends._types_and_constants import exact_fraction
from ..algebras.threads.util import _exact_builtin_paths
from ..core.arrays import matrix_dgcv
from ..core.arrays._matrix_engine import _builtin_unsupported_matrix
from . import _factor, _functions, _poly
from ._linsolve import _divisor_values
from ._normal_form import RF, _fold_fraction, _value_from_rf
from ._normal_form import _nf as _builtin_nf

_SEED_STRIDE = 7919
_EIGEN_ATTEMPTS = 4
_DET_ATTEMPTS = 2
_PROBE_POINT_MAX = 97
_KRYLOV_ENTRY_MAX = 9
_ROOT_ATTEMPTS = 3
_ROOT_POINT_SHIFT = 8
_ROOT_IMAGE_BIT_BUDGET = 1 << 17


def _unsupported(feature):
    return _builtin_unsupported_matrix(feature)


def _random_point(atoms, rng, bound):
    return {a: Fraction(rng.randint(2, bound), rng.randint(2, bound)) for a in atoms}


def _rational_value(value):
    fr = exact_fraction(value)
    if fr is not None:
        return fr
    rf = _builtin_nf(value)
    if rf is None or not rf.is_constant:
        return None
    cv = rf.constant_value()
    if isinstance(cv, float):
        return None
    return Fraction(cv)


def _rational_rows(mat):
    n = mat.nrows
    rows = []
    for i in range(n):
        row = [_rational_value(mat[i, j]) for j in range(n)]
        if None in row:
            return None
        rows.append(row)

    return rows


def _rational_roots(coeffs):
    roots, rest = _poly.univariate_rational_roots(coeffs)
    if len(rest) > 1:
        raise _unsupported("eigenvalues (not all found as rational numbers)")
    return roots


def eigen_data(mat):
    n = mat.nrows
    if n != mat.ncols:
        raise ValueError("eigenvalues require a square matrix")
    rows = _rational_rows(mat)
    if rows is not None:
        return rows, _rational_roots(_charpoly_rational(matrix_dgcv(rows)))
    atoms = sorted(get_free_symbols(mat), key=str)
    rng = random.Random(len(atoms) * _SEED_STRIDE + n)
    for _attempt in range(_EIGEN_ATTEMPTS):
        point = _random_point(atoms, rng, _PROBE_POINT_MAX)
        try:
            srows = _rational_rows(mat.subs(point))
            if srows is not None:
                return None, _rational_roots(_charpoly_rational(matrix_dgcv(srows)))
        except NotImplementedError:
            continue

    raise _unsupported("eigenvalues")


def _generic_det(mat):
    atoms = sorted(get_free_symbols(mat), key=str)
    if not atoms:
        return None
    n = mat.nrows
    rng = random.Random(len(atoms) * _SEED_STRIDE + n)
    for _attempt in range(_DET_ATTEMPTS):
        point = _random_point(atoms, rng, _PROBE_POINT_MAX)
        try:
            entries = {}
            for i in range(n):
                for j in range(n):
                    v = subs(mat[i, j], point)
                    if not _scalar_is_zero(v):
                        entries[(i, j)] = v
            value = matrix_dgcv(entries, shape=(n, n)).det()
        except (NotImplementedError, ZeroDivisionError):
            return None
        if not _scalar_is_zero(value):
            return value
    return None


def generic_det(mat):
    if _exact_builtin_paths():
        return None
    return _generic_det(mat)


def _charpoly_rational(mat):
    n = mat.nrows
    identity = matrix_dgcv.identity(n)
    coeffs = [None] * (n + 1)
    coeffs[n] = Fraction(1)
    current = identity
    for k in range(1, n + 1):
        current = mat @ current
        trace = 0
        for i in range(n):
            trace = trace + current[i, i]
        tr = _rational_value(trace)
        if tr is None:
            return None
        coeffs[n - k] = -tr / k
        if k < n:
            current = current + _fold_fraction(coeffs[n - k]) * identity
    return coeffs


def _minimal_polynomial(mat, degree, rng):
    n = mat.nrows
    krylov = [matrix_dgcv([[rng.randint(1, _KRYLOV_ENTRY_MAX)] for _ in range(n)])]
    for _ in range(degree):
        krylov.append(mat @ krylov[-1])

    system = matrix_dgcv.from_cols(
        [[vector[i, 0] for i in range(n)] for vector in krylov[:degree]]
    )
    coeffs, pivots = system.solve_right(
        krylov[degree], return_divisors=True, parametric_vars=[0] * degree
    )
    if coeffs is None or len(pivots) < degree:
        return None
    return coeffs


def _quadratic_square(g):
    if g == _poly.I_GEN:
        return RF({(): -1})
    reduction = _poly.REDUCTIONS.get(g)
    if reduction is None or reduction[0] != 2:
        return None
    return reduction[1]()


def _quadratic_tower(gens):
    closure = set()
    squares = {}
    todo = set(gens)
    while todo:
        g = todo.pop()
        closure.add(g)
        square = _quadratic_square(g)
        if square is not None:
            squares[g] = square
            todo |= square.gens() - closure

    tower = []
    placed = set()
    while len(tower) < len(squares):
        ready = [
            g
            for g in sorted(squares)
            if g not in placed
            and all(h in placed for h in squares[g].gens() if h in squares)
        ]
        if not ready:
            return None, closure
        tower.append((ready[0], squares[ready[0]]))
        placed.add(ready[0])

    return tower, closure


def _split_quadratic(rf, g, square):
    parts = []
    for poly in (rf.num, rf.den):
        by_degree = _poly._split_by_gen(poly, g)
        if any(e > 1 for e in by_degree):
            return None
        parts += [RF(by_degree.get(0, {})), RF(by_degree.get(1, {}))]

    n0, n1, e0, e1 = parts
    if e1.is_zero:
        return n0 / e0, n1 / e0
    norm = e0 * e0 - e1 * e1 * square
    return (n0 * e0 - n1 * e1 * square) / norm, (n1 * e0 - n0 * e1) / norm


def _rationalized(rf, tower):
    for g, square in reversed(tower):
        if g in _poly.p_gens(rf.den):
            parts = _split_quadratic(rf, g, square)
            if parts is None:
                return rf
            rf = parts[0] + parts[1] * RF(_poly.p_gen(g))

    return rf


def _free_square_root(rf):
    const, factors = _factor.factor_poly(_poly.p_mul(rf.num, rf.den))
    root = _factor._integer_root(const, 2) if const > 0 else 0
    if root * root != const or any(mult % 2 for _, mult in factors):
        return None
    outside = {(): root}
    for poly, mult in factors:
        outside = _poly.p_mul(outside, _poly.p_pow(poly, mult // 2))

    return RF(outside) / RF(rf.den)


def _square_in_tower(value, tower):
    if value.is_zero:
        return value
    if not tower:
        return _free_square_root(value)
    g, square = tower[-1]
    lower = tower[:-1]
    d0, d1 = value, None
    if g in value.gens():
        parts = _split_quadratic(value, g, square)
        if parts is None:
            return None
        d0, d1 = parts

    if d1 is None or d1.is_zero:
        root = _square_in_tower(d0, lower)
        if root is not None:
            return root
        root = _square_in_tower(d0 * square, lower)
        return None if root is None else root * RF(_poly.p_gen(g)) / square

    norm = _square_in_tower(d0 * d0 - d1 * d1 * square, lower)
    if norm is None:
        return None
    two = RF({(): 2})
    for half in ((d0 + norm) / two, (d0 - norm) / two):
        s0 = _square_in_tower(half, lower)
        if s0 is not None and not s0.is_zero:
            return s0 + d1 / (s0 + s0) * RF(_poly.p_gen(g))

    return None


def _adjoined(root, closure):
    return {g for g in root.gens() - closure if g == _poly.I_GEN or g in _poly._RELATED}


def _square_root(rf, present):
    if rf.is_zero or _poly.p_has_float(rf.num) or _poly.p_has_float(rf.den):
        return None
    tower, closure = _quadratic_tower(present | rf.gens())
    if tower is None:
        return None
    free = rf.gens().isdisjoint(g for g, _ in tower)
    if free:
        const, factors = _factor.factor_poly(_poly.p_mul(rf.num, rf.den))
        outside = {(): 1}
        inside = {(): 1}
        for poly, mult in factors:
            if mult > 1:
                outside = _poly.p_mul(outside, _poly.p_pow(poly, mult // 2))
            if mult % 2:
                inside = _poly.p_mul(inside, poly)

        if const < 0 and not _poly.p_is_const(inside):
            const, inside = -const, _poly.p_neg(inside)
        root = _functions._radical_const(Fraction(const), 2) * RF(outside) / RF(rf.den)
        if not _poly.p_is_const(inside):
            root = root * _functions.radical(RF(inside), 2)

        polys = [poly for poly, _ in factors]
        adjoined = _adjoined(root, closure)
        if not adjoined:
            return root, polys, adjoined

    inner = _square_in_tower(rf, tower)
    if inner is not None:
        return inner, polys if free else [inner.num, inner.den], set()
    if any(g in _poly._RELATED and _quadratic_square(g) is None for g in closure):
        return None
    if not free:
        radicand = RF(rf.num) * RF(rf.den)
        root = _functions.radical(radicand, 2) / RF(rf.den)
        polys = [radicand.num]
        adjoined = _adjoined(root, closure)

    extended, _closure = _quadratic_tower(closure | adjoined)
    if extended is None:
        return None
    for level, (_g, square) in enumerate(extended):
        if _square_in_tower(square, extended[:level]) is not None:
            return None

    return root, polys, adjoined


def _monic_integral(rfs):
    lcm = {(): 1}
    for rf in rfs:
        common, _complete = _poly.p_gcd(lcm, rf.den)
        lcm = _poly.p_mul(lcm, _poly.p_div_exact(rf.den, common))

    coeffs = []
    power = {(): 1}
    for rf in reversed(rfs):
        numerator = _poly.p_mul(rf.num, _poly.p_div_exact(lcm, rf.den))
        coeffs.append(_poly.p_neg(_poly.p_mul(numerator, power)))
        power = _poly.p_mul(power, lcm)

    coeffs.reverse()
    return lcm, coeffs


def _substituted(poly, gid, image):
    out = {}
    for exponent, part in _poly._split_by_gen(poly, gid).items():
        if exponent:
            part = _poly.p_mul(part, _poly.p_pow(image, exponent), reduce_i=False)
        out = _poly.p_add(out, part)

    return out


def _unreduced(coeffs):
    eliminated = set()
    for g in sorted(set().union(*[_poly.p_gens(f) for f in coeffs])):
        square = _quadratic_square(g)
        if g == _poly.I_GEN or square is None or square.den != {(): 1}:
            continue
        for t in sorted(_poly.p_gens(square.num) - eliminated):
            lead = square.num.get(((t, 1),))
            rest = {m: c for m, c in square.num.items() if m != ((t, 1),)}
            rest_gens = _poly.p_gens(rest)
            if (
                lead not in (1, -1)
                or t == _poly.I_GEN
                or t in _poly._RELATED
                or t in rest_gens
                or not eliminated.isdisjoint(rest_gens)
            ):
                continue
            image = _poly.p_scale(_poly.p_sub(_poly.p_gen(g, 2), rest), lead)
            coeffs = [_substituted(f, t, image) for f in coeffs]
            eliminated.add(t)
            break

    return coeffs


def _field_roots(rfs):
    if any(_poly.p_has_float(rf.num) or _poly.p_has_float(rf.den) for rf in rfs):
        return []
    degree = len(rfs)
    lcm, coeffs = _monic_integral(rfs)
    images = _unreduced(coeffs)
    gens = sorted(set().union(*[_poly.p_gens(f) for f in images]))
    roots = []
    for attempt in range(_ROOT_ATTEMPTS):
        values = images
        points = []
        for x in gens:
            bound = max(
                _factor._integer_root(_poly._p_maxnorm(f), degree - k)
                for k, f in enumerate(values)
            )
            xi = (2 * bound + 4) << (_ROOT_POINT_SHIFT * attempt)
            width = max(_poly.p_degree_in(f, x) for f in values)
            if xi.bit_length() * width > _ROOT_IMAGE_BIT_BUDGET:
                return []
            points.append(xi)
            values = [_poly._p_eval_gen(f, x, xi) for f in values]

        found, _rest = _poly.univariate_rational_roots(
            [_poly.p_const_value(f) for f in values] + [1]
        )
        roots = []
        for r in found:
            candidate = _poly.p_const(int(r))
            for x, xi in zip(reversed(gens), reversed(points)):
                candidate = _poly._poly_from_digits(candidate, x, xi)

            if _poly.p_degree_in(candidate, _poly.I_GEN) > 1:
                continue
            value = RF(candidate)
            residue = RF({(): 1})
            for f in reversed(coeffs):
                residue = residue * value + RF(f)

            if residue.is_zero:
                roots.append(value / RF(lcm))

        if len(roots) == len(found):
            break

    return roots


def _spectrum(mat, coeffs):
    degree = len(coeffs)
    numbers = [_rational_value(c) for c in coeffs]
    if None not in numbers:
        roots, _rest = _poly.univariate_rational_roots([-c for c in numbers] + [1])
        if len(roots) == degree:
            return [_fold_fraction(r) for r in roots], [], set(), None

    entries = [_builtin_nf(value) for value in mat._data.values()]
    rfs = [_builtin_nf(c) for c in coeffs]
    if any(rf is None for rf in entries + rfs):
        return None
    present = set()
    for rf in entries:
        present |= rf.gens()

    roots = _field_roots(rfs) if degree > 2 else []
    if roots and len(roots) == degree:
        tower = _quadratic_tower(present)[0] or []
        return [_value_from_rf(r) for r in roots], [], set(), tower
    if len(roots) != degree - 2:
        return None
    linear, constant = coeffs[1], coeffs[0]
    if roots:
        quotient = [-rf for rf in rfs] + [RF({(): 1})]
        for r in roots:
            carry = quotient[-1]
            lower = [None] * (len(quotient) - 1)
            for k in range(len(quotient) - 2, -1, -1):
                lower[k] = carry
                carry = carry * r + quotient[k]
            quotient = lower

        linear = _value_from_rf(-quotient[1])
        constant = _value_from_rf(-quotient[0])

    discriminant = _builtin_nf(linear * linear + 4 * constant)
    found = _square_root(discriminant, present)
    if found is None:
        return None
    root, polys, adjoined = found
    half = Fraction(1, 2)
    root = _value_from_rf(root)
    evals = [_value_from_rf(r) for r in roots]
    evals += [half * (linear - root), half * (linear + root)]
    tower = None
    if degree > 2:
        tower = _quadratic_tower(present | adjoined)[0] or []
    return evals, polys, adjoined, tower


def _without_units(poly, adjoined):
    content = _poly.p_monomial_content(poly)
    return _poly.p_divide_monomial(
        poly, tuple((g, e) for g, e in content if g in adjoined or g == _poly.I_GEN)
    )


def projector_eigenspaces(specialized, expected, rng):
    n = specialized.nrows
    coeffs = _minimal_polynomial(specialized, expected, rng)
    if coeffs is None:
        return None
    spectrum = _spectrum(specialized, coeffs)
    if spectrum is None:
        return None
    evals, polys, adjoined, tower = spectrum
    identity = matrix_dgcv.identity(n)
    shifted = [specialized - r * identity for r in evals]
    annihilated = identity
    for shift in shifted:
        annihilated = annihilated @ shift
    if any(annihilated.iter_nonzero_items()):
        return None

    blocks = []
    for j in range(expected):
        proj = identity
        for k in range(expected):
            if k != j:
                proj = proj @ shifted[k]

        columns, pivots = proj.pivot_columns(record_divisors=True)
        scale = None
        if tower is not None:
            scale = RF({(): 1})
            for k in range(expected):
                if k != j:
                    scale = scale * _builtin_nf(evals[j] - evals[k])

        block = []
        for col in columns:
            column = [proj[i, col] for i in range(n)]
            if scale is not None:
                entries = [
                    _rationalized(_builtin_nf(value) / scale, tower) for value in column
                ]
                polys.extend(_without_units(e.den, adjoined) for e in entries)
                column = [_value_from_rf(e) for e in entries]
            block.append(column)

        blocks.append(block)
        for pivot in pivots:
            ratio = _builtin_nf(pivot)
            if scale is not None:
                ratio = _rationalized(ratio / scale, tower)
                polys.append(_without_units(ratio.den, adjoined))
            polys.append(_without_units(ratio.num, adjoined))

    if sum(len(block) for block in blocks) != n:
        return None
    return blocks, _divisor_values(polys)
