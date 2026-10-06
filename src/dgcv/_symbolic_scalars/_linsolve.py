# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

import contextlib
import math
from fractions import Fraction
from typing import NamedTuple, Optional

from ..core.solvers._predicates import _as_zero_expr
from ..eds._atoms import zero_form_atom
from ..eds._constants import _BuiltinLeaf
from ..eds._zero_forms import zero_form_class
from . import _poly
from ._normal_form import (
    RF,
    OpaqueLeaf,
    _CompoundLeaf,
    _mentions,
    _nf,
    _value_from_poly,
    _value_from_rf,
)

_INCONSISTENT = object()
_ZERO_RF = RF({})
_NONLINEAR = object()
_seen_keys = {}
_transform_cache = {}
_SEEN_LIMIT = 64
_CACHE_LIMIT = 8
_TRANSFORM_AFTER = 2
_NO_TRANSFORM = -1
_pivot_guard = None


@contextlib.contextmanager
def pivot_guard(guard):
    global _pivot_guard
    previous = _pivot_guard
    _pivot_guard = guard
    try:
        yield guard
    finally:
        _pivot_guard = previous


def _leaf_of(v):
    if isinstance(v, zero_form_class):
        v = v.base
    return v if isinstance(v, (zero_form_atom, _BuiltinLeaf)) else None


def _build_rows(eqns, gids):
    col_of = {g: i for i, g in enumerate(gids)}
    var_leaves = [_poly.gen_leaf(g) for g in gids]
    var_set = set(var_leaves)
    compound = {}
    seen = set()
    rows = []
    numeric = True

    def _compound(g):
        cp = compound.get(g)
        if cp is None:
            leaf = _poly.gen_leaf(g)
            cp = isinstance(leaf, _CompoundLeaf) and any(
                a in var_set for a in leaf.free_symbols
            )
            compound[g] = cp
        return cp

    for eq in eqns:
        expr = _as_zero_expr(eq)
        rf = _nf(expr)
        if rf is None:
            return None, False
        num, den = rf.num, rf.den
        if not num:
            continue
        if _poly.p_has_float(num) or _poly.p_has_float(den):
            return None, False
        if not _poly.p_is_const(den):
            for g in _poly.p_gens(den):
                if g in col_of or _compound(g):
                    return _NONLINEAR, False
            numeric = False
        coeffs = {}
        const = {}
        for m, c in num.items():
            col = None
            rest = []
            for g, e in m:
                if _compound(g):
                    return _NONLINEAR, False
                j = col_of.get(g)
                if j is None:
                    rest.append((g, e))
                    seen.add(g)
                    numeric = False
                elif col is not None or e != 1:
                    return _NONLINEAR, False
                else:
                    col = j
            target = const if col is None else coeffs.setdefault(col, {})
            target[tuple(rest)] = c
        if not coeffs:
            if const:
                return _INCONSISTENT, False
            continue
        rows.append((coeffs, _poly.p_neg(const)))
    if not numeric:
        for g in seen:
            leaf = _poly.gen_leaf(g)
            if isinstance(leaf, OpaqueLeaf) and any(
                _mentions(leaf.base, v) for v in var_leaves
            ):
                return None, False
    return rows, numeric


def _poly_key(poly):
    if len(poly) == 1:
        return tuple(poly.items())
    return tuple(sorted(poly.items()))


def _canonical_rows(rows):
    keyed = []
    for coeffs, rhs in rows:
        sig = tuple(sorted((col, _poly_key(poly)) for col, poly in coeffs.items()))
        keyed.append((sig, coeffs, rhs))
    keyed.sort(key=lambda item: item[0])
    return (
        tuple(item[0] for item in keyed),
        [item[1] for item in keyed],
        [item[2] for item in keyed],
    )


def _remember(key):
    count = _seen_keys.get(key, 0)
    if count == _NO_TRANSFORM:
        return count
    count += 1
    if count == 1 and len(_seen_keys) >= _SEEN_LIMIT:
        _seen_keys.clear()
    _seen_keys[key] = count
    return count


def _store(key, prepared):
    if len(_transform_cache) >= _CACHE_LIMIT:
        _transform_cache.clear()
    _transform_cache[key] = prepared


def _sparse_rref(rows, n_cols):
    work = []
    for r in rows:
        work.append({k: RF(v) for k, v in r.items()})

    col_rows = {}
    for i, row in enumerate(work):
        for j in row:
            if isinstance(j, int):
                col_rows.setdefault(j, set()).add(i)

    order = list(range(len(work)))
    pivots = []
    pivot_polys = []
    guard = _pivot_guard
    r = 0

    for col in range(n_cols):
        holders = col_rows.get(col)
        if not holders:
            continue
        pick = None
        best = None
        for pos in range(r, len(order)):
            i = order[pos]
            if i in holders:
                entry = work[i][col]
                if guard is not None:
                    if not guard.nonzero(entry.num):
                        continue
                    key = (
                        0 if entry.is_constant else 1,
                        len(work[i]),
                        guard.weight(entry.num),
                    )
                else:
                    key = (0 if entry.is_constant else 1, len(work[i]))

                if best is None or key < best:
                    best = key
                    pick = pos

        if pick is None:
            continue
        order[r], order[pick] = order[pick], order[r]
        i = order[r]
        prow = work[i]
        piv = prow[col]
        pivot_polys.append(piv.num)
        if not piv.is_one:
            for k in list(prow):
                prow[k] = prow[k] / piv

        for t in list(holders):
            if t == i:
                continue
            row = work[t]
            f = row[col]
            for k, v in prow.items():
                old = row.get(k)
                new = old - f * v if old is not None else -(f * v)
                if new.is_zero:
                    if old is not None:
                        del row[k]
                        if isinstance(k, int):
                            col_rows[k].discard(t)
                elif old is None:
                    row[k] = new
                    if isinstance(k, int):
                        col_rows.setdefault(k, set()).add(t)
                else:
                    row[k] = new

        pivots.append(col)
        r += 1

    return [work[i] for i in order], pivots, pivot_polys


class _Factors:
    __slots__ = ("polys", "index")

    def __init__(self):
        self.polys = []
        self.index = {}

    def register(self, poly):
        key = tuple(sorted(poly.items()))
        idx = self.index.get(key)
        if idx is None:
            idx = len(self.polys)
            self.polys.append(poly)
            self.index[key] = idx
        return idx


def _cancel(num, den, F):
    if not num:
        return {}, {}
    if not den:
        return num, den
    c = _poly.p_int_content(num)
    if c != 1:
        num = {m: v // c for m, v in num.items()}

    out = {}
    for idx, e in den.items():
        f = F.polys[idx]
        while e > 0:
            q = _poly.p_div_exact(num, f)
            if q is None:
                break
            num = q
            e -= 1
        if e:
            out[idx] = e
    den = out
    if c != 1:
        num = {m: v * c for m, v in num.items()}
    return num, den


def _den_poly(den, F):
    out = {(): 1}
    for idx, e in den.items():
        for _ in range(e):
            out = _poly.p_mul(out, F.polys[idx])
    return out


def _mul(a, b, F):
    an, ad = a
    bn, bd = b
    if not bd and len(bn) == 1 and () in bn:
        k = bn[()]
        if k == 1:
            return a
        return {m: x * k for m, x in an.items()}, ad

    if len(an) == 1 and len(bn) == 1:
        ((ma, ca),) = an.items()
        ((mb, cb),) = bn.items()
        m, sign = _poly._mono_mul(ma, mb)
        num = {m: ca * cb if sign == 1 else -(ca * cb)}
        if not ad and not bd:
            return num, {}
        den = dict(ad)
        for idx, e in bd.items():
            den[idx] = den.get(idx, 0) + e

        return _cancel(num, den, F)

    den = dict(ad)
    for idx, e in bd.items():
        den[idx] = den.get(idx, 0) + e

    return _cancel(_poly.p_mul(an, bn), den, F)


def _sub_scaled(a, c, b, F):
    an, ad = a
    bn, bd = b
    if c != 1:
        an = {m: v * c for m, v in an.items()}

    if ad == bd:
        return _cancel(_poly.p_sub(an, bn), dict(ad), F)
    lcm = dict(ad)
    for idx, e in bd.items():
        if lcm.get(idx, 0) < e:
            lcm[idx] = e

    ma = {idx: e - ad.get(idx, 0) for idx, e in lcm.items() if e - ad.get(idx, 0) > 0}
    mb = {idx: e - bd.get(idx, 0) for idx, e in lcm.items() if e - bd.get(idx, 0) > 0}
    if ma:
        an = _poly.p_mul(an, _den_poly(ma, F))

    if mb:
        bn = _poly.p_mul(bn, _den_poly(mb, F))

    return _cancel(_poly.p_sub(an, bn), lcm, F)


def _to_rf(entry, F, scale=1):
    num, den = entry
    if not num:
        return RF({}, canonical=True)
    if (
        not den
        and len(num) == 1
        and type(scale) is int
        and not (_poly._REDUCIBLE and _poly.p_needs_reduction(num))
    ):
        ((m, v),) = num.items()
        if type(v) is int:
            c = math.gcd(v, scale)
            if c != 1:
                v //= c
                s = scale // c
            else:
                s = scale

            if s < 0:
                v = -v
                s = -s

            return RF({m: v}, {(): s}, canonical=True)

    d = _den_poly(den, F) if den else {(): 1}
    if scale != 1:
        d = _poly.p_scale(d, scale)

    return RF(num, d)


def _sparse_rref_factored(rows, n_cols):
    F = _Factors()
    work = []
    for r in rows:
        row = {}
        consts = []
        for k, v in r.items():
            if not v:
                continue
            if len(v) == 1 and type(next(iter(v.values()))) is int:
                row[k] = (v, 1)
                continue

            L = _poly.p_denominator_lcm(v)
            num = _poly.p_to_int(_poly.p_scale(v, L)) if L != 1 else v
            row[k] = (num, L)
            consts.append(L)

        L = 1
        for c in consts:
            L = L * c // math.gcd(L, c)

        out = {}
        for k, (num, c) in row.items():
            s = L // c
            out[k] = ({m: v * s for m, v in num.items()} if s != 1 else num, {})

        work.append(out)

    col_rows = {}
    width = [0] * len(work)
    for i, row in enumerate(work):
        for j in row:
            if isinstance(j, int):
                col_rows.setdefault(j, set()).add(i)
                width[i] += 1

    order = list(range(len(work)))
    pos_of = list(range(len(work)))
    pivots = []
    pivot_polys = []
    guard = _pivot_guard
    r = 0

    for col in range(n_cols):
        holders = col_rows.get(col)
        if not holders:
            continue
        pick = None
        best = None
        for i in holders:
            pos = pos_of[i]
            if pos < r:
                continue
            num, den = work[i][col]
            const = not den and _poly.p_is_const(num)
            if const:
                size = 1
            elif guard is not None:
                if not guard.nonzero(num):
                    continue
                size = guard.weight(num)
            else:
                size = len(num)

            key = (0 if const else 1, width[i], size, pos)
            if best is None or key < best:
                best = key
                pick = pos

        if pick is None:
            continue
        a, b = order[r], order[pick]
        order[r], order[pick] = b, a
        pos_of[b], pos_of[a] = r, pick
        i = order[r]
        prow = work[i]
        pn, pd = prow[col]
        pivot_polys.append(pn)
        pf, _ = _poly._prim(pn)
        scale = _den_poly(pd, F)
        if _poly._holds_i(pf):
            pf, z = _poly._gaussian_primitive(pf)
            if z != (1, 0):
                scale = _poly.p_mul(scale, {(): z[0], ((_poly.I_GEN, 1),): -z[1]})

        if _poly.p_is_const(pf):
            inv = (scale, {})
        else:
            fidx = F.register(pf)
            inv = (scale, {fidx: 1})

        for k in list(prow):
            prow[k] = _mul(prow[k], inv, F)

        pcn, pcd = prow[col]
        if pcd or not _poly.p_is_const(pcn):
            return None
        c = pcn[()]
        for t in list(holders):
            if t == i:
                continue
            row = work[t]
            f = row[col]
            if len(prow) == 1:
                del row[col]
                col_rows[col].discard(t)
                width[t] -= 1
                if c != 1:
                    for k, old in row.items():
                        row[k] = ({m: x * c for m, x in old[0].items()}, old[1])

                prow_items = ()
            else:
                prow_items = prow.items()
                if c != 1:
                    for k, old in row.items():
                        if k not in prow:
                            row[k] = ({m: x * c for m, x in old[0].items()}, old[1])

            for k, v in prow_items:
                old = row.get(k)
                fv = _mul(f, v, F)
                if old is None:
                    new = ({m: -x for m, x in fv[0].items()}, fv[1])
                else:
                    new = _sub_scaled(old, c, fv, F)

                if not new[0]:
                    if old is not None:
                        del row[k]
                        if isinstance(k, int):
                            col_rows[k].discard(t)
                            width[t] -= 1
                else:
                    if old is None and isinstance(k, int):
                        col_rows.setdefault(k, set()).add(t)
                        width[t] += 1

                    row[k] = new

            g = 0
            for num, _d in row.values():
                g = math.gcd(g, _poly.p_int_content(num))
                if g == 1:
                    break

            if g > 1:
                for k in list(row):
                    num, d = row[k]
                    row[k] = ({m: x // g for m, x in num.items()}, d)

        pivots.append(col)
        r += 1

    out_rows = []
    for pos, i in enumerate(order):
        sc = 1
        if pos < len(pivots):
            pn, pd = work[i][pivots[pos]]
            if pd or not _poly.p_is_const(pn):
                return None
            sc = pn[()]

        out_rows.append({k: _to_rf(v, F, sc) for k, v in work[i].items()})

    return out_rows, pivots, pivot_polys


def _rows_related(rows):
    related = _poly._RELATED
    for row in rows:
        for poly in row.values():
            for m in poly:
                for g, _ in m:
                    if g in related:
                        return True
    return False


def _symbolic_rref(rows, n_cols, key=None):
    if not (_poly._RELATED and _rows_related(rows)):
        out = _sparse_rref_factored([dict(r) for r in rows], n_cols)
        if out is not None:
            return out
        if key is not None:
            _seen_keys[key] = _NO_TRANSFORM
    return _sparse_rref(rows, n_cols)


def _numeric_rref(rows, n_cols):
    m = len(rows)
    work = [dict(r) for r in rows]
    holders = {}

    for i, row in enumerate(work):
        for j in row:
            holders.setdefault(j, set()).add(i)

    pivots = []
    r = 0
    for col in range(n_cols):
        hold = holders.get(col)
        if not hold:
            continue
        pick = None
        for i in hold:
            if i >= r and (pick is None or i < pick) and work[i][col]:
                pick = i

        if pick is None:
            continue
        if pick != r:
            moved = work[r]
            work[r], work[pick] = work[pick], moved
            for j in moved:
                holders[j].discard(r)

            for j in work[r]:
                holders[j].discard(pick)

            for j in moved:
                holders[j].add(pick)

            for j in work[r]:
                holders[j].add(r)

        prow = work[r]
        p = prow[col]
        if p != 1:
            inv = Fraction(1) / p
            for j in list(prow):
                prow[j] = prow[j] * inv

        for i in list(hold):
            if i == r:
                continue
            row = work[i]
            f = row.get(col)
            if not f:
                continue
            for j, b in prow.items():
                v = row.get(j, 0) - f * b
                if v:
                    if j not in row:
                        holders[j].add(i)

                    row[j] = v
                elif j in row:
                    del row[j]
                    holders[j].discard(i)

        pivots.append(col)
        r += 1
        if r == m:
            break

    return work, pivots


def _numeric_rows(rows):
    out = []
    for coeffs, rhs in rows:
        row = {col: Fraction(poly[()]) for col, poly in coeffs.items()}
        row["rhs"] = Fraction(rhs.get((), 0))
        out.append(row)
    return out


def _numeric_solution(rhs, rows, pivots, n_cols, gids):
    for i in range(len(pivots), len(rhs)):
        if rhs[i]:
            return None

    pivot_set = set(pivots)
    free_cols = [j for j in range(n_cols) if j not in pivot_set]
    sol = [None] * n_cols

    for j in free_cols:
        sol[j] = _poly.gen_leaf(gids[j])

    for k, col in enumerate(pivots):
        row = rows[k]
        poly = {}
        if rhs[k]:
            poly[()] = rhs[k]

        for j in free_cols:
            c = row.get(j)
            if c:
                poly[((gids[j], 1),)] = -c

        sol[col] = _value_from_poly(poly)

    return sol


def _symbolic_solution(rhs, rows, pivots, n_cols, gids):
    for i in range(len(pivots), len(rhs)):
        if not rhs[i].is_zero:
            return None

    pivot_set = set(pivots)
    free_cols = [j for j in range(n_cols) if j not in pivot_set]
    sol = [None] * n_cols

    for j in free_cols:
        sol[j] = _poly.gen_leaf(gids[j])

    for k, col in enumerate(pivots):
        row = rows[k]
        rf = rhs[k]
        for j in free_cols:
            c = row.get(j)
            if c is not None:
                rf = rf - c * RF(_poly.p_gen(gids[j]))

        sol[col] = _value_from_rf(rf)

    return sol


def _numeric_images(T, rhs_list):
    b = [Fraction(v.get((), 0)) for v in rhs_list]
    images = []
    for trow in T:
        total = Fraction(0)
        for j, t in trow.items():
            if b[j]:
                total += t * b[j]

        images.append(total)

    return images


def _symbolic_images(T, rhs_list):
    nonzero = [(j, RF(v)) for j, v in enumerate(rhs_list) if v]
    images = []
    for trow in T:
        total = _ZERO_RF
        for j, v in nonzero:
            t = trow.get(j)
            if t is not None:
                total = total + t * v

        images.append(total)

    return images


class _Transform(NamedTuple):
    transform: list
    reduced: list
    pivots: list
    pivot_polys: Optional[list]


def _transform_record(work, pivots, pivot_polys=None):
    return _Transform(
        [{k[1]: v for k, v in row.items() if isinstance(k, tuple)} for row in work],
        [{j: v for j, v in row.items() if not isinstance(j, tuple)} for row in work],
        pivots,
        pivot_polys,
    )


def _divisor_values(pivot_polys):
    out = []
    seen = set()
    for p in pivot_polys:
        if _poly.p_is_const(p):
            continue
        if len(p) <= 2 and _poly.p_gens(p) == {_poly.I_GEN}:
            continue
        key = _poly_key(p)
        if key in seen:
            continue
        seen.add(key)

        p, c = _poly._prim(p)
        if c != 1:
            key = _poly_key(p)
            if key in seen:
                continue
            seen.add(key)
        out.append(_value_from_poly(p))
    return out


def builtin_build_and_solve(eqns, vars_, return_divisors):
    gids = []
    for v in vars_:
        leaf = _leaf_of(v)
        if leaf is None:
            return None
        gids.append(_poly.gen_id(leaf))
    n_cols = len(gids)
    rows, numeric = _build_rows(eqns, gids)
    if rows is None:
        return None
    if rows is _NONLINEAR:
        raise NotImplementedError(
            "dgcv's builtin symbolic engine solves systems that are linear in the unknowns; this system is nonlinear. Install sympy or sage and set `default_engine` accordingly"
        )
    if rows is _INCONSISTENT:
        return [], []
    if not rows:
        sol = [_poly.gen_leaf(g) for g in gids]
        return (dict(zip(vars_, sol)), [])
    signature, coeff_rows, rhs_rows = _canonical_rows(rows)
    key = (signature, n_cols, numeric)
    seen = _remember(key) if _pivot_guard is None else 0
    prepared = _transform_cache.get(key) if _pivot_guard is None else None
    divisors = []
    if numeric:
        if prepared is None and seen > _TRANSFORM_AFTER:
            aug = []
            for i, coeffs in enumerate(coeff_rows):
                row = {col: Fraction(poly[()]) for col, poly in coeffs.items()}
                row[("t", i)] = Fraction(1)
                aug.append(row)

            work, pivots = _numeric_rref(aug, n_cols)
            prepared = _transform_record(work, pivots)
            _store(key, prepared)

        if prepared is not None:
            T, R, pivots, _ = prepared
            rhs = _numeric_images(T, rhs_rows)
        else:
            nrows = _numeric_rows(list(zip(coeff_rows, rhs_rows)))
            R, pivots = _numeric_rref(nrows, n_cols)
            rhs = [row.get("rhs", 0) for row in R]

        sol = _numeric_solution(rhs, R, pivots, n_cols, gids)
    else:
        if prepared is None and seen > _TRANSFORM_AFTER:
            built = None
            if not (_poly._RELATED and _rows_related(coeff_rows)):
                aug = []
                for i, coeffs in enumerate(coeff_rows):
                    row = dict(coeffs)
                    row[("t", i)] = {(): 1}
                    aug.append(row)

                built = _sparse_rref_factored(aug, n_cols)

            if built is None:
                _seen_keys[key] = _NO_TRANSFORM
            else:
                work, pivots, pivot_polys = built
                prepared = _transform_record(work, pivots, pivot_polys)
                _store(key, prepared)

        if prepared is not None:
            T, R, pivots, pivot_polys = prepared
            rhs = _symbolic_images(T, rhs_rows)
        else:
            aug = []
            for coeffs, rhs in zip(coeff_rows, rhs_rows):
                row = dict(coeffs)
                if rhs:
                    row["rhs"] = rhs

                aug.append(row)

            R, pivots, pivot_polys = _symbolic_rref(aug, n_cols, key if seen else None)
            rhs = [row.get("rhs", _ZERO_RF) for row in R]

        divisors = _divisor_values(pivot_polys) if return_divisors else []
        sol = _symbolic_solution(rhs, R, pivots, n_cols, gids)

    if sol is None:
        return [], divisors
    return (dict(zip(vars_, sol)), divisors)
