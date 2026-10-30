# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

import math
import numbers
from fractions import Fraction

from .._aux._backends._polynomials import PolyBackendError
from ..eds._zero_forms import _as_leaf, zero_form_class
from . import _factor, _poly
from ._normal_form import RF, _nf, nf_from_base
from ._normal_form import poly_terms as _rf_terms
from ._poly import _prim


def poly_terms(raw, gens):
    gens_t = tuple(gens)
    leaves = []
    for g in gens_t:
        leaf = _as_leaf(g)
        if leaf is None:
            raise PolyBackendError(
                f"dgcv: builtin polynomial generators must be zero-form atoms, received {type(g).__name__}"
            )

        leaves.append(leaf)

    rf = nf_from_base(raw)
    terms = _rf_terms(rf, [_poly.gen_id(leaf) for leaf in leaves])
    if terms is None:
        raise PolyBackendError(
            "dgcv: the builtin engine received a non-polynomial expression for polynomial term extraction"
        )

    monoms = [tuple(int(e) for e in exps) for exps, _ in terms]
    coeffs = []
    for _, c in terms:
        zf = zero_form_class._from_nf(c)
        if _poly.p_is_const(c.den) and (
            _poly.p_is_const(c.num)
            or (
                len(c.num) == 1
                and len(next(iter(c.num))) == 1
                and next(iter(c.num))[0][1] == 1
                and next(iter(c.num.values())) == 1
                and c.den == {(): 1}
            )
        ):
            base = zf.base
            coeffs.append(base if not isinstance(base, tuple) else zf)
        else:
            coeffs.append(zf)

    return gens_t, monoms, coeffs


def complex_poly_terms(expr, gens):
    rf = _nf(expr)
    if rf is None or not _poly.p_is_const(rf.den):
        return None
    den = rf.den.get((), 1)
    if not isinstance(den, (int, Fraction)):
        return None
    index = {}
    for i, g in enumerate(gens):
        leaf = _as_leaf(g)
        if leaf is None:
            return None
        index[_poly.gen_id(leaf)] = i
    n = len(index)
    real = {}
    imag = {}
    for m, c in rf.num.items():
        if not isinstance(c, (int, Fraction)):
            return None
        exps = [0] * n
        has_i = False
        for g, e in m:
            if g == _poly.I_GEN:
                has_i = True
            else:
                i = index.get(g)
                if i is None:
                    return None
                exps[i] = e
        (imag if has_i else real)[tuple(exps)] = Fraction(c) / den
    return real, imag


def linear_form(pairs):
    pairs = list(pairs)
    den = 1
    for c, _u in pairs:
        if not isinstance(c, (int, Fraction)):
            return None
        den = den * c.denominator // math.gcd(den, c.denominator) if isinstance(c, Fraction) else den
    num = {}
    for c, u in pairs:
        leaf = _as_leaf(u)
        if leaf is None:
            return None
        key = ((_poly.gen_id(leaf), 1),)
        num[key] = num.get(key, 0) + int(c * den)
    num = {m: v for m, v in num.items() if v}
    return zero_form_class._from_nf(RF(num, {(): den}))


_FACTOR_TERM_LIMIT = 200


def primitive_factors(expr, try_hard=False):
    if isinstance(expr, numbers.Number):
        return []
    rf = _nf(expr)
    if rf is None:
        return None
    p = rf.num
    if not p or _poly.p_is_const(p) or _poly.p_has_float(p):
        return []
    mono = _poly.p_monomial_content(p)
    factors = [_poly.p_gen(g) for g, _e in mono if g != _poly.I_GEN]
    q, _ = _prim(_poly.p_divide_monomial(p, mono) if mono else p)
    gens = sorted(_poly.p_gens(q) - {_poly.I_GEN})
    if len(q) > _FACTOR_TERM_LIMIT:
        factors.append(q)
    elif gens:
        if _poly._holds_i(q):
            q, _ = _prim(_poly._gaussian_primitive(q)[0])

        pieces = _factor._split_factors(q, gens)
        if try_hard:
            pieces = _factor._refine(1, pieces, True)[1]
        factors.extend(f for f, _ in pieces)

    return [zero_form_class._from_nf(RF(f)) for f in factors]
