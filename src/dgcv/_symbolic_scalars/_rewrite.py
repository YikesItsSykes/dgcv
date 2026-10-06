# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

import math
from fractions import Fraction

from .._aux._backends._types_and_constants import to_active_engine
from ..eds._constants import I
from ..eds._zero_forms import _nf_of, zero_form_class
from . import _poly
from ._functions import (
    _CIRCULAR,
    _HYPERBOLIC,
    ElementaryFunction,
    Radical,
    elementary,
    radical,
)
from ._normal_form import (
    RF,
    _value_from_rf,
    rf_gen,
    rf_map_gens,
    rf_number,
    rf_one,
    rf_zero,
)

_ANGLE_EXPANSION_LIMIT = 16
_ANGLE_KINDS = _poly.K_CIRCULAR | _poly.K_HYPERBOLIC
_TIER_KINDS = _ANGLE_KINDS | _poly.K_EXP


def _transform(rf, rule):
    mapping = {}
    for g in rf.gens():
        leaf = _poly.gen_leaf(g)
        if isinstance(leaf, ElementaryFunction):
            old = leaf._arg_rf()
            arg = _transform(old, rule)
            out = rule(leaf.name, arg)
            if out is None and arg is not old:
                out = elementary(leaf.name, arg)
        elif isinstance(leaf, Radical):
            old = leaf._arg_rf()
            arg = _transform(old, rule)
            out = radical(arg, leaf.q) if arg is not old else None
        else:
            continue
        if out is not None:
            mapping[g] = out
    return rf_map_gens(rf, mapping) if mapping else rf


def _quotients(k, s, c):
    if k == 0:
        return s
    if k == 1:
        return c
    if k == 2:
        return s / c
    if k == 3:
        return c / s
    if k == 4:
        return rf_one() / c
    return rf_one() / s


def _exp_rule(name, arg):
    if name in _CIRCULAR:
        u = elementary("exp", arg * rf_gen(I))
        v = rf_one() / u
        s = (v - u) * rf_gen(I) * rf_number(Fraction(1, 2))
        c = (u + v) * rf_number(Fraction(1, 2))
        return _quotients(_CIRCULAR.index(name), s, c)
    if name in _HYPERBOLIC:
        u = elementary("exp", arg)
        v = rf_one() / u
        s = (u - v) * rf_number(Fraction(1, 2))
        c = (u + v) * rf_number(Fraction(1, 2))
        return _quotients(_HYPERBOLIC.index(name), s, c)
    return None


def _trig_rule(name, arg):
    if name != "exp" or not arg.num:
        return None
    if _poly.p_has_float(arg.num) or _poly.p_has_float(arg.den):
        return None
    if all(m and m[0][0] == _poly.I_GEN for m in arg.num):
        b = -(arg * rf_gen(I))
        return elementary("cos", b) + rf_gen(I) * elementary("sin", b)
    return elementary("cosh", arg) + elementary("sinh", arg)


def _angle_terms(arg):
    if not arg.num or _poly.p_has_float(arg.num) or _poly.p_has_float(arg.den):
        return None
    cd = _poly.p_int_content(arg.den)
    den = {m: v // cd for m, v in arg.den.items()} if cd != 1 else arg.den
    shape = tuple(sorted(den.items()))
    return den, [(m, shape, Fraction(c, cd)) for m, c in arg.num.items()]


def _collect_bases(rf, table):
    for g in rf.gens():
        leaf = _poly.gen_leaf(g)
        if isinstance(leaf, (ElementaryFunction, Radical)):
            arg = leaf._arg_rf()
            _collect_bases(arg, table)
            if isinstance(leaf, ElementaryFunction) and (
                leaf.name in _CIRCULAR or leaf.name in _HYPERBOLIC
            ):
                terms = _angle_terms(arg)
                if terms is not None:
                    family = leaf.name in _CIRCULAR
                    for m, shape, c in terms[1]:
                        key = (family, m, shape)
                        table[key] = math.lcm(table.get(key, 1), c.denominator)


def _pair_product(circular, p, q):
    c1, s1 = p
    c2, s2 = q
    if circular:
        return c1 * c2 - s1 * s2, s1 * c2 + c1 * s2
    return c1 * c2 + s1 * s2, s1 * c2 + c1 * s2


def _pair_power(circular, p, n):
    if n < 0:
        p = (p[0], -p[1])
        n = -n
    out = None
    while n:
        if n & 1:
            out = p if out is None else _pair_product(circular, out, p)
        n >>= 1
        if n:
            p = _pair_product(circular, p, p)
    return out


def _angle_rule(table):
    def rule(name, arg):
        circular = name in _CIRCULAR
        if not circular and name not in _HYPERBOLIC:
            return None
        terms = _angle_terms(arg)
        if terms is None:
            return None
        den, terms = terms
        multiples = []
        for m, shape, c in terms:
            scale = table.get((circular, m, shape), c.denominator)
            scale = math.lcm(scale, c.denominator)
            multiples.append((m, scale, int(c * scale)))
        if len(multiples) == 1 and multiples[0][2] == 1:
            return None
        if sum(abs(n) for _, _, n in multiples) > _ANGLE_EXPANSION_LIMIT:
            return None
        names = ("cos", "sin") if circular else ("cosh", "sinh")
        den_rf = RF(den, canonical=True)
        total = None
        for m, scale, n in multiples:
            b = RF({m: 1}, canonical=True) / (den_rf * rf_number(scale))
            p = _pair_power(
                circular, (elementary(names[0], b), elementary(names[1], b)), n
            )
            total = p if total is None else _pair_product(circular, total, p)
        c, s = total
        table_of_names = _CIRCULAR if circular else _HYPERBOLIC
        return _quotients(table_of_names.index(name), s, c)

    return rule


def _to_rf(x):
    return _nf_of(to_active_engine(x))


def to_exp_rf(rf):
    return _transform(rf, _exp_rule)


def from_exp_rf(rf):
    return _transform(rf, _trig_rule)


def expand_angles_rf(rf):
    table = {}
    _collect_bases(rf, table)
    return _transform(rf, _angle_rule(table))


def to_exp(expr):
    return _value_from_rf(to_exp_rf(_to_rf(expr)))


def from_exp(expr):
    return _value_from_rf(from_exp_rf(_to_rf(expr)))


def expand_angles(expr):
    return _value_from_rf(expand_angles_rf(_to_rf(expr)))


def zero_by_exp(expr):
    return to_exp_rf(_to_rf(expr)).is_zero


def _size(rf):
    return len(rf.num) + len(rf.den)


def simplify_tier(zf):
    rf = zf._nf
    kinds = rf.kinds()
    if not kinds & _TIER_KINDS:
        return None
    best = None
    try:
        if kinds & _ANGLE_KINDS:
            if to_exp_rf(rf).is_zero:
                return zero_form_class._from_nf(rf_zero())
            alt = expand_angles_rf(rf)
            if _size(alt) < _size(rf):
                best = alt
        if kinds & _poly.K_EXP and (best is None or best.num):
            alt = expand_angles_rf(from_exp_rf(rf))
            if _size(alt) < _size(rf if best is None else best):
                best = alt
    except (ZeroDivisionError, NotImplementedError):
        return None
    if best is None:
        return None
    return zero_form_class._from_nf(best)
