# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

import math
import numbers
import re
from collections import namedtuple
from fractions import Fraction

from .._aux.printing.printing._eds import split_conjugation_prefix
from .._aux.printing.printing._string_processing import latex_superscript
from ..eds._atoms import zero_form_atom
from ..eds._constants import E, _BuiltinLeaf
from ..eds._zero_forms import _base_free_symbols, _nf_of, zero_form_class
from . import _poly

_ONE = {(): 1}
_HOOKS = {}
LeafKey = namedtuple("LeafKey", "conjugated kind rank name order args")
NUMBER_RADICAL, CONSTANT, ATOM, FUNCTION, ELEMENTARY, RADICAL, OPAQUE = range(7)
_DIGIT_RUNS = re.compile(r"(\d+)")


class OpaqueLeaf:
    __slots__ = ("base", "label")

    def __init__(self, base):
        self.base = base
        self.label = _opaque_label(base)

    def __eq__(self, other):
        return isinstance(other, OpaqueLeaf) and other.base == self.base

    def __hash__(self):
        return hash(("dgcv.builtin.opaque", self.base))

    def __repr__(self):
        return f"OpaqueLeaf({self.base!r})"

    def __str__(self):
        return self.label


class _CompoundLeaf(_BuiltinLeaf):
    is_constant = False
    _compound = True
    _dgcv_category = "builtin_function"
    _head_kind = 0

    def _kind_bits(self):
        k = self._head_kind
        for a in self.args:
            k |= nf_from_base(a).kinds()
        return k

    @property
    def label(self):
        return str(self)

    @property
    def free_symbols(self):
        out = set()
        for a in self.args:
            out |= _base_free_symbols(a)
        return out

    def _arg_rf(self):
        return nf_from_base(self.args[0])

    def diff(self, *args, **kwargs):
        return zero_form_class(self).diff(*args, **kwargs)

    def _eval_conjugate(self):
        return zero_form_class._from_nf(self.conj_rf())

    def conjugate(self):
        return self._eval_conjugate()

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return self._hash

    def _power_latex(self, e):
        return latex_superscript(self._latex(), str(e))


def _mentions(base, leaf):
    if _is_expr(base):
        base = base.base
    if isinstance(base, tuple):
        return any(_mentions(a, leaf) for a in base[1:])
    if base == leaf:
        return True
    if isinstance(base, _CompoundLeaf):
        return any(_mentions(a, leaf) for a in base.args)
    return False


def _opaque_label(base):
    if isinstance(base, tuple):
        op, *args = base
        return f"{op}({', '.join(_opaque_label(a) for a in args)})"
    return str(base)


def natural_key(label):
    return tuple(
        int(part) if k % 2 else part for k, part in enumerate(_DIGIT_RUNS.split(label))
    )


def _leaf_sort_key(gid):
    leaf = _poly.gen_leaf(gid)

    if isinstance(leaf, zero_form_atom):
        label, conjugated = str(leaf), False
        if not leaf.coframe_derivatives:
            label, conjugated = split_conjugation_prefix(leaf.label)
        return LeafKey(conjugated, ATOM, 0, natural_key(label), 0, "")

    if isinstance(leaf, OpaqueLeaf):
        return LeafKey(False, OPAQUE, 0, natural_key(leaf.label), 0, "")

    if isinstance(leaf, _CompoundLeaf):
        return leaf._sort_key()

    return LeafKey(False, CONSTANT, 0, natural_key(str(leaf)), 0, "")


def _leaf_keys(gens):
    return {g: _leaf_sort_key(g) for g in gens if g != _poly.I_GEN}


class RF:
    __slots__ = ("num", "den", "complete", "has_float", "gens_cache", "partials_cache")

    def __init__(
        self,
        num,
        den=None,
        canonical=False,
        reduced=False,
        complete=True,
        has_float=None,
    ):
        if den is None:
            den = _ONE
        if not canonical:
            num, den, done = _canon(num, den, reduced=reduced)
            if not done:
                complete = False
        self.num = num
        self.den = den
        self.complete = complete
        self.has_float = has_float
        self.gens_cache = None
        self.partials_cache = None

    def retry_reduction(self):
        if self.complete:
            return self
        return RF(self.num, self.den)

    @property
    def is_zero(self):
        return not self.num

    @property
    def is_constant(self):
        return _poly.p_is_const(self.num) and _poly.p_is_const(self.den)

    @property
    def is_polynomial(self):
        return _poly.p_is_const(self.den)

    def constant_value(self):
        n = _poly.p_const_value(self.num)
        d = _poly.p_const_value(self.den)
        if isinstance(n, float) or isinstance(d, float):
            return n / d
        return _fold_fraction(Fraction(n, d))

    @property
    def is_one(self):
        return self.is_constant and self.constant_value() == 1

    @property
    def is_minus_one(self):
        return self.is_constant and self.constant_value() == -1

    def gens(self):
        gs = self.gens_cache
        if gs is None:
            gs = _poly.p_gens(self.num) | _poly.p_gens(self.den)
            self.gens_cache = gs
        return gs

    def kinds(self):
        if _poly._kinds_stale[0]:
            _poly.refresh_kinds()
        k = 0
        kind = _poly.GEN_KIND
        for g in self.gens():
            b = kind[g]
            if b is None:
                b = kind[g] = _poly.leaf_kind(_poly._gen_leaves[g])
            k |= b
        return k

    def _combine(self, other, sign):
        ok = self.complete and other.complete
        if other.den is self.den or other.den == self.den:
            num = (
                _poly.p_add(self.num, other.num)
                if sign > 0
                else _poly.p_sub(self.num, other.num)
            )
            if len(self.den) == 1 and _mono_ready(self, other):
                if not num:
                    return RF({}, canonical=True)
                g, exps = _mono_gcd(num, self.den)
                if g != 1 or exps:
                    return _mono_result(
                        _mono_quot(num, g, exps), _mono_quot(self.den, g, exps), ok
                    )

                return _mono_result(num, self.den, ok)

            return RF(num, self.den, complete=ok)

        if len(self.den) == 1 and len(other.den) == 1 and _mono_ready(self, other):
            return _add_mono(self.num, self.den, other.num, other.den, sign, ok)
        if _poly.p_is_const(self.den) or _poly.p_is_const(other.den):
            a = _poly.p_mul(self.num, other.den)
            b = _poly.p_mul(other.num, self.den)
            return RF(
                _poly.p_add(a, b) if sign > 0 else _poly.p_sub(a, b),
                _poly.p_mul(self.den, other.den),
                reduced=True,
                complete=ok,
            )

        g, complete = _poly.p_gcd(self.den, other.den)
        if not complete:
            ok = False

        if _trivial(g):
            a = _poly.p_mul(self.num, other.den)
            b = _poly.p_mul(other.num, self.den)
            return RF(
                _poly.p_add(a, b) if sign > 0 else _poly.p_sub(a, b),
                _poly.p_mul(self.den, other.den),
                reduced=ok,
                complete=ok,
            )

        d1 = _poly.p_div_exact(self.den, g)
        d2 = _poly.p_div_exact(other.den, g)
        a = _poly.p_mul(self.num, d2)
        b = _poly.p_mul(other.num, d1)
        num = _poly.p_add(a, b) if sign > 0 else _poly.p_sub(a, b)
        if not num:
            return RF({}, canonical=True)
        h, complete = _gcd_with_real(num, g)
        if not complete:
            ok = False

        den = _poly.p_mul(self.den, d2)
        if not _trivial(h):
            qn = _poly.p_div_exact(num, h)
            qd = _poly.p_div_exact(den, h)
            if qn is not None and qd is not None:
                num, den = qn, qd

        return RF(num, den, reduced=ok, complete=ok)

    def __add__(self, other):
        return self._combine(other, 1)

    def __sub__(self, other):
        return self._combine(other, -1)

    def __neg__(self):
        return RF(
            _poly.p_neg(self.num),
            self.den,
            canonical=True,
            complete=self.complete,
            has_float=self.has_float,
        )

    def __mul__(self, other):
        n1, d1, n2, d2 = self.num, self.den, other.num, other.den
        if not n1 or not n2:
            return RF({}, canonical=True)
        if _poly.p_is_const(n2) and _poly.p_is_const(d2):
            c2 = n2[()]
            if d2[()] == 1 and type(c2) is int:
                if c2 == 1:
                    return self
                if c2 == -1:
                    return RF(
                        _poly.p_neg(n1),
                        d1,
                        canonical=True,
                        complete=self.complete,
                        has_float=self.has_float,
                    )

            return _scaled(n1, d1, c2, d2[()], self.complete)

        if _poly.p_is_const(n1) and _poly.p_is_const(d1):
            c1 = n1[()]
            if d1[()] == 1 and type(c1) is int:
                if c1 == 1:
                    return other
                if c1 == -1:
                    return RF(
                        _poly.p_neg(n2),
                        d2,
                        canonical=True,
                        complete=other.complete,
                        has_float=other.has_float,
                    )

            return _scaled(n2, d2, c1, d1[()], other.complete)

        if len(d1) == 1 and len(d2) == 1 and _mono_ready(self, other):
            return _mul_mono(n1, d1, n2, d2, self.complete and other.complete)
        n1, d2, first = _cross_cancel(n1, d2)
        n2, d1, second = _cross_cancel(n2, d1)
        ok = self.complete and other.complete and first and second
        both_i = _poly.I_GEN in self.gens() and _poly.I_GEN in other.gens()
        return RF(
            _poly.p_mul(n1, n2),
            _poly.p_mul(d1, d2),
            reduced=ok and not both_i,
            complete=ok,
        )

    def __truediv__(self, other):
        if not other.num:
            raise ZeroDivisionError("builtin engine: division by zero")
        n1, d1, n2, d2 = self.num, self.den, other.den, other.num
        if not n1:
            return RF({}, canonical=True)
        if _poly.p_is_const(n2) and _poly.p_is_const(d2):
            return _scaled(n1, d1, n2[()], d2[()], self.complete)
        n1, d2, first = _cross_cancel(n1, d2)
        n2, d1, second = _cross_cancel(n2, d1)
        ok = self.complete and other.complete and first and second
        return RF(_poly.p_mul(n1, n2), _poly.p_mul(d1, d2), reduced=ok, complete=ok)

    def __pow__(self, n):
        if type(n) is not int:
            exponent = int(n)
            if exponent != n:
                raise TypeError(
                    f"builtin engine: an RF power takes an integer exponent, received {n!r}"
                )
            n = exponent

        if n == 0:
            return RF(dict(_ONE), canonical=True)
        if n < 0:
            if not self.num:
                raise ZeroDivisionError("builtin engine: division by zero")
            return RF(
                _poly.p_pow(self.den, -n),
                _poly.p_pow(self.num, -n),
                reduced=self.complete,
                complete=self.complete,
            )
        return RF(
            _poly.p_pow(self.num, n),
            _poly.p_pow(self.den, n),
            reduced=self.complete,
            complete=self.complete,
        )

    def equal(self, other):
        d = _poly.p_sub(
            _poly.p_mul(self.num, other.den), _poly.p_mul(other.num, self.den)
        )
        if not d:
            return True
        if _poly._MIXABLE:
            mix = _poly.p_mixture(d)
            if mix:
                return _collapse_poly(d, mix).is_zero
        if _poly._REDUCIBLE and _poly.p_needs_reduction(d):
            return _reduce_poly(d).is_zero
        return False

    def diff(self, gid):
        gens = self.gens()
        target = _poly.gen_leaf(gid)
        total = self._partial(gid) if gid in gens else RF({}, canonical=True)

        for g in gens:
            if g == gid:
                continue
            leaf = _poly.gen_leaf(g)
            if isinstance(leaf, _CompoundLeaf):
                if not _mentions(leaf, target):
                    continue
                inner = RF({}, canonical=True)
                for pk, arg in leaf.partials():
                    d_arg = nf_from_base(arg).diff(gid)
                    if not d_arg.is_zero:
                        inner = inner + pk * d_arg

                if not inner.is_zero:
                    total = total + self._partial(g) * inner
            elif isinstance(leaf, OpaqueLeaf) and _mentions(leaf.base, target):
                raise NotImplementedError(
                    f"dgcv's builtin symbolic engine cannot differentiate through `{leaf.label}`; install sympy or sage and set `default_engine` accordingly"
                )

        return total

    def _partial(self, gid):
        dn = _poly.p_diff(self.num, gid)
        if _poly.p_is_const(self.den):
            return RF(dn, self.den)
        dd = _poly.p_diff(self.den, gid)
        return RF(
            _poly.p_sub(_poly.p_mul(dn, self.den), _poly.p_mul(self.num, dd)),
            _poly.p_mul(self.den, self.den),
        )

    def numer_denom(self):
        return RF(self.num, canonical=True), RF(self.den, canonical=True)

    def __repr__(self):
        return f"RF({self.num!r}, {self.den!r})"


def _trivial(g):
    return len(g) == 1 and g.get(()) == 1


def _has_float(rf):
    fl = rf.has_float
    if fl is None:
        fl = _poly.p_has_float(rf.num) or _poly.p_has_float(rf.den)
        rf.has_float = fl
    return fl


def _mono_ready(a, b):
    if _has_float(a) or _has_float(b):
        return False
    return (
        type(next(iter(a.den.values()))) is int
        and type(next(iter(b.den.values()))) is int
    )


def _mono_result(num, den, ok):
    if _poly._MIXABLE and _poly.p_mixture(num, den):
        return RF(num, den, complete=ok)
    if _poly._REDUCIBLE and (
        _poly.p_needs_reduction(num) or _poly.p_needs_reduction(den)
    ):
        return RF(num, den, complete=ok)
    return RF(num, den, canonical=True, complete=ok, has_float=False)


def _mono_gcd(num, den):
    ((m, c),) = den.items()
    g = c if c > 0 else -c
    exps = dict(m)

    for mm, cc in num.items():
        if g != 1:
            g = math.gcd(g, cc)

        if exps:
            hit = 0
            for gen, e in mm:
                cur = exps.get(gen)
                if cur is not None:
                    hit += 1
                    if e < cur:
                        exps[gen] = e

            if hit != len(exps):
                present = {gen for gen, _ in mm}
                for gen in [x for x in exps if x not in present]:
                    del exps[gen]
        elif g == 1:
            break

    return g, exps


def _mono_quot(p, g, exps):
    out = {}
    if exps:
        for mm, cc in p.items():
            nm = tuple(
                (gen, e - exps[gen]) if gen in exps else (gen, e) for gen, e in mm
            )
            out[tuple(t for t in nm if t[1])] = cc // g
    else:
        for mm, cc in p.items():
            out[mm] = cc // g
    return out


def _mul_mono(n1, d1, n2, d2, ok):
    g, exps = _mono_gcd(n1, d2)
    if g != 1 or exps:
        n1 = _mono_quot(n1, g, exps)
        d2 = _mono_quot(d2, g, exps)
    g, exps = _mono_gcd(n2, d1)
    if g != 1 or exps:
        n2 = _mono_quot(n2, g, exps)
        d1 = _mono_quot(d1, g, exps)
    return _mono_result(_poly.p_mul(n1, n2), _poly.p_mul(d1, d2), ok)


def rf_msum(pairs):
    dens = []
    L = 1
    lexp = {}
    ok = True

    for a, b in pairs:
        if len(a.den) != 1 or len(b.den) != 1 or _has_float(a) or _has_float(b):
            return None
        ((ma, ca),) = a.den.items()
        ((mb, cb),) = b.den.items()
        if type(ca) is not int or type(cb) is not int:
            return None
        ok = ok and a.complete and b.complete
        c = ca * cb
        L = L * c // math.gcd(L, c)
        e = dict(ma)

        for g, x in mb:
            e[g] = e.get(g, 0) + x

        for g, x in e.items():
            if lexp.get(g, 0) < x:
                lexp[g] = x

        dens.append((e, c))

    acc = {}
    for (a, b), (e, c) in zip(pairs, dens):
        if not a.num or not b.num:
            continue
        cof = tuple(
            (g, x - e.get(g, 0)) for g, x in sorted(lexp.items()) if x - e.get(g, 0)
        )
        k = L // c
        an = a.num
        bn = b.num
        if cof or k != 1:
            if len(bn) <= len(an):
                bn = _poly.p_mul(bn, {cof: k})
            else:
                an = _poly.p_mul(an, {cof: k})

        _poly.p_mul_into(acc, an, bn)

    if not acc:
        return RF({}, canonical=True)
    den = {tuple(sorted(lexp.items())): L}
    g, exps = _mono_gcd(acc, den)
    if g != 1 or exps:
        acc = _mono_quot(acc, g, exps)
        den = _mono_quot(den, g, exps)

    return _mono_result(acc, den, ok)


def rf_partial_fast(rf, gid):
    pt = rf.partials_cache
    if pt is None:
        pt = rf.partials_cache = {}
    else:
        out = pt.get(gid)
        if out is not None:
            return out

    den = rf.den
    if den == _ONE or (len(den) == 1 and () in den):
        dn = _poly.p_diff(rf.num, gid)
        if not dn:
            out = RF({}, canonical=True)
        elif rf.has_float is not False and _has_float(rf):
            out = RF(dn, den)
        else:
            out = (
                RF(dn, den, canonical=True, has_float=False)
                if _content_free(dn, den)
                else RF(dn, den)
            )
    elif len(den) == 1 and not _has_float(rf) and type(next(iter(den.values()))) is int:
        out = _partial_mono(rf.num, den, gid, rf.complete)
    else:
        out = rf._partial(gid)

    pt[gid] = out
    return out


def _partial_mono(num, den, gid, ok):
    ((m, c),) = den.items()
    dn = _poly.p_diff(num, gid)
    e = 0

    for g, x in m:
        if g == gid:
            e = x
            break

    if e == 0:
        if not dn:
            return RF({}, canonical=True)
        nn, dd = dn, den
    else:
        nn = _poly.p_sub(_poly.p_mul(dn, {((gid, 1),): 1}), _poly.p_scale(num, e))
        if not nn:
            return RF({}, canonical=True)
        dd = {tuple((g, x + 1) if g == gid else (g, x) for g, x in m): c}

    g, exps = _mono_gcd(nn, dd)
    if g != 1 or exps:
        nn = _mono_quot(nn, g, exps)
        dd = _mono_quot(dd, g, exps)

    return _mono_result(nn, dd, ok)


def _content_free(num, den):
    c = den[()]
    if c == 1:
        return True
    return math.gcd(c, _poly.p_int_content(num)) == 1


def _add_mono(n1, d1, n2, d2, sign, ok):
    ((m1, c1),) = d1.items()
    ((m2, c2),) = d2.items()
    e1 = dict(m1)
    e2 = dict(m2)
    lexp = dict(e1)

    for gen, e in e2.items():
        if lexp.get(gen, 0) < e:
            lexp[gen] = e

    L = c1 * c2 // math.gcd(c1, c2)
    cof1 = tuple((gen, e - e1.get(gen, 0)) for gen, e in sorted(lexp.items()))
    cof1 = tuple(t for t in cof1 if t[1])
    cof2 = tuple((gen, e - e2.get(gen, 0)) for gen, e in sorted(lexp.items()))
    cof2 = tuple(t for t in cof2 if t[1])
    k1 = L // c1
    k2 = L // c2
    a = n1 if not cof1 and k1 == 1 else _poly.p_mul(n1, {cof1: k1})
    b = n2 if not cof2 and k2 == 1 else _poly.p_mul(n2, {cof2: k2})
    num = _poly.p_add(a, b) if sign > 0 else _poly.p_sub(a, b)
    if not num:
        return RF({}, canonical=True)
    den = {tuple(sorted(lexp.items())): L}
    g, exps = _mono_gcd(num, den)
    if g != 1 or exps:
        num = _mono_quot(num, g, exps)
        den = _mono_quot(den, g, exps)

    return _mono_result(num, den, ok)


def _scaled(num, den, cn, cd, ok):
    if (
        isinstance(cn, float)
        or isinstance(cd, float)
        or isinstance(cn, Fraction)
        or isinstance(cd, Fraction)
    ):
        c = (
            Fraction(cn, cd)
            if not isinstance(cn, float) and not isinstance(cd, float)
            else cn / cd
        )
        return RF(_poly.p_scale(num, c), den, reduced=True, complete=ok)

    if cd < 0:
        cn, cd = -cn, -cd

    if cd != 1:
        g = math.gcd(cn, cd)
        if g != 1:
            cn //= g
            cd //= g

    if _poly.p_has_float(num) or _poly.p_has_float(den):
        return RF(_poly.p_scale(num, Fraction(cn, cd)), den, reduced=True, complete=ok)
    num = _poly.p_scale(num, cn)
    if cd != 1:
        den = _poly.p_scale(den, cd)

    g = math.gcd(_poly.p_int_content(num), _poly.p_int_content(den))
    if g != 1:
        num = {m: v // g for m, v in num.items()}
        den = {m: v // g for m, v in den.items()}

    return RF(num, den, canonical=True, complete=ok)


def _gcd_with_real(n, d):
    if _poly.I_GEN in _poly.p_gens(n):
        return _poly.p_gcd_real(n, d)
    return _poly.p_gcd(n, d)


def _cross_cancel(n, d):
    if _poly.p_is_const(n) or _poly.p_is_const(d):
        return n, d, True
    gn = _poly.I_GEN in _poly.p_gens(n)
    gd = _poly.I_GEN in _poly.p_gens(d)
    if gn and not gd:
        g, complete = _poly.p_gcd_real(n, d)
    elif gd and not gn:
        g, complete = _poly.p_gcd_real(d, n)
    else:
        g, complete = _poly.p_gcd(n, d)

    if _trivial(g):
        return n, d, complete
    qn = _poly.p_div_exact(n, g)
    qd = _poly.p_div_exact(d, g)
    if qn is None or qd is None:
        return n, d, complete
    return qn, qd, complete


def _split_i(p):
    return _poly.p_split_i(p)


def _conj_i(p):
    out = {}
    for m, c in p.items():
        for g, e in m:
            if g == _poly.I_GEN and e % 2:
                c = -c
                break
        out[m] = c
    return out


def _fold_fraction(x):
    if isinstance(x, Fraction) and x.denominator == 1:
        return x.numerator
    return x


def _nf(expr):
    if isinstance(expr, zero_form_class):
        return expr._nf
    if isinstance(expr, (zero_form_atom, _BuiltinLeaf)) or (
        isinstance(expr, (numbers.Real, complex)) and not isinstance(expr, bool)
    ):
        return _nf_of(expr)
    return None


def _value_from_poly(num):
    if not num:
        return 0
    if _poly.p_is_const(num):
        return _fold_fraction(Fraction(num[()]))
    return zero_form_class._from_nf(RF(num))


def _value_from_rf(rf):
    if rf.is_zero:
        return 0
    if rf.is_constant:
        return _fold_fraction(rf.constant_value())
    return zero_form_class._from_nf(rf)


def _canon(num, den, reduced=False):
    if not den:
        raise ZeroDivisionError("builtin engine: division by zero")
    if not num:
        return {}, dict(_ONE), True
    if _poly._MIXABLE:
        mix = _poly.p_mixture(num, den)
        if mix:
            rf = _collapse_poly(num, mix) / _collapse_poly(den, mix)
            return rf.num, rf.den, rf.complete

    if _poly._REDUCIBLE and (
        _poly.p_needs_reduction(num) or _poly.p_needs_reduction(den)
    ):
        rf = _reduce_poly(num) / _reduce_poly(den)
        return rf.num, rf.den, rf.complete

    if _poly.p_has_float(num) or _poly.p_has_float(den):
        if _poly.p_leading_sign(den) < 0:
            return _poly.p_neg(num), _poly.p_neg(den), True
        return num, den, True

    if _poly.I_GEN in _poly.p_gens(den):
        bar = _conj_i(den)
        num = _poly.p_mul(num, bar)
        den = _poly.p_mul(den, bar)
        reduced = False

    L = _poly.p_denominator_lcm(num)
    Ld = _poly.p_denominator_lcm(den)
    L = L * Ld // math.gcd(L, Ld)
    if L != 1:
        num = _poly.p_to_int(_poly.p_scale(num, L))
        den = _poly.p_to_int(_poly.p_scale(den, L))
    else:
        num = _poly.p_to_int(num)
        den = _poly.p_to_int(den)

    c = math.gcd(_poly.p_int_content(num), _poly.p_int_content(den))
    if c != 1:
        num = {m: v // c for m, v in num.items()}
        den = {m: v // c for m, v in den.items()}

    complete = True
    if not reduced and not _poly.p_is_const(den):
        g, complete = _gcd_with_real(num, den)
        if not (len(g) == 1 and g.get(()) == 1):
            qn = _poly.p_div_exact(num, g)
            qd = _poly.p_div_exact(den, g)
            if qn is not None and qd is not None:
                num, den = _intify_pair(qn, qd)

    if _poly.p_leading_sign(den) < 0:
        num = _poly.p_neg(num)
        den = _poly.p_neg(den)

    return num, den, complete


def _eval_poly(poly, power, mapped):
    groups = {}
    for m, c in poly.items():
        rest = []
        subs_part = []
        for g, e in m:
            if g in mapped:
                subs_part.append((g, e))
            else:
                rest.append((g, e))

        groups.setdefault(tuple(subs_part), {})[tuple(rest)] = c

    total = RF({}, canonical=True)
    for subs_part, rest_poly in groups.items():
        term = RF(rest_poly)
        for g, e in subs_part:
            term = term * power(g, e)

        total = total + term

    return total


def rf_map_gens(rf, mapping):
    if not mapping:
        return rf
    powers = {}

    def power(g, e):
        key = (g, e)
        val = powers.get(key)
        if val is None:
            val = mapping[g] ** e
            powers[key] = val
        return val

    num = _eval_poly(rf.num, power, mapping)
    if any(g in mapping for g in _poly.p_gens(rf.den)):
        return num / _eval_poly(rf.den, power, mapping)
    return num / RF(rf.den, canonical=True)


def _reduce_poly(poly):
    gens = _poly.p_gens(poly)
    mapped = {g for g in gens if g in _poly._REDUCIBLE}
    if not mapped:
        return RF(poly)
    cache = {}

    def power(g, e):
        thr, factory = _poly.REDUCTIONS[g]
        if e < thr:
            return RF(_poly.p_gen(g, e), canonical=True)
        rep = cache.get(g)
        if rep is None:
            rep = factory()
            cache[g] = rep
        out = rep ** (e // thr)
        if e % thr:
            out = out * RF(_poly.p_gen(g, e % thr), canonical=True)
        return out

    return _eval_poly(poly, power, mapped)


def _collapse_poly(poly, mix):
    cache = {}

    def power(g, e):
        rep = cache.get(g)
        if rep is None:
            named = _poly._NAMED.get(g)
            if named is not None:
                rep = named()
            else:
                key, index, factory = _poly._GRADED[g]
                common = index
                for h in mix:
                    other = _poly._GRADED.get(h)
                    if other is not None and other[0] == key:
                        common = math.lcm(common, other[1])
                rep = factory(common) ** (common // index)
            cache[g] = rep
        return rep if e == 1 else rep**e

    return _eval_poly(poly, power, mix)


def conj_rf(rf):
    mapping = {}
    for g in rf.gens():
        if g == _poly.I_GEN:
            continue
        leaf = _poly.gen_leaf(g)
        if isinstance(leaf, zero_form_atom):
            image = leaf._eval_conjugate()
            if image != leaf:
                mapping[g] = rf_gen(image)
        elif isinstance(leaf, _CompoundLeaf):
            mapping[g] = leaf.conj_rf()
        elif isinstance(leaf, OpaqueLeaf):
            raise NotImplementedError(
                f"dgcv's builtin symbolic engine cannot conjugate through `{leaf.label}`"
            )

    base = RF(_conj_i(rf.num), _conj_i(rf.den), canonical=True)
    return rf_map_gens(base, mapping)


def _intify_pair(p, q):
    L = _poly.p_denominator_lcm(p)
    Lq = _poly.p_denominator_lcm(q)
    L = L * Lq // math.gcd(L, Lq)
    if L != 1:
        p = _poly.p_scale(p, L)
        q = _poly.p_scale(q, L)
    p = _poly.p_to_int(p)
    q = _poly.p_to_int(q)
    c = math.gcd(_poly.p_int_content(p), _poly.p_int_content(q))
    if c != 1:
        p = {m: v // c for m, v in p.items()}
        q = {m: v // c for m, v in q.items()}
    return p, q


def rf_zero():
    return RF({}, canonical=True)


def rf_one():
    return RF(dict(_ONE), canonical=True)


def _not_finite(value):
    return NotImplementedError(f"`{value}` is not a finite number")


def rf_number(x):
    if isinstance(x, bool):
        raise TypeError("builtin engine does not accept booleans as scalars")
    if isinstance(x, (int, Fraction)):
        return RF(_poly.p_const(x))
    if isinstance(x, float):
        if not math.isfinite(x):
            raise _not_finite(x)
        return RF(_poly.p_const(x), canonical=True)
    if isinstance(x, complex):
        re, im = x.real, x.imag
        if not (math.isfinite(re) and math.isfinite(im)):
            raise _not_finite(x)
        if re == int(re):
            re = int(re)

        if im == int(im):
            im = int(im)

        return RF(
            _poly.p_add(_poly.p_const(re), _poly.p_scale(_poly.p_gen(_poly.I_GEN), im))
        )

    if isinstance(x, numbers.Integral):
        return RF(_poly.p_const(int(x)))
    if isinstance(x, numbers.Rational):
        num = x.numerator
        den = x.denominator
        if callable(num):
            num = num()

        if callable(den):
            den = den()

        return RF(_poly.p_const(Fraction(int(num), int(den))))

    if isinstance(x, numbers.Real):
        v = float(x)
        if not math.isfinite(v):
            raise _not_finite(x)
        return RF(_poly.p_const(v), canonical=True)
    raise TypeError(f"builtin engine cannot use {type(x).__name__} as a scalar")


def rf_gen(leaf):
    return RF(_poly.p_gen(_poly.gen_id(leaf)), canonical=True)


def rf_opaque(base):
    return rf_gen(OpaqueLeaf(base))


def _is_expr(x):
    return getattr(x, "_dgcv_category", None) == "abstract_ZF"


def nf_from_base(base):
    if _is_expr(base):
        return base._nf
    if isinstance(base, zero_form_atom):
        return base._nf
    if isinstance(base, _BuiltinLeaf):
        return rf_gen(base)
    if isinstance(base, numbers.Number):
        return rf_number(base)
    if isinstance(base, tuple):
        op, *args = base
        if op == "add":
            out = rf_zero()
            for a in args:
                out = out + nf_from_base(a)

            return out

        if op == "mul":
            out = rf_one()
            for a in args:
                out = out * nf_from_base(a)

            return out

        if op == "sub":
            return nf_from_base(args[0]) - nf_from_base(args[1])
        if op == "div":
            return nf_from_base(args[0]) / nf_from_base(args[1])
        if op == "pow":
            b, e = args
            e_rf = nf_from_base(e)
            if e_rf.is_constant:
                ev = e_rf.constant_value()
                if isinstance(ev, numbers.Integral):
                    return nf_from_base(b) ** int(ev)
                if isinstance(ev, Fraction):
                    b_rf = nf_from_base(b)
                    p, q = ev.numerator, ev.denominator
                    out = _HOOKS["radical"](b_rf, q) ** (p % q)
                    if p // q:
                        out = out * b_rf ** (p // q)

                    return out

            if isinstance(b, _BuiltinLeaf) and b == E:
                return _HOOKS["exp"](e_rf)

            return rf_opaque(("pow", canonical_base(b), canonical_base(e)))

        return rf_opaque((op, *[canonical_base(a) for a in args]))

    return rf_opaque(base)


def canonical_base(x):
    if isinstance(x, (zero_form_atom, _BuiltinLeaf, numbers.Number)):
        return x
    return base_from_nf(nf_from_base(x))


def _leaf_base(gid):
    leaf = _poly.gen_leaf(gid)
    if isinstance(leaf, OpaqueLeaf):
        return leaf.base
    return leaf


def _term_key(m, keys):
    powers = sorted((keys[g], -e) for g, e in m if g != _poly.I_GEN)
    has_i = len(powers) < len(m)
    return (sum(e for _, e in powers), tuple(powers), has_i)


def _term_base(m, c):
    factors = []
    for g, e in m:
        lb = _leaf_base(g)
        factors.append(("pow", lb, e) if e != 1 else lb)
    if not factors:
        return c
    if c == 1:
        return ("mul", *factors) if len(factors) > 1 else factors[0]
    if c == -1:
        return ("mul", -1, *factors)
    return ("mul", c, *factors)


def poly_base(p, scale=1):
    if not p:
        return 0
    keys = _leaf_keys(_poly.p_gens(p))
    terms = sorted(p.items(), key=lambda kv: _term_key(kv[0], keys))
    parts = []

    for m, c in terms:
        if scale != 1:
            c = (
                _fold_fraction(Fraction(c) * scale)
                if not isinstance(c, float)
                else c * scale
            )

        parts.append(_term_base(m, c))

    if len(parts) == 1:
        return parts[0]
    return ("add", *parts)


def base_from_nf(rf):
    if not rf.num:
        return 0
    if _poly.p_is_const(rf.den):
        d = _poly.p_const_value(rf.den)
        if d == 1:
            return poly_base(rf.num)
        if isinstance(d, float):
            return poly_base(rf.num, 1.0 / d)
        return poly_base(rf.num, Fraction(1, d))
    return ("div", poly_base(rf.num), poly_base(rf.den))


def collect_base(rf, gids):
    gids = set(gids)
    groups = {}
    for m, c in rf.num.items():
        key = tuple((g, e) for g, e in m if g in gids)
        rest = tuple((g, e) for g, e in m if g not in gids)
        groups.setdefault(key, {})[rest] = c

    keys = _leaf_keys(gids)
    parts = []
    for key in sorted(groups, key=lambda m: _term_key(m, keys)):
        coeff = base_from_nf(RF(groups[key], rf.den))
        if key:
            factors = [
                ("pow", _leaf_base(g), e) if e != 1 else _leaf_base(g) for g, e in key
            ]
            parts.append(("mul", coeff, *factors))
        else:
            parts.append(coeff)

    if not parts:
        return 0
    if len(parts) == 1:
        return parts[0]
    return ("add", *parts)


def poly_terms(rf, gids):
    if not _poly.p_is_const(rf.den):
        return None
    gids = list(gids)
    index = {g: i for i, g in enumerate(gids)}
    groups = {}

    for m, c in rf.num.items():
        exps = [0] * len(gids)
        rest = []
        for g, e in m:
            if g in index:
                exps[index[g]] = e
            else:
                rest.append((g, e))

        groups.setdefault(tuple(exps), {})[tuple(rest)] = c

    out = []
    for exps in sorted(groups, key=lambda t: (-sum(t), tuple(-e for e in t))):
        out.append((exps, RF(groups[exps], rf.den)))

    return out
