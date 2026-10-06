# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

import math
from fractions import Fraction

from ..eds._zero_forms import zero_form_class
from . import _poly
from ._latex import _NFPrinter
from ._normal_form import RF, _term_key, nf_from_base
from ._poly import _prim

_BINOMIAL_MAX_ROOT = 6


def _squarefree_decomposition(p, gens):
    out = []
    rest = p
    k = 1

    while not _poly.p_is_const(rest):
        common = None
        for g in gens:
            d = _poly.p_diff(rest, g)
            if not d:
                continue
            if common is None:
                common = d
            else:
                common, complete = _poly.p_gcd(common, d)
                if not complete:
                    return None

            if _poly.p_is_const(common):
                break

        if common is None or _poly.p_is_const(common):
            out.append((rest, k))
            return out

        h, complete = _poly.p_gcd(rest, common)
        if not complete:
            return None
        if _poly.p_is_const(h):
            out.append((rest, k))
            return out

        sqf = _poly.p_div_exact(rest, h)
        if sqf is None:
            return None

        out.append((sqf, k))
        rest = h
        k += 1

    return out


def _exact_multiplicities(parts):
    factors = []
    for i, (poly, mult) in enumerate(parts):
        if i + 1 < len(parts):
            poly = _poly.p_div_exact(poly, parts[i + 1][0])
            if poly is None:
                return None

        if not _poly.p_is_const(poly):
            factors.append((poly, mult))

    return factors


def _univariate_split(poly, g):
    deg = _poly.p_degree_in(poly, g)
    coeffs = [0] * (deg + 1)
    for m, v in poly.items():
        e = m[0][1] if m else 0
        coeffs[e] = v

    roots, rest = _poly.univariate_rational_roots(coeffs)
    factors = []
    for r in sorted(roots):
        lin = {((g, 1),): r.denominator}
        if r.numerator:
            lin[()] = -r.numerator

        factors.append((lin, roots[r]))

    if len(rest) > 1:
        rem = {}
        L = 1
        for c in rest:
            L = L * c.denominator // math.gcd(L, c.denominator)

        for e, v in enumerate(rest):
            if v:
                rem[((g, e),) if e else ()] = int(v * L)

        factors.append((rem, 1))

    return factors


def _integer_root(a, n):
    if a < 2:
        return a

    r = 1 << -(-a.bit_length() // n)
    while True:
        s = ((n - 1) * r + a // r ** (n - 1)) // n
        if s >= r:
            return r
        r = s


def _binomial_split(poly):
    if len(poly) != 2:
        return None
    (m1, c1), (m2, c2) = poly.items()
    if c1 < 0:
        (m1, c1), (m2, c2) = (m2, c2), (m1, c1)

    if c2 >= 0:
        return None
    gexp = 0
    for _g, e in m1 + m2:
        gexp = math.gcd(gexp, e)

    a = abs(c1)
    b = abs(c2)
    for n in range(min(gexp, _BINOMIAL_MAX_ROOT), 1, -1):
        ra = _integer_root(a, n)
        rb = _integer_root(b, n)
        if ra**n == a and rb**n == b and gexp % n == 0:
            u = {tuple((g, e // n) for g, e in m1): ra}
            v = {tuple((g, e // n) for g, e in m2): rb}
            if n % 2 == 0 and n > 2:
                continue
            if n == 2:
                return [(_poly.p_sub(u, v), 1), (_poly.p_add(u, v), 1)]
            s = _poly.p_sub(u, v)
            quot = _poly.p_div_exact(poly, s)
            if quot is not None:
                return [(s, 1), (quot, 1)]

    return None


def _split_factors(p, gens):
    parts = _squarefree_decomposition(p, gens)
    if parts is not None:
        parts = _exact_multiplicities(parts)

    if parts is None:
        return [(p, 1)]
    factors = []
    for sqf, mult in parts:
        pieces = [sqf]
        if len(gens) == 1 and not _poly._holds_i(sqf):
            pieces = [f for f, _ in _univariate_split(sqf, gens[0])]
        else:
            split = _binomial_split(sqf)
            if split is not None:
                pieces = [f for f, _ in split]

        for piece in pieces:
            piece, _ = _prim(piece)
            if not _poly.p_is_const(piece):
                factors.append((piece, mult))

    return factors


def factor_poly(p):
    if not p:
        return 0, []
    p, c = _prim(p)
    factors = []
    mono = _poly.p_monomial_content(p)
    unit = 0

    for g, e in mono:
        if g == _poly.I_GEN:
            unit = e % 4
        else:
            factors.append((_poly.p_gen(g), e))

    if mono:
        p = _poly.p_divide_monomial(p, mono)

    if unit >= 2:
        c = -c
        unit -= 2

    if unit:
        factors.append((_poly.p_gen(_poly.I_GEN), 1))

    gens = sorted(g for g in _poly.p_gens(p) if g != _poly.I_GEN)
    if not gens or _poly._holds_i(p):
        if not _poly.p_is_const(p):
            factors.append((p, 1))

        return c, factors

    factors.extend(_split_factors(p, gens))
    return c, factors


def _factor_tree(const, num_factors, den_factors):
    parts = []
    if const != 1:
        parts.append(const)

    for poly, e in num_factors:
        zf = zero_form_class._from_nf(RF(poly))
        parts.append(("pow", zf, e) if e != 1 else zf)

    if not parts:
        num = 1
    elif len(parts) == 1:
        num = parts[0]
    else:
        num = ("mul", *parts)

    if not den_factors:
        return num
    dparts = []
    for poly, e in den_factors:
        zf = zero_form_class._from_nf(RF(poly))
        dparts.append(("pow", zf, e) if e != 1 else zf)

    den = dparts[0] if len(dparts) == 1 else ("mul", *dparts)
    return ("div", num, den)


def _needs_secondary(factors, gens):
    for poly, _e in factors:
        if len(poly) > 1 and (len(gens) > 1 or _poly.p_degree_in(poly, gens[0]) > 1):
            return True
    return False


def _secondary_factor_list(poly):
    from ._secondary import SecondaryBudgetExpired, secondary_kind, with_secondary

    if secondary_kind() is None:
        return None
    zf = zero_form_class._from_nf(RF(poly))

    def fn(mod, lowered, bridge):
        (e,) = lowered
        if bridge.kind == "sympy":
            const, pairs = mod.factor_list(e)
        else:
            pairs = list(e.factor_list())
            const = 1
        return [(const, 1)] + [(f, int(k)) for f, k in pairs]

    try:
        lifted = with_secondary([zf], fn, feature="factor")
    except (NotImplementedError, SecondaryBudgetExpired):
        return None
    out = []
    for f, k in lifted:
        rf = nf_from_base(f)
        if not rf.is_polynomial:
            return None
        out.append((rf.num, k))
    return out


def _refine(const, factors, try_hard):
    if not try_hard:
        return const, factors
    gens = sorted(
        g for poly, _e in factors for g in _poly.p_gens(poly) if g != _poly.I_GEN
    )
    if not gens or not _needs_secondary(factors, gens):
        return const, factors
    refined = []
    for poly, e in factors:
        pgens = sorted(g for g in _poly.p_gens(poly) if g != _poly.I_GEN)
        if len(poly) > 1 and (len(pgens) > 1 or _poly.p_degree_in(poly, pgens[0]) > 1):
            pieces = _secondary_factor_list(poly)
            if pieces is None:
                refined.append((poly, e))
                continue

            for piece, k in pieces:
                if _poly.p_is_const(piece):
                    const = const * Fraction(piece.get((), 1)) ** (k * e)
                    continue

                cc = _poly.p_int_content(piece)
                if cc != 1:
                    piece = {m: v // cc for m, v in piece.items()}
                    const = const * Fraction(cc) ** (k * e)

                if _poly.p_leading_sign(piece) < 0:
                    piece = _poly.p_neg(piece)
                    if (k * e) % 2:
                        const = -const

                refined.append((piece, k * e))
        else:
            refined.append((poly, e))

    return const, refined


def factor_zf(zf, try_hard=False):
    rf = zf._nf
    if rf.is_zero or rf.is_constant:
        return zf
    if _poly.p_has_float(rf.num) or _poly.p_has_float(rf.den):
        return zf
    cn, nfac = factor_poly(rf.num)
    cd, dfac = factor_poly(rf.den)
    cn, nfac = _refine(Fraction(cn), nfac, try_hard)
    cd, dfac = _refine(Fraction(cd), dfac, try_hard)
    const = cn / cd
    return zero_form_class._from_factored(rf, (const, nfac, dfac))


def factored_base(factored):
    const, nfac, dfac = factored
    num = _factor_tree(const.numerator, nfac, [])
    if dfac or const.denominator != 1:
        return ("div", num, _factor_tree(const.denominator, dfac, []))
    return num


def factored_latex(const, num_factors, den_factors):
    gens = set()
    for poly, _e in (*num_factors, *den_factors):
        gens |= _poly.p_gens(poly)

    printer = _NFPrinter(gens)
    keys = printer.keys

    def order(item):
        poly, e = item
        terms = sorted(poly.items(), key=lambda kv: _term_key(kv[0], keys))
        term_keys = tuple(_term_key(m, keys) for m, _c in terms)
        coeffs = tuple(c for _m, c in terms)
        return (len(poly) > 1, -term_keys[0][0], term_keys, coeffs, e)

    def render(poly, e, alone):
        if len(poly) == 1:
            ((exps, re_, im),) = printer.grouped_terms(poly)
            coeff = abs(re_ or im) ** e
            return printer.mono_tex(
                tuple(v * e for v in exps),
                None if coeff == 1 else str(coeff),
                bool(im),
                None,
            )

        tex = printer.poly_tex(poly)
        if e == 1 and alone:
            return tex
        tex = f"\\left({tex}\\right)"
        if e == 1:
            return tex
        return f"{tex}^{e}" if e < 10 else f"{tex}^{{{e}}}"

    def group(factors, coeff):
        factors = sorted(factors, key=order)
        parts = []
        if coeff != 1:
            parts.append(str(coeff))
        alone = len(factors) == 1 and coeff == 1
        for poly, e in factors:
            parts.append(render(poly, e, alone))
        return " ".join(parts) if parts else "1"

    const = Fraction(const)
    sign = "- " if const < 0 else ""
    num = group(num_factors, abs(const.numerator))
    if den_factors or const.denominator != 1:
        den = group(den_factors, const.denominator)
        return f"{sign}\\frac{{{num}}}{{{den}}}"
    return sign + num
