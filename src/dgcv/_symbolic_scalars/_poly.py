# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

import math
import random
from fractions import Fraction

from ..eds._constants import I

I_GEN = 0
K_IMAGINARY = 1
K_CONJUGATED = 2
K_EXP = 4
K_LOG = 8
K_CIRCULAR = 16
K_HYPERBOLIC = 32
K_INVERSE_CIRCULAR = 64
K_INVERSE_HYPERBOLIC = 128
K_RADICAL = 256
K_FUNCTION = 512
_gen_ids = {I: I_GEN}
_gen_leaves = [I]
GEN_KIND = [K_IMAGINARY]
_GCD_TERM_BUDGET = 200000
TRIAL_DIVISION_LIMIT = 10**12
REDUCTIONS = {}
_REDUCIBLE = set()
GEN_GROUP = {}
_NAMED = {}
_GRADED = {}
_MIXABLE = set()
_RELATED = set()
_graded_members = {}
_group_ids = {}


_kinds_stale = [False]


def leaf_kind(leaf):
    return 0


def refresh_kinds():
    _kinds_stale[0] = False
    GEN_KIND[1:] = [None] * (len(GEN_KIND) - 1)


def register_reduction(gid, threshold, factory):
    REDUCTIONS[gid] = (threshold, factory)
    _REDUCIBLE.add(gid)
    _RELATED.add(gid)


def register_family(gid, family, arg, factory=None):
    key = (family, arg)
    grp = _group_ids.get(key)
    if grp is None:
        grp = _group_ids[key] = len(_group_ids)
    GEN_GROUP[gid] = grp
    if factory is not None:
        _NAMED[gid] = factory
        _MIXABLE.add(gid)
        _RELATED.add(gid)


def register_graded(gid, key, index, factory):
    if gid in _GRADED:
        return
    _GRADED[gid] = (key, index, factory)
    members = _graded_members.setdefault(key, [])
    members.append(gid)
    if len(members) > 1:
        key = ("graded", key)
        grp = _group_ids.get(key)
        if grp is None:
            grp = _group_ids[key] = len(_group_ids)
        for g in members:
            GEN_GROUP[g] = grp
            _MIXABLE.add(g)
            _RELATED.add(g)


def p_mixture(num, den=None):
    grp = GEN_GROUP
    groups = None
    for p in (num, den) if den else (num,):
        for m in p:
            for g, _ in m:
                k = grp.get(g)
                if k is not None:
                    if groups is None:
                        groups = {k: {g}}
                    else:
                        s = groups.get(k)
                        if s is None:
                            groups[k] = {g}
                        else:
                            s.add(g)

    if groups is None:
        return None
    out = None
    for s in groups.values():
        if len(s) > 1:
            for g in s:
                if g in _MIXABLE:
                    if out is None:
                        out = set()

                    out.add(g)

    return out


def has_relation(gens):
    return not _RELATED.isdisjoint(gens)


def p_needs_reduction(p):
    for m in p:
        for g, e in m:
            if g in _REDUCIBLE and e >= REDUCTIONS[g][0]:
                return True
    return False


class GCDFailure(Exception):
    pass


_GCD_WORK_BUDGET = 2000000
_gcd_depth = 0
_gcd_budget = 0


def _gcd_charge(units):
    global _gcd_budget
    if _gcd_depth:
        _gcd_budget -= units
        if _gcd_budget < 0:
            raise GCDFailure


def gen_id(leaf):
    gid = _gen_ids.get(leaf)
    if gid is None:
        gid = len(_gen_leaves)
        _gen_ids[leaf] = gid
        _gen_leaves.append(leaf)
        GEN_KIND.append(None)
    elif _gen_leaves[gid] is not leaf and getattr(leaf, "coframe", None) is not None:
        del _gen_ids[leaf]
        _gen_ids[leaf] = gid
        _gen_leaves[gid] = leaf
    return gid


def gen_leaf(gid):
    return _gen_leaves[gid]


def p_const(c):
    return {(): c} if c else {}


def p_gen(gid, exp=1):
    return {((gid, exp),): 1}


def p_is_const(p):
    return not p or (len(p) == 1 and () in p)


def p_const_value(p):
    return p.get((), 0)


def p_gens(p):
    out = set()
    for m in p:
        for g, _ in m:
            out.add(g)
    return out


def p_degree_in(p, gid):
    best = 0
    for m in p:
        for g, e in m:
            if g == gid and e > best:
                best = e
    return best


def p_add(p, q):
    if not p:
        return dict(q)
    if not q:
        return dict(p)
    out = dict(p)
    for m, c in q.items():
        s = out.get(m)
        if s is None:
            out[m] = c
        else:
            s = s + c
            if s:
                out[m] = s
            else:
                del out[m]

    return out


def p_neg(p):
    return {m: -c for m, c in p.items()}


def p_sub(p, q):
    return p_add(p, p_neg(q))


def p_scale(p, c):
    if not c:
        return {}
    return {m: v * c for m, v in p.items()}


def _mono_merge(a, b):
    la = len(a)
    lb = len(b)
    res = []
    i = 0
    j = 0

    while i < la and j < lb:
        ga, ea = a[i]
        gb, eb = b[j]
        if ga == gb:
            res.append((ga, ea + eb))
            i += 1
            j += 1
        elif ga < gb:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1

    if i < la:
        res.extend(a[i:])
    elif j < lb:
        res.extend(b[j:])

    return res


def _mono_mul(a, b):
    if not a:
        return b, 1
    if not b:
        return a, 1
    res = _mono_merge(a, b)
    g0, ei = res[0]
    if g0 == I_GEN and ei >= 2:
        sign = -1 if (ei // 2) % 2 else 1
        if ei % 2:
            res[0] = (I_GEN, 1)
        else:
            del res[0]

        return tuple(res), sign

    return tuple(res), 1


def _mono_mul_raw(a, b):
    if not a:
        return b
    if not b:
        return a
    return tuple(_mono_merge(a, b))


def p_mul(p, q, reduce_i=True):
    if not p or not q:
        return {}
    out = {}
    if reduce_i:
        for ma, ca in p.items():
            for mb, cb in q.items():
                m, sign = _mono_mul(ma, mb)
                c = ca * cb if sign == 1 else -(ca * cb)
                s = out.get(m)
                if s is None:
                    out[m] = c
                else:
                    s = s + c
                    if s:
                        out[m] = s
                    else:
                        del out[m]
    else:
        for ma, ca in p.items():
            for mb, cb in q.items():
                m = _mono_mul_raw(ma, mb)
                c = ca * cb
                s = out.get(m)
                if s is None:
                    out[m] = c
                else:
                    s = s + c
                    if s:
                        out[m] = s
                    else:
                        del out[m]

    return out


def p_mul_into(out, p, q):
    if not p or not q:
        return out
    for ma, ca in p.items():
        for mb, cb in q.items():
            m, sign = _mono_mul(ma, mb)
            c = ca * cb if sign == 1 else -(ca * cb)
            s = out.get(m)
            if s is None:
                out[m] = c
            else:
                s = s + c
                if s:
                    out[m] = s
                else:
                    del out[m]

    return out


def p_pow(p, n):
    if n < 0:
        raise ValueError("p_pow expects a non-negative exponent")
    result = {(): 1}
    base = p
    while n:
        if n & 1:
            result = p_mul(result, base)
        n >>= 1
        if n:
            base = p_mul(base, base)
    return result


def p_diff(p, gid):
    out = {}
    for m, c in p.items():
        for idx, (g, e) in enumerate(m):
            if g == gid:
                if e == 1:
                    nm = m[:idx] + m[idx + 1 :]
                else:
                    nm = m[:idx] + ((g, e - 1),) + m[idx + 1 :]

                v = c * e
                s = out.get(nm)
                if s is None:
                    out[nm] = v
                else:
                    s = s + v
                    if s:
                        out[nm] = s
                    else:
                        del out[nm]

                break

    return out


def p_has_float(p):
    return any(isinstance(c, float) for c in p.values())


def p_denominator_lcm(p):
    L = 1
    for c in p.values():
        if isinstance(c, Fraction):
            L = L * c.denominator // math.gcd(L, c.denominator)
    return L


def p_int_content(p):
    g = 0
    for c in p.values():
        g = math.gcd(g, int(c))
        if g == 1:
            return 1
    return g


def p_to_int(p):
    return {m: int(c) for m, c in p.items()}


def mono_key(m):
    return (sum(e for _, e in m), tuple((-g, e) for g, e in m))


def p_leading_monomial(p):
    return max(p, key=mono_key)


def p_leading_sign(p):
    if not p:
        return 1
    return 1 if p[p_leading_monomial(p)] > 0 else -1


def _prim(poly):
    c = p_int_content(poly)
    if c != 1:
        poly = {m: v // c for m, v in poly.items()}
    if p_leading_sign(poly) < 0:
        poly = p_neg(poly)
        c = -c
    return poly, c


def p_monomial_content(p):
    if not p:
        return ()
    common = None
    for m in p:
        d = dict(m)
        if common is None:
            common = d
        else:
            common = {g: min(e, d[g]) for g, e in common.items() if g in d}
        if not common:
            return ()
    return tuple(sorted(common.items()))


def p_divide_monomial(p, mono):
    if not mono:
        return dict(p)
    sub = dict(mono)
    out = {}
    for m, c in p.items():
        d = dict(m)
        for g, e in sub.items():
            r = d[g] - e
            if r:
                d[g] = r
            else:
                del d[g]

        out[tuple(sorted(d.items()))] = c

    return out


def _split_by_gen(p, gid):
    out = {}
    for m, c in p.items():
        e = 0
        rest = m
        for idx, (g, ex) in enumerate(m):
            if g == gid:
                e = ex
                rest = m[:idx] + m[idx + 1 :]
                break
        out.setdefault(e, {})[rest] = c
    return out


def _join_by_gen(parts, gid):
    out = {}
    for e, poly in parts.items():
        for m, c in poly.items():
            if e:
                d = dict(m)
                d[gid] = e
                m = tuple(sorted(d.items()))
            out[m] = c
    return out


def _const_div(a, b):
    if isinstance(a, int) and isinstance(b, int):
        q, r = divmod(a, b)
        if r:
            return Fraction(a, b)
        return q
    q = a / b
    if isinstance(q, Fraction) and q.denominator == 1:
        return q.numerator
    return q


def p_div_exact(p, g):
    if not g:
        raise ZeroDivisionError("builtin engine: division by zero")
    _gcd_charge(len(p))
    if not p:
        return {}
    gens = p_gens(g)
    if not gens:
        lc = g[()]
        return {m: _const_div(c, lc) for m, c in p.items()}
    if len(gens) == 1 and I_GEN in gens:
        return _gaussian_div_exact(p, g)

    reduce_i = I_GEN in gens
    x = max(gens)
    G = _split_by_gen(g, x)
    dg = max(G)
    lc_g = G[dg]
    rem = _split_by_gen(p, x)
    quot = {}

    while rem:
        dp = max(rem)
        if dp < dg:
            return None
        c = p_div_exact(rem[dp], lc_g)
        if c is None:
            return None
        quot[dp - dg] = c
        for j, gj in G.items():
            k = dp - dg + j
            r = p_sub(rem.get(k, {}), p_mul(c, gj, reduce_i=reduce_i))
            if r:
                rem[k] = r
            else:
                rem.pop(k, None)

        if dp in rem:
            return None

    return _join_by_gen(quot, x)


def _p_eval_gen(p, gid, value):
    _gcd_charge(len(p) * (1 + value.bit_length() // 64))
    out = {}
    for m, c in p.items():
        d = dict(m)
        e = d.pop(gid, 0)
        v = c * value**e if e else c
        nm = tuple(sorted(d.items()))
        s = out.get(nm)
        if s is None:
            out[nm] = v
        else:
            s = s + v
            if s:
                out[nm] = s
            else:
                del out[nm]

    return out


def _p_maxnorm(p):
    return max((abs(c) for c in p.values()), default=0)


def _symmetric_mod(c, xi):
    r = c % xi
    if r > xi // 2:
        r -= xi
    return r


def _poly_from_digits(g, x, xi):
    G = {}
    i = 0
    while g:
        gi = {}
        for m, c in g.items():
            r = _symmetric_mod(c, xi)
            if r:
                gi[m] = r
        if gi:
            G = p_add(G, p_mul(gi, p_gen(x, i) if i else {(): 1}, reduce_i=False))
        g = {m: (c - gi.get(m, 0)) // xi for m, c in g.items()}
        g = {m: c for m, c in g.items() if c}
        i += 1

    return G


def _is_probable_prime(n):
    if n < 2:
        return False
    for sp in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % sp == 0:
            return n == sp

    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1

    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False

    return True


def _pick_gaussian_prime():
    n = (1 << 61) - 1
    while not (n % 4 == 1 and _is_probable_prime(n)):
        n -= 2
    return n


_COPRIME_PRIME = _pick_gaussian_prime()
_COPRIME_I = pow(
    next(
        a
        for a in range(2, 1000)
        if pow(a, (_COPRIME_PRIME - 1) // 2, _COPRIME_PRIME) == _COPRIME_PRIME - 1
    ),
    (_COPRIME_PRIME - 1) // 4,
    _COPRIME_PRIME,
)
_COPRIME_SEED = 24301
_coprime_rng = random.Random(_COPRIME_SEED)


def _univariate_images(p, xs, point, prime, inv, pw):
    total = 0
    per = {x: {} for x in xs}
    for m, c in p.items():
        v = c
        for ge in m:
            w = pw.get(ge)
            if w is None:
                w = pw[ge] = pow(point[ge[0]], ge[1], prime)

            v *= w

        v %= prime
        if not v:
            continue
        total += v
        for g, e in m:
            d = per.get(g)
            if d is not None:
                w = inv.get((g, e))
                if w is None:
                    w = inv[(g, e)] = pow(inv[g], e, prime)

                d[e] = d.get(e, 0) + v * w

    out = {}
    for x in xs:
        d = per[x]
        c0 = total
        for e, ve in d.items():
            c0 -= ve * pw[(x, e)]

        img = [c0 % prime] + [
            d.get(i, 0) % prime for i in range(1, max(d, default=0) + 1)
        ]
        while img and not img[-1]:
            img.pop()

        if not img:
            return None
        out[x] = img

    return out


def p_eval_mod(p, point, prime):
    total = 0
    for m, c in p.items():
        if not isinstance(c, int):
            return None
        v = c % prime
        for g, e in m:
            v = v * pow(point[g], e, prime) % prime
        total = (total + v) % prime
    return total


def _univariate_gcd_degree(a, b, prime):
    while b and not b[-1]:
        b.pop()

    while a and not a[-1]:
        a.pop()

    while b:
        if len(a) < len(b):
            a, b = b, a
            continue

        inv = pow(b[-1], prime - 2, prime)
        while len(a) >= len(b) and a:
            f = a[-1] * inv % prime
            shift = len(a) - len(b)
            for i, bc in enumerate(b):
                a[shift + i] = (a[shift + i] - f * bc) % prime

            while a and not a[-1]:
                a.pop()

        a, b = b, a

    return len(a) - 1


def _modular_gcd_degrees(p, q, gens, common=None):
    prime = _COPRIME_PRIME
    xs = [x for x in (gens if common is None else common) if x != I_GEN]
    if not xs:
        return {}
    point = {g: _coprime_rng.randrange(2, prime - 1) for g in gens}
    point[I_GEN] = _COPRIME_I
    inv = {x: pow(point[x], prime - 2, prime) for x in xs}
    pw = {}
    a = _univariate_images(p, xs, point, prime, inv, pw)
    b = _univariate_images(q, xs, point, prime, inv, pw) if a is not None else None
    if a is None or b is None:
        return None
    return {x: _univariate_gcd_degree(a[x], b[x], prime) for x in xs}


def _divisor_probe(p, q, gp, gq, degs):
    if len(p) <= len(q):
        small, large, gs, gl = p, q, gp, gq
    else:
        small, large, gs, gl = q, p, gq, gp
    if not gs <= gl:
        return None
    for x, d in degs.items():
        if d != p_degree_in(small, x):
            return None
    if p_div_exact(large, small) is None:
        return None
    return small


def p_split_i(p):
    re_part = {}
    im_part = {}
    for m, c in p.items():
        for idx, (g, _) in enumerate(m):
            if g == I_GEN:
                im_part[m[:idx] + m[idx + 1 :]] = c
                break
        else:
            re_part[m] = c
    return re_part, im_part


def _holds_i(p):
    for m in p:
        if m and m[0][0] == I_GEN:
            return True
    return False


def p_gcd_real(n, d):
    re_part, im_part = p_split_i(n)
    g, complete = p_gcd(re_part if re_part else im_part, d)
    if re_part and im_part and not (len(g) == 1 and g.get(()) == 1):
        g, second = p_gcd(g, im_part)
        complete = complete and second

    return g, complete


_HEU_TRIES = 6
_HEU_STEP_NUM = 73794
_HEU_STEP_DEN = 27011
_HEU_BIT_BUDGET = 16610  # 5000 decimal digits
_HEU_UNIVARIATE_BIT_BUDGET = 8 * _HEU_BIT_BUDGET


def _heugcd(p, q, gens, gaussian=False):
    """Heuristic gcd, Algorithm 7.4 (GCDHEU) of Geddes, Czapor and Labahn,
    Algorithms for Computer Algebra (1992), p. 330.

    Returns the gcd of p and q over the integers, or over the Gaussian integers
    when gaussian is set, integer content included; None when every evaluation
    point fails. Raises GCDFailure at the size guard.
    """
    if not p:
        return dict(q)
    if not q:
        return dict(p)

    if not gens:
        if gaussian:
            g = _gaussian_int_gcd(_gaussian_of(p), _gaussian_of(q))
            return _poly_of_gaussian(g)
        return p_const(math.gcd(p_const_value(p), p_const_value(q)))

    x = gens[0]
    if gaussian:
        common = _gaussian_int_gcd(_gaussian_content(p), _gaussian_content(q))
    else:
        common = math.gcd(p_int_content(p), p_int_content(q))

    xi = 2 * min(_p_maxnorm(p), _p_maxnorm(q)) + 2
    dx = max(p_degree_in(p, x), p_degree_in(q, x), 1)
    bit_budget = _HEU_BIT_BUDGET
    if len(gens) == 1 and not gaussian:
        bit_budget = _HEU_UNIVARIATE_BIT_BUDGET

    for _ in range(_HEU_TRIES):
        if xi.bit_length() * dx > bit_budget:
            raise GCDFailure

        pv = _p_eval_gen(p, x, xi)
        qv = _p_eval_gen(q, x, xi)
        G = _poly_from_digits(_heugcd(pv, qv, gens[1:], gaussian), x, xi)
        if G:
            if gaussian:
                G, _ = _gaussian_primitive(G)
            else:
                content = p_int_content(G)
                if content != 1:
                    G = {m: c // content for m, c in G.items()}

            if p_div_exact(p, G) is not None and p_div_exact(q, G) is not None:
                if gaussian:
                    return p_mul(G, _poly_of_gaussian(common))
                return p_scale(G, common) if common != 1 else G

        xi = xi * _HEU_STEP_NUM // _HEU_STEP_DEN

    return None


def p_gcd(p, q):
    global _gcd_depth, _gcd_budget
    if not p:
        return dict(q), True
    if not q:
        return dict(p), True
    if p_has_float(p) or p_has_float(q):
        return {(): 1}, True
    Lp, Lq = p_denominator_lcm(p), p_denominator_lcm(q)
    p = p_to_int(p_scale(p, Lp) if Lp != 1 else p)
    q = p_to_int(p_scale(q, Lq) if Lq != 1 else q)
    cp, cq = p_int_content(p), p_int_content(q)
    c = math.gcd(cp, cq)
    if cp != 1:
        p = {m: v // cp for m, v in p.items()}

    if cq != 1:
        q = {m: v // cq for m, v in q.items()}

    mono = ()
    mp, mq = p_monomial_content(p), p_monomial_content(q)
    if mp and mq:
        dp, dq = dict(mp), dict(mq)
        common = {g: min(e, dq[g]) for g, e in dp.items() if g in dq and g != I_GEN}
        mono = tuple(sorted(common.items()))

    if mp:
        p = p_divide_monomial(p, mp)

    if mq:
        q = p_divide_monomial(q, mq)

    base = {mono: c}
    over_i = _holds_i(p) or _holds_i(q)
    if over_i:
        p, zp = _gaussian_primitive(p)
        q, zq = _gaussian_primitive(q)
        z = _first_quadrant(
            _gaussian_int_gcd((cp * zp[0], cp * zp[1]), (cq * zq[0], cq * zq[1]))
        )
        base = p_mul({mono: 1}, _poly_of_gaussian(z))

    if p_is_const(p) or p_is_const(q):
        return base, True
    if len(p) + len(q) > _GCD_TERM_BUDGET:
        return base, False

    gp = p_gens(p)
    gq = p_gens(q)
    common = sorted(g for g in gp & gq if g != I_GEN)
    if not common:
        return base, True
    gens = sorted(gp | gq)
    gaussian = gens[0] == I_GEN
    degs = _modular_gcd_degrees(p, q, gens, common)
    if degs is not None:
        if not any(degs.values()):
            return base, True
        G = _divisor_probe(p, q, gp, gq, degs)
        if G is not None:
            if gaussian:
                G = _unit_normal(G)
            elif p_leading_sign(G) < 0:
                G = p_neg(G)

            return p_mul(base, G, reduce_i=over_i), True

    if _gcd_depth == 0:
        _gcd_budget = _GCD_WORK_BUDGET

    _gcd_depth += 1
    complete = True
    try:
        if gaussian:
            G, complete = _gaussian_gcd(p, q, gens, degs=degs)
        else:
            try:
                G = _heugcd(p, q, gens)
            except GCDFailure:
                G = None

            if G is None:
                G, complete = _prs_gcd(p, q, gens, degs=degs)
                if G is None or p_div_exact(p, G) is None or p_div_exact(q, G) is None:
                    return base, False
    except GCDFailure:
        return base, False
    finally:
        _gcd_depth -= 1

    if p_is_const(G):
        return base, complete
    if gaussian:
        G = _unit_normal(G)
        if len(G) == 1 and G.get(()) == 1:
            return base, complete
    elif p_leading_sign(G) < 0:
        G = p_neg(G)

    return p_mul(base, G, reduce_i=over_i), complete


def _round_div(n, d):
    return (2 * n + d) // (2 * d)


def _gaussian_int_gcd(a, b):
    while b != (0, 0):
        br, bi = b
        n = br * br + bi * bi
        ar, ai = a
        xr = ar * br + ai * bi
        xi = ai * br - ar * bi
        qr = _round_div(xr, n)
        qi = _round_div(xi, n)
        a, b = b, (ar - (qr * br - qi * bi), ai - (qr * bi + qi * br))
    return a


def _gaussian_of(p):
    return (int(p.get((), 0)), int(p.get(((I_GEN, 1),), 0)))


def _poly_of_gaussian(z):
    out = {}
    re_, im_ = z
    if re_:
        out[()] = re_
    if im_:
        out[((I_GEN, 1),)] = im_
    return out


def _first_quadrant(z):
    a, b = z
    while a <= 0 or b < 0:
        a, b = -b, a
    return a, b


def _gaussian_content(p):
    re_part, im_part = p_split_i(p)
    z = (0, 0)
    for m in re_part.keys() | im_part.keys():
        z = _gaussian_int_gcd(z, (re_part.get(m, 0), im_part.get(m, 0)))
        if z[0] * z[0] + z[1] * z[1] == 1:
            return (1, 0)
    return _first_quadrant(z)


def _gaussian_primitive(p):
    z = _gaussian_content(p)
    if z == (1, 0):
        return p, z
    return _gaussian_div_exact(p, _poly_of_gaussian(z)), z


def _unit_normal(p):
    re_part, im_part = p_split_i(p)
    lead = max(re_part.keys() | im_part.keys(), key=mono_key)
    a, b = re_part.get(lead, 0), im_part.get(lead, 0)
    k = 0
    while a <= 0 or b < 0:
        a, b = -b, a
        k += 1
    if k == 0:
        return p
    if k == 2:
        return p_neg(p)
    return p_mul(p, {((I_GEN, 1),): 1 if k == 1 else -1})


def _gaussian_gcd(p, q, gens, degs=None):
    if len(gens) <= 1:
        return {(): 1}, True
    vars_ = [g for g in gens if g != I_GEN]
    try:
        G = _heugcd(p, q, vars_, gaussian=True)
    except GCDFailure:
        G = None
    if G is not None:
        return G, True
    G, complete = _prs_gcd(p, q, gens, gaussian=True, degs=degs)
    if G is None or p_div_exact(p, G) is None or p_div_exact(q, G) is None:
        return {(): 1}, False
    return G, complete


_PRS_TERM_BUDGET = 400000


def _coefficient_words(parts):
    bits = max(c.bit_length() for poly in parts.values() for c in poly.values())
    return 1 + bits // 64


def _pseudo_remainder(f, g):
    dg = max(g)
    lc_g = g[dg]
    g_terms = len(lc_g) + max(len(c) for c in g.values())
    g_words = _coefficient_words(g)
    r = dict(f)

    while r:
        dr = max(r)
        if dr < dg:
            break
        lead = r[dr]
        _gcd_charge(
            sum(len(c) for c in r.values())
            * g_terms
            * (_coefficient_words(r) + g_words - 1)
        )
        r = {d: p_mul(c, lc_g) for d, c in r.items()}
        for j, gj in g.items():
            k = dr - dg + j
            v = p_sub(r.get(k, {}), p_mul(lead, gj))
            if v:
                r[k] = v
            else:
                r.pop(k, None)
        if sum(len(c) for c in r.values()) > _PRS_TERM_BUDGET:
            raise GCDFailure
    return r


def _poly_content(parts):
    g = {}
    complete = True
    for c in parts.values():
        if g:
            g, step = p_gcd(g, c)
            complete = complete and step
        else:
            g = dict(c)

        if p_is_const(g):
            return {(): 1}, complete

    return g, complete


def _exact_parts(parts, divisor):
    out = {d: p_div_exact(c, divisor) for d, c in parts.items()}
    if any(c is None for c in out.values()):
        raise GCDFailure
    return out


def _prs_gcd(p, q, gens, gaussian=False, degs=None):
    try:
        return _prs_gcd_inner(p, q, gens, gaussian, degs)
    except GCDFailure:
        return None, False


def _prs_main_gen(p, q, gens, degs=None):
    gq = p_gens(q)
    best = None
    key = None

    for g in gens:
        if g == I_GEN or g not in gq:
            continue
        dp = p_degree_in(p, g)
        if not dp:
            continue
        dq = p_degree_in(q, g)
        dg = degs.get(g, 0) if degs is not None else 0
        k = (degs is not None and not dg, min(dp, dq) - dg, max(dp, dq), -g)
        if key is None or k < key:
            best, key = g, k

    return best


def _prs_gcd_inner(p, q, gens, gaussian, degs=None):
    x = _prs_main_gen(p, q, gens, degs)
    if x is None:
        return {(): 1}, True
    F = _split_by_gen(p, x)
    G = _split_by_gen(q, x)
    cf, complete = _poly_content(F)
    cg, step = _poly_content(G)
    complete = complete and step
    if not p_is_const(cf):
        F = _exact_parts(F, cf)
    if not p_is_const(cg):
        G = _exact_parts(G, cg)
    cont, step = p_gcd(cf, cg)
    complete = complete and step
    if max(F) < max(G):
        F, G = G, F
    while G:
        R = _pseudo_remainder(F, G)
        if not R:
            break
        cr, step = _poly_content(R)
        complete = complete and step
        if not p_is_const(cr):
            R = _exact_parts(R, cr)

        _gcd_charge(sum(len(c) for c in R.values()) * _coefficient_words(R))
        ic = 0
        for c in R.values():
            ic = math.gcd(ic, p_int_content(c))
        if ic > 1:
            R = {d: {m: v // ic for m, v in c.items()} for d, c in R.items()}
        F, G = G, R
    result = _join_by_gen(G if G else F, x)

    if gaussian:
        result, _ = _gaussian_primitive(result)
    else:
        ic = p_int_content(result)
        if ic > 1:
            result = {m: v // ic for m, v in result.items()}

    return p_mul(cont, result, reduce_i=gaussian), complete


def _gaussian_div_exact(p, g):
    a = g.get((), 0)
    b = g.get(((I_GEN, 1),), 0)
    n = a * a + b * b
    conj = {}
    if a:
        conj[()] = a

    if b:
        conj[((I_GEN, 1),)] = -b

    return {m: _const_div(c, n) for m, c in p_mul(p, conj).items()}


_ROOT_PRIME_START = 13


def _list_divmod(a, b):
    a = list(a)
    quotient = [Fraction(0)] * max(len(a) - len(b) + 1, 0)
    while len(a) >= len(b):
        q = Fraction(a[-1], b[-1])
        shift = len(a) - len(b)
        quotient[shift] = q
        for i, c in enumerate(b):
            a[shift + i] -= q * c

        while a and a[-1] == 0:
            a.pop()

    return quotient, a


def _integer_primitive(coeffs):
    L = 1
    for c in coeffs:
        L = L * c.denominator // math.gcd(L, c.denominator)

    ints = [int(c * L) for c in coeffs]
    content = math.gcd(*ints)
    return [c // content for c in ints]


def _horner_mod(coeffs, x, modulus):
    value = 0
    for c in reversed(coeffs):
        value = (value * x + c) % modulus
    return value


def _simple_integer_roots(monic):
    degree = len(monic) - 1
    derivative = [k * c for k, c in enumerate(monic)][1:]
    bound = 2 * (1 + max(abs(c) for c in monic[:-1]))
    p = max(_ROOT_PRIME_START, degree * degree)

    while True:
        while not _is_probable_prime(p):
            p += 1

        reduced = [c % p for c in monic]
        reduced_derivative = [c % p for c in derivative]
        starts = [t for t in range(p) if not _horner_mod(reduced, t, p)]
        if all(_horner_mod(reduced_derivative, t, p) for t in starts):
            break
        p += 1

    lifted = []
    for t in starts:
        modulus = p
        while modulus < bound:
            modulus *= modulus
            slope = _horner_mod(derivative, t, modulus)
            t = (t - _horner_mod(monic, t, modulus) * pow(slope, -1, modulus)) % modulus

        lifted.append(t - modulus if 2 * t > modulus else t)

    return lifted


def univariate_rational_roots(coeffs):
    roots = {}
    poly = [Fraction(c) for c in coeffs]
    while poly and poly[-1] == 0:
        poly.pop()

    while len(poly) > 1 and poly[0] == 0:
        roots[Fraction(0)] = roots.get(Fraction(0), 0) + 1
        poly = poly[1:]

    if len(poly) <= 1:
        return roots, poly
    rest = _integer_primitive(poly)
    repeated = rest
    derivative = [k * c for k, c in enumerate(rest)][1:]
    while derivative:
        remainder = _list_divmod(repeated, derivative)[1]
        repeated = derivative
        derivative = _integer_primitive(remainder) if remainder else remainder

    squarefree = _integer_primitive(_list_divmod(rest, repeated)[0])
    lead = squarefree[-1]
    degree = len(squarefree) - 1
    monic = [c * lead ** (degree - 1 - k) for k, c in enumerate(squarefree[:-1])]
    candidates = [Fraction(t, lead) for t in _simple_integer_roots(monic + [1])]

    for r in sorted(candidates, key=lambda r: (r.denominator, abs(r.numerator), r < 0)):
        while len(rest) > 1:
            quotient = [0] * (len(rest) - 1)
            carry = Fraction(0)
            for k in range(len(rest) - 1, 0, -1):
                carry = carry * r + rest[k]
                quotient[k - 1] = carry

            if carry * r + rest[0] != 0:
                break
            rest = _integer_primitive(quotient)
            roots[r] = roots.get(r, 0) + 1

    return roots, [Fraction(c) for c in rest]
