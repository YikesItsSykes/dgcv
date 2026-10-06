# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

import math
import random
from typing import NamedTuple

from . import _poly
from ._factor import factor_poly
from ._normal_form import (
    RF,
    _fold_fraction,
    _nf,
    _value_from_poly,
    _value_from_rf,
    rf_msum,
)
from ._poly import _prim

_PRIME = _poly._COPRIME_PRIME
_FACTOR_TABLE_SEED = 1027184
_NO_MULTIPLICITY_BOUND = 1 << 30
_REPEATED_PRODUCT_MAX_TERMS = 4
_NOT_DIVISIBLE = object()


class _Point(dict):
    __slots__ = ("rng",)

    def __init__(self):
        self.rng = random.Random(_FACTOR_TABLE_SEED)

    def __missing__(self, g):
        v = self.rng.randrange(2, _PRIME - 1)
        self[g] = v
        return v


def _image_divmod(a, b):
    a = list(a)
    db = len(b) - 1
    if len(a) <= db:
        return None, False
    inv = pow(b[-1], _PRIME - 2, _PRIME)
    q = [0] * (len(a) - db)
    while len(a) > db:
        f = a[-1] * inv % _PRIME
        shift = len(a) - 1 - db
        q[shift] = f
        if f:
            for i in range(db):
                a[shift + i] = (a[shift + i] - f * b[i]) % _PRIME

        a.pop()
        while a and not a[-1]:
            a.pop()

    while q and not q[-1]:
        q.pop()

    return q, not a


class _Overflow(Exception):
    def __init__(self, bound):
        self.bound = bound


def _packed_mul(a, b):
    if not a or not b:
        return {}
    if len(a) < len(b):
        a, b = b, a

    out = {}
    for mb, cb in b.items():
        for ma, ca in a.items():
            m = ma + mb
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


def _packed_add(a, b):
    if not a:
        return dict(b)
    if not b:
        return dict(a)
    out = dict(a)
    for m, c in b.items():
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


def _packed_sub(a, b):
    if not b:
        return dict(a)
    out = dict(a)
    for m, c in b.items():
        s = out.get(m)
        if s is None:
            out[m] = -c
        else:
            s = s - c
            if s:
                out[m] = s
            else:
                del out[m]

    return out


def _packed_scale(p, c):
    if not c:
        return {}
    return {m: v * c for m, v in p.items()}


def _packed_is_const(p):
    return not p or (len(p) == 1 and 0 in p)


def _packed_split(p, sh, mask):
    hi = mask << sh
    out = {}
    for m, c in p.items():
        e = (m >> sh) & mask
        out.setdefault(e, {})[m & ~hi] = c
    return out


def _packed_join(parts, sh):
    out = {}
    for e, poly in parts.items():
        add = e << sh
        for m, c in poly.items():
            out[m + add] = c
    return out


def _check_overflow(p, guard):
    acc = 0
    for m in p:
        acc |= m
    if acc & guard:
        raise _Overflow(None)


def _packed_main_gen(g, T):
    acc = 0
    for m in g:
        acc |= m

    W = T.width
    mask = T.mask
    best = None
    i = 0

    while acc:
        if acc & mask:
            gid = T.slots[i]
            if best is None or gid > best:
                best = gid

        acc >>= W
        i += 1

    return best


def _packed_div_exact(p, g, T):
    if not g:
        raise ZeroDivisionError("builtin engine: division by zero")
    if not p:
        return {}
    x = _packed_main_gen(g, T)
    if x is None:
        lc = g[0]
        return {m: _poly._const_div(c, lc) for m, c in p.items()}

    sh = T.shift[x]
    mask = T.mask
    guard = T.overflow_bits
    G = _packed_split(g, sh, mask)
    dg = max(G)
    lc_g = G[dg]
    rem = _packed_split(p, sh, mask)
    quot = {}

    while rem:
        dp = max(rem)
        if dp < dg:
            return None
        c = _packed_div_exact(rem[dp], lc_g, T)
        if c is None:
            return None
        quot[dp - dg] = c
        _check_overflow(c, guard)
        for j, gj in G.items():
            k = dp - dg + j
            r = _packed_sub(rem.get(k, {}), _packed_mul(c, gj))
            if r:
                rem[k] = r
            else:
                rem.pop(k, None)

        if dp in rem:
            return None

    return _packed_join(quot, sh)


def _packed_div_linear(num, info, T):
    sh, am, amf, ac, b = info
    mask = T.mask
    guard = T.overflow_bits
    rem = _packed_split(num, sh, mask)
    quot = {}

    while rem:
        dp = max(rem)
        if dp < 1:
            return None
        top = rem.pop(dp)
        c = {}
        for m, v in top.items():
            for s, e in amf:
                if (m >> s) & mask < e:
                    return None

            if ac != 1:
                if v % ac:
                    return None
                v //= ac

            c[m - am] = v

        quot[dp - 1] = c
        if b:
            _check_overflow(c, guard)
            r = _packed_sub(rem.get(dp - 1, {}), _packed_mul(c, b))
            if r:
                rem[dp - 1] = r
            else:
                rem.pop(dp - 1, None)

    return _packed_join(quot, sh)


def _packed_images(p, xs, T):
    W = T.width
    mask = T.mask
    slots = T.slots
    pw = T.point_powers
    point = T.point
    inv = T.point_inverse
    total = 0
    per = {x: {} for x in xs}
    xsl = [(x, T.shift[x]) for x in xs]

    for m, c in p.items():
        v = c
        mm = m
        i = 0

        while mm:
            e = mm & mask
            if e:
                d = pw[i]
                u = d.get(e)
                if u is None:
                    u = d[e] = pow(point[slots[i]], e, _PRIME)

                v *= u

            mm >>= W
            i += 1

        v %= _PRIME
        if not v:
            continue
        total += v
        for x, s in xsl:
            e = (m >> s) & mask
            if e:
                u = inv.get((x, e))
                if u is None:
                    u = inv[(x, e)] = pow(inv[x], e, _PRIME)

                d = per[x]
                d[e] = d.get(e, 0) + v * u

    out = {}
    for x, s in xsl:
        d = per[x]
        c0 = total
        pws = pw[s // W]

        for e, ve in d.items():
            c0 -= ve * pws[e]

        img = [c0 % _PRIME] + [
            d.get(i, 0) % _PRIME for i in range(1, max(d, default=0) + 1)
        ]
        while img and not img[-1]:
            img.pop()

        if img:
            out[x] = img

    return out


def _image_trim(a):
    while a and not a[-1]:
        a.pop()
    return a


def _image_mul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return _image_trim([v % _PRIME for v in out])


def _image_add(a, b, sign):
    if len(a) < len(b):
        a = a + [0] * (len(b) - len(a))
    out = list(a)
    if sign > 0:
        for i, y in enumerate(b):
            out[i] = (out[i] + y) % _PRIME
    else:
        for i, y in enumerate(b):
            out[i] = (out[i] - y) % _PRIME
    return _image_trim(out)


def _images_scale(imgs, k):
    k %= _PRIME
    if not k:
        return {}
    if k == 1:
        return imgs
    return {x: [v * k % _PRIME for v in img] for x, img in imgs.items()}


def _images_lift(imgs, exps, T):
    out = {}
    for x, img in imgs.items():
        for fid, e in exps.items():
            fy = T.factor_image(fid, x)
            if fy is None:
                img = None
                break
            for _ in range(e):
                img = _image_mul(img, fy)
        if img:
            out[x] = img
    return out


def _images_add(ia, ib, sign):
    out = {}
    for x, a in ia.items():
        b = ib.get(x)
        if b is None:
            continue
        r = _image_add(a, b, sign)
        if r:
            out[x] = r
    return out


def _images_mul(ia, ib):
    out = {}
    for x, a in ia.items():
        b = ib.get(x)
        if b is None:
            continue
        r = _image_mul(a, b)
        if r:
            out[x] = r
    return out


def _max_exp(poly):
    deg = 0
    for m in poly:
        for _, e in m:
            if e > deg:
                deg = e
    return deg


def _pack(poly, T):
    deg = _max_exp(poly)
    if deg > T.exp_limit:
        T.widen(deg)

    sh = T.shift
    out = {}
    for m, c in poly.items():
        k = 0
        for g, e in m:
            s = sh.get(g)
            if s is None:
                s = T.add_slot(g)

            k += e << s

        out[k] = c

    return out, deg


def _unpack(p, T):
    W = T.width
    mask = T.mask
    slots = T.slots
    out = {}

    for k, c in p.items():
        m = []
        i = 0
        while k:
            e = k & mask
            if e:
                m.append((slots[i], e))

            k >>= W
            i += 1

        m.sort()
        out[tuple(m)] = c

    return out


def _rewiden(p, oldW, newW):
    om = (1 << oldW) - 1
    out = {}
    for k, c in p.items():
        r = 0
        i = 0
        while k:
            e = k & om
            if e:
                r += e << (i * newW)
            k >>= oldW
            i += 1
        out[r] = c
    return out


def _sync(e, T):
    if e.width != T.width:
        e.num = _rewiden(e.num, e.width, T.width)
        e.width = T.width


def _factors_degree(exps, T):
    return sum(e * T.degree[fid] for fid, e in exps.items())


class _LinearFactor(NamedTuple):
    main_gen: int
    lead_monomial: tuple
    lead_coeff: int
    rest: dict


class _PackedLinearFactor(NamedTuple):
    shift: int
    lead_monomial: int
    lead_exponents: list
    lead_coeff: int
    rest: dict


class _FactorTable:
    __slots__ = (
        "polys",
        "index",
        "powers",
        "live",
        "decomp",
        "point",
        "point_inverse",
        "main_gen",
        "main_image",
        "linear",
        "images",
        "width",
        "mask",
        "exp_limit",
        "overflow_bits",
        "shift",
        "slots",
        "packed_polys",
        "packed_powers",
        "packed_linear",
        "degree",
        "point_powers",
    )

    def __init__(self, width=16):
        self.polys = []
        self.index = {}
        self.powers = {}
        self.live = []
        self.decomp = {}
        self.point = _Point()
        self.point_inverse = {}
        self.main_gen = []
        self.main_image = []
        self.linear = []
        self.images = {}
        self.width = width
        self.mask = (1 << width) - 1
        self.exp_limit = (1 << (width - 1)) - 1
        self.overflow_bits = 0
        self.shift = {}
        self.slots = []
        self.packed_polys = []
        self.packed_powers = {}
        self.packed_linear = []
        self.degree = []
        self.point_powers = []

    def add_slot(self, g):
        i = len(self.slots)
        s = i * self.width
        self.shift[g] = s
        self.slots.append(g)
        self.point_powers.append({})
        self.overflow_bits |= 1 << (s + self.width - 1)
        return s

    def ensure(self, bound):
        if bound > self.exp_limit:
            raise _Overflow(bound)

    def retry(self, attempt):
        while True:
            try:
                return attempt()
            except _Overflow as overflow:
                self.widen(overflow.bound)

    def widen(self, bound):
        oldW = self.width
        W = oldW
        while True:
            W *= 2
            if bound is None or bound <= (1 << (W - 1)) - 1:
                break

        self.width = W
        self.mask = (1 << W) - 1
        self.exp_limit = (1 << (W - 1)) - 1
        self.shift = {g: i * W for i, g in enumerate(self.slots)}
        self.overflow_bits = 0

        for i in range(len(self.slots)):
            self.overflow_bits |= 1 << (i * W + W - 1)

        self.packed_polys = [_rewiden(p, oldW, W) for p in self.packed_polys]
        self.packed_powers = {
            k: _rewiden(p, oldW, W) for k, p in self.packed_powers.items()
        }
        for fid in range(len(self.packed_linear)):
            self.packed_linear[fid] = self._linear_pack(fid)

    def _linear_pack(self, fid):
        info = self.linear[fid]
        if info is None:
            return None
        x, am, ac, b = info
        amk = 0
        amf = []
        for g, e in am:
            amk += e << self.shift[g]
            amf.append((self.shift[g], e))
        bp, _ = _pack(b, self)
        return _PackedLinearFactor(self.shift[x], amk, amf, ac, bp)

    def image(self, poly, x):
        if x not in self.point_inverse:
            self.point_inverse[x] = pow(self.point[x], _PRIME - 2, _PRIME)
        if x not in self.shift:
            self.add_slot(x)
        return _packed_images(poly, [x], self).get(x)

    def factor_image(self, fid, y):
        key = (fid, y)
        img = self.images.get(key, False)
        if img is False:
            img = self.image(self.packed_polys[fid], y)
            if img is not None and len(img) - 1 != _poly.p_degree_in(
                self.polys[fid], y
            ):
                img = None
            self.images[key] = img
        return img

    def screen(self, fid, num, cache):
        x = self.main_gen[fid]
        if x is None:
            return None
        fim = self.main_image[fid]
        if fim is None:
            return None
        nim = cache.get(x)
        if nim is None:
            nim = self.image(num, x)
            if nim is None:
                return None
            cache[x] = nim

        if len(nim) < len(fim):
            return _NOT_DIVISIBLE
        q, ok = _image_divmod(nim, fim)
        if not ok:
            return _NOT_DIVISIBLE
        return q

    def divide(self, fid, num):
        info = self.packed_linear[fid]
        if info is not None:
            return _packed_div_linear(num, info, self)
        return _packed_div_exact(num, self.packed_polys[fid], self)

    def divide_power(self, fid, k, num):
        self.ensure(k * self.degree[fid])
        return _packed_div_exact(num, self.packed_power(fid, k), self)

    def packed_power(self, fid, e):
        key = (fid, e)
        out = self.packed_powers.get(key)
        if out is None:
            if e == 1:
                out = self.packed_polys[fid]
            else:
                out = _packed_mul(self.packed_power(fid, e - 1), self.packed_polys[fid])
            self.packed_powers[key] = out
        return out

    def update(self, fid, cache, x, q, k=1):
        new = {}
        if x is not None and q is not None:
            new[x] = q

        for y, img in cache.items():
            if y == x:
                continue
            fy = self.factor_image(fid, y)
            if fy is None:
                continue
            ok = True
            for _ in range(k):
                img, ok = _image_divmod(img, fy)
                if not ok:
                    break

            if ok:
                new[y] = img

        return new

    def strip(self, fid, num, e, cache):
        removed = 0
        x = self.main_gen[fid]
        while e > 0:
            q = self.screen(fid, num, cache)
            if q is _NOT_DIVISIBLE:
                break
            if q is not None and e > 1:
                fim = self.main_image[fid]
                k = 1
                qk = q

                while k < e:
                    q2, ok = _image_divmod(qk, fim)
                    if not ok:
                        break
                    qk = q2
                    k += 1

                if k > 1:
                    qn = self.divide_power(fid, k, num)
                    if qn is not None:
                        num = qn
                        e -= k
                        removed += k
                        cache = self.update(fid, cache, x, qk, k)
                        break

            qn = self.divide(fid, num)
            if qn is None:
                break
            num = qn
            e -= 1
            removed += 1
            cache = self.update(fid, cache, x, q)

        return num, removed, cache

    def power(self, fid, e):
        key = (fid, e)
        out = self.powers.get(key)
        if out is None:
            if e == 1:
                out = self.polys[fid]
            else:
                out = _poly.p_mul(self.power(fid, e - 1), self.polys[fid])
            self.powers[key] = out
        return out

    def product(self, den):
        out = {(): 1}
        for fid, e in den.items():
            out = _poly.p_mul(out, self.power(fid, e))
        return out

    def normalize(self, den):
        decomp = self.decomp
        if not decomp or not any(fid in decomp for fid in den):
            return den
        out = {}
        stack = list(den.items())
        while stack:
            fid, e = stack.pop()
            parts = decomp.get(fid)
            if parts is None:
                out[fid] = out.get(fid, 0) + e
            else:
                for g, k in parts.items():
                    stack.append((g, k * e))

        return out

    def split(self, poly):
        return self.retry(lambda: self._split_once(poly))

    def _split_once(self, poly):
        pp, _deg = _pack(poly, self)
        exps = {}
        cache = {}
        for fid in self.live:
            pp, removed, cache = self.strip(fid, pp, _NO_MULTIPLICITY_BOUND, cache)
            if removed:
                exps[fid] = removed

        return _unpack(pp, self), exps

    def _new(self, prim):
        pp, deg = _pack(prim, self)
        fid = len(self.polys)
        self.polys.append(prim)
        self.index[tuple(sorted(prim.items()))] = fid
        self.live.append(fid)
        self.packed_polys.append(pp)
        self.degree.append(deg)
        gens = _poly.p_gens(prim)
        info = None
        if gens:
            x = max(gens)
            self.main_gen.append(x)
            img = self.image(pp, x)
            if img is None or len(img) - 1 != _poly.p_degree_in(prim, x):
                img = None

            self.main_image.append(img)
            G = _poly._split_by_gen(prim, x)
            if max(G) == 1 and len(G[1]) == 1:
                ((am, ac),) = G[1].items()
                info = _LinearFactor(x, am, ac, G.get(0, {}))
        else:
            self.main_gen.append(None)
            self.main_image.append(None)

        self.linear.append(info)
        self.packed_linear.append(self._linear_pack(fid))
        return fid

    def register(self, prim):
        res, exps = self.split(prim)
        if _poly.p_is_const(res):
            return exps
        prim, _c = _prim(res)
        fid = self.index.get(tuple(sorted(prim.items())))
        if fid is not None and fid not in self.decomp:
            exps[fid] = exps.get(fid, 0) + 1
            return exps

        c, pieces = factor_poly(prim)
        if len(pieces) > 1 or (pieces and pieces[0][1] > 1):
            if c in (1, -1):
                for piece, mult in pieces:
                    for k, e in self.register(piece).items():
                        exps[k] = exps.get(k, 0) + e * mult

                return exps

        for fid in list(self.live):
            f = self.polys[fid]
            g, _ = _poly.p_gcd(prim, f)
            g = _prim(g)[0]
            if _poly.p_is_const(g):
                continue
            h = _poly.p_div_exact(f, g)
            self.live.remove(fid)
            parts = self.register(g)
            if not _poly.p_is_const(h):
                for k, e in self.register(_prim(h)[0]).items():
                    parts[k] = parts.get(k, 0) + e

            self.decomp[fid] = parts
            out = self.register(prim)
            for k, e in exps.items():
                out[k] = out.get(k, 0) + e

            return out

        exps[self._new(prim)] = 1
        return exps

    def admit(self, poly):
        poly, c = _prim(poly)
        if _poly.p_is_const(poly):
            return c * poly[()], {}
        return c, self.register(poly)


def _divide_out(num, den, T, cache):
    out = {}
    for fid, e in den.items():
        num, removed, cache = T.strip(fid, num, e, cache)
        if e - removed:
            out[fid] = e - removed
    return num, out, cache


def _mul_factors(num, exps, T):
    for fid, e in exps.items():
        f = T.packed_polys[fid]
        if len(f) <= _REPEATED_PRODUCT_MAX_TERMS:
            for _ in range(e):
                num = _packed_mul(num, f)
        else:
            num = _packed_mul(num, T.packed_power(fid, e))
    return num


def _cancel(num, d, den, T, cache):
    if not num:
        return {}, 1, {}, {}
    den = T.normalize(den)
    c = _poly.p_int_content(num)
    if c != 1:
        num = {m: v // c for m, v in num.items()}
        if cache:
            cache = (
                _images_scale(cache, pow(c % _PRIME, _PRIME - 2, _PRIME))
                if c % _PRIME
                else {}
            )

    if den:
        num, den, cache = _divide_out(num, den, T, cache)
        c2 = _poly.p_int_content(num)
        if c2 != 1:
            num = {m: v // c2 for m, v in num.items()}
            c *= c2
            if cache:
                cache = (
                    _images_scale(cache, pow(c2 % _PRIME, _PRIME - 2, _PRIME))
                    if c2 % _PRIME
                    else {}
                )

    if d < 0:
        d = -d
        c = -c

    g = math.gcd(c, d)
    if g != 1:
        c //= g
        d //= g

    if c != 1:
        num = {m: v * c for m, v in num.items()}
        if cache:
            cache = _images_scale(cache, c)

    return num, d, den, cache


class _FactoredElement:
    __slots__ = ("num", "int_den", "den", "table", "degree", "width", "images")

    def __init__(self, num, int_den, den, table, degree, width, images=None):
        self.num = num
        self.int_den = int_den
        self.den = den
        self.table = table
        self.degree = degree
        self.width = width
        self.images = images if images is not None else {}

    @property
    def is_zero(self):
        return not self.num

    @property
    def is_constant(self):
        return not self.den and _packed_is_const(self.num)

    @property
    def is_one(self):
        return not self.den and len(self.num) == 1 and self.num.get(0) == self.int_den

    def __neg__(self):
        return _FactoredElement(
            {m: -c for m, c in self.num.items()},
            self.int_den,
            self.den,
            self.table,
            self.degree,
            self.width,
            _images_scale(self.images, _PRIME - 1),
        )

    def _combine(self, other, sign):
        return self.table.retry(lambda: self._combine_once(other, sign))

    def _combine_once(self, other, sign):
        T = self.table
        _sync(self, T)
        _sync(other, T)
        an, bn = self.num, other.num
        if not bn:
            return self
        if not an:
            return other if sign > 0 else -other
        ad, bd = T.normalize(self.den), T.normalize(other.den)
        da, db = self.int_den, other.int_den
        dga, dgb = self.degree, other.degree
        ia, ib = self.images, other.images
        if ad == bd:
            lcm = dict(ad)
        else:
            lcm = dict(ad)
            for fid, e in bd.items():
                if lcm.get(fid, 0) < e:
                    lcm[fid] = e

            ma = {
                fid: e - ad.get(fid, 0)
                for fid, e in lcm.items()
                if e - ad.get(fid, 0) > 0
            }
            mb = {
                fid: e - bd.get(fid, 0)
                for fid, e in lcm.items()
                if e - bd.get(fid, 0) > 0
            }
            if ma:
                dga += _factors_degree(ma, T)

            if mb:
                dgb += _factors_degree(mb, T)

            T.ensure(max(dga, dgb))
            if ma:
                an = _mul_factors(an, ma, T)
                if ia:
                    ia = _images_lift(ia, ma, T)

            if mb:
                bn = _mul_factors(bn, mb, T)
                if ib:
                    ib = _images_lift(ib, mb, T)

        if da != db:
            L = da * db // math.gcd(da, db)
            ka, kb = L // da, L // db
            if ka != 1:
                an = {m: v * ka for m, v in an.items()}
                if ia:
                    ia = _images_scale(ia, ka)

            if kb != 1:
                bn = {m: v * kb for m, v in bn.items()}
                if ib:
                    ib = _images_scale(ib, kb)
        else:
            L = da

        num = _packed_add(an, bn) if sign > 0 else _packed_sub(an, bn)
        if not num:
            return _FactoredElement({}, 1, {}, T, 0, T.width)
        cache = _images_add(ia, ib, sign) if ia and ib else {}
        num, L, lcm, cache = _cancel(num, L, lcm, T, cache)
        return _FactoredElement(num, L, lcm, T, max(dga, dgb), T.width, cache)

    def __add__(self, other):
        return self._combine(other, 1)

    def __sub__(self, other):
        return self._combine(other, -1)

    def __mul__(self, other):
        return self.table.retry(lambda: self._mul_once(other))

    def _mul_once(self, other):
        T = self.table
        _sync(self, T)
        _sync(other, T)
        an, bn = self.num, other.num
        if not an or not bn:
            return _FactoredElement({}, 1, {}, T, 0, T.width)
        ad, bd = dict(T.normalize(self.den)), dict(T.normalize(other.den))
        ia, ib = self.images, other.images
        if ad and not _packed_is_const(bn):
            only = {fid: e for fid, e in ad.items() if fid not in bd}
            if only:
                bn, rest, ib = _divide_out(bn, only, T, dict(ib))
                for fid in only:
                    ad.pop(fid)

                ad.update(rest)

        if bd and not _packed_is_const(an):
            only = {fid: e for fid, e in bd.items() if fid not in ad}
            if only:
                an, rest, ia = _divide_out(an, only, T, dict(ia))
                for fid in only:
                    bd.pop(fid)

                bd.update(rest)

        den = ad
        for fid, e in bd.items():
            den[fid] = den.get(fid, 0) + e

        if _packed_is_const(an):
            num = _packed_scale(bn, an[0])
            deg = other.degree
            cache = _images_scale(ib, an[0]) if ib else {}
        elif _packed_is_const(bn):
            num = _packed_scale(an, bn[0])
            deg = self.degree
            cache = _images_scale(ia, bn[0]) if ia else {}
        else:
            deg = self.degree + other.degree
            T.ensure(deg)
            num = _packed_mul(an, bn)
            cache = _images_mul(ia, ib) if ia and ib else {}

        num, d, den, cache = _cancel(num, self.int_den * other.int_den, den, T, cache)
        return _FactoredElement(num, d, den, T, deg, T.width, cache)

    def __truediv__(self, other):
        if not other.num:
            raise ZeroDivisionError("builtin engine: division by zero")
        return self.table.retry(lambda: self._div_once(other))

    def _div_once(self, other):
        T = self.table
        _sync(other, T)
        if not self.num:
            return _FactoredElement({}, 1, {}, T, 0, T.width)
        c, oexps = T.admit(_unpack(other.num, T))
        _sync(self, T)
        num = self.num
        deg = self.degree
        cache = self.images
        den = dict(T.normalize(self.den))

        for fid, e in T.normalize(other.den).items():
            den[fid] = den.get(fid, 0) - e

        up = {fid: -e for fid, e in den.items() if e < 0}
        den = {fid: e for fid, e in den.items() if e > 0}
        if up:
            deg += _factors_degree(up, T)
            T.ensure(deg)
            num = _mul_factors(num, up, T)
            if cache:
                cache = _images_lift(cache, up, T)

        for fid, e in oexps.items():
            den[fid] = den.get(fid, 0) + e

        if other.int_den != 1:
            num = {m: v * other.int_den for m, v in num.items()}
            if cache:
                cache = _images_scale(cache, other.int_den)

        num, d, den, cache = _cancel(num, self.int_den * c, den, T, dict(cache))
        return _FactoredElement(num, d, den, T, deg, T.width, cache)

    def to_rf(self):
        T = self.table
        _sync(self, T)
        den = T.product(T.normalize(self.den)) if self.den else {(): 1}
        if self.int_den != 1:
            den = _poly.p_scale(den, self.int_den)
        return RF(_unpack(self.num, T), den)

    def weight(self):
        T = self.table
        return (
            len(self.num)
            + sum(e * len(T.polys[fid]) for fid, e in T.normalize(self.den).items())
            + 1
        )


class _SpanField:
    __slots__ = ("one",)

    def __init__(self):
        self.one = _nf(1)

    def fresh(self, vectors):
        factored = False
        for vec in vectors:
            for value in vec.values():
                rf = _nf(value)
                if rf is None:
                    return self
                if rf.is_zero:
                    continue
                num, den = rf.num, rf.den
                if _poly.p_has_float(num) or _poly.p_has_float(den):
                    return self
                if _poly.I_GEN in _poly.p_gens(num) or _poly.I_GEN in _poly.p_gens(den):
                    return self
                if _poly._RELATED and _poly.has_relation(rf.gens()):
                    return self
                if any(type(c) is not int for c in num.values()) or any(
                    type(c) is not int for c in den.values()
                ):
                    return self

                if len(den) > 1:
                    factored = True

        if factored:
            return _FactoredSpanField()
        return self

    @staticmethod
    def convert(vec):
        out = {}
        for row, value in vec.items():
            rf = _nf(value)
            if rf is None:
                return None
            if not rf.is_zero:
                out[row] = rf
        return out

    @staticmethod
    def weight(b):
        return sum(len(e.num) + len(e.den) for e in b.values())

    @staticmethod
    def pivot_key(e):
        if e.is_constant:
            tier = 0
        elif len(e.num) == 1 and len(e.den) == 1:
            tier = 1
        else:
            tier = 2
        return tier, len(e.num) + len(e.den)

    @staticmethod
    def pivot_divisor(piv):
        if _poly.p_is_const(piv.num):
            return None
        return _value_from_poly(piv.num)

    msum = staticmethod(rf_msum)
    lift = staticmethod(_value_from_rf)
    lift_number = staticmethod(_fold_fraction)


span_field = _SpanField()


class _FactoredSpanField:
    __slots__ = ("one", "table")

    def __init__(self, width=16):
        self.table = _FactorTable(width)
        self.one = _FactoredElement({0: 1}, 1, {}, self.table, 0, self.table.width)

    def fresh(self, vectors):
        return self

    def convert(self, vec):
        T = self.table
        out = {}
        for row, value in vec.items():
            rf = _nf(value)
            if rf is None:
                return None
            if rf.is_zero:
                continue
            num, den = rf.num, rf.den
            if _poly.p_has_float(num) or _poly.p_has_float(den):
                return None
            if _poly.I_GEN in _poly.p_gens(num) or _poly.I_GEN in _poly.p_gens(den):
                return None
            if _poly._RELATED and _poly.has_relation(rf.gens()):
                return None
            if any(type(c) is not int for c in num.values()) or any(
                type(c) is not int for c in den.values()
            ):
                return None

            c, exps = T.admit(den)
            out[row] = T.retry(lambda: self._element(num, c, exps))

        return out

    def _element(self, num, c, exps):
        T = self.table
        pn, deg = _pack(num, T)
        pn, d, exps, cache = _cancel(pn, c, exps, T, {})
        return _FactoredElement(pn, d, exps, T, deg, T.width, cache)

    @staticmethod
    def weight(b):
        return sum(e.weight() for e in b.values())

    @staticmethod
    def pivot_key(e):
        if e.is_constant:
            tier = 0
        elif len(e.num) == 1 and not e.den:
            tier = 1
        else:
            tier = 2
        return tier, e.weight()

    @staticmethod
    def pivot_divisor(piv):
        num = piv.to_rf().num
        if _poly.p_is_const(num):
            return None
        return _value_from_poly(num)

    @staticmethod
    def msum(pairs):
        return None

    @staticmethod
    def lift(e):
        return _value_from_rf(e.to_rf())

    lift_number = staticmethod(_fold_fraction)
