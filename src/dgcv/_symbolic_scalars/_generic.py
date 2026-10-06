# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

from fractions import Fraction

from ..eds._atoms import zero_form_atom
from ..eds._zero_forms import zero_form_class
from . import _linsolve, _poly
from ._normal_form import RF, _value_from_rf, rf_map_gens


class GenericConstants:
    __slots__ = ("prefix", "table", "atoms", "mapping", "lengths", "memo", "weights")

    def __init__(self, prefix):
        self.prefix = prefix
        self.table = {}
        self.atoms = []
        self.mapping = {}
        self.lengths = {}
        self.memo = {}
        self.weights = {}

    @property
    def size(self):
        return len(self.atoms)

    def substitute(self, v):
        nf = getattr(v, "_nf", None)
        if nf is None:
            return v, False
        num, den = nf.num, nf.den
        if _poly.p_is_const(num):
            return v, False
        if (
            not _poly.p_is_const(den)
            or _poly.p_has_float(num)
            or _poly.p_has_float(den)
        ):
            return None, False

        L = _poly.p_denominator_lcm(num)
        poly = _poly.p_to_int(_poly.p_scale(num, L)) if L != 1 else _poly.p_to_int(num)
        mono = _poly.p_monomial_content(poly)
        if mono:
            poly = _poly.p_divide_monomial(poly, mono)

        c = _poly.p_int_content(poly)
        if c != 1:
            poly = {m: x // c for m, x in poly.items()}

        sign = _poly.p_leading_sign(poly)
        if sign < 0:
            poly = _poly.p_neg(poly)

        if len(poly) == 1 and poly.get((), None) == 1:
            return v, False
        key = tuple(sorted(poly.items()))
        gid = self.table.get(key)
        if gid is None:
            atom = zero_form_atom(
                f"{self.prefix}{len(self.atoms) + 1}",
                _markers=frozenset({"engine_symbol"}),
            )
            gid = _poly.gen_id(atom)
            self.atoms.append(atom)
            self.table[key] = gid
            self.mapping[gid] = RF(poly)
            self.lengths[gid] = len(poly)

        scalar = Fraction(sign * c, L) / Fraction(den[()])
        return zero_form_class._from_nf(
            RF(_poly.p_mul({mono: scalar}, _poly.p_gen(gid)))
        ), True

    def expand(self, v):
        nf = getattr(v, "_nf", None)
        if nf is None:
            return v
        gens = nf.gens()
        if not (gens & self.mapping.keys()):
            return v
        key = (tuple(sorted(nf.num.items())), tuple(sorted(nf.den.items())))
        out = self.memo.get(key)
        if out is None:
            out = _value_from_rf(rf_map_gens(nf, self.mapping))
            self.memo[key] = out
        return out

    def weight(self, num):
        total = 0
        lengths = self.lengths
        weights = self.weights

        for m in num:
            w = weights.get(m)
            if w is None:
                w = 1
                for g, e in m:
                    L = lengths.get(g)
                    if L is not None:
                        w *= L**e

                weights[m] = w

            total += w

        return total

    def nonzero(self, num):
        if len(num) <= 1:
            return True
        lengths = self.lengths
        if not any(g in lengths for m in num for g, _ in m):
            return True
        key = tuple(sorted(num.items()))
        v = self.memo.get(("nonzero", key))
        if v is None:
            v = not rf_map_gens(RF(num), self.mapping).is_zero
            self.memo[("nonzero", key)] = v
        return v

    def pivots(self):
        return _linsolve.pivot_guard(self)
