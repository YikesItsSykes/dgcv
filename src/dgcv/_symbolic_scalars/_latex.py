# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

import math
from fractions import Fraction

from .._aux.printing.printing._string_processing import latex_superscript
from ..eds._atoms import zero_form_atom
from . import _poly
from ._normal_form import OpaqueLeaf, _leaf_keys, _term_key

_merge_monomial = None
_split_exp_denominator = None


class _Unprintable(Exception):
    pass


def _gaussian_tex(re_, im):
    out = str(re_)
    if im < 0:
        out += " - " + ("i" if im == -1 else f"{-im} i")
    else:
        out += " + " + ("i" if im == 1 else f"{im} i")
    return out


class _NFPrinter:
    def __init__(self, gens):
        self.keys = _leaf_keys(gens)
        self.leaves = {g: _poly.gen_leaf(g) for g in self.keys}

        if any(isinstance(leaf, OpaqueLeaf) for leaf in self.leaves.values()):
            raise _Unprintable

        self.tex = {g: leaf._latex() for g, leaf in self.leaves.items()}
        self.order = sorted(self.keys, key=self.keys.get)
        self.extra = None
        self.index = {g: i for i, g in enumerate(self.order)}

    def monomial(self, exps):
        return tuple((g, e) for g, e in zip(self.order, exps) if e)

    def grouped_terms(self, poly):
        groups = {}
        for m, c in poly.items():
            exps = [0] * len(self.order)
            has_i = False
            for g, e in m:
                if g == _poly.I_GEN:
                    has_i = True
                else:
                    exps[self.index[g]] = e
            key = tuple(exps)
            re_, im = groups.get(key, (0, 0))
            if has_i:
                im += c
            else:
                re_ += c
            groups[key] = (re_, im)

        terms = []
        for k, (re_, im) in groups.items():
            if re_ and im and not any(k):
                terms.append((k, re_, 0))
                terms.append((k, 0, im))
            else:
                terms.append((k, re_, im))

        terms.sort(
            key=lambda t: (_term_key(self.monomial(t[0]), self.keys), bool(t[2]))
        )
        return terms

    def mono_tex(self, exps, coeff_tex, has_i, gauss):
        parts = []
        if gauss is not None:
            parts.append(f"\\left({_gaussian_tex(*gauss)}\\right)")
        if coeff_tex:
            parts.append(coeff_tex)
        if has_i:
            parts.append("i")

        shown = self.monomial(exps)
        if _merge_monomial is not None:
            shown = _merge_monomial(shown, self.extra) or shown
        present = {g: e for g, e in shown if type(g) is int}
        merged = {}
        for g, e in present.items():
            leaf = self.leaves[g]
            q = getattr(leaf, "q", None)
            if q is None or not isinstance(leaf.args[0], zero_form_atom):
                continue
            atom_gid = _poly.gen_id(leaf.args[0])
            if atom_gid in present and atom_gid not in merged:
                merged[atom_gid] = (present[atom_gid] * q + e, q)
                merged[g] = None

        for g, e in shown:
            if type(g) is not int:
                parts.append(g._latex() if e == 1 else g._power_latex(e))
                continue
            if g in merged:
                if merged[g] is not None:
                    num, q = merged[g]
                    power = f"\\frac{{{num}}}{{{q}}}"
                    parts.append(latex_superscript(self.tex[g], power))
                continue
            leaf = self.leaves[g]
            if e == 1:
                parts.append(self.tex[g])
            elif hasattr(leaf, "_power_latex"):
                parts.append(leaf._power_latex(e))
            else:
                parts.append(latex_superscript(self.tex[g], str(e)))

        if not parts:
            return "1"
        return " ".join(parts)

    def term_tex(self, exps, re_, im, den):
        if re_ and im:
            g = 0
            for v in (re_, im, den):
                g = math.gcd(g, abs(v))

            re_, im, d = re_ // g, im // g, den // g
            body = self.mono_tex(exps, None, False, (re_, im))
            if d != 1:
                body = f"\\frac{{{body}}}{{{d}}}"

            return False, body

        c = re_ if re_ else im
        has_i = not re_
        neg = c < 0
        cabs = -c if neg else c
        frac = Fraction(cabs, den)
        cabs, d = frac.numerator, frac.denominator
        body = self.mono_tex(exps, None if cabs == 1 else str(cabs), has_i, None)
        if d != 1:
            body = f"\\frac{{{body}}}{{{d}}}"

        return neg, body

    def poly_tex(self, poly, den=1):
        out = ""
        for i, (exps, re_, im) in enumerate(self.grouped_terms(poly)):
            neg, body = self.term_tex(exps, re_, im, den)
            if i == 0:
                if neg:
                    if not any(exps) and im == 0 and den == 1:
                        out = f"-{body}"
                    else:
                        out = f"- {body}"
                else:
                    out = body
            else:
                out += (" - " if neg else " + ") + body

        return out


def nf_latex(rf):
    num, den = rf.num, rf.den
    if not num:
        return "0"
    if _poly.p_has_float(num) or _poly.p_has_float(den):
        raise _Unprintable
    printer = _NFPrinter(_poly.p_gens(num) | _poly.p_gens(den))
    if _split_exp_denominator is not None:
        den, printer.extra = _split_exp_denominator(rf)
    if _poly.p_is_const(den):
        d = den[()]
        if d == 1:
            return printer.poly_tex(num)
        return printer.poly_tex(num, den=d)

    _, re_, im = printer.grouped_terms(den)[0]
    if (re_ or im) < 0:
        num, den = _poly.p_neg(num), _poly.p_neg(den)

    den_tex = printer.poly_tex(den)
    terms = printer.grouped_terms(num)
    if len(terms) == 1:
        exps, re_, im = terms[0]
        neg, body = printer.term_tex(exps, re_, im, 1)
        return ("- " if neg else "") + f"\\frac{{{body}}}{{{den_tex}}}"
    return f"\\frac{{{printer.poly_tex(num)}}}{{{den_tex}}}"
