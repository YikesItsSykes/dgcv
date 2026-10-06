# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

from fractions import Fraction

from .._aux._backends._symbolic_router import _scalar_is_zero
from .._aux._vmf.vmf import vmf_lookup
from ..eds._zero_forms import _as_leaf, zero_form_class
from . import _poly
from ._normal_form import (
    RF,
    OpaqueLeaf,
    _CompoundLeaf,
    _has_float,
    _nf,
    rf_msum,
    rf_number,
    rf_partial_fast,
)

_WRT_HOL, _WRT_ANTIHOL, _WRT_REAL = range(3)


def _builtin_apply(vf, other):
    rf = _nf(other)
    if rf is None:
        return None
    total = None
    for k, c in vf.coeff_dict.items():
        if _scalar_is_zero(c):
            continue
        if len(k) // 3 != 1:
            continue
        vs = vf._variable_spaces.get(k[2])
        if not isinstance(vs, tuple | dict):
            continue
        leaf = _as_leaf(vs[k[0]])
        if leaf is None:
            return None
        d = rf.diff(_poly.gen_id(leaf))
        if d.is_zero:
            continue
        cn = _nf(c)
        if cn is None:
            return None
        term = cn * d
        total = term if total is None else total + term

    if total is None or total.is_zero:
        return 0
    return zero_form_class._from_nf(total)


_wirtinger_consts = {}


def _wirtinger_plan(vf):
    plan = {}
    vst = vf.variable_spaces_types
    for sys, vs in vf._variable_spaces.items():
        if not isinstance(vs, tuple):
            return None
        gids = []
        for v in vs:
            leaf = _as_leaf(v)
            if leaf is None:
                return None
            gids.append(_poly.gen_id(leaf))

        info = vst.get(sys)
        if info is None:
            return None
        if info.get("type") != "complex":
            plan[sys] = (0, gids)
            continue

        n = len(vs) // 4
        if n == 0 or 4 * n != len(vs):
            return None
        rel = vmf_lookup(vs[0], flattened_relatives=True).get("flattened_relatives")
        if rel != (vs[0], vs[n], vs[2 * n], vs[3 * n]):
            return None
        plan[sys] = (n, gids)

    return plan


def _apply_gids(vf, plan):
    out = set()
    for k, c in vf.coeff_dict.items():
        if _scalar_is_zero(c):
            continue
        if len(k) // 3 != 1:
            continue
        entry = plan.get(k[2])
        if entry is None:
            return None
        n, gids = entry
        idx = k[0]
        if n == 0:
            out.add(gids[idx])
        else:
            r = idx % n
            out.update((gids[r], gids[n + r], gids[2 * n + r], gids[3 * n + r]))

    return frozenset(out)


def _cached_plan(vf):
    cached = getattr(vf, "_builtin_wirtinger_plan", None)
    if cached is None:
        plan = _wirtinger_plan(vf)
        cached = (plan, None if plan is None else _apply_gids(vf, plan))
        vf._builtin_wirtinger_plan = cached

    return cached


def _apply_dependence(vf):
    plan, used = _cached_plan(vf)
    if plan is None or used is None:
        return None
    gens = set()
    for c in vf.coeff_dict.values():
        if _scalar_is_zero(c):
            continue
        rf = _nf(c)
        if rf is None:
            return None
        for g in rf.gens():
            leaf = _poly.gen_leaf(g)
            if isinstance(leaf, (_CompoundLeaf, OpaqueLeaf)):
                return None
            gens.add(g)

    return used, frozenset(gens)


def _builtin_apply_complex(vf, other):
    plan, used = _cached_plan(vf)
    if plan is None:
        return None
    rf = _nf(other)
    if rf is None:
        return None
    consts = _wirtinger_consts
    if not consts:
        consts["half"] = rf_number(Fraction(1, 2))
        consts["i"] = RF(_poly.p_gen(_poly.I_GEN), canonical=True)
        consts["zero"] = RF({}, canonical=True)

    half = consts["half"]
    imu = consts["i"]
    zero = consts["zero"]
    gens = rf.gens()

    for g in gens:
        leaf = _poly.gen_leaf(g)
        if isinstance(leaf, (_CompoundLeaf, OpaqueLeaf)):
            return None

    if used is not None and used.isdisjoint(gens):
        return 0
    den = rf.den
    if (
        len(den) == 1
        and () in den
        and type(den[()]) is int
        and (rf.has_float is False or not _has_float(rf))
    ):
        batched = _apply_complex_batched(vf, rf, plan, gens)
        if batched is not None:
            return batched

    pairs = []
    for k, c in vf.coeff_dict.items():
        if _scalar_is_zero(c):
            continue
        if len(k) // 3 != 1:
            continue
        entry = plan.get(k[2])
        if entry is None:
            return None
        n, gids = entry
        idx = k[0]
        if n == 0:
            g = gids[idx]
            if g not in gens:
                continue
            term = rf_partial_fast(rf, g)
        else:
            q, r = divmod(idx, n)
            gz, gzb, gx, gy = gids[r], gids[n + r], gids[2 * n + r], gids[3 * n + r]
            if gz not in gens and gzb not in gens and gx not in gens and gy not in gens:
                continue
            dz = rf_partial_fast(rf, gz) if gz in gens else zero
            dzb = rf_partial_fast(rf, gzb) if gzb in gens else zero
            dx = rf_partial_fast(rf, gx) if gx in gens else zero
            dy = rf_partial_fast(rf, gy) if gy in gens else zero
            if q == _WRT_HOL:
                term = dz + half * (dx - imu * dy)
            elif q == _WRT_ANTIHOL:
                term = dzb + half * (dx + imu * dy)
            elif q == _WRT_REAL:
                term = dx + dz + dzb
            else:
                term = dy + imu * (dz - dzb)

        if term.is_zero:
            continue
        cn = _nf(c)
        if cn is None:
            return None
        pairs.append((cn, term))

    if not pairs:
        return 0
    total = rf_msum(pairs)
    if total is None:
        for cn, term in pairs:
            term = cn * term
            total = term if total is None else total + term

    if total.is_zero:
        return 0
    return zero_form_class._from_nf(total)


def _apply_complex_batched(vf, rf, plan, gens):
    num = rf.num
    den1 = rf.den
    den2 = {(): 2 * den1[()]}
    imono = {((_poly.I_GEN, 1),): 1}
    partials = {}

    def part(g):
        if g not in gens:
            return None
        out = partials.get(g)
        if out is None:
            out = _poly.p_diff(num, g)
            partials[g] = out
        return out or None

    pairs = []
    for k, c in vf.coeff_dict.items():
        if _scalar_is_zero(c):
            continue
        if len(k) // 3 != 1:
            continue
        entry = plan.get(k[2])
        if entry is None:
            return None
        n, gids = entry
        idx = k[0]
        if n == 0:
            tn = part(gids[idx])
            if tn is None:
                continue
            tden = den1
        else:
            q, r = divmod(idx, n)
            pz = part(gids[r])
            pzb = part(gids[n + r])
            px = part(gids[2 * n + r])
            py = part(gids[3 * n + r])
            tn = {}
            if q == _WRT_HOL:
                if pz:
                    tn = _poly.p_scale(pz, 2)
                if px:
                    tn = _poly.p_add(tn, px)
                if py:
                    tn = _poly.p_sub(tn, _poly.p_mul(py, imono))
                tden = den2
            elif q == _WRT_ANTIHOL:
                if pzb:
                    tn = _poly.p_scale(pzb, 2)
                if px:
                    tn = _poly.p_add(tn, px)
                if py:
                    tn = _poly.p_add(tn, _poly.p_mul(py, imono))
                tden = den2
            elif q == _WRT_REAL:
                for pp in (px, pz, pzb):
                    if pp:
                        tn = _poly.p_add(tn, pp)
                tden = den1
            else:
                if py:
                    tn = dict(py)
                if pz and pzb:
                    diff_z = _poly.p_sub(pz, pzb)
                elif pz:
                    diff_z = pz
                elif pzb:
                    diff_z = _poly.p_neg(pzb)
                else:
                    diff_z = None
                if diff_z:
                    tn = _poly.p_add(tn, _poly.p_mul(diff_z, imono))
                tden = den1
            if not tn:
                continue
        cn = _nf(c)
        if cn is None:
            return None
        pairs.append((cn, RF(tn, tden, canonical=True, has_float=False)))
    if not pairs:
        return 0
    total = rf_msum(pairs)
    if total is None:
        return None
    if total.is_zero:
        return 0
    return zero_form_class._from_nf(total)
