# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

import numbers
from fractions import Fraction

from .._aux._backends._engine import sympy_module_if_available
from .._aux._backends._symbolic_router import _SAGE_METHOD_ALIASES
from ..eds import _atoms as _atoms_mod
from ..eds import _zero_forms as _zf_mod
from ..eds._atoms import zero_form_atom
from ..eds._constants import I, ImaginaryUnit, NamedConstant, _BuiltinLeaf
from ..eds._zero_forms import (
    _as_leaf,
    _base_free_symbols,
    _builtin_leaf_types,
    _builtin_unsupported,
    _nf_of,
    zero_form_class,
)
from ..eds._zf_ops import op_spec
from . import _functions, _poly, _rewrite, _secondary
from ._factor import factor_zf, factored_base, factored_latex
from ._latex import _Unprintable, nf_latex
from ._normal_form import (
    RF,
    OpaqueLeaf,
    _CompoundLeaf,
    _mentions,
    _split_i,
    base_from_nf,
    collect_base,
    conj_rf,
    nf_from_base,
    rf_gen,
    rf_map_gens,
    rf_msum,
)

_TAN_REWRITE_MAX_ARGUMENTS = 2
_OPS_GROWTH_FACTOR = 2
_OPS_GROWTH_SLACK = 8


def simplify(zf):
    rf = zf._nf
    if not rf.complete:
        rf = rf.retry_reduction()
    return zero_form_class._from_nf(rf)


def secondary_simplify(zf):
    rf = zf._nf
    if not any(isinstance(_poly.gen_leaf(g), _CompoundLeaf) for g in rf.gens()):
        return zf
    alt = _rewrite.simplify_tier(zf)
    if alt is not None:
        zf = alt
        rf = zf._nf
        if rf.is_zero:
            return zf
    kind = _secondary.simplify_secondary_kind()
    if kind is None:
        return zf
    try:
        alt = zf._secondary_method("simplify", {}, kind=kind)
    except Exception:
        return zf
    try:
        alt_rf = alt._nf
    except (ZeroDivisionError, NotImplementedError):
        return zf
    if len(alt_rf.num) + len(alt_rf.den) < len(rf.num) + len(rf.den):
        return alt
    return zf


def secondary_method(zf, name, kwargs, kind=None):
    kwargs = {k: v for k, v in kwargs.items() if k != "try_hard"}

    def fn(mod, lowered, bridge):
        (e,) = lowered
        if bridge.kind == "sympy":
            if name == "simplify":
                return _sympy_simplify_candidates(mod, e, bridge, _nf_size(zf))
            return getattr(mod, name)(e, **kwargs)
        alias = _SAGE_METHOD_ALIASES.get(name, name)
        return getattr(e, alias)()

    out = _secondary.with_secondary(
        [zf],
        fn,
        feature=name,
        kind=kind,
        lift=not (name == "simplify" and kind == "sympy"),
    )
    if isinstance(out, zero_form_class):
        return out
    return zero_form_class._from_nf(_nf_of(out))


def _nf_size(x):
    try:
        rf = _nf_of(x)
    except (ZeroDivisionError, NotImplementedError):
        return None
    return len(rf.num) + len(rf.den)


def _sympy_simplify_candidates(sp, e, bridge, target):
    trig_atoms = e.atoms(sp.sin, sp.cos, sp.tan)
    candidates = []
    if (
        trig_atoms
        and len({a.args[0] for a in trig_atoms}) <= _TAN_REWRITE_MAX_ARGUMENTS
    ):
        candidates.append(lambda: sp.cancel(e.rewrite(sp.tan)))

    candidates.append(lambda: sp.simplify(e))
    best = None
    best_size = target
    ops = sp.count_ops(e)

    for make in candidates:
        try:
            out = make()
            if sp.count_ops(out) > _OPS_GROWTH_FACTOR * ops + _OPS_GROWTH_SLACK:
                continue
            lifted = bridge.lift(out)
        except (NotImplementedError, ZeroDivisionError, ValueError, TypeError):
            continue

        size = _nf_size(lifted)
        if size is None:
            continue
        if best_size is None or size < best_size:
            best, best_size = lifted, size
            if target is not None:
                return best

    return best if best is not None else bridge.lift(e)


def factor(zf, try_hard=False):
    return factor_zf(zf, try_hard)


def expand(zf):
    return zero_form_class._from_nf(zf._nf)


def numer_denom(zf):
    n, d = zf._nf.numer_denom()
    return zero_form_class._from_nf(n), zero_form_class._from_nf(d)


def collect(zf, syms):
    gids = []
    for s in syms:
        leaf = _as_leaf(s)
        if leaf is None:
            received = (
                "an expression" if isinstance(s, zero_form_class) else type(s).__name__
            )
            raise TypeError(
                f"builtin `collect` expects atoms or function leaves, received {received}"
            )
        gids.append(_poly.gen_id(leaf))
    return zero_form_class(collect_base(zf._nf, gids))


def _diff_args(args):
    out = []
    i = 0
    n = len(args)
    while i < n:
        var = args[i]
        if i + 1 < n and isinstance(args[i + 1], numbers.Integral):
            out.append((var, int(args[i + 1])))
            i += 2
        else:
            out.append((var, 1))
            i += 1
    return out


def diff(zf, *args):
    expr = zf
    for var, order in _diff_args(args):
        leaf = _as_leaf(var)
        if leaf is None:
            raise TypeError(
                f"builtin `diff` expects zero-form atoms as differentiation variables, received {type(var).__name__}"
            )
        rf = expr._nf
        gid = _poly.gen_id(leaf)
        for _ in range(order):
            rf = rf.diff(gid)
        expr = zero_form_class._from_nf(rf)
    return expr


def _split_real_imag(zf):
    all_real = _functions._all_real
    try:
        rf = zf._nf
    except (ZeroDivisionError, NotImplementedError):
        return None

    if not all_real(RF(rf.num, rf.den, canonical=True)) and any(
        g != _poly.I_GEN and not all_real(RF(_poly.p_gen(g), canonical=True))
        for g in rf.gens()
    ):
        return None

    if _poly.I_GEN in _poly.p_gens(rf.den):
        return None
    re_part, im_part = _split_i(rf.num)
    return (
        zero_form_class._from_nf(RF(re_part, rf.den)),
        zero_form_class._from_nf(RF(im_part, rf.den)),
    )


def re(zf):
    r = zf.to_real()
    fast = _split_real_imag(r)
    if fast is not None:
        return fast[0]
    return simplify(
        zero_form_class(("mul", Fraction(1, 2), ("add", r, r._eval_conjugate())))
    )


def im(zf):
    r = zf.to_real()
    fast = _split_real_imag(r)
    if fast is not None:
        return fast[1]
    return simplify(
        zero_form_class(("mul", -1, I, Fraction(1, 2), ("sub", r, r._eval_conjugate())))
    )


def nf_free_symbols(rf):
    FS = set()
    for g in rf.gens():
        leaf = _poly.gen_leaf(g)
        if isinstance(leaf, zero_form_atom):
            FS.add(leaf)
        elif isinstance(leaf, OpaqueLeaf):
            FS |= _base_free_symbols(leaf.base)
        elif isinstance(leaf, _BuiltinLeaf):
            FS |= leaf.free_symbols
    return FS


def conjugate(zf):
    try:
        return zero_form_class._from_nf(conj_rf(zf._nf))
    except (NotImplementedError, ZeroDivisionError):
        return None


def base_from_factored(factored):
    return factored_base(factored)


def latex(zf):
    try:
        if zf._factored is not None:
            return factored_latex(*zf._factored)
        return nf_latex(zf._nf)
    except (_Unprintable, ZeroDivisionError, NotImplementedError):
        return None


class _SubsFallback(Exception):
    pass


def _subs_value(value):
    if isinstance(value, _zf_mod.abstract_differential_form_atom):
        if value.degree != 0:
            raise _SubsFallback
        value = value.coeff
    if not (
        isinstance(value, (zero_form_class, tuple, _builtin_leaf_types))
        or (isinstance(value, numbers.Number) and not isinstance(value, bool))
    ):
        raise _SubsFallback
    try:
        return _nf_of(value)
    except (ZeroDivisionError, NotImplementedError, TypeError, ValueError):
        raise _SubsFallback from None


def _subs_rf(rf, data):
    mapping = {}
    for g in rf.gens():
        leaf = _poly.gen_leaf(g)
        if leaf in data:
            mapping[g] = _subs_value(data[leaf])
        elif isinstance(leaf, _CompoundLeaf):
            new_args = []
            changed = False
            for a in leaf.args:
                a_rf = nf_from_base(a)
                n_rf = _subs_rf(a_rf, data)
                if n_rf is not a_rf:
                    changed = True
                    new_args.append(base_from_nf(n_rf))
                else:
                    new_args.append(a)

            if changed:
                mapping[g] = leaf.with_args(new_args)
        elif isinstance(leaf, OpaqueLeaf):
            for key in data:
                if _mentions(
                    leaf.base, _as_leaf(key) if _as_leaf(key) is not None else key
                ):
                    raise _SubsFallback

    if not mapping:
        return rf
    try:
        return rf_map_gens(rf, mapping)
    except (ZeroDivisionError, NotImplementedError):
        raise _SubsFallback from None


def subs(zf, data):
    try:
        rf = zf._nf
    except (ZeroDivisionError, NotImplementedError):
        return None

    try:
        out = _subs_rf(rf, data)
    except _SubsFallback:
        return None

    if out is rf:
        for key in data:
            if isinstance(key, zero_form_class) and _as_leaf(key) is None:
                raise TypeError(
                    "builtin `subs` expects atoms or function leaves as keys, received an expression"
                )

        return simplify(zf)

    return zero_form_class._from_nf(out)


def linear_one_hot(values, atoms):
    gids = []
    for a in atoms:
        leaf = _as_leaf(a)
        if leaf is None:
            return None
        gids.append(_poly.gen_id(leaf))

    gid_set = set(gids)
    out = [dict() for _ in gids]
    for key, value in values.items():
        if isinstance(value, numbers.Number) and not isinstance(value, bool):
            if value:
                for d in out:
                    d[key] = value

            continue

        if type(value) is not zero_form_class:
            return None
        rf = value._nf
        gens = rf.gens()
        if not (gens & gid_set):
            if not rf.is_zero:
                for d in out:
                    d[key] = value

            continue

        if _poly.p_gens(rf.den) & gid_set:
            return None
        const = {}
        parts = {g: {} for g in gids}
        for m, c in rf.num.items():
            hit = None
            for g, e in m:
                if g in gid_set:
                    if hit is not None or e != 1:
                        return None
                    hit = g

            if hit is None:
                const[m] = c
            else:
                parts[hit][tuple(t for t in m if t[0] != hit)] = c

        for d, g in zip(out, gids):
            num = _poly.p_add(const, parts[g]) if const else parts[g]
            if num:
                d[key] = zero_form_class._from_nf(RF(num, rf.den))

    return out


def exact_nonzero(x):
    rf = nf_from_base(x)
    if rf.is_constant:
        return not rf.is_zero
    return None


def scalars_equal(a, b):
    return nf_from_base(a).equal(nf_from_base(b))


def to_sympy(base):
    sp = sympy_module_if_available()
    if sp is None:
        raise _builtin_unsupported("to_sympy")
    bridge = _secondary._bridge_for("sympy", sp)

    def conv(b):
        if isinstance(b, zero_form_class):
            return conv(b.base)
        if isinstance(b, tuple):
            op, *args = b
            vals = [conv(a) for a in args]
            return op_spec(op).lower(vals)

        if isinstance(b, ImaginaryUnit):
            return sp.I
        if isinstance(b, (NamedConstant, _CompoundLeaf)):
            try:
                return bridge.leaf_cas(b, conv)
            except _secondary.SecondaryUnavailable:
                raise _builtin_unsupported(
                    "to_sympy of a function derivative with non-symbol arguments"
                ) from None

        if isinstance(b, zero_form_atom):
            if b.role == "antiholomorphic":
                return sp.Symbol(str(b))
            return bridge.symbol_for(b)

        if isinstance(b, Fraction):
            return sp.Rational(b.numerator, b.denominator)
        return sp.sympify(b)

    return conv(base)


_atoms_mod._nf_hooks.update(
    rf_gen=rf_gen,
    nf_from_base=nf_from_base,
    base_from_nf=base_from_nf,
    base_from_factored=base_from_factored,
    display_base=_functions.display_base,
    simplify=simplify,
    secondary_simplify=secondary_simplify,
    secondary_method=secondary_method,
    factor=factor,
    expand=expand,
    numer_denom=numer_denom,
    collect=collect,
    diff=diff,
    re=re,
    im=im,
    nf_free_symbols=nf_free_symbols,
    conjugate=conjugate,
    latex=latex,
    subs=subs,
    msum=rf_msum,
    to_sympy=to_sympy,
)
