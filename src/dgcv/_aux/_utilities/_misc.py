"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv._aux._utilities._misc.py


---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import math
import random
import uuid
from fractions import Fraction

from .._backends._types_and_constants import (
    _disposable_symbol,
    imag_unit,
    rational,
    symbol,
)


def zip_sum(*args, init=0):
    if len(args) == 2:
        pairs = list(zip(*args))
        if pairs:
            hook = getattr(type(pairs[0][1]), "_dgcv_multiadd_scaled", None)
            if hook is not None:
                return hook(pairs, init)
        return sum([a * b for a, b in pairs], init)
    if len(args) == 0:
        return
    if len(args) == 1:
        return args[0]
    return zip_sum(zip_sum(*args[:-1]), args[-1])


def linear_combination(
    elements,
    coef_label: str = None,
    separate_real_and_imag_parts=False,
    coefficient_assumptions=None,
    initialIndex: int = 1,
    _disposable: bool = False,
    **kwargs,
):
    if separate_real_and_imag_parts:
        imag = imag_unit()
        if isinstance(coef_label, str):
            vl = coef_label
            vlr, vli = coef_label + "r", coef_label + "i"
        else:
            prefix = kwargs.pop("prefix", None) or "_dgcvvar_"
            prefr, prefi = prefix + "r" or "_dgcvr_", prefix + "i" or "_dgcvi_"
            suff = uuid.uuid4().hex[:6]
            vlr, vli = prefr + suff, prefi + suff
        n, ad = len(elements), {"real": True}
        vr = [
            symbol(f"{vlr}{x}", assumptions=ad)
            for x in range(initialIndex, initialIndex + n)
        ]
        vi = [
            symbol(f"{vli}{x}", assumptions=ad)
            for x in range(initialIndex, initialIndex + n)
        ]
        v = [a + imag * b for a, b in zip(vr, vi)]
        return zip_sum(v, elements), vr, vi
    if isinstance(coef_label, str):
        vl = coef_label
    else:
        pref = kwargs.pop("prefix", None) or "_dgcvvar_"
        vl = pref + uuid.uuid4().hex[:6]
    if coefficient_assumptions is None and (_disposable or not isinstance(coef_label, str)):
        v = [
            _disposable_symbol(f"{vl}{x}")
            for x in range(initialIndex, initialIndex + len(elements))
        ]
    else:
        v = [
            symbol(f"{vl}{x}", assumptions=coefficient_assumptions)
            for x in range(initialIndex, initialIndex + len(elements))
        ]
    return zip_sum(v, elements), v


_NUMBER_WORDS = ("integer", "rational", "float")
_COMPLEX_WORDS = ("complex", "gaussian_integer", "gaussian_rational")
_REALITY_WORDS = ("real", "imaginary")
_SIGN_WORDS = ("positive", "negative", "nonnegative", "nonpositive", "nonzero")
_DOMAIN_WORDS = _NUMBER_WORDS + _COMPLEX_WORDS + _REALITY_WORDS + _SIGN_WORDS
_SIGN_CONFLICTS = (
    {"positive", "negative"},
    {"positive", "nonpositive"},
    {"negative", "nonnegative"},
    {"nonnegative", "nonpositive"},
)


def _parse_domain(words, away_from_zero):
    from ._config import dgcv_warning

    if isinstance(words, str):
        words = [words]
    words = {str(w).lower() for w in (words or ())}
    unknown = sorted(words - set(_DOMAIN_WORDS))
    if unknown:
        dgcv_warning(f"Ignoring unrecognized domain assumptions {unknown}.")
    numbers = [w for w in _NUMBER_WORDS if w in words]
    if "gaussian_integer" in words:
        numbers.append("integer")
    if "gaussian_rational" in words:
        numbers.append("rational")
    numbers = set(numbers)
    if len(numbers) > 1:
        dgcv_warning(
            f"Domain assumptions name several number types {sorted(numbers)}; using the most general one."
        )
    number = next(
        (w for w in ("float", "rational", "integer") if w in numbers), "integer"
    )
    is_real, is_imag = "real" in words, "imaginary" in words
    if is_real and is_imag:
        dgcv_warning(
            "Domain assumptions 'real' and 'imaginary' conflict; ignoring both."
        )
        is_real = is_imag = False
    signs = words & (set(_SIGN_WORDS) - {"nonzero"})
    for pair in _SIGN_CONFLICTS:
        if pair <= signs:
            dgcv_warning(
                f"Domain assumptions {sorted(signs)} conflict; ignoring the sign words."
            )
            signs = set()
            break
    return {
        "number": number,
        "complex": bool(words & set(_COMPLEX_WORDS)),
        "real": is_real,
        "imaginary": is_imag,
        "sign": next(iter(signs)) if signs else None,
        "nonzero": bool(away_from_zero) or "nonzero" in words,
    }


def _refine_domain(domain, sym):
    out = dict(domain)
    flags = {
        name: getattr(sym, f"is_{name}", None) is True
        for name in (
            "real",
            "imaginary",
            "positive",
            "negative",
            "nonnegative",
            "nonpositive",
            "integer",
            "nonzero",
        )
    }
    if flags["real"] and not out["imaginary"]:
        out["real"] = True
    if flags["imaginary"] and not out["real"]:
        out["imaginary"] = True
    if flags["integer"]:
        out["number"] = "integer"
    if flags["nonzero"]:
        out["nonzero"] = True
    if out["sign"] is None:
        for name in ("positive", "negative", "nonnegative", "nonpositive"):
            if flags[name]:
                out["sign"] = name
                break
    return out


def _sample_component(rng, domain, bounds, denominator_bound):
    lo, hi = bounds
    exclude_zero = domain["nonzero"]
    sign = domain["sign"]
    if sign in ("positive", "nonnegative"):
        lo = max(lo, 0)
    if sign in ("negative", "nonpositive"):
        hi = min(hi, 0)
    exclude_zero = exclude_zero or sign in ("positive", "negative")
    if lo > hi or (exclude_zero and lo == hi == 0):
        raise ValueError("The sampling range contains no admissible value.")
    number = domain["number"]
    for _ in range(200):
        if number == "integer":
            ilo, ihi = math.ceil(lo), math.floor(hi)
            if ilo > ihi:
                raise ValueError("The sampling range contains no admissible integer.")
            value = rng.randint(ilo, ihi)
        elif number == "rational":
            den = rng.randint(1, max(1, int(denominator_bound)))
            value = Fraction(
                rng.randint(math.ceil(lo * den), math.floor(hi * den)), den
            )
        else:
            value = rng.uniform(float(lo), float(hi))
        if not (exclude_zero and value == 0):
            return value
    raise ValueError("The sampling range contains no admissible value.")


def _sample_parts(rng, domain, real_bounds, im_bounds, denominator_bound, complex_ok):
    want_imag = complex_ok and (domain["complex"] or domain["imaginary"])
    a = (
        0
        if domain["imaginary"] and complex_ok
        else _sample_component(rng, domain, real_bounds, denominator_bound)
    )
    b = (
        _sample_component(rng, domain, im_bounds, denominator_bound)
        if want_imag and not domain["real"]
        else 0
    )
    return a, b


def _value_builder(obj_kind):
    if obj_kind == "sympy":
        import sympy

        def conv(x):
            if isinstance(x, float):
                return sympy.Float(x)
            return sympy.Rational(x) if isinstance(x, Fraction) else sympy.Integer(x)

        return conv, sympy.I
    if obj_kind == "sage":
        import sage.all as sage

        def conv(x):
            if isinstance(x, float):
                return sage.RR(x)
            return sage.QQ(x) if isinstance(x, Fraction) else sage.ZZ(x)

        return conv, sage.I

    def conv(x):
        if isinstance(x, Fraction):
            return rational(x.numerator, x.denominator)
        return x

    return conv, imag_unit()


def _object_kind(obj):
    if isinstance(obj, (list, tuple, dict)):
        items = obj.values() if isinstance(obj, dict) else obj
        for item in items:
            kind = _object_kind(item)
            if kind != "dgcv":
                return kind
        return "dgcv"
    module = type(obj).__module__ or ""
    if getattr(obj, "_dgcv_category", None) is not None:
        return "dgcv"
    if module.startswith("sympy"):
        return "sympy"
    if module.startswith("sage"):
        return "sage"
    return "dgcv"


def _free_symbols(obj, kind):
    from .._backends._symbolic_router import get_free_symbols

    if isinstance(obj, (list, tuple, dict)):
        items = obj.values() if isinstance(obj, dict) else obj
        out = set()
        for item in items:
            out |= _free_symbols(item, kind)
        return out
    if kind == "sage":
        variables = getattr(obj, "variables", None)
        return set(variables()) if callable(variables) else set()
    if kind == "sympy":
        return set(getattr(obj, "free_symbols", ()))
    try:
        return set(get_free_symbols(obj))
    except Exception:
        return set(getattr(obj, "free_symbols", ()))


def _substitute(obj, point, kind):
    from .._backends._symbolic_router import subs

    if isinstance(obj, dict):
        return {k: _substitute(v, point, kind) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_substitute(o, point, kind) for o in obj)
    if kind == "dgcv":
        return subs(obj, point)
    sub = getattr(obj, "subs", None)
    return obj if sub is None else sub(point)


def _selection_atoms(items):
    from ...core.conversions.conversions import _scope_atoms
    from ._config import get_variable_registry

    if items is None:
        return None
    if isinstance(items, str) or not isinstance(items, (list, tuple, set, frozenset)):
        items = [items]
    items = list(items)
    if not items:
        return None
    vr = get_variable_registry()
    out = set(_scope_atoms(vr, items) or ())
    standard = vr.get("standard_variable_systems", {})
    for it in items:
        if isinstance(it, str):
            out |= set(standard.get(it, {}).get("family_values", ()))
        else:
            out.add(it)
    return out


def _parts_of_fixed(value):
    from .._backends._symbolic_router import im, re

    if isinstance(value, (int, Fraction, float)):
        return value, 0
    if isinstance(value, complex):
        return value.real, value.imag
    return re(value), im(value)


def evaluate_at_random_point(
    obj,
    *,
    domain_assumptions=(),
    away_from_zero=True,
    range=(-10, 10),
    real_range=None,
    im_range=None,
    independent_conjugates=False,
    skipVar=None,
    variables_scope=None,
    fixed=None,
    denominator_bound=10,
    seed=None,
    return_point=False,
):
    """
    Substitute random values for every free symbol of `obj` (a dgcv object, a sympy or
    sage expression, or a list/tuple/dict of them). Returns the evaluated object, or
    `(evaluated, point)` when `return_point=True`.

    Variables of one dgcv complex variable system are sampled as a single complex number
    (`z = a + i b`, `BARz = a - i b`, `x = a`, `y = b`); `independent_conjugates=True`
    samples every symbol separately. Real coordinates and bare symbols default to real
    values. `fixed={sym: value}` pins values (relatives of a pinned complex coordinate
    follow it) and `seed` fixes the draw. `skipVar` and `variables_scope` follow the
    `dgcv.core.conversions` syntax: a variable, a complex system label, or an iterable of
    them; a variable stands for its whole `(z, BARz, x, y)` family. Skipped or out-of-scope
    symbols stay symbolic.

    `range` bounds both the real and imaginary parts; `real_range` and `im_range`
    override it. Values are exact (`int`/rational) unless `'float'` is requested;
    `away_from_zero` rejects zero components.

    `domain_assumptions` is an iterable of words:
      number type: 'integer' (default), 'rational' (denominators up to
        `denominator_bound`), 'float';
      complex values: 'complex', 'gaussian_integer' (= complex + integer),
        'gaussian_rational' (= complex + rational); without one of these, bare symbols
        are real while complex variable systems are still complex;
      reality: 'real' (imaginary parts 0), 'imaginary' (real parts 0);
      sign, applied to every sampled real component: 'positive', 'negative',
        'nonnegative', 'nonpositive', 'nonzero' (same as `away_from_zero=True`).
    Several number types keep the most general one; 'real' with 'imaginary', or
    opposite sign words, cancel each other; unknown words are ignored. Each case raises a
    dgcv warning. Symbol assumptions (sympy `positive=True`, dgcv atom markers) further
    restrict individual symbols. 'imaginary' never applies to real coordinates.
    """
    from .._vmf.vmf import vmf_lookup
    from ._config import dgcv_warning

    rng = random.Random(seed) if seed is not None else random
    symbols = _free_symbols(obj, _object_kind(obj))
    allowed = _selection_atoms(variables_scope)
    if allowed is not None:
        symbols &= allowed
    skipped = _selection_atoms(skipVar)
    if skipped:
        symbols -= skipped
    domain = _parse_domain(domain_assumptions, away_from_zero)
    real_bounds = tuple(real_range if real_range is not None else range)
    im_bounds = tuple(im_range if im_range is not None else range)
    for bounds in (real_bounds, im_bounds):
        if len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError("Ranges must be pairs (low, high) with low <= high.")
    fixed = dict(fixed or {})
    kind = _object_kind(obj)
    conv, imag = _value_builder(kind)

    def number(a, b):
        return conv(a) if b == 0 else conv(a) + imag * conv(b)

    point = dict(fixed)
    seen = set(fixed)
    warned_imaginary = False
    for sym in sorted(symbols, key=str):
        if sym in seen:
            continue
        info = vmf_lookup(sym, relatives=True) or {}
        rel = info.get("relatives") or {}
        sub_type = info.get("sub_type")
        coordinate = info.get("type") == "coordinate"
        if (
            coordinate
            and sub_type in ("holo", "anti", "real", "imag")
            and not independent_conjugates
        ):
            members = {
                role: rel.get(role)
                for role in ("holo", "anti", "real", "imag")
                if rel.get(role) in symbols
            }
            seen |= set(members.values())
            local = _refine_domain(domain, rel.get("holo"))
            if not local["real"]:
                local = dict(local, complex=True)
            a = b = None
            for role, flip in (("holo", 1), ("anti", -1)):
                if members.get(role) in fixed:
                    a, b = _parts_of_fixed(fixed[members[role]])
                    b = flip * b
                    break
            if a is None:
                a0, b0 = _sample_parts(
                    rng, local, real_bounds, im_bounds, denominator_bound, True
                )
                a = fixed.get(members.get("real"), a0)
                b = fixed.get(members.get("imag"), b0)
            values = {
                "holo": number(a, b),
                "anti": number(a, -b),
                "real": conv(a),
                "imag": conv(b),
            }
            for role, member in members.items():
                if member not in fixed:
                    point[member] = values[role]
            continue
        seen.add(sym)
        local = _refine_domain(domain, sym)
        real_valued = coordinate and sub_type in ("standard", "real", "imag")
        if real_valued and local["imaginary"] and not warned_imaginary:
            dgcv_warning(
                "'imaginary' does not apply to real coordinates; sampling them as real."
            )
            warned_imaginary = True
        if real_valued:
            local = dict(local, imaginary=False, complex=False)
        if coordinate and sub_type in ("holo", "anti") and not local["real"]:
            local = dict(local, complex=True)
        a, b = _sample_parts(
            rng, local, real_bounds, im_bounds, denominator_bound, not real_valued
        )
        point[sym] = number(a, b)
    result = _substitute(obj, point, kind)
    return (result, point) if return_point else result
