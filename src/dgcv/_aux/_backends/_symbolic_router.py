"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv._aux._backends

module: dgcv.bac_aux._backendskends._symbolic_router


Description: manages dgcv's interfacing with available CAS libraries.

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
import math
import numbers
from fractions import Fraction

from .._utilities._config import dgcv_warning, get_variable_registry
from ._engine import engine_capability, engine_kind, engine_module
from ._types_and_constants import (
    as_fraction,
    constant_scalar_types,
    expr_types,
    fast_scalar_types,
    is_builtin_scalar,
    one,
    to_active_engine,
    zero,
)


def _builtin_unsupported(name):
    return NotImplementedError(
        f"`{name}` is not available in dgcv's builtin symbolic engine; install sympy or sage and set `default_engine` accordingly"
    )


def _builtin_zf(x):
    from ...eds._zero_forms import zero_form_class

    return x if isinstance(x, zero_form_class) else zero_form_class(x)


# -----------------------------------------------------------------------------
# uilities
# -----------------------------------------------------------------------------
_plain_number_types = (int, Fraction, float)
_zero_tests = {}


def register_zero_test(tp, fn):
    _zero_tests[tp] = fn


def _scalar_is_zero(x) -> bool:
    if x is None:
        return False
    if type(x) in _plain_number_types:
        return x == 0
    fn = _zero_tests.get(type(x))
    if fn is not None:
        return fn(x)

    z = getattr(x, "is_literal_zero", None)
    if z is None:
        z = getattr(x, "is_trivial_zero", None)
    if z is None:
        z = getattr(x, "is_zero", None)
    if z is True:
        return True
    if z is False:
        return False
    if callable(z):
        try:
            v = z()
            if isinstance(v, bool):
                return v
        except Exception:
            pass
    try:
        if isinstance(x, constant_scalar_types()) and not isinstance(x, bool):
            return x == 0
    except Exception:
        pass
    try:
        eq = x == 0
        return eq if isinstance(eq, bool) else False
    except Exception:
        return False


_gaussian_field = None


def _exact_number_fields():
    global _gaussian_field
    sage = engine_module()
    if _gaussian_field is None:
        _gaussian_field = sage.QQ[sage.I]
    return (sage.QQ, _gaussian_field)


def exact_nonzero(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, numbers.Rational):
        return x != 0

    kind = engine_kind()
    if kind == "builtin":
        if isinstance(x, numbers.Number):
            return x != 0
        if is_builtin_scalar(x):
            return engine_capability("exact_nonzero")(x)
        return None
    try:
        if kind == "sage":
            for field in _exact_number_fields():
                try:
                    return not field(x).is_zero()
                except (TypeError, ValueError, ArithmeticError, NotImplementedError):
                    continue
            return None
        if kind == "sympy":
            if getattr(x, "is_Rational", False):
                return x != 0
            if getattr(x, "is_number", False):
                real_part, imag_part = x.as_real_imag()
                if getattr(real_part, "is_Rational", False) and getattr(
                    imag_part, "is_Rational", False
                ):
                    return real_part != 0 or imag_part != 0
            return None
    except Exception:
        return None
    return None


def is_zero_knowing_zero_is_expected(x) -> bool:
    if _scalar_is_zero(x):
        return True
    try:
        return _scalar_is_zero(simplify(x))
    except Exception:
        return False


def _scalar_is_one(x) -> bool:
    if type(x) in _plain_number_types:
        return x == 1
    io = getattr(x, "is_one", None)
    if isinstance(io, bool):
        return io
    try:
        return _scalar_is_zero(x - 1)
    except Exception:
        return False


def _scalar_is_minus_one(x) -> bool:
    im1 = getattr(x, "is_minus_one", None)
    if isinstance(im1, bool):
        return im1
    try:
        return _scalar_is_zero(x + 1)
    except Exception:
        return False


def scalars_equal(a, b) -> bool:
    if engine_kind() == "builtin" and (is_builtin_scalar(a) or is_builtin_scalar(b)):
        if not all(
            isinstance(v, numbers.Number) or is_builtin_scalar(v) for v in (a, b)
        ):
            return False
        try:
            return engine_capability("scalars_equal")(a, b)
        except Exception:
            return False
    if isinstance(a, expr_types()) or isinstance(b, expr_types()):
        try:
            return _scalar_is_zero(a - b)
        except Exception:
            return False
    try:
        eq = a == b
    except Exception:
        return False
    return eq if isinstance(eq, bool) else False


class IndeterminateSignError(Exception):
    """
    Raised when the sign of a scalar cannot be certified.
    """


def _scalar_sign(x) -> int:
    """
    Sign of a scalar as -1, 0, or 1.

    Raises
    ------
    IndeterminateSignError
        If the sign cannot be certified.
    """
    if _scalar_is_zero(x):
        return 0

    for attr, val in (("is_positive", 1), ("is_negative", -1)):
        z = getattr(x, attr, None)
        if callable(z):
            try:
                z = z()
            except Exception:
                z = None
        if z is True:
            return val

    try:
        if isinstance(x, constant_scalar_types()) and not isinstance(x, bool):
            return 1 if x > 0 else -1
    except Exception:
        pass

    try:
        gt = x > 0
        if isinstance(gt, bool):
            return 1 if gt else -1
    except Exception:
        pass

    raise IndeterminateSignError(
        f"Cannot determine the sign of {x!r}; it may depend on free parameters."
    )


def get_free_symbols(expr):
    """
    Return the set of atomic elements in symbolic expr
    """
    expr = to_active_engine(expr)
    if hasattr(expr, "free_symbols"):
        return expr.free_symbols

    if hasattr(expr, "variables"):
        return set(expr.variables())

    return set()


def simplify(expr, method=None, try_hard=False, **kwargs):
    expr = to_active_engine(expr)
    f = getattr(expr, "__dgcv_simplify__", None)
    if callable(f):
        try:
            if try_hard:
                return f(method=method, try_hard=True, **kwargs)
            return f(method=method, **kwargs)
        except Exception:
            return expr

    kind = engine_kind()

    if kind == "sympy":
        sp = engine_module()
        try:
            if method is None or method == "simplify":
                return sp.simplify(expr, **kwargs)
            fn = getattr(sp, method, None)
            if callable(fn):
                return fn(expr, **kwargs)
            return expr
        except Exception:
            return expr

    if kind == "sage":
        try:
            if method is None or method == "simplify":
                n = getattr(expr, "normalize", None)
                if callable(n):
                    try:
                        expr = n()
                    except Exception:
                        pass
                f = getattr(expr, "simplify", None)
                if callable(f):
                    return f(**kwargs)
                f = getattr(expr, "simplify_full", None)
                if callable(f):
                    return f(**kwargs)
                return expr
            f = getattr(expr, method, None)
            if callable(f):
                return f(**kwargs)
            return expr
        except Exception:
            return expr

    return expr


def _fast_simplify(expr, method=None, **kwargs):
    return simplify(expr, method=method, **kwargs)


def _resolve_subs_keys(expr, subs_data):
    items = getattr(subs_data, "items", None)
    if items is None or not any(isinstance(k, str) for k in subs_data):
        return subs_data

    atoms = {str(a): a for a in get_free_symbols(expr)}
    return {atoms.get(k, k) if isinstance(k, str) else k: v for k, v in items()}


def subs(expr, subs_data, **kwargs):
    expr = to_active_engine(expr)
    if not subs_data:
        return expr
    f = getattr(expr, "subs", None)
    if f is None:
        return expr
    subs_data = _resolve_subs_keys(expr, subs_data)
    try:
        return f(subs_data, **kwargs)
    except Exception:
        return f(subs_data)


def _conjugation_swaps(expr):
    cd = get_variable_registry()["conversion_dictionaries"]["conjugation"]
    swaps = {}
    for a in get_free_symbols(expr):
        image = cd.get(str(a))
        if image is not None:
            swaps[a] = image
    return swaps


def _symbolic_conjugate(expr):
    if engine_kind() == "builtin":
        return conjugate(expr)
    swaps = _conjugation_swaps(expr)

    if engine_kind() == "sage":
        f = getattr(expr, "conjugate", None)
        if not callable(f):
            return expr
        out = f()
        undo = {}
        for a, image in ((a, swaps.get(a, a)) for a in get_free_symbols(expr)):
            g = getattr(a, "conjugate", None)
            if not callable(g):
                continue
            ca = g()
            if str(ca) == str(a):
                continue
            undo[ca] = image
        return subs(out, undo)

    from ._types_and_constants import imag_unit

    imag = imag_unit()
    return subs(expr, {**swaps, imag: -imag}, simultaneous=True)


def conjugate(expr, symbolic=False):
    expr = to_active_engine(expr)
    f = getattr(expr, "__dgcv_conjugate__", None)
    if callable(f):
        return f(symbolic=symbolic)
    if symbolic is True:
        return _symbolic_conjugate(expr)

    kind = engine_kind()

    if kind == "sympy":
        f = getattr(expr, "conjugate", None)
        if f is not None:
            return f()
        return expr

    if kind == "sage":
        f = getattr(expr, "conjugate", None)
        if f is None:
            return expr

        c = f()

        registry = get_variable_registry()
        subs_map = registry.get("dgcv_enforced_real_atoms")
        if subs_map:
            g = getattr(c, "subs", None)
            if g is not None:
                return g(subs_map)
        return c

    f = getattr(expr, "conjugate", None)
    if f is not None:
        return f()
    return expr


def ratio(x, y=1):
    y = getattr(y, "to_sym_engine_expr", y)

    kind = engine_kind()

    if kind is None or kind == "builtin":
        if isinstance(x, (float, complex)) or isinstance(y, (float, complex)):
            return x / y
        if isinstance(x, numbers.Rational) and isinstance(y, numbers.Rational):
            return as_fraction(x) / as_fraction(y)
        return x / y

    eng = engine_module()

    if kind == "sympy":
        sp = eng

        if isinstance(x, sp.Float) or isinstance(y, sp.Float):
            return x / y
        if isinstance(x, (float, complex)) or isinstance(y, (float, complex)):
            return sp.sympify(x) / sp.sympify(y)

        if isinstance(x, numbers.Integral) and isinstance(y, numbers.Integral):
            return sp.Rational(int(x), int(y))

        if isinstance(x, (sp.Integer, sp.Rational)) and isinstance(
            y, (sp.Integer, sp.Rational)
        ):
            return x / y

        if isinstance(x, Fraction):
            x = sp.Rational(x.numerator, x.denominator)
        if isinstance(y, Fraction):
            y = sp.Rational(y.numerator, y.denominator)

        sx = x if isinstance(x, sp.Basic) else sp.sympify(x)
        sy = y if isinstance(y, sp.Basic) else sp.sympify(y)

        return sx / sy

    if kind == "sage":
        sage = eng

        if isinstance(x, (float, complex)) or isinstance(y, (float, complex)):
            return x / y

        QQ = sage.QQ

        def _exact(value):
            if isinstance(value, Fraction):
                return QQ(value.numerator) / QQ(value.denominator)
            if isinstance(value, numbers.Integral) and not isinstance(value, bool):
                return QQ(int(value))
            return value

        exact_x, exact_y = _exact(x), _exact(y)
        exact_types = fast_scalar_types()
        if isinstance(exact_x, exact_types) and isinstance(exact_y, exact_types):
            return exact_x / exact_y

        SR = sage.SR
        return SR(x) / SR(y)
    return x / y


def re(expr):
    """
    Return the real part of expr in the active symbolic engine.
    """
    expr = to_active_engine(expr)
    f = getattr(expr, "__dgcv_re__", None)
    if callable(f):
        return f()
    kind = engine_kind()

    if kind == "builtin" and is_builtin_scalar(expr):
        return simplify((expr + conjugate(expr)) / 2)

    if kind == "sympy":
        sp = engine_module()
        return sp.re(expr)

    if kind == "sage":
        f = getattr(expr, "real_part", None)
        if f is not None:
            return f()
        f = getattr(expr, "real", None)
        if f is not None:
            return f() if callable(f) else f
        return expr

    z = getattr(expr, "real", None)
    if callable(z):
        return z()
    if z is not None:
        return z
    if isinstance(expr, numbers.Number):
        return expr.real
    return expr


def im(expr):
    """
    Return the imaginary part of expr in the active symbolic engine.
    """
    expr = to_active_engine(expr)
    f = getattr(expr, "__dgcv_im__", None)
    if callable(f):
        return f()
    kind = engine_kind()

    if kind == "builtin" and is_builtin_scalar(expr):
        from ._types_and_constants import imag_unit

        return simplify((expr - conjugate(expr)) / (2 * imag_unit()))

    if kind == "sympy":
        sp = engine_module()
        return sp.im(expr)

    if kind == "sage":
        f = getattr(expr, "imag_part", None)
        if f is not None:
            return f()
        f = getattr(expr, "imag", None)
        if f is not None:
            return f() if callable(f) else f
        return zero()

    z = getattr(expr, "imag", None)
    if callable(z):
        return z()
    if z is not None:
        return z
    if isinstance(expr, numbers.Number):
        return expr.imag
    return zero()


def log(expr):
    expr = to_active_engine(expr)
    f = getattr(expr, "__dgcv_log__", None)
    if callable(f):
        return f()
    return engine_module().log(expr)


def ln(expr):
    return log(expr)


def _trig(name, expr):
    expr = to_active_engine(expr)
    f = getattr(expr, f"__dgcv_{name}__", None)
    if callable(f):
        return f()
    return getattr(engine_module(), name)(expr)


def sin(expr):
    return _trig("sin", expr)


def cos(expr):
    return _trig("cos", expr)


def tan(expr):
    return _trig("tan", expr)


def cot(expr):
    return _trig("cot", expr)


def sec(expr):
    return _trig("sec", expr)


def csc(expr):
    return _trig("csc", expr)


def sinh(expr):
    return _trig("sinh", expr)


def cosh(expr):
    return _trig("cosh", expr)


def tanh(expr):
    return _trig("tanh", expr)


def coth(expr):
    return _trig("coth", expr)


def sech(expr):
    return _trig("sech", expr)


def csch(expr):
    return _trig("csch", expr)


def asin(expr):
    return _trig("asin", expr)


def acos(expr):
    return _trig("acos", expr)


def atan(expr):
    return _trig("atan", expr)


def acot(expr):
    return _trig("acot", expr)


def asec(expr):
    return _trig("asec", expr)


def acsc(expr):
    return _trig("acsc", expr)


def asinh(expr):
    return _trig("asinh", expr)


def acosh(expr):
    return _trig("acosh", expr)


def atanh(expr):
    return _trig("atanh", expr)


def acoth(expr):
    return _trig("acoth", expr)


def asech(expr):
    return _trig("asech", expr)


def acsch(expr):
    return _trig("acsch", expr)


def exp(expr):
    expr = to_active_engine(expr)
    f = getattr(expr, "__dgcv_exp__", None)
    if callable(f):
        return f()

    return engine_module().exp(expr)


def as_numer_denom(expr):
    """
    Return (numerator, denominator) for expr in the active symbolic engine.
    """
    expr = to_active_engine(expr)
    kind = engine_kind()

    f = getattr(expr, "as_numer_denom", None)
    if callable(f):
        return f()
    if kind == "builtin":
        if isinstance(expr, numbers.Rational) and not isinstance(expr, bool):
            return expr.numerator, expr.denominator
        if is_builtin_scalar(expr):
            return _builtin_zf(expr).as_numer_denom()
        return expr, one()
    if kind == "sympy":
        try:
            sp = engine_module()
            return sp.sympify(expr).as_numer_denom()
        except Exception:
            pass

    if kind == "sage":
        f = getattr(expr, "numerator", None)
        g = getattr(expr, "denominator", None)
        if callable(f) and callable(g):
            try:
                return f(), g()
            except Exception:
                return expr, one()

    return expr, one()


def common_multiple(*exprs):
    return lcm_routed(*exprs)


def lcm_routed(*exprs):
    try:
        if len(exprs) == 0:
            return
        if len(exprs) == 1:
            return exprs[0]
        return common_multiple(engine_module().lcm(exprs[0], exprs[1]), *exprs[2:])
    except Exception:
        return math.prod(exprs)


def gcd_routed(*exprs):
    try:
        if len(exprs) == 0:
            return
        if len(exprs) == 1:
            if _scalar_is_zero(exprs[0]):
                return 1
            return exprs[0]
        return gcd_routed(engine_module().gcd(exprs[0], exprs[1]), *exprs[2:])
    except Exception:
        return one()


def ilcm(*ints):
    """
    Integer least common multiple in the active symbolic engine.
    """
    ints = [int(x) for x in ints if x is not None]
    if not ints:
        return 1
    if len(ints) == 1:
        return ints[0]

    kind = engine_kind()

    if kind == "sympy":
        sp = engine_module()
        return sp.ilcm(*ints)

    if kind == "sage":
        sage = engine_module()
        fn = getattr(sage, "lcm", None)
        if fn is not None:
            from functools import reduce

            return reduce(fn, ints)
    return math.lcm(*ints)


def clear_denominators(seq, *, return_scale=False):
    """
    Multiply a sequence of scalars by the LCM of their denominators (when detectable),
    returning a new list.
    """
    if seq is None:
        return None

    denoms = []
    for x in seq:
        _, d = as_numer_denom(x)
        try:
            denoms.append(int(d))
        except Exception:
            pass

    L = ilcm(*denoms) if denoms else 1
    out = list(seq) if L == 1 else [L * x for x in seq]
    return (out, L) if return_scale else out


_SAGE_METHOD_ALIASES = {
    "simplify": "simplify_full",
    "factor": "factor",
    "expand": "expand",
    "trigsimp": "simplify_trig",
    "cancel": "simplify_rational",
    "ratsimp": "simplify_rational",
    "together": "combine",
    "apart": "partial_fraction",
    "logcombine": "simplify_log",
    "expand_log": "expand_log",
    "expand_trig": "expand_trig",
}
_engine_method_warned = set()


def engine_method(expr, name, **kwargs):
    expr = to_active_engine(expr)
    kind = engine_kind()
    if kind == "builtin":
        if isinstance(expr, numbers.Number):
            return expr
        if is_builtin_scalar(expr):
            return _builtin_zf(expr)._engine_method(name, **kwargs)
        raise _builtin_unsupported(name)
    if kind == "sympy":
        fn = getattr(engine_module(), name, None)
        if not callable(fn):
            return expr
        try:
            return fn(expr, **kwargs)
        except Exception:
            return expr
    if kind == "sage":
        alias = _SAGE_METHOD_ALIASES.get(name)
        if alias is None:
            if name not in _engine_method_warned:
                _engine_method_warned.add(name)
                dgcv_warning(
                    f"`{name}` has no sage equivalent; the expression was returned unchanged."
                )
            return expr
        fn = getattr(expr, alias, None)
        if not callable(fn):
            return expr
        try:
            return fn(**kwargs)
        except Exception:
            try:
                return fn()
            except Exception:
                return expr
    return expr


def expand(expr, **kwargs):
    """
    Expand expr using the active symbolic engine, intended as a backend hook for
    expand_dgcv (and polynomial expansion).
    """

    expr = to_active_engine(expr)
    kind = engine_kind()
    if kind == "builtin":
        if isinstance(expr, numbers.Number):
            return expr
        if is_builtin_scalar(expr):
            return _builtin_zf(expr)._builtin_expand()
    f = getattr(expr, "__dgcv_expand__", None)
    if f:
        try:
            return f(**kwargs)
        except TypeError:
            return f()
    f = getattr(expr, "__dgcv_apply__", None)
    if f:
        try:
            return f(expand, **kwargs)
        except TypeError:
            return f(expand)  ###!!! remove try/excepts here

    if kind == "sympy":
        try:
            return engine_module().expand(expr, **kwargs)
        except Exception:
            return expr

    if kind == "sage":
        f = getattr(expr, "expand", None)
        if callable(f):
            try:
                return f(**kwargs)
            except TypeError:
                return f()

    return expr


def factor(expr, try_hard=False, **kwargs):
    """
    Factor expr using the active symbolic engine, intended as a backend hook for
    factor_dgcv (and polynomial factoring).
    """
    expr = to_active_engine(expr)
    kind = engine_kind()

    if kind == "builtin":
        if isinstance(expr, numbers.Number):
            return expr
        if is_builtin_scalar(expr):
            return _builtin_zf(expr)._engine_method("factor", try_hard=try_hard)

    f = getattr(expr, "__dgcv_apply__", None)
    if f:
        if try_hard:
            kwargs = dict(kwargs, try_hard=True)
        try:
            return f(factor, **kwargs)
        except TypeError:
            return f(factor)

    if isinstance(expr, constant_scalar_types()):
        return expr

    if kind == "sympy":
        sp = engine_module()
        return sp.factor(expr, **kwargs)

    if kind == "sage":
        f = getattr(expr, "factor", None)
        if callable(f):
            try:
                return f(**kwargs)
            except TypeError:
                try:
                    return f()
                except Exception:
                    return expr
            except Exception:
                return expr
        return expr

    return expr


def cancel(expr, **kwargs):
    """
    Cancel common factors in a rational expression using the active symbolic engine,
    intended as a backend hook for cancel_dgcv (and rational simplification).
    """
    expr = to_active_engine(expr)
    kind = engine_kind()

    if kind == "builtin":
        if isinstance(expr, numbers.Number):
            return expr
        if is_builtin_scalar(expr):
            return _builtin_zf(expr)._builtin_simplify()

    f = getattr(expr, "__dgcv_apply__", None)
    if f:
        try:
            return f(cancel, **kwargs)
        except TypeError:
            return f(cancel)

    if kind == "sympy":
        sp = engine_module()
        return sp.cancel(expr, **kwargs)

    if kind == "sage":
        for name in ("normalize", "cancel"):
            f = getattr(expr, name, None)
            if callable(f):
                try:
                    return f(**kwargs)
                except TypeError:
                    return f()

    return expr


def collect(expr, syms):
    expr = to_active_engine(expr)
    kind = engine_kind()

    if isinstance(syms, (list, tuple, set, frozenset)):
        syms = list(syms)
    else:
        syms = [syms]

    if kind == "builtin":
        if isinstance(expr, numbers.Number):
            return expr
        if is_builtin_scalar(expr):
            return _builtin_zf(expr).collect(syms)

    f = getattr(expr, "__dgcv_apply__", None)
    if f:
        return f(lambda e: collect(e, syms))

    if not syms:
        return expr

    if kind == "sympy":
        sp = engine_module()
        for s in syms:
            try:
                expr = sp.collect(expr, s)
            except Exception:
                return expr
        return expr

    if kind == "sage":
        for s in syms:
            g = getattr(expr, "collect", None)
            if not callable(g):
                return expr
            try:
                expr = g(s)
            except Exception:
                return expr
        return expr

    return expr


def defloat(expr, *, heuristic=False, **kwargs):
    """
    Attempt to coerce floating point numbers within expressions to exact symbolic ratios.

    This should not be relied upon for exact computation programmatically. Instead, it is intended as a convenience utility for copy/pasting printed math, as printed expressions tipically format exact ratios into syntax that compiles with floating point numbers.
    """
    expr = to_active_engine(expr)
    if isinstance(expr, list):
        return [defloat(inner, heuristic=heuristic, **kwargs) for inner in expr]
    if isinstance(expr, tuple):
        return tuple(defloat(inner, heuristic=heuristic, **kwargs) for inner in expr)
    if isinstance(expr, dict):
        return {
            defloat(k, heuristic=heuristic, **kwargs): defloat(
                v, heuristic=False, **kwargs
            )
            for k, v in expr.items()
        }
    kind = engine_kind()
    if kind == "builtin" and is_builtin_scalar(expr):
        return _builtin_defloat(expr, heuristic=heuristic)
    f = getattr(expr, "__dgcv_apply__", None)
    if f:
        return f(defloat, heuristic=heuristic, **kwargs)
    if kind == "sympy":
        return engine_module().nsimplify(expr)
    if kind == "sage":
        return _sage_defloat(expr, heuristic=heuristic)

    if isinstance(expr, float):
        return _float_to_fraction(expr, heuristic)
    f = getattr(expr, "as_integer_ratio", None)
    if callable(f):
        n, d = f()
        return Fraction(n, d)
    return expr


def _float_to_fraction(x, heuristic):
    if heuristic:
        return Fraction(x).limit_denominator(10**6)
    return Fraction(str(x))


def _builtin_defloat(expr, *, heuristic=False):
    from ...eds._zero_forms import zero_form_class

    def walk(b):
        if isinstance(b, zero_form_class):
            return walk(b.base)
        if isinstance(b, tuple):
            return (b[0], *[walk(a) for a in b[1:]])
        if isinstance(b, float):
            return _float_to_fraction(b, heuristic)
        return b

    if isinstance(expr, zero_form_class):
        return zero_form_class(walk(expr.base))
    return expr


def _sage_defloat(expr, *, heuristic=False, **kwargs):
    sage = engine_module()

    try:
        from sage.rings.real_double import RealDoubleElement  # type: ignore
        from sage.rings.real_mpfr import RealNumber as SageRealNumber  # type: ignore

        real_types = (SageRealNumber, RealDoubleElement)
    except Exception:
        real_types = ()

    def convert(x):
        if isinstance(x, real_types):
            try:
                return x.nearby_rational() if heuristic else sage.QQ(str(x))
            except Exception:
                return x
        return x

    try:
        return expr.map(lambda x: convert(x))
    except Exception:
        try:
            op = expr.operator()
            args = expr.operands()
            if not args:
                return convert(expr)
            return op(*[_sage_defloat(a, heuristic=heuristic) for a in args])
        except Exception:
            return convert(expr)


def sqrt(expr):
    return engine_module().sqrt(expr)
