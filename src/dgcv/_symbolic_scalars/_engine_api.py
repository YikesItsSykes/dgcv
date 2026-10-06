# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

import cmath
import math
import numbers
from fractions import Fraction

from .._aux._backends._types_and_constants import is_builtin_scalar
from ..eds._atoms import zero_form_atom, zeroFormAtom
from ..eds._constants import PI, E, I, named_constant
from ..eds._zero_forms import _builtin_unsupported, abstract_ZF, zero_form_class
from . import _bindings, _poly
from ._foreign import from_foreign
from ._functions import FunctionHead, elementary, float_value, radical
from ._generic import GenericConstants
from ._linalg import eigen_data, generic_det, projector_eigenspaces
from ._linsolve import builtin_build_and_solve
from ._normal_form import RF, nf_from_base, rf_number
from ._polynomials import poly_terms, primitive_factors
from ._probes import basis_pick
from ._secondary import integrate, secondary_kind, solve_with_secondary
from ._span_field import span_field
from ._vector_fields import _apply_dependence, _builtin_apply, _builtin_apply_complex

__all__ = [
    "symbol",
    "I",
    "E",
    "PI",
    "named_constant",
    "gcd",
    "lcm",
    "sqrt",
    "integrate",
    "log",
    "exp",
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "sinh",
    "cosh",
    "tanh",
    "coth",
    "sech",
    "csch",
    "asin",
    "acos",
    "atan",
    "acot",
    "asec",
    "acsc",
    "asinh",
    "acosh",
    "atanh",
    "acoth",
    "asech",
    "acsch",
    "dgcv_capabilities",
    "abstract_ZF",
    "zeroFormAtom",
    "zero_form_class",
    "zero_form_atom",
]

dgcv_capabilities = {
    "linsolve": builtin_build_and_solve,
    "secondary_kind": secondary_kind,
    "solve_secondary": solve_with_secondary,
    "eigen_data": eigen_data,
    "generic_det": generic_det,
    "projector_eigenspaces": projector_eigenspaces,
    "basis_pick": basis_pick,
    "span_field": span_field,
    "from_foreign": from_foreign,
    "apply_vector_field": _builtin_apply,
    "apply_complex_vector_field": _builtin_apply_complex,
    "apply_dependence": _apply_dependence,
    "linear_one_hot": _bindings.linear_one_hot,
    "poly_terms": poly_terms,
    "primitive_factors": primitive_factors,
    "function_head": FunctionHead,
    "generic_constants": GenericConstants,
    "exact_nonzero": _bindings.exact_nonzero,
    "scalars_equal": _bindings.scalars_equal,
}


def symbol(label, assumptions=None):
    markers = set()
    if isinstance(assumptions, dict):
        for key in (
            "real",
            "positive",
            "negative",
            "nonnegative",
            "nonpositive",
            "nonzero",
            "integer",
        ):
            if assumptions.get(key, False) is True:
                markers.add(key)

        if markers & {"positive", "negative", "nonnegative", "nonpositive", "integer"}:
            markers.add("real")

    markers.add("engine_symbol")
    return zero_form_atom(str(label), _markers=frozenset(markers))


def _poly_of(x, feature):
    rf = nf_from_base(x)
    if not rf.is_polynomial:
        raise _builtin_unsupported(f"{feature} of non-polynomial expressions")
    return rf


def gcd(a, b):
    if isinstance(a, numbers.Integral) and isinstance(b, numbers.Integral):
        return math.gcd(int(a), int(b))
    ra, rb = _poly_of(a, "gcd"), _poly_of(b, "gcd")
    g, _ = _poly.p_gcd(ra.num, rb.num)
    return zero_form_class._from_nf(RF(g))


def lcm(a, b):
    if isinstance(a, numbers.Integral) and isinstance(b, numbers.Integral):
        return math.lcm(int(a), int(b))
    ra, rb = _poly_of(a, "lcm"), _poly_of(b, "lcm")
    g, _ = _poly.p_gcd(ra.num, rb.num)
    q = _poly.p_div_exact(ra.num, g)
    return zero_form_class._from_nf(RF(_poly.p_mul(q, rb.num)))


def _float_result(v):
    if isinstance(v, complex):
        rf = rf_number(v)
        return rf.constant_value() if rf.is_constant else zero_form_class._from_nf(rf)
    return v


def sqrt(x):
    if isinstance(x, numbers.Integral) and not isinstance(x, bool) and x >= 0:
        r = math.isqrt(int(x))
        if r * r == x:
            return r
    if isinstance(x, Fraction) and x >= 0:
        rn, rd = math.isqrt(x.numerator), math.isqrt(x.denominator)
        if rn * rn == x.numerator and rd * rd == x.denominator:
            return Fraction(rn, rd)
    if isinstance(x, float):
        if x < 0:
            return _float_result(cmath.sqrt(x))
        return math.sqrt(x)
    if is_builtin_scalar(x) or isinstance(x, numbers.Number):
        return zero_form_class._from_nf(radical(nf_from_base(x), 2))
    raise _builtin_unsupported("sqrt")


def _elementary(name, x):
    if is_builtin_scalar(x) or isinstance(x, numbers.Rational):
        return abstract_ZF._from_nf(elementary(name, nf_from_base(x)))
    return _float_result(float_value(name, x))


def log(x):
    return _elementary("log", x)


def exp(x):
    return _elementary("exp", x)


def sin(x):
    return _elementary("sin", x)


def cos(x):
    return _elementary("cos", x)


def tan(x):
    return _elementary("tan", x)


def cot(x):
    return _elementary("cot", x)


def sec(x):
    return _elementary("sec", x)


def csc(x):
    return _elementary("csc", x)


def sinh(x):
    return _elementary("sinh", x)


def cosh(x):
    return _elementary("cosh", x)


def tanh(x):
    return _elementary("tanh", x)


def coth(x):
    return _elementary("coth", x)


def sech(x):
    return _elementary("sech", x)


def csch(x):
    return _elementary("csch", x)


def asin(x):
    return _elementary("asin", x)


def acos(x):
    return _elementary("acos", x)


def atan(x):
    return _elementary("atan", x)


def acot(x):
    return _elementary("acot", x)


def asec(x):
    return _elementary("asec", x)


def acsc(x):
    return _elementary("acsc", x)


def asinh(x):
    return _elementary("asinh", x)


def acosh(x):
    return _elementary("acosh", x)


def atanh(x):
    return _elementary("atanh", x)


def acoth(x):
    return _elementary("acoth", x)


def asech(x):
    return _elementary("asech", x)


def acsch(x):
    return _elementary("acsch", x)
