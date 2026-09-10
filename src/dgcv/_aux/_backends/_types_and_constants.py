"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv._aux._backends

module: dgcv._aux._backends._types_and_constants


Description: aggregates various classes into mathematically meaningful categories

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers

from ._engine import (
    _get_sage_module,
    _get_sympy_module,
    engine_kind,
    is_sage_available,
    is_sympy_available,
)

# types
_atomic_pred = None
_constant_scalar_types = None
_expr_types = None
_expr_numeric_types = None
_fast_scalar_types = None


def invalidate_types_and_constants_cache():
    global \
        _atomic_pred, \
        _constant_scalar_types, \
        _expr_types, \
        _expr_numeric_types, \
        _fast_scalar_types, \
        _one_obj, \
        _zero_obj, \
        _I_obj, \
        _half_obj, \
        _sympy_conj_head, \
        _sympy_re_head, \
        _sympy_im_head, \
        _e_obj, \
        _op_table, \
        _op_engine, \
        _sage_constant_strs
    _atomic_pred = None
    _constant_scalar_types = None
    _expr_types = None
    _expr_numeric_types = None
    _fast_scalar_types = None
    _one_obj = None
    _zero_obj = None
    _I_obj = None
    _half_obj = None
    _e_obj = None
    _sympy_conj_head = None
    _sympy_re_head = None
    _sympy_im_head = None
    _op_table = None
    _op_engine = None
    _sage_constant_strs = None


def fast_scalar_types():
    global _fast_scalar_types
    if _fast_scalar_types is not None:
        return _fast_scalar_types

    types = []

    if is_sage_available():
        try:
            from sage.rings.integer import Integer as SageInteger  # type: ignore
            from sage.rings.rational import Rational as SageRational  # type: ignore

            types.extend([SageInteger, SageRational])
        except Exception:
            pass

    if is_sympy_available():
        try:
            sp = _get_sympy_module()
            types.extend([sp.Integer, sp.Rational])
        except Exception:
            pass

    _fast_scalar_types = tuple(types)
    return _fast_scalar_types


def constant_scalar_types():
    global _constant_scalar_types
    if _constant_scalar_types is not None:
        return _constant_scalar_types

    kind = engine_kind()

    types = [numbers.Number]  # catches int/float/complex/Fraction/Decimal/etc.

    if kind == "sympy":
        sp = _get_sympy_module()
        types.append(sp.Number)  # catches sympy Float, Integer, Rational, etc.
    elif kind == "sage":
        types.extend(list(fast_scalar_types()))
        from sage.rings.complex_mpfr import ComplexNumber  # type: ignore
        from sage.rings.real_mpfr import RealNumber  # type: ignore

        types.extend([RealNumber, ComplexNumber])

    _constant_scalar_types = tuple(types)
    return _constant_scalar_types


def expr_types():
    global _expr_types
    if _expr_types is not None:
        return _expr_types

    types = []

    if is_sage_available():
        try:
            from sage.symbolic.expression import (  # type: ignore
                Expression as SageExpression,  # type: ignore
            )

            types.append(SageExpression)
        except Exception:
            pass

    if is_sympy_available():
        try:
            sp = _get_sympy_module()
            types.append(sp.Expr)
        except Exception:
            pass

    _expr_types = tuple(types)
    return _expr_types


def expr_numeric_types():
    global _expr_numeric_types
    if _expr_numeric_types is not None:
        return _expr_numeric_types

    types = [numbers.Number]

    if is_sympy_available():
        try:
            sp = _get_sympy_module()
            types.append(sp.Expr)
        except Exception:
            pass

    if is_sage_available():
        try:
            from sage.structure.element import Element as SageElement  # type: ignore

            types.append(SageElement)
        except Exception:
            pass

    _expr_numeric_types = tuple(types)
    return _expr_numeric_types


def atomic_predicate():
    global _atomic_pred
    if _atomic_pred is not None:
        return _atomic_pred

    sp = None
    if is_sympy_available():
        try:
            sp = _get_sympy_module()
        except Exception:
            sp = None

    SageExpr = None
    if is_sage_available():
        try:
            from sage.symbolic.expression import (  # type: ignore
                Expression as SageExpr,  # type: ignore[assignment]
            )
        except Exception:
            SageExpr = None

    def pred(x):
        if SageExpr is not None and isinstance(x, SageExpr):
            try:
                return bool(x.is_symbol())
            except Exception:
                return False
        if sp is not None and isinstance(x, sp.Basic):
            try:
                return bool(x.is_Atom)
            except Exception:
                return False
        return False

    _atomic_pred = pred
    return _atomic_pred


def is_atomic(expr):
    return atomic_predicate()(expr)


# expression-tree introspection
_op_table = None
_op_engine = None

OP_ADD = "add"
OP_MUL = "mul"
OP_POW = "pow"
OP_SYMBOL = "symbol"
OP_NUMBER = "number"
OP_CONJUGATE = "conjugate"
OP_EXP = "exp"
OP_LOG = "log"
OP_FUNCTION = "function"
OP_CONSTANT = "constant"

_sage_constant_strs = None


def _build_op_table():
    global _sage_constant_strs
    kind = engine_kind()
    table = {}
    _sage_constant_strs = None

    if kind == "sage":
        try:
            sage = _get_sage_module()
            _sage_constant_strs = {
                str(c) for c in (sage.e, sage.pi, sage.I) if c is not None
            }
        except Exception:
            _sage_constant_strs = None

        try:
            sage = _get_sage_module()
            x = sage.SR.var("_dgcv_head_probe_x")
            y = sage.SR.var("_dgcv_head_probe_y")
            for expr, tag in (
                (x + y, OP_ADD),
                (x * y, OP_MUL),
                (x**y, OP_POW),
                (sage.exp(x), OP_EXP),
                (sage.log(x), OP_LOG),
                (sage.conjugate(x), OP_CONJUGATE),
            ):
                try:
                    op = expr.operator()
                except Exception:
                    continue
                if op is not None:
                    table[op] = tag
        except Exception:
            pass

    elif kind == "sympy":
        try:
            sp = _get_sympy_module()
            probe = sp.Symbol("_dgcv_head_probe")
            for expr, tag in (
                (sp.conjugate(probe), OP_CONJUGATE),
                (sp.exp(probe), OP_EXP),
                (sp.log(probe), OP_LOG),
            ):
                try:
                    table[expr.func] = tag
                except Exception:
                    continue
        except Exception:
            pass

    return table


def _get_op_table():
    global _op_table, _op_engine
    kind = engine_kind()
    if _op_table is None or _op_engine != kind:
        _op_table = _build_op_table()
        _op_engine = kind
    return _op_table


def op_expr(expr):
    kind = engine_kind()
    table = _get_op_table()

    if kind == "sympy":
        sp = _get_sympy_module()
        if isinstance(expr, sp.Basic):
            if isinstance(expr, sp.Number):
                return OP_NUMBER
            if isinstance(expr, sp.NumberSymbol) or expr is sp.I:
                return OP_CONSTANT
            if isinstance(expr, sp.Symbol):
                return OP_SYMBOL
            if isinstance(expr, sp.Add):
                return OP_ADD
            if isinstance(expr, sp.Mul):
                return OP_MUL
            if isinstance(expr, sp.Pow):
                return OP_POW
            tag = table.get(getattr(expr, "func", None))
            if tag is not None:
                return tag
            if isinstance(expr, sp.Function):
                return OP_FUNCTION
            return None
        if isinstance(expr, numbers.Number) and not isinstance(expr, bool):
            return OP_NUMBER
        return None

    if kind == "sage":
        try:
            from sage.symbolic.expression import (  # type: ignore
                Expression as SageExpression,  # type: ignore
            )
        except Exception:
            SageExpression = None

        if SageExpression is not None and isinstance(expr, SageExpression):
            try:
                if expr.is_symbol():
                    return OP_SYMBOL
            except Exception:
                pass
            if _sage_constant_strs:
                try:
                    if str(expr) in _sage_constant_strs:
                        return OP_CONSTANT
                except Exception:
                    pass
            try:
                op = expr.operator()
            except Exception:
                op = None
            if op is None:
                try:
                    if expr.is_numeric():
                        return OP_NUMBER
                except Exception:
                    pass
                return None
            tag = table.get(op)
            if tag is not None:
                return tag
            return OP_FUNCTION

        if _sage_constant_strs and not isinstance(expr, numbers.Number):
            try:
                if str(expr) in _sage_constant_strs:
                    return OP_CONSTANT
            except Exception:
                pass

        if isinstance(expr, constant_scalar_types()) and not isinstance(expr, bool):
            return OP_NUMBER
        return None

    if isinstance(expr, numbers.Number) and not isinstance(expr, bool):
        return OP_NUMBER
    return None


def expr_operands(expr):
    if op_expr(expr) == OP_CONSTANT:
        return ()

    kind = engine_kind()

    if kind == "sage":
        f = getattr(expr, "operands", None)
        if callable(f):
            try:
                return tuple(f())
            except Exception:
                return ()

    args = getattr(expr, "args", None)
    if args is None:
        return ()
    try:
        return tuple(args)
    except Exception:
        return ()


dgcv_expr_types = {"expression", "polynomial"}
q_dgcv_c = None


def check_dgcv_scalar(expr):
    global q_dgcv_c
    if q_dgcv_c is None:
        from .._vmf._safeguards import query_dgcv_categories

        q_dgcv_c = query_dgcv_categories
    return q_dgcv_c(expr, dgcv_expr_types) or isinstance(expr, expr_numeric_types())


# elements
_one_obj = None
_zero_obj = None
_e_obj = None
_I_obj = None
_half_obj = None


def symbol(name, assumptions=None):
    kind = engine_kind()

    if kind == "sympy":
        sp = _get_sympy_module()
        if not isinstance(assumptions, dict):
            return sp.Symbol(str(name))
        if assumptions.get("positive", False) is True:
            return sp.Symbol(str(name), positive=True)
        if assumptions.get("nonnegative", False) is True:
            return sp.Symbol(str(name), nonnegative=True)
        if assumptions.get("real", False) is True:
            return sp.Symbol(str(name), real=True)
        return sp.Symbol(str(name))

    if kind == "sage":
        sage = _get_sage_module()
        v = getattr(sage, "var", None)
        if v is None:
            raise RuntimeError(
                "dgcv: Sage backend is active but sage.var is unavailable"
            )

        nm = str(name)
        if not isinstance(assumptions, dict):
            return v(nm, domain="complex")
        if assumptions.get("positive", False) is True:
            var = v(nm)
            sage.assume(var > 0)
            return var
        if assumptions.get("nonnegative", False) is True:
            var = v(nm)
            sage.assume(var >= 0)
            return var
        if assumptions.get("real", False) is True:
            return v(nm, domain="real")
        return v(nm, domain="complex")

    return str(name)


def _disposable_symbol(name):
    kind = engine_kind()

    if kind == "sympy":
        return _get_sympy_module().Symbol(str(name))

    if kind == "sage":
        return _get_sage_module().var(str(name))

    return str(name)


def _disposable_symbols(prefix, count, start=0):
    return [_disposable_symbol(f"{prefix}{idx}") for idx in range(start, start + count)]


def one():
    global _one_obj
    if _one_obj is not None:
        return _one_obj
    kind = engine_kind()
    if kind == "sage":
        _one_obj = _get_sage_module().Integer(1)
    elif kind == "sympy":
        _one_obj = _get_sympy_module().S.One
    else:
        _one_obj = 1
    return _one_obj


def zero():
    global _zero_obj
    if _zero_obj is not None:
        return _zero_obj
    kind = engine_kind()
    if kind == "sage":
        _zero_obj = _get_sage_module().Integer(0)
    elif kind == "sympy":
        _zero_obj = _get_sympy_module().S.Zero
    else:
        _zero_obj = 0
    return _zero_obj


def imag_unit():
    global _I_obj
    if _I_obj is not None:
        return _I_obj

    kind = engine_kind()

    if kind == "sage":
        _I_obj = _get_sage_module().I
        return _I_obj

    if kind == "sympy":
        _I_obj = _get_sympy_module().I
        return _I_obj

    _I_obj = 1j
    return _I_obj


def half():
    global _half_obj
    if _half_obj is not None:
        return _half_obj
    _half_obj = rational(1, 2)
    return _half_obj


def e_constant():
    global _e_obj
    if _e_obj is not None:
        return _e_obj

    kind = engine_kind()

    if kind == "sage":
        _e_obj = _get_sage_module().e
        return _e_obj

    if kind == "sympy":
        _e_obj = _get_sympy_module().E
        return _e_obj

    import math

    _e_obj = math.e
    return _e_obj


def integer(n):
    n = int(n)
    kind = engine_kind()
    if kind == "sage":
        return _get_sage_module().Integer(n)
    if kind == "sympy":
        return _get_sympy_module().Integer(n)
    return n


def as_engine_scalar(x):
    if isinstance(x, bool) or not isinstance(x, numbers.Number):
        return x
    if isinstance(x, numbers.Integral):
        return integer(x)

    kind = engine_kind()
    if kind == "sympy":
        return _get_sympy_module().sympify(x)
    if kind == "sage":
        return _get_sage_module().SR(x)
    return x


def rational(p, q=1):
    """
    Construct an exact rational number from integer inputs.

    This function is intended **only** for ratios of integer-like values.

    Parameters
    ----------
    p : int-like
        Integer numerator.
    q : int-like, optional
        Integer denominator (default is 1).

    Returns
    -------
    scalar
        An exact rational scalar in the active symbolic engine.

    Warnings
    --------
    This function assumes that both `p` and `q` are integer-like and will coerce
    them using `int(p)` and `int(q)`. Passing symbolic rationals, expressions, or
    non-integer numeric types may lead to unintended truncation or division by zero.

    In particular, this function should **not** be used for general scalar division
    or inversion of symbolic quantities.

    For more general exact ratios of scalars, use the `ratio` function defined in
    `dgcv/backends/_symbolic_router.py` instead.

    See Also
    --------
    ratio : computes ratios of general scalar types.
    """
    p = int(p)
    q = int(q)
    kind = engine_kind()
    if kind == "sage":
        sage = _get_sage_module()
        return sage.Integer(p) / sage.Integer(q)
    if kind == "sympy":
        return _get_sympy_module().Rational(p, q)
    from fractions import Fraction

    return Fraction(p, q)


_sympy_conj_head = None


def _get_sympy_conj_head():
    global _sympy_conj_head
    if _sympy_conj_head is not None:
        return _sympy_conj_head

    if not is_sympy_available():
        _sympy_conj_head = None
        return None

    try:
        sp = _get_sympy_module()
        probe = sp.Symbol("_dgcv_conj_probe")
        _sympy_conj_head = sp.conjugate(probe).func
    except Exception:
        _sympy_conj_head = None

    return _sympy_conj_head


def verify_conjugates_free(expr) -> bool:
    kind = engine_kind()

    if kind == "sympy":
        h = getattr(expr, "has", None)
        if not callable(h):
            return False
        head = _get_sympy_conj_head()
        if head is None:
            return False
        try:
            return not bool(h(head))
        except Exception:
            return False

    if kind == "sage":
        try:
            sage = _get_sage_module()
            conj = getattr(sage, "conjugate", None)
            if conj is None:
                return False

            h = getattr(expr, "has", None)
            if callable(h):
                try:
                    return not bool(h(conj))
                except Exception:
                    pass

            seen = set()

            def _has_conj(e) -> bool:
                oid = id(e)
                if oid in seen:
                    return False
                seen.add(oid)

                op = getattr(e, "operator", None)
                if callable(op):
                    try:
                        if op() == conj:
                            return True
                    except Exception:
                        pass

                operands = getattr(e, "operands", None)
                if callable(operands):
                    try:
                        for a in operands():
                            if _has_conj(a):
                                return True
                    except Exception:
                        pass

                args = getattr(e, "args", None)
                if args is not None:
                    try:
                        for a in args:
                            if _has_conj(a):
                                return True
                    except Exception:
                        pass

                return False

            if _has_conj(expr):
                return False

            s = str(expr)
            if "conjugate(" in s or "Conjugate(" in s:
                return False

            return True

        except Exception:
            return False

    return False


_sympy_re_head = None
_sympy_im_head = None


def _get_sympy_re_head():
    global _sympy_re_head
    if _sympy_re_head is not None:
        return _sympy_re_head
    if not is_sympy_available():
        _sympy_re_head = None
        return None
    try:
        sp = _get_sympy_module()
        probe = sp.Symbol("_dgcv_re_probe")
        _sympy_re_head = sp.re(probe).func
    except Exception:
        _sympy_re_head = None
    return _sympy_re_head


def _get_sympy_im_head():
    global _sympy_im_head
    if _sympy_im_head is not None:
        return _sympy_im_head
    if not is_sympy_available():
        _sympy_im_head = None
        return None
    try:
        sp = _get_sympy_module()
        probe = sp.Symbol("_dgcv_im_probe")
        _sympy_im_head = sp.im(probe).func
    except Exception:
        _sympy_im_head = None
    return _sympy_im_head


def verify_conjugate_re_im_free(expr) -> bool:
    """
    Return True iff `expr` appears free of conjugation / re / im operators
    in the active symbolic engine.
    """
    kind = engine_kind()

    if kind == "sympy":
        h = getattr(expr, "has", None)
        if not callable(h):
            return False

        heads = []
        c = _get_sympy_conj_head()
        if c is not None:
            heads.append(c)
        r = _get_sympy_re_head()
        if r is not None:
            heads.append(r)
        i = _get_sympy_im_head()
        if i is not None:
            heads.append(i)

        if not heads:
            return False

        try:
            return not any(bool(h(head)) for head in heads)
        except Exception:
            return False

    if kind == "sage":
        try:
            sage = _get_sage_module()
            conj = getattr(sage, "conjugate", None)

            seen = set()

            def _has_bad_op(e) -> bool:
                oid = id(e)
                if oid in seen:
                    return False
                seen.add(oid)

                op = getattr(e, "operator", None)
                if callable(op):
                    try:
                        o = op()
                        if conj is not None and o is not None and bool(o == conj):
                            return True
                    except Exception:
                        pass

                operands = getattr(e, "operands", None)
                if callable(operands):
                    try:
                        for a in operands():
                            if _has_bad_op(a):
                                return True
                    except Exception:
                        pass

                args = getattr(e, "args", None)
                if args is not None:
                    try:
                        for a in args:
                            if _has_bad_op(a):
                                return True
                    except Exception:
                        pass

                return False

            if _has_bad_op(expr):
                return False

            s = str(expr)
            if "conjugate(" in s or "Conjugate(" in s:
                return False
            if "real(" in s or "RealPart(" in s or ".real()" in s:
                return False
            if "imag(" in s or "ImagPart(" in s or ".imag()" in s:
                return False

            return True

        except Exception:
            return False

    return False
