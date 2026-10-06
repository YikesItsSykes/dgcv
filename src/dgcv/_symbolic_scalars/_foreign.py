# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

import math
import numbers
import operator as _op
from fractions import Fraction

from .._aux._backends._engine import (
    _get_sage_module,
    _get_sympy_module,
    sympy_module_if_available,
)
from ..eds._atoms import zero_form_atom
from ..eds._constants import I, _BuiltinLeaf, named_constant
from ..eds._zero_forms import zero_form_class
from . import _poly
from ._functions import ELEMENTARY_NAMES, FunctionAtom, elementary, radical
from ._normal_form import (
    RF,
    _not_finite,
    base_from_nf,
    conj_rf,
    nf_from_base,
    rf_gen,
    rf_msum,
    rf_one,
)

_SAGE_HEADS = {
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
    "arccot": "acot",
    "arcsec": "asec",
    "arccsc": "acsc",
    "arcsinh": "asinh",
    "arccosh": "acosh",
    "arctanh": "atanh",
    "arccoth": "acoth",
    "arcsech": "asech",
    "arccsch": "acsch",
}


def _unsupported(what):
    return NotImplementedError(
        f"dgcv's builtin symbolic engine cannot convert the foreign expression head `{what}`"
    )


def _atom_for(name, real=False):
    return zero_form_atom(
        str(name), _markers=frozenset({"real"}) if real else frozenset()
    )


def _canon_arg(x):
    return base_from_nf(nf_from_base(x))


def _build(op, args):
    return zero_form_class((op, *args))


def _leaf_rf(leaf):
    if isinstance(leaf, (zero_form_atom, _BuiltinLeaf)):
        return rf_gen(leaf)
    return nf_from_base(leaf)


def _sympy_number(sp, e):
    if isinstance(e, sp.Integer):
        return int(e)
    if isinstance(e, sp.Rational):
        return Fraction(int(e.p), int(e.q))
    if isinstance(e, sp.Float):
        return float(e)
    return None


def _sympy_atom(symbol_map, e):
    mapped = symbol_map.get(e)
    if mapped is not None:
        return mapped
    return _atom_for(e.name, real=bool(e.is_real))


def _sympy_leaf(sp, symbol_map, e):
    value = _sympy_number(sp, e)
    if value is not None:
        return value
    if e is sp.I:
        return I
    if e is sp.E:
        return named_constant("e")
    if e is sp.pi:
        return named_constant("pi")
    if isinstance(e, sp.Symbol):
        return _sympy_atom(symbol_map, e)
    return None


def _sympy_function_head(sp, e):
    AppliedUndef = sp.core.function.AppliedUndef
    if isinstance(e, AppliedUndef):
        return e.func.__name__, e.args, None
    if isinstance(e, sp.Derivative):
        inner = e.expr
        if not isinstance(inner, AppliedUndef) or not all(
            isinstance(a, sp.Symbol) for a in inner.args
        ):
            raise _unsupported("Derivative")

        derivs = [0] * len(inner.args)
        for var, count in e.variable_count:
            if var not in inner.args:
                raise _unsupported("Derivative")
            derivs[list(inner.args).index(var)] += int(count)

        return inner.func.__name__, inner.args, derivs

    return None


def from_sympy(expr, symbol_map=None):
    sp = _get_sympy_module()
    heads = {getattr(sp, name): name for name in ELEMENTARY_NAMES}
    symbol_map = symbol_map or {}

    def conv(e):
        leaf = _sympy_leaf(sp, symbol_map, e)
        if leaf is not None:
            return leaf
        if isinstance(e, sp.Add):
            return _build("add", [conv(a) for a in e.args])
        if isinstance(e, sp.Mul):
            return _build("mul", [conv(a) for a in e.args])
        if isinstance(e, sp.Pow):
            b, ex = e.args
            return _build("pow", [conv(b), conv(ex)])

        if isinstance(e, sp.conjugate):
            return zero_form_class(conv(e.args[0]))._eval_conjugate()
        name = heads.get(getattr(e, "func", None))
        if name is not None:
            return zero_form_class._from_nf(
                elementary(name, nf_from_base(conv(e.args[0])))
            )

        head = _sympy_function_head(sp, e)
        if head is not None:
            label, args, derivs = head
            return zero_form_class(
                FunctionAtom(label, tuple(_canon_arg(conv(a)) for a in args), derivs)
            )

        if isinstance(e, bool):
            raise TypeError("builtin engine does not accept booleans as scalars")
        if isinstance(e, numbers.Number):
            return e
        raise _unsupported(type(e).__name__)

    return conv(expr)


def _sage_operators():
    from sage.symbolic.operators import (  # type: ignore
        FDerivativeOperator,
        add_vararg,
        mul_vararg,
    )

    return FDerivativeOperator, add_vararg, mul_vararg


def _sage_rational(sage, v):
    try:
        return int(sage.ZZ(v))
    except Exception:
        pass
    try:
        q = sage.QQ(v)
        return Fraction(int(q.numerator()), int(q.denominator()))
    except Exception:
        return None


def _sage_number(sage, v):
    value = _sage_rational(sage, v)
    if value is not None:
        return value
    parts = None
    try:
        parts = list(v.list())
    except Exception:
        pass
    if parts is not None and len(parts) == 2 and str(v.parent().gen()) == "I":
        re, im = (
            Fraction(int(sage.QQ(c).numerator()), int(sage.QQ(c).denominator()))
            for c in parts
        )
        return ("add", re, ("mul", im, I))
    try:
        value = float(v)
    except Exception:
        raise _unsupported(type(v).__name__) from None

    if not math.isfinite(value):
        raise _not_finite(v)
    return value


def _sage_leaf(sage, names, e):
    if isinstance(e, bool):
        raise TypeError("builtin engine does not accept booleans as scalars")
    if isinstance(e, numbers.Number):
        return e
    if not hasattr(e, "operator"):
        return _sage_number(sage, e)
    s = str(e)
    if e.is_symbol():
        mapped = names.get(s)
        if mapped is not None:
            return mapped
        return _atom_for(s)

    if s == "I":
        return I
    if s == "e":
        return named_constant("e")
    if s == "pi":
        return named_constant("pi")
    if e.operator() is not None:
        return None
    if e.is_numeric():
        return _sage_number(sage, e.pyobject())
    raise _unsupported(s)


def _sage_function_head(op, nargs, FDerivativeOperator):
    if isinstance(op, FDerivativeOperator):
        derivs = [0] * nargs
        for k in op.parameter_set():
            derivs[int(k)] += 1

        return str(op.function()), derivs

    if hasattr(op, "name") and callable(op.name):
        return str(op.name()), None
    return None


def from_sage(expr, symbol_map=None):
    FDerivativeOperator, add_vararg, mul_vararg = _sage_operators()
    sage = _get_sage_module()
    names = {str(k): v for k, v in (symbol_map or {}).items()}

    def conv(e):
        leaf = _sage_leaf(sage, names, e)
        if leaf is not None:
            return leaf
        op = e.operator()
        args = [conv(a) for a in e.operands()]
        if op is add_vararg:
            return _build("add", args)
        if op is mul_vararg:
            return _build("mul", args)
        if op is _op.pow:
            return _build("pow", args)
        name = str(op)
        name = _SAGE_HEADS.get(name, name)
        if name in ELEMENTARY_NAMES:
            return zero_form_class._from_nf(elementary(name, nf_from_base(args[0])))
        if name == "conjugate":
            return zero_form_class(args[0])._eval_conjugate()
        head = _sage_function_head(op, len(args), FDerivativeOperator)
        if head is not None:
            label, derivs = head
            return zero_form_class(
                FunctionAtom(label, tuple(_canon_arg(a) for a in args), derivs)
            )

        raise _unsupported(name)

    return conv(
        sage.SR(expr)
        if not hasattr(expr, "operator") and not isinstance(expr, numbers.Number)
        else expr
    )


def _rf_sum(args, monomial, conv):
    acc = {}
    parts = []
    for a in args:
        term = monomial(a)
        if term is None:
            parts.append(conv(a))
        else:
            m, c = term
            v = acc.get(m)
            if v is None:
                acc[m] = c
            else:
                v = v + c
                if v:
                    acc[m] = v
                else:
                    del acc[m]

    rest = []
    for rf in parts:
        if rf.is_polynomial:
            d = rf.den.get((), 1)
            num = rf.num
            if d != 1:
                num = _poly.p_scale(
                    num, (1.0 / d) if isinstance(d, float) else Fraction(1, d)
                )
            acc = _poly.p_add(acc, num) if acc else dict(num)
        else:
            rest.append(rf)

    out = RF(acc) if acc else None
    if rest:
        tail = rf_msum([(r, rf_one()) for r in rest])
        if tail is None:
            tail = rest[0]
            for r in rest[1:]:
                tail = tail + r
        out = tail if out is None else out + tail
    if out is None:
        return RF({}, canonical=True)
    return out


def from_sympy_rf(expr, symbol_map=None):
    sp = _get_sympy_module()
    heads = {getattr(sp, name): name for name in ELEMENTARY_NAMES}
    symbol_map = symbol_map or {}

    tree_failed = False

    def tree(e):
        nonlocal tree_failed
        try:
            return nf_from_base(from_sympy(e, symbol_map=symbol_map))
        except NotImplementedError:
            tree_failed = True
            raise

    def gid(e):
        if isinstance(e, sp.Symbol):
            return _poly.gen_id(_sympy_atom(symbol_map, e))
        if e is sp.I:
            return _poly.I_GEN
        return None

    def factor(e):
        g = gid(e)
        if g is not None:
            return (g, 1)
        if isinstance(e, sp.Pow):
            b, ex = e.args
            if isinstance(ex, sp.Integer) and int(ex) > 0:
                g = gid(b)
                if g is not None:
                    return (g, int(ex))
        return None

    def monomial(e):
        c = _sympy_number(sp, e)
        if c is not None:
            return (), c
        if isinstance(e, sp.Mul):
            c = 1
            pairs = []
            for f in e.args:
                v = _sympy_number(sp, f)
                if v is not None:
                    c = c * v
                    continue

                pair = factor(f)
                if pair is None:
                    return None
                pairs.append(pair)

            if len({g for g, _ in pairs}) != len(pairs):
                return None
            return tuple(sorted(pairs)), c

        pair = factor(e)
        return None if pair is None else ((pair,), 1)

    def conv(e):
        try:
            return _conv(e)
        except NotImplementedError:
            if tree_failed:
                raise
            return tree(e)

    def _conv(e):
        leaf = _sympy_leaf(sp, symbol_map, e)
        if leaf is not None:
            return _leaf_rf(leaf)
        if isinstance(e, sp.Add):
            return _rf_sum(e.args, monomial, conv)
        if isinstance(e, sp.Mul):
            out = rf_one()
            for a in e.args:
                out = out * conv(a)

            return out

        if isinstance(e, sp.Pow):
            b, ex = e.args
            if isinstance(ex, sp.Integer):
                return conv(b) ** int(ex)
            if isinstance(ex, sp.Rational):
                b_rf = conv(b)
                p, q = int(ex.p), int(ex.q)
                out = radical(b_rf, q) ** (p % q)
                if p // q:
                    out = out * b_rf ** (p // q)

                return out

            if b is sp.E:
                return elementary("exp", conv(ex))
            return tree(e)

        if isinstance(e, sp.conjugate):
            return conj_rf(conv(e.args[0]))
        name = heads.get(getattr(e, "func", None))
        if name is not None:
            return elementary(name, conv(e.args[0]))
        head = _sympy_function_head(sp, e)
        if head is not None:
            label, args, derivs = head
            return rf_gen(
                FunctionAtom(label, tuple(base_from_nf(conv(a)) for a in args), derivs)
            )

        return tree(e)

    return conv(expr)


def from_sage_rf(expr, names=None):
    FDerivativeOperator, add_vararg, mul_vararg = _sage_operators()
    sage = _get_sage_module()
    if names is None:
        names = {}

    tree_failed = False

    def tree(e):
        nonlocal tree_failed
        try:
            return nf_from_base(from_sage(e, symbol_map=names))
        except NotImplementedError:
            tree_failed = True
            raise

    def gid(x):
        if not hasattr(x, "operator") or not x.is_symbol():
            return None
        s = str(x)
        mapped = names.get(s)
        return _poly.gen_id(mapped if mapped is not None else _atom_for(s))

    def factor_pair(f):
        g = gid(f)
        if g is not None:
            return (g, 1)
        if hasattr(f, "operator") and f.operator() is _op.pow:
            b, ex = f.operands()
            n = _sage_rational(sage, ex)
            if isinstance(n, int) and n > 0:
                g = gid(b)
                if g is not None:
                    return (g, n)
        return None

    def monomial(x):
        if isinstance(x, numbers.Number) and not isinstance(x, bool):
            return (), x
        if not hasattr(x, "operator"):
            c = _sage_rational(sage, x)
            return None if c is None else ((), c)

        if x.is_numeric():
            c = _sage_rational(sage, x.pyobject())
            return None if c is None else ((), c)

        if x.is_symbol():
            return ((gid(x), 1),), 1
        op = x.operator()
        if op is _op.pow:
            pair = factor_pair(x)
            return None if pair is None else ((pair,), 1)

        if op is mul_vararg:
            c = 1
            pairs = []
            for f in x.operands():
                if hasattr(f, "is_numeric") and f.is_numeric():
                    v = _sage_rational(sage, f.pyobject())
                    if v is None:
                        return None
                    c = c * v
                    continue

                pair = factor_pair(f)
                if pair is None:
                    return None
                pairs.append(pair)

            if len({g for g, _ in pairs}) != len(pairs):
                return None
            return tuple(sorted(pairs)), c

        return None

    def conv(e):
        try:
            return _conv(e)
        except NotImplementedError:
            if tree_failed:
                raise
            return tree(e)

    def _conv(e):
        leaf = _sage_leaf(sage, names, e)
        if leaf is not None:
            return _leaf_rf(leaf)
        op = e.operator()
        operands = e.operands()
        if op is add_vararg:
            return _rf_sum(operands, monomial, conv)
        if op is mul_vararg:
            out = rf_one()
            for a in operands:
                out = out * conv(a)

            return out

        if op is _op.pow:
            b, ex = operands
            try:
                q = sage.QQ(ex)
            except Exception:
                q = None

            if q is not None:
                p, qd = int(q.numerator()), int(q.denominator())
                if qd == 1:
                    return conv(b) ** p
                b_rf = conv(b)
                out = radical(b_rf, qd) ** (p % qd)
                if p // qd:
                    out = out * b_rf ** (p // qd)

                return out

            if not b.is_symbol() and str(b) == "e":
                return elementary("exp", conv(ex))
            return tree(e)

        name = str(op)
        name = _SAGE_HEADS.get(name, name)
        if name in ELEMENTARY_NAMES:
            return elementary(name, conv(operands[0]))
        if name == "conjugate":
            return conj_rf(conv(operands[0]))
        head = _sage_function_head(op, len(operands), FDerivativeOperator)
        if head is not None:
            label, derivs = head
            return rf_gen(
                FunctionAtom(
                    label, tuple(base_from_nf(conv(a)) for a in operands), derivs
                )
            )

        return tree(e)

    return conv(
        sage.SR(expr)
        if not hasattr(expr, "operator") and not isinstance(expr, numbers.Number)
        else expr
    )


def from_foreign(x):
    mod = type(x).__module__ or ""
    sp = sympy_module_if_available()
    if mod.startswith("sympy") or (sp is not None and isinstance(x, sp.Basic)):
        return from_sympy(x)
    if mod.startswith("sage"):
        return from_sage(x)
    return x
