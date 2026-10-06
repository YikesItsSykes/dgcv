# dgcv/backends/_display.py

from __future__ import annotations

import numbers
import threading

from .._vmf._safeguards import check_dgcv_category
from ._engine import (
    _get_sympy_module,
    engine_kind,
    is_sage_available,
    is_sympy_available,
)

_tls = threading.local()


def _enter_guard(obj) -> bool:
    stack = getattr(_tls, "stack", None)
    if stack is None:
        stack = set()
        _tls.stack = stack
    oid = id(obj)
    if oid in stack:
        return False
    stack.add(oid)
    return True


def _exit_guard(obj) -> None:
    stack = getattr(_tls, "stack", None)
    if stack is None:
        return
    stack.discard(id(obj))


def _strip_math_delims(s: str) -> str:
    t = s.strip()
    if t.startswith("$$") and t.endswith("$$") and len(t) >= 4:
        return t[2:-2].strip()
    if t.startswith("$") and t.endswith("$") and len(t) >= 2:
        return t[1:-1].strip()
    return t


def _is_sympy_obj(x) -> bool:
    if not is_sympy_available():
        return False
    try:
        sp = _get_sympy_module()
        return isinstance(x, sp.Basic)
    except Exception:
        return False


def latex(expr, **kwargs) -> str:
    if expr is None:
        return ""

    if not _enter_guard(expr):
        try:
            return str(expr)
        except Exception:
            return repr(expr)

    try:
        if check_dgcv_category(expr):
            return _strip_math_delims(expr._repr_latex_(raw=True))

        if isinstance(expr, numbers.Rational) and not isinstance(
            expr, (bool, numbers.Integral)
        ):
            n, d = expr.numerator, expr.denominator
            if n < 0:
                return f"- \\frac{{{-n}}}{{{d}}}"
            return f"\\frac{{{n}}}{{{d}}}"

        if _is_sympy_obj(expr):
            try:
                from sympy.printing.latex import LatexPrinter  # type: ignore

                return LatexPrinter(settings=kwargs).doprint(expr)
            except Exception:
                try:
                    sp = _get_sympy_module()
                    return sp.latex(expr, **kwargs)
                except Exception:
                    try:
                        return str(expr)
                    except Exception:
                        return repr(expr)

        if engine_kind() == "sage" and is_sage_available():
            try:
                from sage.misc.latex import latex as sage_latex  # type: ignore

                return str(sage_latex(expr))
            except Exception:
                pass

        f = getattr(expr, "_repr_latex_", None)
        if callable(f):
            try:
                s = f()
                if isinstance(s, str):
                    return _strip_math_delims(s)
            except Exception:
                pass

        try:
            return str(expr)
        except Exception:
            return repr(expr)

    finally:
        _exit_guard(expr)


def fast_printable(expr, max_nodes=500, return_count=False):

    is_sympy = _is_sympy_obj(expr)

    count = 0
    stack = [expr]

    while stack:
        current = stack.pop()
        count += 1
        if count > max_nodes:
            if return_count:
                return count
            return False

        base = getattr(current, "base", None)
        if (
            base is not None
            and getattr(current, "_dgcv_category", None) == "abstract_ZF"
        ):
            stack.append(base)
            continue
        if isinstance(current, tuple) and current and isinstance(current[0], str):
            stack.extend(current[1:])
            continue
        if is_sympy:
            if hasattr(current, "args") and current.args:
                stack.extend(current.args)
        else:  # assuming sage
            if hasattr(current, "operands"):
                try:
                    ops = current.operands()
                    if ops:
                        stack.extend(ops)
                except (AttributeError, TypeError):
                    pass
    if return_count:
        return count
    return True
