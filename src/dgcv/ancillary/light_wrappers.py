"""
package: dgcv - Differential Geometry with Complex Variables
module: dgcv.light_wrappers

Author (of this module): David Gamble Sykes
Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes
Licensed under the Apache License, Version 2.0
SPDX-License-Identifier: Apache-2.0
"""

from .._aux._backends._engine import engine_capability, engine_kind, engine_module
from .._aux._backends._types_and_constants import _get_sympy_conj_head
from .._aux._vmf._function_arguments import (
    complete_function_arguments,
    function_properties,
)
from .._aux._vmf._safeguards import retrieve_passkey
from .._aux.printing.printing._dgcv_display import LaTeX
from .._aux.printing.printing._string_processing import _process_label, _verbose_labels

__all__ = ["function_dgcv"]


def _sympy_derivative_head(root, derivs):
    cache = root.__dict__.get("_dgcv_derivative_heads")
    if cache is None:
        cache = {}
        root._dgcv_derivative_heads = cache
    head = cache.get(derivs)
    if head is None:
        indices = "_".join(str(k + 1) for k, n in enumerate(derivs) for _ in range(n))
        head = type(
            f"D_{indices}({root.__name__})",
            (root,),
            {"_dgcv_derivs": derivs, "_dgcv_root": root},
        )
        cache[derivs] = head
    return head


def _sympy_function_dgcv(name, real=False, deriv_style="subscript"):
    sp = engine_module()
    if deriv_style not in ("subscript", "partial"):
        raise ValueError(
            f"`deriv_style` is 'subscript' or 'partial', received {deriv_style!r}"
        )

    class _function_dgcv(sp.Function):
        _dgcv_label = name
        _dgcv_derivs = None
        _dgcv_root = None
        _dgcv_real = bool(real)
        _dgcv_partial_style = deriv_style == "partial"

        @classmethod
        def eval(cls, *args):
            return None

        def fdiff(self, argindex=1):
            derivs = list(self._dgcv_derivs or (0,) * len(self.args))
            derivs[argindex - 1] += 1
            root = self._dgcv_root or self.func
            return _sympy_derivative_head(root, tuple(derivs))(*self.args)

        def _dgcv_indices(self, tex):
            out = []
            for k, n in enumerate(self._dgcv_derivs):
                a = self.args[k]
                if isinstance(a, sp.Symbol):
                    lab = _process_label(str(a)) if tex else str(a)
                elif isinstance(a, _get_sympy_conj_head()) and isinstance(a.args[0], sp.Symbol):
                    inner = a.args[0]
                    lab = (
                        f"\\overline{{{_process_label(str(inner))}}}"
                        if tex
                        else f"BAR{inner}"
                    )
                else:
                    lab = str(k + 1)
                out.extend([lab] * n)
            return out

        def _sympystr(self, printer):
            head = self._dgcv_label
            if self._dgcv_derivs and any(self._dgcv_derivs):
                head = f"D_{'_'.join(self._dgcv_indices(False))}({head})"
            return head

        def _latex(self, printer, **kwargs):
            tex = _process_label(self._dgcv_label)
            if self._dgcv_derivs and any(self._dgcv_derivs):
                indices = self._dgcv_indices(True)
                if self._dgcv_partial_style:
                    order = len(indices)
                    sup = f"^{{{order}}}" if order > 1 else ""
                    dens = "".join(f"\\partial {lab}" for lab in indices)
                    tex = f"\\frac{{\\partial{sup} {tex}}}{{{dens}}}"
                elif "_" in tex or "^" in tex:
                    tex = f"{{{tex}}}_{{{' '.join(indices)}}}"
                else:
                    tex = f"{tex}_{{{' '.join(indices)}}}"
            exp = kwargs.get("exp")
            tail = ""
            if _verbose_labels() and self.args:
                tail = LaTeX(self.args)
            if exp:
                tex = f"{tex}^{{{exp}}}"
            return tex + tail

        def _repr_latex_(self, raw=False, **kwargs):
            tex = sp.latex(self)
            return tex if raw else f"$\\displaystyle {tex}$"

    return type(name, (_function_dgcv,), {})


def _sage_function_dgcv(name):
    sage = engine_module()
    tex = _process_label(name)

    def print_latex(self, *args):
        if _verbose_labels() and args:
            return f"{tex}\\left({', '.join(str(arg) for arg in args)}\\right)"
        return tex

    return sage.function(name, print_latex_func=print_latex)


class _function_dgcv_head:
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "function"

    def __init__(self, head, name, dependence, properties):
        self._head = head
        self.label = name
        self.__name__ = name
        self.dependence = dependence
        self.properties = properties

    def __call__(self, *args):
        return self._head(
            *complete_function_arguments(self.label, args, self.dependence)
        )

    def __repr__(self):
        names = ", ".join(repr(p) for p in sorted(self.properties))
        return f"function_dgcv({self.label!r}, properties={{{names}}})"

    def __str__(self):
        return self.label


def function_dgcv(
    name: str, real: bool = False, deriv_style: str = "subscript", properties=None
):
    name = str(name)
    real, dependence, names = function_properties(name, properties, real)
    kind = engine_kind()
    if kind == "sympy":
        head = _sympy_function_dgcv(name, real=real, deriv_style=deriv_style)
    elif kind == "sage":
        head = _sage_function_dgcv(name)
    else:
        head = None
    if head is None:
        head = engine_capability("function_head")(
            name, real=real, deriv_style=deriv_style
        )
    else:
        head._dgcv_class_check = retrieve_passkey()
        head._dgcv_category = "function"
    if dependence is None:
        return head
    return _function_dgcv_head(head, name, dependence, names)
