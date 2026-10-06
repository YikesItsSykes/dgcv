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
from .._aux._vmf._safeguards import retrieve_passkey
from .._aux.printing.printing._dgcv_display import LaTeX
from .._aux.printing.printing._string_processing import _process_label, _verbose_labels

__all__ = ["function_dgcv"]


def _sympy_function_dgcv(name):
    sp = engine_module()

    class _function_dgcv(sp.Function):
        @classmethod
        def eval(cls, *args):
            return None

        def _sympystr(self, printer):
            return self.func.__name__

        def _latex(self, printer, **kwargs):
            tex = _process_label(self.func.__name__)
            exp = kwargs.get("exp")
            tail = ""
            if _verbose_labels() and self.args:
                tail = LaTeX(self.args)
            if exp:
                tex = f"{tex}^{{{exp}}}"
            return tex + tail

    return type(name, (_function_dgcv,), {})


def _sage_function_dgcv(name):
    sage = engine_module()
    tex = _process_label(name)

    def print_latex(self, *args):
        if _verbose_labels() and args:
            return f"{tex}\\left({', '.join(str(arg) for arg in args)}\\right)"
        return tex

    return sage.function(name, print_latex_func=print_latex)


def function_dgcv(name: str, real: bool = False, deriv_style: str = "subscript"):
    name = str(name)
    kind = engine_kind()
    if kind == "sympy":
        cls = _sympy_function_dgcv(name)
    elif kind == "sage":
        cls = _sage_function_dgcv(name)
    else:
        return engine_capability("function_head")(
            name, real=real, deriv_style=deriv_style
        )
    cls._dgcv_class_check = retrieve_passkey()
    cls._dgcv_category = "function"
    return cls
