"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.symbolic_router

module: dgcv.symbolic_router.__init__


---
Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from .._aux._backends._calculus import diff, integrate
from .._aux._backends._exact_arith import exact_reciprocal, factorial
from .._aux._backends._symbolic_router import (
    _scalar_is_minus_one as is_scalar_minus_one,
)
from .._aux._backends._symbolic_router import (
    _scalar_is_one as is_scalar_one,
)
from .._aux._backends._symbolic_router import (
    _scalar_is_zero as is_scalar_zero,
)
from .._aux._backends._symbolic_router import (
    _scalar_sign as scalar_sign,
)
from .._aux._backends._symbolic_router import (
    as_numer_denom,
    cancel,
    clear_denominators,
    collect,
    conjugate,
    defloat,
    exact_nonzero,
    exp,
    expand,
    factor,
    get_free_symbols,
    ilcm,
    im,
    log,
    ratio,
    re,
    simplify,
    sqrt,
    subs,
)
from .._aux._backends._symbolic_router import (
    gcd_routed as gcd,
)
from .._aux._backends._symbolic_router import (
    lcm_routed as lcm,
)
from .._aux._backends._types_and_constants import (
    as_engine_scalar,
    e_constant,
    expr_operands,
    half,
    imag_unit,
    integer,
    is_atomic,
    one,
    op_expr,
    rational,
    symbol,
    zero,
)
from . import polynomials

__all__ = [
    "as_engine_scalar",
    "as_numer_denom",
    "cancel",
    "clear_denominators",
    "collect",
    "conjugate",
    "defloat",
    "diff",
    "e_constant",
    "exact_nonzero",
    "exact_reciprocal",
    "exp",
    "expand",
    "op_expr",
    "expr_operands",
    "factor",
    "factorial",
    "gcd",
    "get_free_symbols",
    "half",
    "ilcm",
    "im",
    "imag_unit",
    "integer",
    "integrate",
    "is_atomic",
    "is_scalar_minus_one",
    "is_scalar_one",
    "is_scalar_zero",
    "lcm",
    "log",
    "one",
    "polynomials",
    "ratio",
    "rational",
    "re",
    "scalar_sign",
    "simplify",
    "sqrt",
    "subs",
    "symbol",
    "zero",
]


def symbols(*args):
    labels = []
    for arg in args:
        labels += [
            symbol(label)
            for word in arg.split(",")
            if word
            for label in word.split(" ")
            if label
        ]
    return labels
