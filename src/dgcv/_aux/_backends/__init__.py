"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv._aux._backends

module: dgcv._aux._backends.__init__

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
# calculus-like operations
from ._calculus import diff as diff_dgcv
from ._calculus import integrate as integrate_dgcv
from ._groebner import groebner as groebner_dgcv

# expression/router operations
from ._symbolic_router import (
    as_numer_denom,
    get_free_symbols,
    ilcm,
    im,
    ratio,
    re,
)
from ._symbolic_router import (
    cancel as cancel_dgcv,
)
from ._symbolic_router import (
    conjugate as conjugate_dgcv_sym_router,
)
from ._symbolic_router import (
    expand as expand_dgcv,
)
from ._symbolic_router import (
    factor as factor_dgcv,
)
from ._symbolic_router import (
    simplify as simplify_dgcv,
)
from ._symbolic_router import sqrt as sqrt_dgcv
from ._symbolic_router import (
    subs as subs_dgcv,
)
from ._types_and_constants import symbol

__all__ = [
    # calculus
    "diff_dgcv",
    "integrate_dgcv",
    # router
    "as_numer_denom",
    "cancel_dgcv",
    "conjugate_dgcv_sym_router",
    "expand_dgcv",
    "factor_dgcv",
    "get_free_symbols",
    "groebner_dgcv",
    "ilcm",
    "im",
    "re",
    "ratio",
    "simplify_dgcv",
    "sqrt_dgcv",
    "subs_dgcv",
    "symbol",
]
