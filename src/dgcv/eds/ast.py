"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

module: dgcv.eds.ast

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from .._aux._utilities._config import dgcv_warning, dgcvDeprecationWarning
from ._zero_forms import zero_form_class

__all__ = []


def __getattr__(name):
    if name == "abstract_ZF_test":
        dgcv_warning(
            "`dgcv.eds.ast.abstract_ZF_test` has been folded into `dgcv.eds.abstract_ZF`.",
            dgcvDeprecationWarning,
            stacklevel=2,
            old_kw="abstract_ZF_test",
            new_kw="abstract_ZF",
            sunset="2027",
        )
        return zero_form_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
