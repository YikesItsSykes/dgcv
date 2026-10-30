"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv.secondary_library


---
Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from ._aux._backends._symbolic_router import _scalar_is_zero as is_scalar_zero
from ._aux._backends._symbolic_router import defloat
from ._aux._utilities._misc import (
    evaluate_at_random_point,
    linear_combination,
    zip_sum,
)
from .branching_analysis import case_tree
from .core.arrays import frozen_array_dgcv, frozen_matrix_dgcv
from .core.vector_fields_and_differential_forms.decomposition import (
    _decompose_over_number_field as decompose_over_number_field,
)
from .core.vector_fields_and_differential_forms.decomposition import (
    _extract_basis_over_function_ring as extract_basis_over_function_ring,
)
from .core.vector_fields_and_differential_forms.decomposition import (
    _extract_basis_over_number_field as extract_basis_over_number_field,
)
from .core.vector_fields_and_differential_forms.independence import (
    _extract_basis_by_wedge_vectorized as extract_basis_by_wedge_vectorized,
)

__all__ = [
    "case_tree",
    "decompose_over_number_field",
    "defloat",
    "evaluate_at_random_point",
    "extract_basis_by_wedge_vectorized",
    "extract_basis_over_function_ring",
    "extract_basis_over_number_field",
    "frozen_matrix_dgcv",
    "frozen_array_dgcv",
    "is_scalar_zero",
    "linear_combination",
    "zip_sum",
]
