"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.core.vector_fields_and_differential_forms


---
Author (of this sub-package): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from .annihilators import annihilator
from .cartan import (
    LieDerivative,
    _prep_symb_set_for_ext_der,
    exteriorDerivative,
    interiorProduct,
    makeZeroForm,
)
from .coframes import get_coframe, get_frame
from .complex_assembly import (
    assembleFromAntiholVFC,
    assembleFromCompVFC,
    assembleFromHolVFC,
)
from .decomposition import decompose
from .independence import _extract_basis_by_wedge_vectorized
from .retrieval import (
    coordinate_differential_form,
    coordinate_vector_field,
    get_DF,
    get_VF,
)

__all__ = [
    "_extract_basis_by_wedge_vectorized",
    "LieDerivative",
    "annihilator",
    "assembleFromAntiholVFC",
    "assembleFromCompVFC",
    "assembleFromHolVFC",
    "_prep_symb_set_for_ext_der",
    "coordinate_differential_form",
    "coordinate_vector_field",
    "decompose",
    "exteriorDerivative",
    "get_DF",
    "get_VF",
    "get_coframe",
    "get_frame",
    "interiorProduct",
    "makeZeroForm",
]
