"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.core

Description:
------------
Defining dgcv's core classes and utilities.

Dependency note: Positioned upstream from all non-utility subpackages in the library.


---
Author (of this sub-package): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from .arrays import (
    array_dgcv,
    assemble_block_matrix,
    frozen_array_dgcv,
    frozen_matrix_dgcv,
    matrix_dgcv,
)
from .combinatorics import Baker_Campbell_Hausdorff, permSign
from .conversions import (
    allToHol,
    allToReal,
    allToSym,
    cleanUpConjugation,
    holToReal,
    holToSym,
    realToHol,
    realToSym,
    symToHol,
    symToReal,
)
from .dgcv_core import (
    VF_bracket,
    antiholVF_coeffs,
    assemble_tensor_field,
    complex_struct_op,
    complexVFC,
    conj_with_hol_coor,
    conj_with_real_coor,
    conjugate_dgcv,
    createMatrixCoordinates,
    createVariables,
    differential_form_class,
    exteriorProduct,
    holVF_coeffs,
    im_with_hol_coor,
    im_with_real_coor,
    polynomial_dgcv,
    re_with_hol_coor,
    re_with_real_coor,
    realPartOfVF,
    symmetric_product,
    tensor_field_class,
    tensor_product,
    vector_field_class,
    wedge,
)
from .morphisms import coordinate_map, homomorphism
from .polynomials import (
    createBigradPolynomial,
    createPolynomial,
    createRational,
    getWeightedTerms,
    monomialWeight,
)
from .solvers import solve_dgcv
from .tensors import multi_tensor_product, tensorProduct
from .vector_fields_and_differential_forms import (
    LieDerivative,
    annihilator,
    assembleFromAntiholVFC,
    assembleFromCompVFC,
    assembleFromHolVFC,
    coordinate_differential_form,
    coordinate_vector_field,
    decompose,
    exteriorDerivative,
    get_coframe,
    get_frame,
    get_DF,
    get_VF,
    interiorProduct,
    makeZeroForm,
)

__all__ = [
    "Baker_Campbell_Hausdorff",
    "LieDerivative",
    "VF_bracket",
    "allToHol",
    "allToReal",
    "allToSym",
    "annihilator",
    "antiholVF_coeffs",
    "array_dgcv",
    "assembleFromAntiholVFC",
    "assembleFromCompVFC",
    "assembleFromHolVFC",
    "assemble_block_matrix",
    "assemble_tensor_field",
    "cleanUpConjugation",
    "complexVFC",
    "complex_struct_op",
    "conj_with_hol_coor",
    "conj_with_real_coor",
    "conjugate_dgcv",
    "coordinate_differential_form",
    "coordinate_map",
    "coordinate_vector_field",
    "createBigradPolynomial",
    "createMatrixCoordinates",
    "createPolynomial",
    "createRational",
    "createVariables",
    "decompose",
    "differential_form_class",
    "exteriorDerivative",
    "exteriorProduct",
    "frozen_array_dgcv",
    "frozen_matrix_dgcv",
    "getWeightedTerms",
    "get_DF",
    "get_VF",
    "get_coframe",
    "get_frame",
    "holToReal",
    "holToSym",
    "holVF_coeffs",
    "homomorphism",
    "im_with_hol_coor",
    "im_with_real_coor",
    "interiorProduct",
    "makeZeroForm",
    "matrix_dgcv",
    "monomialWeight",
    "multi_tensor_product",
    "permSign",
    "polynomial_dgcv",
    "re_with_hol_coor",
    "re_with_real_coor",
    "realPartOfVF",
    "realToHol",
    "realToSym",
    "solve_dgcv",
    "symToHol",
    "symToReal",
    "symmetric_product",
    "tensorProduct",
    "tensor_field_class",
    "tensor_product",
    "vector_field_class",
    "wedge",
]
