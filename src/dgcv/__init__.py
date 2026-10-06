"""
package: dgcv - Differential Geometry with Complex Variables

Description:
------------
The dgcv package (Differential Geometry with Complex Variables) provides
general tools for differential geometry together a framework for efficiently
working with complex variables.

This library is oriented toward supporting convenient syntax that is intuitive
in the context of its mathematics applications.

Dependencies:
-------------
There is no required dependency, but using dgcv along side at least one
supported CAS is recommended (currently supported: SymPy and Sage).
Optional integrations (e.g., IPython) are used automatically if present.

---
Author: David Gamble Sykes,

Project page for help and documentation: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# -----------------------------------------------------------------------------
# preliminary imports
# -----------------------------------------------------------------------------
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dgcv")
except PackageNotFoundError:
    __version__ = "unknown"


from ._aux._backends import (
    expand_dgcv,
    factor_dgcv,
    get_free_symbols,
    simplify_dgcv,
    subs_dgcv,
)
from ._aux._utilities._config import (
    canonicalize,
    configure_convenient_labels,
    configure_warnings,
)
from ._aux._utilities._settings import (
    reset_dgcv_settings,
    set_dgcv_settings,
    view_dgcv_settings,
)
from ._aux._utilities._styles import get_dgcv_themes
from ._aux._vmf.vmf import (
    clear_vmf,
    clearVar,
    listVar,
    variableSummary,
    vmf_lookup,
    vmf_summary,
)
from ._aux.printing.printing._dgcv_display import (
    LaTeX,
    LaTeX_eqn_system,
    LaTeX_list,
    show,
)
from ._aux.printing.printing._string_processing import clean_LaTeX
from .algebras import (
    adjointRepresentation,
    algebra_class,
    algebra_element_class,
    algebra_subspace_class,
    algebra_tools,
    algebraDataFromMatRep,
    createAlgebra,
    createSimpleLieAlgebra,
    killingForm,
    linear_representation,
    subalgebra_class,
    subalgebra_element,
    vector_space_endomorphisms,
)
from .algebras.creators.general import createFiniteAlg  # deprecated
from .ancillary.light_wrappers import function_dgcv
from .core import (
    Baker_Campbell_Hausdorff,
    LieDerivative,
    VF_bracket,
    allToHol,
    allToReal,
    allToSym,
    annihilator,
    antiholVF_coeffs,
    array_dgcv,
    assemble_block_matrix,
    assemble_tensor_field,
    assembleFromAntiholVFC,
    assembleFromCompVFC,
    assembleFromHolVFC,
    cleanUpConjugation,
    complex_struct_op,
    complexVFC,
    conj_with_hol_coor,
    conj_with_real_coor,
    conjugate_dgcv,
    coordinate_differential_form,
    coordinate_map,
    coordinate_vector_field,
    createBigradPolynomial,
    createMatrixCoordinates,
    createPolynomial,
    createRational,
    createVariables,
    decompose,
    differential_form_class,
    exteriorDerivative,
    exteriorProduct,
    get_coframe,
    get_DF,
    get_VF,
    getWeightedTerms,
    holToReal,
    holToSym,
    holVF_coeffs,
    homomorphism,
    im_with_hol_coor,
    im_with_real_coor,
    interiorProduct,
    makeZeroForm,
    matrix_dgcv,
    monomialWeight,
    multi_tensor_product,
    polynomial_dgcv,
    re_with_hol_coor,
    re_with_real_coor,
    realPartOfVF,
    realToHol,
    realToSym,
    solve_dgcv,
    symmetric_product,
    symToHol,
    symToReal,
    tensor_field_class,
    tensor_product,
    tensorProduct,
    vector_field_class,
    wedge,
)
from .core.dgcv_core.decprec import (
    DFClass,
    STFClass,
    VF_coeffs,
    VFClass,
    addDF,
    addVF,
    conjComplex,
    scaleDF,
    scaleVF,
    tensorField,
)
from .eds import (
    DF_representation,
    abst_coframe,
    abstract_coframe,
    abstract_DF,
    abstract_differential_form,
    abstract_ZF,
    coframe_derivative,
    createCoframe,
    createDiffForm,
    createZeroForm,
    extDer,
    simplify_with_PDEs,
    transform_coframe,
    zero_form_atom,
    zero_form_class,
    zeroFormAtom,
)
from .special_fields import filtration_tools
from .special_fields.complex_structures import Del, DelBar, KahlerStructure
from .special_fields.CR_geometry import (
    CR_structure,
    findWeightedCRSymmetries,
    model2Nondegenerate,
    tangencyObstruction,
    weightedHomogeneousVF,
)
from .special_fields.filtered_structures import (
    Tanaka_symbol,
    distribution,
    filtration_class,
)
from .special_fields.Riemannian_geometry import (
    metric_from_matrix,
    metricClass,
)

# -----------------------------------------------------------------------------
# broadcasting
# -----------------------------------------------------------------------------
__all__ = [
    "Baker_Campbell_Hausdorff",
    "CR_structure",
    "DF_representation",
    "Del",
    "DelBar",
    "KahlerStructure",
    "LaTeX",
    "LaTeX_eqn_system",
    "LaTeX_list",
    "LieDerivative",
    "Tanaka_symbol",
    "VF_bracket",
    "abst_coframe",
    "abstract_coframe",
    "abstract_DF",
    "abstract_ZF",
    "abstract_differential_form",
    "zero_form_class",
    "adjointRepresentation",
    "algebraDataFromMatRep",
    "algebra_class",
    "algebra_element_class",
    "algebra_subspace_class",
    "algebra_tools",
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
    "canonicalize",
    "cleanUpConjugation",
    "clean_LaTeX",
    "clearVar",
    "clear_vmf",
    "coframe_derivative",
    "complexVFC",
    "complex_struct_op",
    "configure_convenient_labels",
    "conj_with_hol_coor",
    "conj_with_real_coor",
    "conjugate_dgcv",
    "coordinate_differential_form",
    "coordinate_map",
    "coordinate_vector_field",
    "createAlgebra",
    "createBigradPolynomial",
    "createCoframe",
    "createDiffForm",
    "createMatrixCoordinates",
    "createPolynomial",
    "createRational",
    "createSimpleLieAlgebra",
    "createVariables",
    "createZeroForm",
    "decompose",
    "differential_form_class",
    "distribution",
    "expand_dgcv",
    "extDer",
    "exteriorDerivative",
    "exteriorProduct",
    "factor_dgcv",
    "filtration_class",
    "filtration_tools",
    "findWeightedCRSymmetries",
    "function_dgcv",
    "getWeightedTerms",
    "get_DF",
    "get_VF",
    "get_coframe",
    "get_dgcv_themes",
    "get_free_symbols",
    "holToReal",
    "holToSym",
    "holVF_coeffs",
    "homomorphism",
    "im_with_hol_coor",
    "im_with_real_coor",
    "interiorProduct",
    "killingForm",
    "linear_representation",
    "listVar",
    "makeZeroForm",
    "matrix_dgcv",
    "metricClass",
    "metric_from_matrix",
    "model2Nondegenerate",
    "monomialWeight",
    "multi_tensor_product",
    "polynomial_dgcv",
    "re_with_hol_coor",
    "re_with_real_coor",
    "realPartOfVF",
    "realToHol",
    "realToSym",
    "reset_dgcv_settings",
    "set_dgcv_settings",
    "show",
    "simplify_dgcv",
    "simplify_with_PDEs",
    "solve_dgcv",
    "subalgebra_class",
    "subalgebra_element",
    "subs_dgcv",
    "symToHol",
    "symToReal",
    "symmetric_product",
    "tangencyObstruction",
    "tensorField",
    "tensorProduct",
    "tensor_field_class",
    "tensor_product",
    "transform_coframe",
    "variableSummary",
    "vector_field_class",
    "vector_space_endomorphisms",
    "view_dgcv_settings",
    "vmf_lookup",
    "vmf_summary",
    "wedge",
    "weightedHomogeneousVF",
    "zeroFormAtom",
    "zero_form_atom",
] + [  # deprecated
    "DFClass",
    "STFClass",
    "VFClass",
    "VF_coeffs",
    "addDF",
    "addVF",
    "conjComplex",
    "createFiniteAlg",
    "scaleDF",
    "scaleVF",
]


# -----------------------------------------------------------------------------
# additional configurations
# -----------------------------------------------------------------------------
configure_warnings()
