from __future__ import annotations

import numbers

from ..._aux._backends._symbolic_router import get_free_symbols
from ..._aux._backends._types_and_constants import expr_numeric_types
from ..._aux._utilities._config import dgcv_exception_note, dgcv_warning
from ..._aux._vmf._safeguards import create_key, retrieve_passkey, unique_label
from ...core.arrays import matrix_dgcv
from ...core.dgcv_core.spaces.spaces import _vs_card
from ...core.tensors import tensorProduct
from ..aec import algebra_element_class
from ..linear_algebra import (
    _flatten_structure_data,
    _gather_structure_singularities,
    _structure_array,
)
from .validation import _validate_structure_data


def _alg_init(
    target_alg,
    structure_data,
    grading=None,
    base_field="complex",
    format_sparse=False,
    process_matrix_rep=False,
    preferred_representation=None,
    simplify_products_by_default=None,
    assume_skew=False,
    matrix_representation=None,
    tensor_representation=None,
    _basis_labels_parent=None,
    _label=None,
    _basis_labels=None,
    _calledFromCreator=None,
    _callLock=None,
    _print_warning=None,
    _child_print_warning=None,
    _exclude_from_VMF=None,
    _markers={},
):
    if isinstance(structure_data, numbers.Integral):
        if structure_data >= 0:
            structure_data = _structure_array(dict(), structure_data)
    if _calledFromCreator == retrieve_passkey():
        validated_structure_data = structure_data
        params = _markers.get("parameters", set())
    else:
        try:
            vsd = _validate_structure_data(
                structure_data,
                process_matrix_rep=process_matrix_rep,
                assume_skew=assume_skew,
                basis_order_for_supplied_str_eqns=False,
            )
            if process_matrix_rep is True:
                if matrix_representation is not None:
                    dgcv_warning(
                        "The `algebra_class` initializer disregarded the optional parameter value given for `matrix_representation` because `process_matrix_rep` was set to `True`, which forces automated computation of the representation."
                    )
                validated_structure_data, matrix_representation, params = (
                    vsd[0][0],
                    vsd[0][1],
                    vsd[0][2],
                )
            else:
                validated_structure_data, params = vsd
                if not isinstance(params, set):
                    params = set()
                params |= get_free_symbols(
                    validated_structure_data
                )  ###!!! fix vsd to remove redundancy

        except dgcv_exception_note as e:
            raise SystemExit(e)
    target_alg.structureData = validated_structure_data
    target_alg.dimension = target_alg.structureData.shape[0]
    target_alg._parameters = params
    target_alg._tex_label = None
    target_alg._tex_basis_labels = None
    target_alg._educed_properties = dict()

    def _assign_composite_labels():
        if _markers.get("registered", None) is False:
            incoming_tex_label = _markers.get("_tex_label", None)
            if incoming_tex_label is None:
                target_alg.label = unique_label(_label)
                target_alg._tex_label = None
            else:
                target_alg.label, target_alg._tex_label = unique_label(
                    _label, tex_label=incoming_tex_label
                )

            incoming_basis = list(_basis_labels or [])
            incoming_tex_basis = list(_markers.get("_tex_basis_labels", []) or [])
            have_tex_basis = len(incoming_basis) > 0 and len(incoming_tex_basis) == len(
                incoming_basis
            )

            new_basis = []
            new_tex_basis = [] if have_tex_basis else None
            batch_protected = set()
            if isinstance(target_alg.label, str):
                batch_protected.add(target_alg.label)
            for idx, base_lbl in enumerate(incoming_basis):
                candidate = base_lbl
                if have_tex_basis:
                    base_tex = incoming_tex_basis[idx]
                    final_lbl, final_tex = unique_label(
                        candidate, tex_label=base_tex, protected=batch_protected
                    )
                    new_basis.append(final_lbl)
                    new_tex_basis.append(final_tex)
                    batch_protected.add(final_lbl)
                else:
                    final_lbl = unique_label(candidate, protected=batch_protected)
                    new_basis.append(final_lbl)
                    batch_protected.add(final_lbl)

            target_alg.basis_labels = new_basis
            if have_tex_basis:
                target_alg._tex_basis_labels = new_tex_basis
            elif target_alg._tex_label is not None:
                target_alg._tex_basis_labels = [
                    f"{target_alg._tex_label}_{{{i + 1}}}"
                    for i in range(target_alg.dimension)
                ]
        else:
            target_alg.label = _label
            target_alg.basis_labels = _basis_labels
            if not target_alg.basis_labels:
                base = (
                    target_alg.label
                    if isinstance(target_alg.label, str) and target_alg.label
                    else "_e"
                )
                target_alg.basis_labels = [
                    f"{base}{i + 1}" for i in range(target_alg.dimension)
                ]
            if _markers.get("_tex_label", None) is not None:
                target_alg._tex_label = _markers["_tex_label"]
            if _markers.get("_tex_basis_labels", None) is not None:
                target_alg._tex_basis_labels = _markers["_tex_basis_labels"]
            elif (
                target_alg._tex_label is not None
                and target_alg._tex_basis_labels is None
            ):
                target_alg._tex_basis_labels = [
                    f"{target_alg._tex_label}_{{{i + 1}}}"
                    for i in range(target_alg.dimension)
                ]

    if _calledFromCreator == retrieve_passkey():
        if isinstance(_markers.get("_educed_properties", None), dict):
            target_alg._educed_properties = _markers.get("_educed_properties", dict())
        if _markers.get("endo", False):
            if _label is None:
                target_alg.label = f"gl_{_markers.get('endo_label', '')}"
                target_alg._tex_label = (
                    f"\\mathfrak{{gl}}\\left({_markers.get('endo_tex', '')}\\right)"
                )
                target_alg.basis_labels = [
                    f"{target_alg.label}{i + 1}" for i in range(target_alg.dimension)
                ]
                target_alg._tex_basis_labels = [
                    f"{target_alg._tex_label}_{{{i + 1}}}"
                    for i in range(target_alg.dimension)
                ]
            else:
                target_alg.label = _label
                target_alg.basis_labels = _basis_labels
        elif _markers.get("sum", False) or _markers.get("prod", False):
            _assign_composite_labels()
        else:
            target_alg.label = _label
            target_alg.basis_labels = _basis_labels
        target_alg._registered = True
    else:
        target_alg.label = "Alg_" + create_key()
        if _basis_labels_parent is True:
            target_alg.basis_labels = [
                f"{target_alg.label}{i + 1}" for i in range(target_alg.dimension)
            ]
        else:
            target_alg.basis_labels = [
                f"_e{i + 1}" for i in range(target_alg.dimension)
            ]
        target_alg._registered = False
    target_alg._basis_labels_parent = _basis_labels_parent
    target_alg._callLock = _callLock
    target_alg._print_warning = _print_warning
    target_alg._child_print_warning = _child_print_warning
    target_alg._exclude_from_VMF = _exclude_from_VMF
    target_alg.is_sparse = format_sparse
    target_alg.structureDataDict = _flatten_structure_data(
        target_alg.structureData, _source="algebra_class"
    )
    if _calledFromCreator == retrieve_passkey():
        target_alg.base_field = _markers.get("base_field", base_field)
    else:
        target_alg.base_field = base_field
    if target_alg.base_field not in ("real", "complex"):
        target_alg.base_field = "complex"
    if target_alg.base_field == "real" and _calledFromCreator != retrieve_passkey():
        target_alg._profile_structure_data()
    target_alg._built_from_matrices = process_matrix_rep
    target_alg.simplify_products_by_default = simplify_products_by_default
    target_alg.semidirect_decomposition = _markers.get("semidirect_decomposition", None)
    target_alg.tensor_decomposition = _markers.get("tensor_decomposition", None)
    target_alg._dgcv_class_check = retrieve_passkey()
    target_alg._dgcv_category = "algebra"
    if target_alg._parameters:
        target_alg._singularities = {
            "structure": _gather_structure_singularities(
                target_alg.structureData, target_alg._parameters
            )
        }
    else:
        target_alg._singularities = {}

    numeric_types = expr_numeric_types()

    def validate_and_adjust_grading_vector(vector, dimension):
        vector = list(vector)
        if len(vector) < dimension:
            dgcv_warning(
                f"Grading vector is shorter than the dimension ({len(vector)} < {dimension}). "
                f"Padding with zeros to match the dimension."
            )
            vector += [0] * (dimension - len(vector))
        elif len(vector) > dimension:
            dgcv_warning(
                f"Grading vector is longer than the dimension ({len(vector)} > {dimension}). "
                f"Truncating to match the dimension.",
            )
            vector = vector[:dimension]

        for i, component in enumerate(vector):
            if not isinstance(component, numeric_types):
                raise ValueError(
                    f"Invalid component in grading vector at index {i}: {component}. "
                    f"Expected scalar"
                ) from None

        return tuple(vector)

    if grading is None:
        target_alg.grading = [tuple([0] * target_alg.dimension)]
    else:
        if isinstance(grading, (list, tuple)) and all(
            isinstance(g, (list, tuple)) for g in grading
        ):
            target_alg.grading = [
                validate_and_adjust_grading_vector(vector, target_alg.dimension)
                for vector in grading
            ]
        else:
            target_alg.grading = [
                validate_and_adjust_grading_vector(grading, target_alg.dimension)
            ]

    target_alg._gradingNumber = len(target_alg.grading)

    for val, lab in zip(
        [matrix_representation, preferred_representation, tensor_representation],
        [
            "matrix_representation",
            "preferred_representation",
            "tensor_representation",
        ],
    ):
        if val is not None and (
            not isinstance(val, (list, tuple)) or len(val) != target_alg.dimension
        ):
            raise TypeError(f"unsupported format for {lab}.") from None
    if preferred_representation is not None and all(
        isinstance(elem, matrix_dgcv) for elem in preferred_representation
    ):
        target_alg._preferred_rep_type = "matrix"
        target_alg._preferred_representation = preferred_representation
    elif preferred_representation is not None and all(
        isinstance(elem, tensorProduct) for elem in preferred_representation
    ):
        target_alg._preferred_rep_type = "tensor"
        target_alg._preferred_representation = preferred_representation
    elif preferred_representation is not None and all(
        isinstance(elem, (list, tuple)) for elem in preferred_representation
    ):
        target_alg._preferred_rep_type = "matrix"
        target_alg._preferred_representation = [
            matrix_dgcv(elem) for elem in preferred_representation
        ]
    elif preferred_representation is not None:
        raise TypeError("unsupported format for `preferred_representation`.") from None
    else:
        target_alg._preferred_rep_type = None
        target_alg._preferred_representation = None

    if matrix_representation is not None and all(
        isinstance(elem, matrix_dgcv) for elem in matrix_representation
    ):
        target_alg._mat_rep = matrix_representation
    elif matrix_representation is not None and all(
        isinstance(elem, (list, tuple)) for elem in matrix_representation
    ):
        target_alg._mat_rep = [matrix_dgcv(elem) for elem in matrix_representation]
    elif matrix_representation is not None:
        raise TypeError("unsupported format for `matrix_representation`.") from None
    else:
        target_alg._mat_rep = None

    if tensor_representation is not None and all(
        isinstance(elem, tensorProduct) for elem in tensor_representation
    ):
        target_alg._tensor_rep = tensor_representation
    elif tensor_representation is not None:
        raise TypeError("unsupported format for `tensor_representation`.") from None
    else:
        target_alg._tensor_rep = None

    target_alg.card = _vs_card(target_alg)
    target_alg.basis = tuple(
        [
            algebra_element_class(
                target_alg,
                [1 if i == j else 0 for j in range(target_alg.dimension)],
                1,
            )
            for i in range(target_alg.dimension)
        ]
    )

    target_alg._basis_labels = tuple(_basis_labels) if _basis_labels else None
    target_alg._grading = tuple(map(tuple, target_alg.grading))

    target_alg._skew_symmetric_cache = None
    target_alg._jacobi_identity_cache = None
    target_alg._lie_algebra_cache = None
    target_alg._is_semisimple_cache = None
    target_alg._is_simple_cache = None
    target_alg._is_nilpotent_cache = None
    target_alg._is_abelian_cache = None
    target_alg._is_solvable_cache = None
    target_alg._rank_approximation = None
    target_alg._center_cache = None
    target_alg._lower_central_series_cache = None
    target_alg._lower_central_series_terminated = None
    target_alg._lower_central_series_depth = None
    target_alg._derived_series_cache = None
    target_alg._derived_series_terminated = None
    target_alg._derived_series_depth = None
    target_alg._grading_compatible = None
    target_alg._grading_report = None
    target_alg._killing_form = None
    target_alg._derived_subalg_cache = None
    target_alg._derived_subalg_lazy_cache = None
    target_alg._radical_cache = None
    target_alg._Levi_deco_cache = None
    target_alg._graded_components = None
    target_alg._endomorphisms = None
    target_alg._coproduct = {idx: None for idx in range(target_alg.dimension)}
    target_alg._structure_data_profile = None
