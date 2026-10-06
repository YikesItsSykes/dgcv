from __future__ import annotations

import numbers

from ..._aux._backends._types_and_constants import expr_numeric_types
from ..._aux._utilities._config import (
    dgcv_exception_note,
    dgcv_warning,
    dgcvDeprecationWarning,
    get_dgcv_settings_registry,
    get_variable_registry,
    update_working_namespace,
)
from ..._aux._vmf._safeguards import (
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_passkey,
    unique_label,
    validate_label,
    validate_label_list,
)
from ..._aux._vmf.vmf import clearVar, listVar
from ...core.arrays import array_dgcv, freeze_matrix, matrix_dgcv
from ...core.base import annotated_container
from ..aec import algebra_element_class
from ..algebras import algebra_class
from ..format_support import (
    _external_library_algebra_processing,
    _validate_structure_data,
)
from ..linear_algebra import _structure_array
from ..specialized import simple_Lie_algebra


def createAlgebra(
    obj,
    label,
    basis_labels=None,
    grading=None,
    base_field=None,
    process_matrix_rep: bool = False,
    preferred_representation=None,
    matrix_representation=None,
    tensor_representation=None,
    verbose: bool = False,
    assume_skew: bool = False,
    basis_order_for_supplied_str_eqns=None,
    _simple=None,
    special_processing_rules: dict = None,
    return_created_object: bool = False,
    forgo_vmf_registry: bool = False,
    simplify_products_by_default: bool = None,
    initial_basis_index=1,
    allow_natural_basis_reordering: bool | None = None,
    jet_determinacy_order_ansatz: int = None,
    process_with_decompose=False,
    _basis_labels_parent=None,
    _markers={},
    **kwargs,
):
    """
    Registers an algebra object and its basis elements in the caller's global namespace,
    and adds them to the variable_registry for tracking in the Variable Management Framework.

    Parameters
    ----------
    obj : algebra, structure data, or list of algebra_element_class
        The algebra object (an instance of algebra), the structure data used to create one,
        or a list of algebra_element_class instances with the same parent algebra.
    label : str
        The label used to reference the algebra object in the global namespace.
    basis_labels : list, optional
        A list of custom labels for the basis elements of the algebra.
        If not provided, default labels will be generated.
    grading : list of lists or list, optional
        A list specifying the grading(s) of the algebra.
    process_matrix_rep : bool, optional
        Whether to compute and store the matrix representation of the algebra.
    verbose : bool, optional
        If True, provides information during the creation process.
    special_processing_rules: dict, optional
        Use this to pass a list of alebra-like objects from another library (i.e., non-dgcv).
        The dictionary should have two string labeled keys "mul" and  "zero_obst". These key's
        values should be callable functions. The value for "mul" must be a function operating
        on two algebra element parameters that returns a new algebra element (assumed to
        represent the algebra product). The value for "zero_obst" should be a function
        operating on single algebra elements, return an iterable of elements that obstruct
        being zero (e.g., list coeffients from a linear combination of basis elements).
        At third optional keyworb "assume_skew" is permitted, in which case the algebra structure
        is computed with skew-shape-aware sparsified formulas.
    """
    if kwargs.get("assume_Lie_alg"):  # old keyword support
        assume_skew = True
    if kwargs.get("return_created_obj"):  # old keyword support
        return_created_object = kwargs.get("return_created_obj")
    notes = {}
    _markers["_educed_properties"] = dict()
    if get_dgcv_category(obj) == "Tanaka_symbol":
        t_message = "True by construction: data --> `Tanaka_symbol` --> `createAlgebra`"
        _markers["_educed_properties"]["is_Lie_algebra"] = t_message
        _markers["_educed_properties"]["is_skew"] = t_message
        _markers["_educed_properties"]["satisfies_Jacobi_ID"] = t_message
        if grading is not None:
            dgcv_warning(
                "When processing a `Tanaka_symbol` object, `createAlgebra` uses the symbol's internally defined grading rather than a manually supplied grading. You are getting this warning because an additional grading was manually supplied. To apply the custom grading instead, extract the symbol object's structure data using `Tanaka_symbol.export_algebra_data()`, and then pass that to `createAlgebra` -- create the algebra first and extract the data from the created `algebra_class` attributes."
            )
        if allow_natural_basis_reordering is None:
            allow_natural_basis_reordering = False
        preserve_negative_part_basis = not allow_natural_basis_reordering
        symbolData = obj.export_algebra_data(
            _internal_call_lock=retrieve_passkey(),
            preserve_negative_part_basis=preserve_negative_part_basis,
            jacobi_threshold=1,
            try_hard=True,
        )
        if isinstance(symbolData, str):
            raise TypeError(
                symbolData + " So `createAlgebra` did not instantiate a new algebra."
            ) from None
        obj = symbolData["structure_data"]
        grading = symbolData["grading"]

    passkey = retrieve_passkey()
    if (_markers.get("sum", False) or _markers.get("prod", False)) and _markers.get(
        "lockKey", None
    ) == passkey:
        incoming_tex_label = _markers.get("_tex_label", None)
        if incoming_tex_label is None:
            label = unique_label(label)
        else:
            label, _markers["_tex_label"] = unique_label(
                label, tex_label=incoming_tex_label
            )
    if (
        label in listVar(algebras_only=True)
        and get_dgcv_settings_registry()["forgo_warnings"] is not True
    ):
        if isinstance(_simple, dict) and _simple.get("lockKey", None) == passkey:
            callFunction = "createSimpleLieAlgebra"
        else:
            callFunction = "createAlgebra"
        dgcv_warning(
            f"`{callFunction}` was called with a `label` parameter already assigned to another algebra, so `{callFunction}` will overwrite the other algebra in the VMF and global namespace."
        )
        clearVar(label)

    def extract_structure_from_elements(elements, markers):
        """
        Computes structure constants and validates linear independence from a list of algebra_element_class.

        Parameters
        ----------
        elements : list of algebra_element_class
            A list of algebra_element_class instances.

        Returns
        -------
        structure_data : list of lists of lists
            The structure constants for the subalgebra spanned by the elements.

        Raises
        ------
        ValueError
            If the elements are not linearly independent or not closed under the algebra product.
        """
        if isinstance(elements, (list, tuple)):
            elements = [
                (
                    elem.ambient_rep
                    if get_dgcv_category(elem) == "subalgebra_element"
                    else elem
                )
                for elem in elements
            ]
        if not elements or not all(
            isinstance(el, algebra_element_class) for el in elements
        ):
            raise ValueError(
                "Invalid input: All elements must be instances of algebra_element_class."
            ) from None
        parent_algebra = elements[0].algebra
        if parent_algebra._lie_algebra_cache is True:
            t_message = "True by inheritance: subalgebra of Lie algebra"
            markers["_educed_properties"]["is_Lie_algebra"] = t_message
            markers["_educed_properties"]["is_skew"] = t_message
            markers["_educed_properties"]["satisfies_Jacobi_ID"] = t_message
        else:
            if parent_algebra._jacobi_identity_cache is True:
                markers["_educed_properties"]["satisfies_Jacobi_ID"] = (
                    "True by inheritance: subalgebra of Jacobi satisfying algebra"
                )
            if parent_algebra._skew_symmetric_cache is True:
                markers["_educed_properties"]["is_skew"] = (
                    "True by inheritance: subalgebra of a skew symmetric algebra"
                )

        if not all(el.algebra == parent_algebra for el in elements):
            raise ValueError(
                "All algebra_element_class instances must share the same parent algebra."
            ) from None
        try:
            result = parent_algebra.is_subspace_subalgebra(
                elements, return_structure_data=True
            )
        except ValueError as e:
            raise ValueError(
                "Error during subalgebra validation. "
                "The input list of algebra_element_class instances must be linearly independent and closed under the algebra product. "
                f"Original error: {e}"
            ) from e
        if not result["linearly_independent"]:
            raise ValueError(
                "The input elements are not linearly independent. "
            ) from None
        if not result["closed_under_product"]:
            raise ValueError(
                "The input elements are not closed under the algebra product. "
            ) from None
        return result["structure_data"]

    if get_dgcv_category(obj) == "algebra_subspace":
        try:
            obj = obj.ambient.subalgebra(obj)
        except dgcv_exception_note as e:
            raise SystemExit(e)

    def _branch_inference(elems):
        gate = False
        types = {"algebra_element", "subalgebra_element"}
        for elem in elems:
            if get_dgcv_category(elem) == "tensorProduct":
                gate = True
            elif get_dgcv_category(elem) not in types:
                return False
        return gate

    if (
        isinstance(obj, annotated_container)
        and obj._dgcv_notes.get("signature", None) == "algebra_str_eqns"
    ):
        if obj._dgcv_notes.get("skew_aware_sparse", False):
            assume_skew = True
        basis_order_for_supplied_str_eqns = obj[1]
        obj = obj[0]
        if basis_labels is None:
            basis_labels = [str(x) for x in basis_order_for_supplied_str_eqns]

    if isinstance(obj, numbers.Integral) and obj >= 0:
        structure_data = _structure_array(dict(), obj)
        dimension = obj
        t_message = "True by construction: abelian data --> `createAlgebra`"
        _markers["_educed_properties"]["is_Lie_algebra"] = t_message
        _markers["_educed_properties"]["is_skew"] = t_message
        _markers["_educed_properties"]["satisfies_Jacobi_ID"] = t_message
        _markers["_educed_properties"]["is_nilpotent"] = t_message
        _markers["_educed_properties"]["is_solvable"] = t_message
        _markers["_educed_properties"]["special_type"] = "abelian"
    elif get_dgcv_category(obj) in {"algebra", "subalgebra"}:
        if verbose:
            print(f"Using existing algebra instance: {label}")
        _markers["_educed_properties"] = getattr(obj, "_educed_properties", dict())
        structure_data = obj.structureData
        dimension = obj.dimension
        if grading is None:
            grading = getattr(obj, "grading", None)
    elif isinstance(obj, (list, tuple)) and len(obj) == 0:
        structure_data = array_dgcv(
            dict(),
            shape=(0, 0),
            null_return=freeze_matrix(matrix_dgcv.zeros(0, 1)),
        )
        dimension = 0
    elif isinstance(special_processing_rules, dict):
        mul, zero_obst = (
            special_processing_rules.get("mul"),
            special_processing_rules.get("zero_obst"),
        )
        if not callable(mul) or not callable(zero_obst):
            raise TypeError(
                "special_processing_rules parameter is in an unsupported format"
            )
        try:
            structure_data = _external_library_algebra_processing(
                obj,
                mul=mul,
                zero_obst=zero_obst,
                assume_skew=special_processing_rules.get("assume_skew", False),
            )
            dimension = len(obj)
        except Exception:
            raise RuntimeError(
                "Could not process data recieved in the first parameter and special_processing_rules parameter"
            )
    elif isinstance(obj, (list, tuple)) and all(
        get_dgcv_category(el) in {"algebra_element", "subalgebra_element"} for el in obj
    ):
        if verbose:
            print("Creating algebra from list of algebra_element_class instances.")
        structure_data = extract_structure_from_elements(obj, _markers)
        dimension = len(obj)
    elif isinstance(obj, (list, tuple)) and _branch_inference(obj):
        notes["process_tensor_rep"] = True
        if verbose:
            print("Creating algebra from list of tensorProduct instances.")
        try:
            vsd = _validate_structure_data(
                [el._convert_to_tp() for el in obj],
                process_matrix_rep=False,
                assume_skew=assume_skew,
                basis_order_for_supplied_str_eqns=basis_order_for_supplied_str_eqns,
                process_tensor_rep=True,
            )
            if tensor_representation is not None:
                dgcv_warning(
                    "The primary object given to `createAlgebra` was a list of tensorProduct instances, but a secondary value fo `tensor_representation` representation was given. The latter was ignored."
                )
            structure_data, tensor_representation = vsd[0][0], vsd[0][1]

        except dgcv_exception_note as e:
            raise SystemExit(e)
        dimension = len(structure_data)
    else:
        if verbose:
            print("processing structure data...")
        try:
            vsd = _validate_structure_data(
                obj,
                process_matrix_rep=process_matrix_rep,
                assume_skew=assume_skew,
                basis_order_for_supplied_str_eqns=basis_order_for_supplied_str_eqns,
                determinacy_order_ansatz=jet_determinacy_order_ansatz,
                process_with_decompose=process_with_decompose,
            )
            if vsd[-1] == "matrix":
                t_message = "True by construction: list of matrices --> `createAlgebra`"
                _markers["_educed_properties"]["is_Lie_algebra"] = t_message
                _markers["_educed_properties"]["is_skew"] = t_message
                _markers["_educed_properties"]["satisfies_Jacobi_ID"] = t_message
                _markers["parameters"] = vsd[0][2]
                structure_data, matrix_representation = vsd[0][0], vsd[0][1]
            elif vsd[-1] == "tensor":
                notes["process_tensor_rep"] = True
                _markers["parameters"] = vsd[0][2]
                structure_data, tensor_representation = vsd[0][0], vsd[0][1]
            else:
                if (
                    isinstance(obj, (list, tuple))
                    and len(obj) > 1
                    and query_dgcv_categories(obj[-1], "vector_field")
                ):
                    t_message = "True by construction: list of vector fields --> `createAlgebra`"
                    _markers["_educed_properties"]["is_Lie_algebra"] = t_message
                    _markers["_educed_properties"]["is_skew"] = t_message
                    _markers["_educed_properties"]["satisfies_Jacobi_ID"] = t_message
                structure_data = vsd[0]
                if len(vsd) > 2 and grading is None:
                    grading = vsd[2]
                _markers["parameters"] = vsd[1]
                simplify_products_by_default = (
                    False
                    if simplify_products_by_default is None
                    else simplify_products_by_default
                )

        except dgcv_exception_note as e:
            raise SystemExit(e)
        dimension = structure_data.shape[0]

    if (_markers.get("sum", False) or _markers.get("prod", False)) and _markers.get(
        "lockKey", None
    ) == passkey:
        if basis_labels is None:
            needs_sep = str(label)[-1].isdigit()
            initial_names = [
                f"{label}_{i + 1}" if needs_sep else f"{label}{i + 1}"
                for i in range(dimension)
            ]
            _basis_labels_parent = True
        elif isinstance(basis_labels, str):
            needs_sep = str(basis_labels)[-1].isdigit()
            initial_names = [
                f"{basis_labels}_{i + initial_basis_index}"
                if needs_sep
                else f"{basis_labels}{i + initial_basis_index}"
                for i in range(dimension)
            ]
        else:
            initial_names = list(basis_labels)

        incoming_tex_basis = list(_markers.get("_tex_basis_labels", []) or [])
        have_tex_basis = (
            len(incoming_tex_basis) == len(initial_names) and len(initial_names) > 0
        )

        batch_protected = {label}
        new_basis = []
        new_tex_basis = [] if have_tex_basis else None
        for idx, base_lbl in enumerate(initial_names):
            if have_tex_basis:
                bl, tl = unique_label(
                    base_lbl,
                    tex_label=incoming_tex_basis[idx],
                    protected=batch_protected,
                )
                new_basis.append(bl)
                new_tex_basis.append(tl)
                batch_protected.add(bl)
            else:
                bl = unique_label(base_lbl, protected=batch_protected)
                new_basis.append(bl)
                batch_protected.add(bl)

        basis_labels = new_basis
        if have_tex_basis:
            _markers["_tex_basis_labels"] = new_tex_basis
    else:  # check for redundancy here
        if basis_labels is None:
            needs_sep = str(label)[-1].isdigit()
            basis_labels = [
                validate_label(f"{label}_{i + 1}" if needs_sep else f"{label}{i + 1}")
                for i in range(dimension)
            ]
            _basis_labels_parent = True
        elif isinstance(basis_labels, str):
            needs_sep = str(basis_labels)[-1].isdigit()
            basis_labels = [
                validate_label(
                    f"{basis_labels}_{i + initial_basis_index}"
                    if needs_sep
                    else f"{basis_labels}{i + initial_basis_index}"
                )
                for i in range(dimension)
            ]
        else:
            validate_label_list(basis_labels)

    if grading is None:
        if notes.get("process_tensor_rep", False) is True:
            w = None
            changed = None
            weights = []
            for elem in tensor_representation:
                wts = elem.compute_weight()
                if isinstance(wts, str):
                    changed = "break"
                    break
                weights.append(wts)
                if w is None:
                    w = len(wts)
                    changed = False
                elif len(wts) < w:
                    w = len(wts)
                    if changed is False:
                        changed = True
            if changed != "break":
                if changed is True:
                    weights = [elem[:w] for elem in weights]
                grading = list(zip(*weights))
        else:
            grading = [tuple([0] * dimension)]
    elif isinstance(grading, (list, tuple)) and all(
        isinstance(w, expr_numeric_types()) for w in grading
    ):
        if len(grading) != dimension:
            raise ValueError(
                f"Grading vector length ({len(grading)}) must match the algebra dimension ({dimension})."
            ) from None
        grading = [tuple(grading)]
    elif isinstance(grading, (list, tuple)) and all(
        isinstance(vec, (list, tuple)) for vec in grading
    ):
        for vec in grading:
            if len(vec) != dimension:
                raise ValueError(
                    f"Grading vector length ({len(vec)}) must match the algebra dimension ({dimension})."
                ) from None
        grading = [tuple(vec) for vec in grading]
    else:
        raise ValueError(
            f"Grading must be a single vector or a list of vectors. Recieved {grading}"
        ) from None
    if isinstance(_simple, dict) and _simple.get("lockKey", None) == passkey:
        algebra_obj = simple_Lie_algebra(
            structure_data=structure_data,
            grading=grading,
            base_field=base_field,
            process_matrix_rep=process_matrix_rep,
            preferred_representation=preferred_representation,
            _label=label,
            _basis_labels=basis_labels,
            _calledFromCreator=passkey,
            _simple_data=_simple,
            _basis_labels_parent=_basis_labels_parent,
        )
    else:
        if _markers.get("lockKey", None) == passkey:
            _markers = {k: v for k, v in _markers.items() if k != "lockKey"}
        elif "_educed_properties" in _markers:
            _markers = {
                "_educed_properties": _markers["_educed_properties"],
                "parameters": _markers.get("parameters", set()),
            }
        else:
            _markers = {"parameters": _markers.get("parameters", set())}
        algebra_obj = algebra_class(
            structure_data=structure_data,
            grading=grading,
            base_field=base_field,
            process_matrix_rep=process_matrix_rep,
            preferred_representation=preferred_representation,
            simplify_products_by_default=simplify_products_by_default,
            matrix_representation=matrix_representation,
            tensor_representation=tensor_representation,
            _label=label,
            _basis_labels=basis_labels,
            _calledFromCreator=passkey,
            _basis_labels_parent=_basis_labels_parent,
            _markers=_markers,
        )

    if forgo_vmf_registry is False:
        update_working_namespace({label: algebra_obj})
        update_working_namespace(zip(basis_labels, algebra_obj.basis))

        variable_registry = get_variable_registry()
        paths = variable_registry.get("paths", None)
        paths[label] = {
            "kind": "finite_algebra_systems",
            "path": ("finite_algebra_systems", label),
        }

        variable_registry["finite_algebra_systems"][label] = {
            "family_type": "algebra",
            "family_names": tuple(basis_labels),
            "family_values": tuple(algebra_obj.basis),
            "dimension": dimension,
            "grading": grading,
            "basis_labels": basis_labels,
            "structure_data": structure_data,
        }
        variable_registry["_labels"][label] = {
            "path": ("finite_algebra_systems", label),
            "children": set(basis_labels),
        }

    if verbose:
        if forgo_vmf_registry is False:
            print(f"Algebra '{label}' registered successfully.")
        print(
            f"Created an algebra with the following properties. Dimension: {dimension}, Grading: {grading}, Basis Labels: {basis_labels}"
        )

    if return_created_object is True:
        return algebra_obj


def createFiniteAlg(
    obj,
    label,
    basis_labels=None,
    grading=None,
    format_sparse=False,
    process_matrix_rep=False,
    preferred_representation=None,
    verbose=False,
    assume_skew=False,
    basis_order_for_supplied_str_eqns=None,
    _simple=None,
    **kwargs,
):
    dgcv_warning(
        "`createFiniteAlg` has been deprecated as it is being replaced with a more general function.",
        dgcvDeprecationWarning,
        stacklevel=2,
        old_kw="createFiniteAlg",
        new_kw="createAlgebra",
        sunset="2026",
    )
    if kwargs.get("assume_Lie_alg", False):
        assume_skew = True
    return createAlgebra(
        obj,
        label,
        basis_labels=basis_labels,
        grading=grading,
        format_sparse=format_sparse,
        process_matrix_rep=process_matrix_rep,
        preferred_representation=preferred_representation,
        verbose=verbose,
        assume_skew=assume_skew,
        basis_order_for_supplied_str_eqns=basis_order_for_supplied_str_eqns,
        _simple=_simple,
    )
