from __future__ import annotations

from ..._aux._backends._polynomials import expr_union_primitives
from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    get_free_symbols,
    is_zero_knowing_zero_is_expected,
    simplify,
)
from ..._aux._utilities._config import dgcv_warning, get_dgcv_settings_registry
from ..._aux._vmf._safeguards import get_dgcv_category, retrieve_passkey
from ..._aux._vmf.vmf import order_coordinates
from ...core.arrays import matrix_dgcv
from ...core.dgcv_core import wedge
from ..linear_algebra import _structure_array
from ..threads import _indep_check
from ..threads.util import _span_solver
from .subspaces import algebra_subspace_class


def subalgebra(
    target_alg,
    basis,
    grading=None,
    span_warning=True,
    simplify_basis=False,
    simplify_products_by_default=None,
    surface_singularities=None,
    base_field=None,
    lazy_structure=False,
):
    from ..subspaces.subalgebras import subalgebra_class

    if surface_singularities is None and target_alg._parameters:
        surface_singularities = True

    if simplify_products_by_default is None:
        simplify_products_by_default = target_alg.simplify_products_by_default
    if get_dgcv_category(basis) in {"algebra_subspace", "algebra"}:
        basis = basis.basis
    use_slices = True
    subIndices = set()
    index_map = dict()
    pos = target_alg._basis_index
    for count, elem in enumerate(basis):
        try:
            idx = pos.get(elem)
        except TypeError:  ###!!! update to optimize break path
            idx = None
        if idx is None:
            use_slices = False
            break
        index_map[idx] = count
        subIndices.add(idx)
    if use_slices:
        sub_dim = len(subIndices)
        if sub_dim != len(basis):
            raise ValueError(
                "The basis provided to `algebra_class.subalgebra` contains "
                f"repeated elements ({len(basis)} given, {sub_dim} distinct)."
            ) from None
        sub_order = sorted(subIndices, key=index_map.get)

        def truncateBySubInd(li):
            return [li[j] for j in sub_order]

        def restrict_structure_data(data):
            new_data = dict()
            inner_shape = (sub_dim, 1)
            for (i, j, k), v in data.items():
                if i in subIndices and j in subIndices:
                    if k in subIndices:
                        outer_key = (index_map[i], index_map[j])
                        if outer_key in new_data:
                            new_data[outer_key][index_map[k]] = v
                        else:
                            new_data[outer_key] = matrix_dgcv(
                                {index_map[k]: v}, shape=inner_shape
                            )
                    elif v is not None and not is_zero_knowing_zero_is_expected(v):
                        raise TypeError(
                            "The basis provided to the `algebra_class.subalgebra` method does not span a subalgebra."
                        )
            return _structure_array(new_data, sub_dim)

        if isinstance(grading, (list, tuple)) and all(
            isinstance(elem, (list, tuple)) for elem in grading
        ):
            gradings = grading
        else:
            if grading is not None:
                dgcv_warning(
                    "The `gradings` keyword given to `algebra_class.subalgebra` was in an unsupported format (i.e., not list of lists), so a valid alternate gradings vector was computed instead inherited from the parent algebra."
                )
            gradings = [truncateBySubInd(vector) for vector in target_alg.grading]
        structureData = restrict_structure_data(target_alg.structureDataDict)
        return subalgebra_class(
            basis,
            target_alg,
            grading=gradings,
            _compressed_structure_data=structureData,
            _internal_lock=retrieve_passkey(),
            base_field=base_field,
        )
    if simplify_basis:
        basis = list(
            target_alg.filter_independent_elements(
                basis,
                apply_light_basis_simplification=True,
                surface_singularities=False,
            )
        )
    if lazy_structure:
        return subalgebra_class(
            basis,
            target_alg,
            grading=grading,
            _internal_lock=retrieve_passkey(),
            span_warning=False,
            simplify_basis=False,
            simplify_products_by_default=simplify_products_by_default,
            base_field=base_field,
            _lazy_structure_data=True,
        )
    testStruct = target_alg.is_subspace_subalgebra(
        basis,
        return_structure_data=True,
        surface_singularities=surface_singularities,
    )
    if surface_singularities:
        testStruct, sing = testStruct
        new_sing = [v for v in sing if get_free_symbols(v)]
        if get_dgcv_settings_registry().get(
            "simplify_singularity_ideals_by_default", True
        ):
            new_sing = expr_union_primitives(
                new_sing,
                order_coordinates(target_alg._parameters),
                process_rationals=True,
                fail_quietly=True,
            )
        ks = {"basis": new_sing}
    else:
        ks = None
    if testStruct["closed_under_product"] is not True:
        raise TypeError(
            "The basis provided to the `algebra_class.subalgebra` method does not span a subalgebra. Suggestion: use `algebra_class.subspace` instead."
        ) from None
    return subalgebra_class(
        basis,
        target_alg,
        grading=grading,
        _compressed_structure_data=testStruct["structure_data"],
        _internal_lock=retrieve_passkey(),
        span_warning=False,
        simplify_basis=False,
        simplify_products_by_default=simplify_products_by_default,
        _known_singularities=ks,
        base_field=base_field,
    )


def new_alg_from_subalgebra(
    target_alg,
    basis,
    grading=None,
    span_warning=True,
    simplify_basis=False,
    label=None,
    basis_labels=None,
    register_in_vmf=False,
    initial_basis_index=None,
    simplify_products_by_default=None,
):
    if simplify_products_by_default is None:
        simplify_products_by_default = target_alg.simplify_products_by_default
    if get_dgcv_category(basis) == "subalgebra" and basis.ambient == target_alg:
        alg = basis
    else:
        alg = target_alg.subalgebra(
            basis,
            grading=grading,
            span_warning=span_warning,
            simplify_basis=simplify_basis,
            simplify_products_by_default=simplify_products_by_default,
        )
    return alg.copy(
        label=label,
        basis_labels=basis_labels,
        register_in_vmf=register_in_vmf,
        initial_basis_index=initial_basis_index,
        simplify_products_by_default=simplify_products_by_default,
    )


def _closure_product_data(target_alg, elems):
    if not elems:
        return None
    first = elems[0]
    valence = getattr(first, "valence", None)
    views = []
    for elem in elems:
        if (
            get_dgcv_category(elem) != "algebra_element"
            or elem.algebra is not target_alg
            or elem.valence != valence
        ):
            return None
        cd = elem.coeff_dict
        if not isinstance(cd, dict):
            return None
        views.append(cd)
    struct = target_alg.structureData
    data = getattr(struct, "_data", None)
    unspool = getattr(struct, "_unspool", None)
    if not isinstance(data, dict) or unspool is None:
        return None
    table = {}
    for flat, col in data.items():
        col_data = getattr(col, "_data", None)
        if not isinstance(col_data, dict):
            return None
        idx1, idx2 = unspool(flat)
        row = table.get(idx1)
        if row is None:
            row = table[idx1] = {}
        row[idx2] = col_data
    sign = 1 if valence == 1 else -1
    spbd = target_alg.simplify_products_by_default is True
    return views, table, sign, spbd


def _closure_product(fast, i, j):
    views, table, sign, spbd = fast
    new = {}
    for idx1, c1 in views[i].items():
        row = table.get(idx1)
        if row is None:
            continue
        for idx2, c2 in views[j].items():
            col = row.get(idx2)
            if not col:
                continue
            scalar = sign * c1 * c2
            for idx3, c3 in col.items():
                new[idx3] = new.get(idx3, 0) + (
                    simplify(scalar * c3) if spbd else scalar * c3
                )
    return {k: v for k, v in new.items() if not _scalar_is_zero(v)}


def is_subspace_subalgebra(
    target_alg,
    elements,
    return_structure_data=False,
    check_linear_independence=False,
    surface_singularities=None,
):
    if surface_singularities is None and target_alg._parameters:
        surface_singularities = True
    filtered_elem = target_alg.filter_independent_elements(
        elements, surface_singularities=surface_singularities
    )
    if surface_singularities:
        filtered_elem, sing = filtered_elem
    new_dim = len(filtered_elem)
    linearly_independent = len(elements) == len(filtered_elem)
    closed_under_product = True
    skew = target_alg.is_skew_symmetric()
    if return_structure_data is True:
        structure_data = _structure_array(dict(), new_dim)
    if not isinstance(return_structure_data, bool):
        return_structure_data = False
    solver = _span_solver.build(filtered_elem)
    fast = None if solver is None else _closure_product_data(target_alg, filtered_elem)
    for count, elem in enumerate(filtered_elem):
        if closed_under_product is False:
            break
        lIdx = count + 1 if skew else 0
        for j in range(lIdx, new_dim):
            if closed_under_product is False:
                break
            if fast is not None:
                fc = solver.reduce_sparse(
                    _closure_product(fast, count, j),
                    surface_singularities=surface_singularities,
                )
                if fc is not None:
                    residual, coeffs, new_sing = fc
                    if surface_singularities:
                        sing += new_sing
                    if residual:
                        closed_under_product = False
                        structure_data = None
                    elif return_structure_data:
                        coeff_array = matrix_dgcv(
                            {
                                idx: coeff
                                for idx, coeff in coeffs.items()
                                if not _scalar_is_zero(coeff)
                            },
                            shape=(new_dim, 1),
                        )
                        structure_data[count, j] = coeff_array
                        if skew:
                            structure_data[j, count] = -coeff_array
                    continue
            product = elem * filtered_elem[j]
            ic = None
            if solver is not None:
                ic = solver.reduce(
                    product,
                    return_decomp_coeffs=return_structure_data,
                    surface_singularities=surface_singularities,
                )
            reduced = ic is not None
            if not reduced:
                ic = _indep_check(
                    filtered_elem,
                    product,
                    return_decomp_coeffs=return_structure_data,
                    surface_singularities=surface_singularities,
                )
            if surface_singularities:
                passCheck, new_sing = (
                    ic if return_structure_data is False else (ic[0], ic[2])
                )
                if passCheck is True and not reduced:
                    ic = _indep_check(
                        filtered_elem,
                        product,
                        return_decomp_coeffs=return_structure_data,
                        surface_singularities=True,
                        _force_eqn_simiplify=True,
                    )
                    passCheck, new_sing = (
                        ic if return_structure_data is False else (ic[0], ic[2])
                    )
                sing += new_sing
            else:
                passCheck = ic if return_structure_data is False else ic[0]
            if passCheck is True:
                closed_under_product = False
                structure_data = None
            elif return_structure_data:
                coeff_array = matrix_dgcv(
                    {
                        idx: coeff
                        for idx, coeff in ic[1][0].items()
                        if not _scalar_is_zero(coeff)
                    },
                    shape=(new_dim, 1),
                )
                structure_data[count, j] = coeff_array
                if skew:
                    structure_data[j, count] = -coeff_array
    if return_structure_data:
        out = {
            "linearly_independent": linearly_independent,
            "closed_under_product": closed_under_product,
            "structure_data": structure_data,
        }
    elif check_linear_independence:
        out = linearly_independent and closed_under_product
    else:
        out = closed_under_product
    if surface_singularities:
        return out, sing
    return out


def is_ideal(self, subspace_elements, assume_basis=False):
    if subspace_elements == self:
        return True
    if (
        isinstance(subspace_elements, algebra_subspace_class)
        and subspace_elements.ambient == self
    ):
        subspace_elements = subspace_elements.basis
        assume_basis = True
    else:
        for el in subspace_elements:
            if not get_dgcv_category(el) == "algebra_element" or el.algebra != self:
                raise ValueError(
                    "All elements in subspace_elements must belong to this algebra."
                ) from None

    if assume_basis:
        b_product = wedge(*subspace_elements)
    skew = self.is_skew_symmetric()
    for el in subspace_elements:
        for other in self.basis:
            products = [el * other] if skew else [el * other, other * el]
            for product in products:
                if assume_basis:
                    if wedge(product, b_product).is_zero:
                        return False
                else:
                    if not self.is_in_span(
                        product, subspace_elements, assume_basis=False
                    ):
                        return False
    return True
