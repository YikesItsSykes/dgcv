"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.algebras.algebra_tools

module: dgcv.algebras.algebra_tools.algebra_tools


---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

SPDX-License-Identifier: Apache-2.0


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
# imports and broadcasting
# -----------------------------------------------------------------------------
from typing import Sequence

from ..._aux._backends._symbolic_router import _scalar_is_zero, get_free_symbols, subs
from ..._aux._backends._types_and_constants import check_dgcv_scalar, symbol
from ..._aux._utilities._config import dgcv_warning
from ..._aux._utilities._misc import linear_combination, zip_sum
from ..._aux._vmf._safeguards import create_key, get_dgcv_category
from ..._aux._vmf.vmf import first_available_label, vmf_lookup
from ...core.arrays import _as_matrix_dgcv, array_dgcv, freeze_matrix, matrix_dgcv
from ...core.combinatorics.combinatorics import Baker_Campbell_Hausdorff
from ...core.dgcv_core import createVariables, wedge
from ...core.morphisms import homomorphism
from ...core.solvers import solve_dgcv
from ...core.vector_fields_and_differential_forms import coordinate_vector_field
from ..algebras import algebra_class
from ..creators import createAlgebra
from ..linear_algebra.util import _mat_to_tensor, linear_representation
from ..subspaces import algebra_subspace_class
from ..subspaces.subalgebras import subalgebra_class
from ..threads import _extract_basis, adjointRepresentation, killingForm

__all__ = [
    "adjointRepresentation",
    "build_linear_representation",
    "derivations",
    "derived_subalgebra",
    "generate_subalgebra",
    "intersection",
    "killingForm",
    "Levi_decomposition",
    "multiply",
    "new_graded_algebra_from_old",
    "quotient_by_ideal",
    "span",
    "vector_field_representation",
    "vector_field_rep_from_linear_rep",
]


# -----------------------------------------------------------------------------
# functions
# -----------------------------------------------------------------------------


def derived_subalgebra(
    algebra: algebra_class | subalgebra_class, return_basis_only=False
) -> subalgebra_class:
    if get_dgcv_category(algebra) not in {"algebra", "subalgebra"}:
        raise TypeError(
            "`derived_subalgebra` only operates on dgcv algebra class objects."
        )
    basis = algebra.basis
    if algebra.is_skew_symmetric():
        products = [a * b for c, a in enumerate(basis) for b in basis[c + 1 :]]
    else:
        products = [a * b for a in basis for b in basis]
    if return_basis_only:
        return span(products, format_as_subspaces=False)
    return algebra.subalgebra(products, span_warning=False)


def killing_form(algebra, assume_Lie_algebra=False):
    return killingForm(algebra, assume_Lie_algebra=assume_Lie_algebra)


def adjoint_representation(
    algebra, list_format=False, assume_Lie_algebra=False
) -> list:
    return adjointRepresentation(
        algebra, list_format=list_format, assume_Lie_algebra=assume_Lie_algebra
    )


def multiply(
    *args, filter_for_linear_independance=False, apply_light_basis_simplification=False
) -> subalgebra_class | algebra_class | list:
    """
    Computes a list of products of elements in the given algebras (or algebra-like objects)

    Parameters:
    -----------
    *args: algebra like objects
        This can be any number of algebra calss objects, or lists of algebra elements, or even single algebra elements. "algebra-like" is broad here, include algebra subspace/subalgebra classes, vector space classes etc.

    filter_for_linear_independance: bool
        If true the computed products are filtered down to a linearly independant subset

    apply_light_basis_simplification: bool
        permits the algorithm to scale elements in a way that attempts to simplify coefficients.

    returns:
    --------
        list of alegra elements
    """

    def scale_or_atom(obj):
        return get_dgcv_category(obj) in {
            "algebra_element",
            "subalgebra_element",
        } or check_dgcv_scalar(obj)

    factors = [[elem] if scale_or_atom(elem) else elem for elem in args]

    def product(*elems):
        if len(elems) == 0:
            return elems
        elif len(elems) == 1:
            return list(elems[0])
        if len(elems) > 2:
            return product(elems[0], product(*elems[1:]))
        out = []
        for e1 in elems[0]:
            for e2 in elems[1]:
                out.append(e1 * e2)
        if filter_for_linear_independance is True and len(out) > 0:
            alg = getattr(out[0], "algebra", None)
            if alg:
                out = alg.filter_independent_elements(
                    out,
                    apply_light_basis_simplification=apply_light_basis_simplification,
                )
        return list(out)

    return product(*factors)


def span(
    *args,
    apply_light_basis_simplification=False,
    separate_by_algebra=False,
    promote_to_parent_algebra=False,
    format_as_subspaces=False,
) -> subalgebra_class | algebra_class | list:
    """
    computes the span of elements in given list of algebra-like objects.

    Parameters:
    -----------
    *args: algebra like objects
        This can be any number of algebra calss objects, or lists of algebra elements, or even single algebra elements. "algebra-like" is broad here, include algebra subspace/subalgebra classes, vector space classes etc.

    apply_light_basis_simplification: bool
        permits the algorithm to scale elements in a way that attempts to simplify coefficients.

    separate_by_algebra: bool
        If true, the returned list is divided into sub-lists grouped by algebra.

    Returns:
    --------
        list of alegra elements or list of lists of algebra_elements (depending on separate_by_algebra setting)

    Notes:
    ------
        Returns empty list for 0-dimensional spaces
    """

    def wrap(obj):
        return (
            [obj]
            if get_dgcv_category(obj)
            in {
                "algebra_element",
                "subalgebra_element",
            }
            or check_dgcv_scalar(obj)
            else obj
        )

    parents = dict()
    for elem in args:
        for x in wrap(elem):
            alg = getattr(x, "vectorSpace")
            if alg is None:
                raise TypeError("algebra_tools.span can only combine algebra elements.")
            if alg.ambient in parents:
                parents[alg.ambient][alg] = parents[alg.ambient].get(alg, []) + [x]
            else:
                parents[alg.ambient] = (  ###!!! promote_to_parent_algebra logic here
                    {alg: [x]}
                )
    space_bases = dict()
    for k, v in parents.items():
        spanners = []
        if len(v) == 1:
            spanners = next(iter(v.values()))
        else:
            for _, val in v:
                spanners += [x.ambient_rep for x in val]
        space_bases[k] = k.filter_independent_elements(
            spanners, apply_light_basis_simplification=apply_light_basis_simplification
        )
    if separate_by_algebra:
        if format_as_subspaces:
            space_bases = {k: k.subspace(v) for k, v in space_bases.items()}
        return space_bases
    out = []
    for k, v in space_bases.items():
        if format_as_subspaces is True:
            out.append(k.subspace(v, span_warning=False))
        else:
            out += v
    if format_as_subspaces and len(out) == 1:
        out = out[0]
    return out


def generate_subalgebra(
    *subspaces,
    simplify_basis=False,
    simplify_products_by_default=None,
) -> subalgebra_class | algebra_class | list:
    """
    Generates a subalgebra from a subset of algebra elements

    Parameters:
    -----------
    *args: algebra like objects
        This can be any number of algebra calss objects, or lists of algebra elements, or even single algebra elements. "algebra-like" is broad here, include algebra subspace/subalgebra classes, vector space classes etc.

    simplify_basis: bool
        Permits the algorithm to scale elements in a way that attempts to simplify coefficients. The resulting bases may be different from subsets of input algebra element lists as a result.

    simplify_products_by_default: bool
        Set's the default simplification behavior on the created algebra objects. Recommended to ignore this if unsure.

    returns:
    --------
        Single algebra/subalgebra object or list of them depending on whether given elements belong to a common algebra or not.
    """

    spaces = span(
        *subspaces,
        separate_by_algebra=True,
        promote_to_parent_algebra=True,
        format_as_subspaces=True,
    )
    out = []
    for _, subspace in spaces.items():
        out.append(
            subspace.generate_subalgebra(
                simplify_basis=simplify_basis,
                simplify_products_by_default=simplify_products_by_default,
            )
        )
    if len(out) == 1 and len(subspaces) == 1:
        return out[0]
    return out


def intersection(*args) -> list:
    """
    computes the intersection of spaces spanned by elements in given set of algebra-like objects.

    Parameters:
    -----------
    *args: algebra like objects
        This can be any number of algebra calss objects, or lists of algebra elements, or even single algebra elements. "algebra-like" is broad here, include algebra subspace/subalgebra classes, vector space classes etc. Each obeject is interpreted as spanning a subspace, so if single elements are given, they represent 1-d. subspaces, tc.

    Returns:
    --------
        list of alegra elements spanning the intersection.

    Notes:
    ------
        Returns empty list for 0-dimensional spaces
    """

    def scale_or_atom(obj):
        return get_dgcv_category(obj) in {
            "algebra_element",
            "subalgebra_element",
        } or check_dgcv_scalar(obj)

    factors = sorted(
        [[elem] if scale_or_atom(elem) else [x for x in elem] for elem in args],
        key=lambda item: len(item),
    )
    alg = None
    for factor in factors:
        f_alg = getattr(factor[0], "algebra") if len(factor) > 0 else None
        f_alg = getattr(f_alg, "ambient", f_alg)
        if f_alg is None:
            return []
        if alg is None:
            alg == f_alg
        elif alg != f_alg:
            return []
    if len(factors) == 0:
        return []
    b1 = _extract_basis(factors[0])

    for f2 in factors[1:]:
        w1 = wedge(*b1)
        b2 = _extract_basis(f2)
        genelem, variables = linear_combination(b2, _disposable=True)
        w2 = wedge(w1, genelem)
        sol = solve_dgcv(w2.coeffs, variables, method="linsolve", simplify_result=False)
        if len(sol) == 0:
            return []
        solelem = subs(genelem, sol[0])
        zeroing = {v: 0 for v in get_free_symbols(solelem) if v in variables}
        b1 = [subs(solelem, {**zeroing, v: 1}) for v in zeroing]
    return list(b1)


def vector_field_rep_from_linear_rep(
    algebra: algebra_class | subalgebra_class | list | tuple,
    coordinate_labels=None,
    register_new_coordinates_in_vmf=False,
) -> list:
    if isinstance(algebra, list | tuple):
        if len(algebra) == 0:
            return 0
        dim = None
        for m in algebra:
            if get_dgcv_category(m) != "array":
                raise TypeError(
                    "vector_field_rep_from_linear_rep can only be applied to Lie algebras or lists of matrices."
                )
            shape = m.shape
            if (
                len(shape) != 2
                or (dim is not None and any(x != dim for x in shape))
                or (dim is None and shape[0] != shape[1])
            ):
                raise TypeError(
                    "vector_field_rep_from_linear_rep can only be applied to Lie algebras or lists of square matrices with the same dimension."
                )
            dim = shape[0]
        matrices = algebra
    else:
        if not algebra.is_Lie_algebra:
            raise TypeError(
                "vector_field_rep_from_linear_rep can only be applied to Lie algebras"
            )
        matrices = (
            adjoint_representation(algebra)
            if algebra._mat_rep is None
            else algebra.matrix_representation
        )
    alg_dim = len(matrices)

    if coordinate_labels is None:
        label, labels = first_available_label(), None
    elif isinstance(coordinate_labels, str):
        label, labels = coordinate_labels, None
    elif (
        isinstance(coordinate_labels, list | tuple)
        and len(coordinate_labels) == alg_dim
        and all(isinstance(lab, str) for lab in coordinate_labels)
    ):
        label, labels = None, coordinate_labels
    if register_new_coordinates_in_vmf is True:
        if label:
            coordinates = createVariables(
                label, len(matrices), return_created_object=True
            )[0]
        else:
            coordinates = [
                createVariables(lab, return_created_object=True)[0][0] for lab in labels
            ]
    else:
        if label:
            if vmf_lookup(label).get("type") != "unregistered":
                dgcv_warning(
                    f"The coordinate label {label} provided to `vector_field_rep_from_linear_rep` is already assigned to an object in the VMF. It is advised to use a different label, or the keyword vector_field_rep_from_linear_rep(...,register_new_coordinates_in_vmf=True) instead."
                )
            coordinates = [symbol(f"{label}{idx}") for idx in range(1, alg_dim + 1)]
        else:
            if any(vmf_lookup(lab).get("type") for lab in labels):
                dgcv_warning(
                    "One or more of the coordinate labels provided to `vector_field_rep_from_linear_rep` is already assigned to an object in the VMF. It is advised to use a different label, or the keyword vector_field_rep_from_linear_rep(...,register_new_coordinates_in_vmf=True) instead."
                )
            coordinates = labels
    basis = []
    for mat in matrices:
        out = 0
        if get_dgcv_category(mat) == "array":
            for idx, val in mat._data.items():
                i, j = mat._unspool(idx)
                out += val * coordinates[i] * coordinate_vector_field(coordinates[j])
        else:
            for i in range(algebra.dimension):
                for j in range(algebra.dimension):
                    try:
                        out += (
                            mat[i, j]
                            * coordinates[i]
                            * coordinate_vector_field(coordinates[j])
                        )
                    except Exception:
                        raise TypeError(
                            "vector_field_rep_from_linear_rep encountered an object that does not behave as dgcv expects matrices to behave, may be an array class from another library."
                        )
        basis.append(out)

    return basis


def vector_field_representation(
    algebra: algebra_class | subalgebra_class | list | tuple,
    coordinate_labels=None,
    register_new_coordinates_in_vmf=False,
) -> list:
    """
    Attempts to find vector field representations of Lie algebras. Currently supports nilpotent and semisimple algebras, and others that are marked with a faithful linear representations.

    Parameters:
    -----------
    algebra: algebra_class | subalgebra

    coordinate_labels: str | list[str]
        Will be used to label coordinates that the vector fields are represented in

    register_new_coordinates_in_vmf: bool
        If true, the created coordinate system will be registered in the dgcv VMF (and active session's namespace.)

    Returns:
    --------
        list of vector fields representing a basis.

    Notes:
    ------
        Not fully implemented --  missing algorithm for general Lie algebras.
    """
    if isinstance(algebra, (list, tuple)):
        return vector_field_rep_from_linear_rep(
            algebra,
            coordinate_labels=coordinate_labels,
            register_new_coordinates_in_vmf=register_new_coordinates_in_vmf,
        )
    if algebra.is_nilpotent():
        order = len(algebra.lower_central_series()) - 1
    elif algebra._mat_rep or algebra.is_semisimple():
        return vector_field_rep_from_linear_rep(
            algebra,
            coordinate_labels=coordinate_labels,
            register_new_coordinates_in_vmf=register_new_coordinates_in_vmf,
        )
    else:
        raise TypeError(
            "Currently, vector_field_representation only supports Lie algebras that are nilpotent or semisimple or marked with a matrix representation in their metadata."
        )
    if coordinate_labels is None:
        label, labels = first_available_label(), None
    elif isinstance(coordinate_labels, str):
        label, labels = coordinate_labels, None
    elif (
        isinstance(coordinate_labels, list | tuple)
        and len(coordinate_labels) == algebra.dimension
        and all(isinstance(lab, str) for lab in coordinate_labels)
    ):
        label, labels = None, coordinate_labels
    if register_new_coordinates_in_vmf is True:
        if label:
            coordinates = createVariables(
                label, algebra.dimension, return_created_object=True
            )[0]
        else:
            coordinates = [
                createVariables(lab, return_created_object=True)[0][0] for lab in labels
            ]
    else:
        if label:
            if vmf_lookup(label).get("type") != "unregistered":
                dgcv_warning(
                    f"The coordinate label {label} provided to `vector_field_representation` is already assigned to an object in the VMF. It is advised to use a different label, or the keyword vector_field_representation(...,register_new_coordinates_in_vmf=True) instead."
                )
            coordinates = [
                symbol(f"{label}{idx}") for idx in range(1, algebra.dimension + 1)
            ]
        else:
            if any(vmf_lookup(lab).get("type") for lab in labels):
                dgcv_warning(
                    "One or more of the coordinate labels provided to `vector_field_representation` is already assigned to an object in the VMF. It is advised to use a different label, or the keyword vector_field_representation(...,register_new_coordinates_in_vmf=True) instead."
                )
            coordinates = labels

    basis = algebra.basis
    vfs = [coordinate_vector_field(coor) for coor in coordinates]
    svars = [symbol(f"_s_dgcv{idx}") for idx in range(algebra.dimension)]
    coor_curves = [c * elem for c, elem in zip(svars, basis)]
    gen_element = coor_curves[0]
    for elem in coor_curves[1:]:
        gen_element = Baker_Campbell_Hausdorff(
            gen_element, elem, truncation_degree=order
        )
    eqns = [
        var - gen_element.coeff_dict.get(idx, 0) for idx, var in enumerate(coordinates)
    ]
    sol = solve_dgcv(eqns, svars, method="linsolve", simplify_result=False)[0]
    vf_rep = []
    t = symbol("_t_dgcv")
    tvf = coordinate_vector_field(t)
    for idx in range(algebra.dimension):
        integral_curve = Baker_Campbell_Hausdorff(
            gen_element, t * basis[idx], truncation_degree=order
        )
        vf = 0
        for idx in range(algebra.dimension):
            c = integral_curve.coeff_dict.get(idx, 0)
            term = subs(tvf(c), sol | {t: 0}) * vfs[idx]
            vf += term
        vf_rep.append(vf)
    return vf_rep


def derivations(
    algebra: algebra_class | subalgebra_class, grading_preserving=True
) -> list:
    """
    Computes derivations of an algebra.

    Parameters:
    -----------
    algebra: algebra_class | subalgebra_class | algebra_dual

    grading_preserving: bool (optional, default=True)
        If the algebra has assigned gradings, then when True, this will restrict the computed derivations to only those that preserve the gradings.

    Returns:
    --------
        list of tensor products A x A^* representing endomorphisms on the algebra A

    Notes:
    ------
        The returns list of tensor product plus the algebra's basis added to the list can be given to dgcv.createAlgebra to build an algebra isomorphic to their semi-direct sum.
    """
    params = algebra._parameters if algebra._parameters else set()
    basis = algebra.basis
    if grading_preserving:
        components = algebra.graded_components
    pref = create_key("v")
    variables, targets = [], []
    w_space = algebra
    for c, x in enumerate(basis):
        if grading_preserving:
            w_space = components.get(x.check_element_weight())
        new_variables = [
            symbol(f"{pref}{c}_{idx2}") for idx2 in range(w_space.dimension)
        ]
        targets.append(zip_sum(w_space.basis, new_variables))
        variables += new_variables
    hom = homomorphism(algebra, algebra, targets)
    eqns = hom._alg_der_eqns
    sol = solve_dgcv(eqns, variables, method="linsolve", simplify_result=False)[0]
    fv = set()
    params = set()
    for expr in sol.values():
        fv |= get_free_symbols(expr)
    gen_derivation = subs(hom.tensor_representation, sol)
    fv = {v for v in fv if v not in params}
    if len(fv) == 0:
        return []
    zeroing = {v: 0 for v in fv}
    derivations = [subs(gen_derivation, {**zeroing, v: 1}) for v in fv]
    return derivations


def Levi_decomposition(
    algebra: algebra_class | subalgebra_class,
    decompose_semisimple_fully: bool = False,
    verbose: bool = False,
    assume_Lie_algebra: bool = False,
) -> tuple[subalgebra_class, subalgebra_class]:
    return algebra.Levi_decomposition(
        decompose_semisimple_fully=decompose_semisimple_fully,
        verbose=verbose,
        assume_Lie_algebra=assume_Lie_algebra,
    )


def quotient_by_ideal(
    algebra: algebra_class | subalgebra_class,
    subalgebra: algebra_subspace_class | algebra_class,
    label: str = None,
    basis_labels: str | Sequence[str] = None,
    initial_basis_index: int = 1,
    simplify_products_by_default: bool = None,
    register_in_vmf=True,
) -> algebra_class:
    """
    Constructs an algebra isomorphic to the quotient by an ideal.

    Parameters:
    -----------
    algebra: algebra_class

    subalgebra: algebra_class
        Needs to be an ideal in the given parent algebra.

    coordinate_labels: str | list[str]
        Will be used to label coordinates that the vector fields are represented in

    register_in_vmf: bool (default, True)
        If true, the created algebra will be registered in the dgcv VMF (and active session's namespace.)

    Returns:
    --------
        new algebra_class isomorphic to the quotient algebra/subalgebra

    Notes:
    ------
        If the given subalgebra `B` is not a subspace in the given algebra `A`, then it is interpreted as a subace in the direct sum A+B and A = (A+B)/B is returned.
    """
    if not isinstance(algebra, (algebra_class, subalgebra_class)) or not isinstance(
        subalgebra, (algebra_subspace_class, algebra_class)
    ):
        raise TypeError(
            "The `algebra` and `subalgebra` class parameters must be among algebra_class, subalgebra_class, or algebra_subspace_class types in the function `quotient_by_ideal`."
        )
    if not isinstance(label, str) and register_in_vmf is True:
        raise TypeError(
            "If setting `register_in_vmf=True` then the `label` parameter must be given a string value."
        )
    if algebra == subalgebra:
        return createAlgebra(
            0,
            label=label if label else "trivial_algebra",
            base_field=algebra.base_field,
            forgo_vmf_registry=not register_in_vmf,
            return_created_object=True,
        )
    if algebra.ambient != subalgebra.ambient:
        return algebra.copy(
            label=label,
            basis_labels=basis_labels,
            register_in_vmf=register_in_vmf,
            initial_basis_index=initial_basis_index,
            simplify_products_by_default=simplify_products_by_default,
        )
    first_basis = (
        [x.ambient_rep for x in algebra.basis]
        if isinstance(algebra, subalgebra_class)
        else algebra.basis
    )
    ss_basis = list(subalgebra.basis)
    running_span = wedge(*ss_basis)
    top_basis = []
    for x in first_basis:
        new_span = wedge(x, running_span)
        if new_span.is_zero:
            continue
        running_span = new_span
        top_basis.append(x)
    dim = len(top_basis)
    sd_out = array_dgcv(
        {},
        shape=(dim, dim),
        null_return=freeze_matrix(matrix_dgcv.zeros(dim, 1)),
    )
    v1, v2 = (
        [symbol(f"_dgcvv1_{idx}") for idx in range(dim)],
        [symbol(f"_dgcvv2_{idx}") for idx in range(len(ss_basis))],
    )
    general_element = zip_sum(v1, top_basis) + zip_sum(v2, ss_basis)
    skew = algebra.is_skew_symmetric()
    for idx1, x1 in enumerate(top_basis):
        start = idx1 + 1 if skew else 0
        for idx, x2 in enumerate(top_basis[start:]):
            idx2 = idx + start
            prod = x1 * x2
            sol = solve_dgcv(
                prod - general_element,
                v1 + v2,
                method="linsolve",
                simplify_result=False,
            )
            if len(sol) == 0:
                raise RuntimeError("Error in `quotient_by_ideal` algorithm")
            s = sol[0]
            new_coeffs = {
                nidx: s.get(v, 0)
                for nidx, v in enumerate(v1)
                if not _scalar_is_zero(s.get(v, 0))
            }
            sd_out[idx1, idx2] = matrix_dgcv(new_coeffs, shape=(dim, 1))
            if skew:
                sd_out[idx2, idx1] = matrix_dgcv(
                    {k: -v for k, v in new_coeffs.items()}, shape=(dim, 1)
                )
    return createAlgebra(
        sd_out,
        label=label,
        basis_labels=basis_labels,
        base_field=algebra.base_field,
        forgo_vmf_registry=not register_in_vmf,
        initial_basis_index=initial_basis_index,
        simplify_products_by_default=simplify_products_by_default,
        return_created_object=True,
    )


def build_linear_representation(
    algebra: algebra_class | subalgebra_class,
    representation_space,
    action: Sequence,
) -> linear_representation:
    if not isinstance(algebra, (algebra_class, subalgebra_class)):
        raise TypeError(
            "`build_linear_representation` expects an algebra as its first argument."
        )
    if get_dgcv_category(representation_space) not in {
        "algebra",
        "subalgebra",
        "vector_space",
    }:
        raise TypeError(
            "`build_linear_representation` expects a vector space or algebra as its second argument."
        )
    if not isinstance(action, (list, tuple)) or len(action) != algebra.dimension:
        raise ValueError(
            f"`action` needs one entry per basis element of the acting algebra ({algebra.dimension} in all)."
        )
    space = representation_space
    basis = list(space.basis)
    zero = 0 * (basis[0] @ basis[0].dual())

    def _is_element(x):
        return hasattr(x, "coeff_dict") and hasattr(x, "dual")

    def _tensor(entry):
        if get_dgcv_category(entry) == "tensorProduct":
            return entry
        if isinstance(entry, dict):
            pairs = list(entry.items())
        elif isinstance(entry, (list, tuple)) and all(
            isinstance(p, (list, tuple)) and len(p) == 2 and _is_element(p[0])
            for p in entry
        ):
            pairs = list(entry)
        elif check_dgcv_scalar(entry) and _scalar_is_zero(entry):
            return zero
        else:
            mat = _as_matrix_dgcv(entry)
            if mat is None or mat.shape != (len(basis), len(basis)):
                raise TypeError(
                    "Each entry of `action` should be a dict or list of (basis element, image) pairs, "
                    "a tensor such as image @ e.dual() + ..., or a square matrix acting on column vectors."
                )
            return _mat_to_tensor(mat, space.dual(), space)
        tp = zero
        for key, image in pairs:
            if key not in basis:
                raise ValueError(
                    f"`action` maps {key}, which is not a basis element of the representation space."
                )
            if check_dgcv_scalar(image) and _scalar_is_zero(image):
                continue
            if not _is_element(image) or getattr(image, "algebra", None) != space:
                raise ValueError(
                    f"The image of {key} should be an element of the representation space; received {image}."
                )
            tp = tp + image @ key.dual()
        return tp

    tensors = [_tensor(entry) for entry in action]
    return linear_representation(homomorphism(algebra, [space, space.dual()], tensors))


def new_graded_algebra_from_old(
    algebra: algebra_class | subalgebra_class,
    *combination_coefficients: Sequence,
    label: str = None,
    basis_labels: str | Sequence[str] = None,
    initial_basis_index: int = 1,
    simplify_products_by_default: bool = None,
    register_in_vmf=True,
) -> algebra_class:
    if not isinstance(algebra, (algebra_class, subalgebra_class)):
        raise TypeError(
            "`new_graded_algebra_from_old` expects an algebra_class or subalgebra_class instance as its first argument."
        )
    if not isinstance(label, str) and register_in_vmf is True:
        raise TypeError(
            "If setting `register_in_vmf=True` then the `label` parameter must be given a string value."
        )
    old_grading = list(getattr(algebra, "grading", None) or [])
    if len(old_grading) == 0:
        raise ValueError(
            "`new_graded_algebra_from_old` requires an algebra with at least one grading vector."
        )
    if len(combination_coefficients) == 0:
        raise ValueError(
            "`new_graded_algebra_from_old` requires at least one list of combination coefficients."
        )
    for coeffs in combination_coefficients:
        if (
            not isinstance(coeffs, (list, tuple))
            or len(coeffs) != len(old_grading)
            or not all(check_dgcv_scalar(c) for c in coeffs)
        ):
            raise ValueError(
                f"Each list of combination coefficients must be a list or tuple of {len(old_grading)} scalars, one per grading vector of the given algebra. Received {coeffs}"
            )
    new_grading = [
        tuple(
            sum(c * weights[j] for c, weights in zip(coeffs, old_grading))
            for j in range(algebra.dimension)
        )
        for coeffs in combination_coefficients
    ]
    return createAlgebra(
        algebra,
        label=label,
        basis_labels=basis_labels,
        grading=new_grading,
        base_field=algebra.base_field,
        forgo_vmf_registry=not register_in_vmf,
        initial_basis_index=initial_basis_index,
        simplify_products_by_default=simplify_products_by_default,
        return_created_object=True,
    )


def center(
    algebra: algebra_class | subalgebra_class,
    format_as_subalgebra=True,
    surface_singularities=False,
) -> list | subalgebra_class:
    return algebra.center(
        surface_singularities=surface_singularities,
        format_as_subalgebra=format_as_subalgebra,
    )


def find_simple_subalgebras(
    alg: algebra_class | subalgebra_class,
    assume_Lie_algebra: bool = False,
    verbose: bool = False,
    surface_singularities: bool | None = None,
    simplify_singularities: bool | None = None,
    force_heavy_solve: bool = False,
) -> list[subalgebra_class]:
    return alg.simple_subalgebras(
        assume_Lie_algebra=assume_Lie_algebra,
        verbose=verbose,
        surface_singularities=surface_singularities,
        simplify_singularities=simplify_singularities,
        force_heavy_solve=force_heavy_solve,
    )
