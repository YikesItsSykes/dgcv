from __future__ import annotations

import numbers

from ..._aux._backends._symbolic_router import (
    _scalar_is_one,
    _scalar_is_zero,
    get_free_symbols,
)
from ..._aux._backends._types_and_constants import expr_numeric_types
from ..._aux._utilities._config import dgcv_warning
from ..._aux._utilities._misc import linear_combination
from ..._aux._vmf._safeguards import create_key, get_dgcv_category, retrieve_passkey
from ...core.arrays import freeze_matrix, matrix_dgcv
from ...core.dgcv_core.spaces.spaces import _vs_card
from ...core.solvers import solve_dgcv
from ..algebras import algebra_class
from ..composition.sums import _sa_direct_sum
from ..creators import createAlgebra
from ..display.multiplication_table import _sa_multiplication_table
from ..linear_algebra import (
    _flatten_structure_data,
    _gather_structure_singularities,
    _structure_array,
)
from ..saec import subalgebra_element
from ..subspaces import algebra_subspace_class
from ..threads import _algebra_methods, killingForm
from ..threads.util import _span_solver


class subalgebra_class(_algebra_methods, algebra_subspace_class):
    def __init__(
        self,
        basis,
        alg,
        grading=None,
        _compressed_structure_data=None,
        _internal_lock=None,
        span_warning=True,
        simplify_basis=False,
        simplify_products_by_default=None,
        base_field=None,
        _markers={},
        _lazy_structure_data=False,
        **kwargs,
    ):
        super().__init__(
            basis,
            alg,
            test_weights=None,
            _grading=grading,
            _internal_lock=_internal_lock,
            span_warning=span_warning,
            simplify_basis=simplify_basis,
        )

        # must go after super().__init__ and before assembling SD
        self.card = _vs_card(self, getattr(self.ambient, "card", None))

        basis = self.filtered_basis
        self._structure_data = None
        self._structure_data_dict = None
        self._lazy_rows = None
        self._lazy_structure_data = bool(
            _lazy_structure_data and _internal_lock == retrieve_passkey()
        )

        if self._lazy_structure_data:
            params = set(getattr(self.ambient, "_parameters", ()) or ())
            for elem in basis:
                for v in elem.coeff_dict.values():
                    params |= set(get_free_symbols(v))
            self._parameters = params
        else:
            if _internal_lock == retrieve_passkey() and simplify_basis is False:
                if _compressed_structure_data is not None:
                    self._structure_data = _compressed_structure_data
            if self._structure_data is None:
                self._structure_data = self.is_subalgebra(
                    return_structure_data=True
                )["structure_data"]
            self._parameters = get_free_symbols(self._structure_data)

        self.basis_in_ambient_alg = tuple(basis)
        self.basis = [
            subalgebra_element(
                self,
                [1 if j == count else 0 for j in range(self.dimension)],
                elem.valence,
            )
            for count, elem in enumerate(basis)
        ]
        amb_index = self.ambient._basis_index
        if all(elem in amb_index for elem in basis):
            self.basis_labels = [elem.__str__() for elem in self.basis]
        else:
            self.basis_labels = [f"_e_{j + 1}" for j in range(self.dimension)]
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "subalgebra"

        if not self._lazy_structure_data:
            self._structure_data_dict = _flatten_structure_data(
                self._structure_data, _source="subalgebra_class"
            )
        _ambient_field = getattr(self.ambient, "base_field", "complex")
        if base_field is None:
            self.base_field = _ambient_field
        elif base_field not in ("real", "complex"):
            self.base_field = "complex"
        elif base_field == "complex" and _ambient_field == "real":
            raise ValueError(
                "A subalgebra cannot be flagged `base_field='complex'` inside a "
                "real ambient algebra."
            ) from None
        else:
            self.base_field = base_field
            if base_field == "real" and _ambient_field != "real":
                self._profile_structure_data()
        if (
            simplify_products_by_default is True
            or self.ambient.simplify_products_by_default is True
        ):
            self.simplify_products_by_default = True
        elif self._lazy_structure_data:
            self.simplify_products_by_default = False
        else:
            self.simplify_products_by_default = simplify_products_by_default
        self._registered = self.ambient._registered
        if self._parameters and not self._lazy_structure_data:
            self._singularities = {
                "structure": _gather_structure_singularities(
                    self._structure_data, self._parameters
                )
            }
        else:
            self._singularities = {}
        known_s = kwargs.get("_known_singularities", False)
        if known_s:
            self._singularities |= known_s
        # cached_properties
        self._jacobi_identity_cache = None
        self._skew_symmetric_cache = None
        self._lie_algebra_cache = None
        self._killing_form = None
        self._derived_subalg_cache = None
        self._derived_subalg_lazy_cache = None
        self._derived_series_cache = None
        self._derived_series_terminated = None
        self._derived_series_depth = None
        self._lower_central_series_cache = None
        self._lower_central_series_terminated = None
        self._lower_central_series_depth = None
        self._center_cache = None
        self._grading_compatible = None
        self._grading_report = None
        self._radical_cache = None
        self._Levi_deco_cache = None
        self._is_semisimple_cache = None
        self._is_simple_cache = None
        self._is_nilpotent_cache = None
        self._is_abelian_cache = None
        self._is_solvable_cache = None
        self._rank_approximation = None
        self._graded_components = None
        self._basis_index_cache = None
        self._structure_data_slices = None
        self._structure_rows_cache = None
        self._structure_data_profile = None
        self._verified_ideal = False
        self._centroid_type = None
        self._educed_properties = dict()
        ep = getattr(self.ambient, "_educed_properties", dict())
        t_message = "True by inheritance: parent algebra --> subalgebra"
        if ep.get("is_Lie_algebra", None) is not None:
            self._educed_properties["is_Lie_algebra"] = t_message
        if ep.get("is_skew", None) is not None:
            self._educed_properties["is_skew"] = t_message
        if ep.get("satisfies_Jacobi_ID", None) is not None:
            self._educed_properties["satisfies_Jacobi_ID"] = t_message
        if ep.get("is_nilpotent", None) is not None:
            self._educed_properties["is_nilpotent"] = t_message
        if ep.get("is_solvable", None) is not None:
            self._educed_properties["is_solvable"] = t_message
        if ep.get("special_type", None) in {"abelian", "solvable", "nilpotent"}:
            self._educed_properties["special_type"] = ep.get("special_type", None)

    @property
    def structureData(self):
        if self._structure_data is None:
            self._materialize_structure_data()
        return self._structure_data

    @structureData.setter
    def structureData(self, value):
        self._structure_data = value
        self._structure_data_dict = None
        self._lazy_structure_data = False

    @property
    def structureDataDict(self):
        if self._structure_data_dict is None:
            if self._structure_data is None:
                self._materialize_structure_data()
            self._structure_data_dict = _flatten_structure_data(
                self._structure_data, _source="subalgebra_class"
            )
        return self._structure_data_dict

    @structureDataDict.setter
    def structureDataDict(self, value):
        self._structure_data_dict = value

    def _materialize_structure_data(self):
        self._is_subalgebra = None
        self._structure_data = self.is_subalgebra(return_structure_data=True)[
            "structure_data"
        ]
        self._structure_data_dict = None
        self._lazy_structure_data = False
        self._lazy_rows = None
        if self._parameters and "structure" not in self._singularities:
            self._singularities["structure"] = _gather_structure_singularities(
                self._structure_data, self._parameters
            )

    def _structure_row(self, idx1, idx2):
        if not self._lazy_structure_data:
            return self.structureData[idx1, idx2]
        rows = self._lazy_rows
        if rows is None:
            solver = _span_solver.build(list(self.basis_in_ambient_alg))
            if solver is None:
                return self.structureData[idx1, idx2]
            rows = self._lazy_rows = {"solver": solver, "skew": None}
        row = rows.get((idx1, idx2))
        if row is not None:
            return row
        amb = self.basis_in_ambient_alg
        product = amb[idx1] * amb[idx2]
        out = rows["solver"].reduce(product, return_decomp_coeffs=True)
        if out is None:
            return self.structureData[idx1, idx2]
        if out[0] is True:
            raise TypeError(
                "The basis of this lazily constructed subalgebra does not span a subalgebra."
            )
        coeffs = {k: v for k, v in out[1][0].items() if not _scalar_is_zero(v)}
        row = matrix_dgcv(coeffs, shape=(self.dimension, 1))
        rows[(idx1, idx2)] = row
        if rows["skew"] is None:
            rows["skew"] = self.ambient.is_skew_symmetric() is True
        if rows["skew"] and idx1 != idx2:
            rows[(idx2, idx1)] = -row
        return row

    @property
    def zero_element(self):
        return subalgebra_element(self, {}, 1)

    def _verbose_subject(self):
        if self.ambient.label is None:
            return "The subalgebra"
        return f"The subalgebra in {self.ambient.label}"

    def __iter__(self):
        return iter(self.basis)

    def __getitem__(self, indices):
        if isinstance(indices, numbers.Integral):
            return self.basis[indices]
        elif isinstance(indices, (list, tuple)):
            if len(indices) == 1:
                return self.basis[indices[0]]
            elif len(indices) == 2:
                return self.structureData[indices[0], indices[1]]
            elif len(indices) == 3:
                return self.structureData[indices[0], indices[1]][indices[2]]
            raise TypeError(
                f"Expected one, two, or three indices. Received {len(indices)}: {indices}"
            ) from None
        else:
            raise TypeError(
                f"To access a subalgebra element or structure data component, provide one index for an element from the basis, two indices for a list of coefficients from the product  of two basis elements, or 3 indices for the corresponding entry in the structure array. Instead of an integer of list of integers, the following was given: {indices}"
            ) from None

    def _class_builder(self, coeffs, valence):
        return subalgebra_element(self, coeffs, valence)

    def multiplication_table(
        self,
        elements=None,
        restrict_to_subspace=False,
        theme=None,
        use_latex=None,
        plain_text: bool | None = None,
        return_displayable: bool = False,
        col_number_limit: int = 10,
        row_number_limit: int = 15,
        cell_char_lim: int = 20,
        table_css_properties: str = None,
        **kwargs,
    ):
        return _sa_multiplication_table(
            self,
            elements=elements,
            restrict_to_subspace=restrict_to_subspace,
            theme=theme,
            use_latex=use_latex,
            plain_text=plain_text,
            return_displayable=return_displayable,
            col_number_limit=col_number_limit,
            row_number_limit=row_number_limit,
            cell_char_lim=cell_char_lim,
            table_css_properties=table_css_properties,
            **kwargs,
        )

    def subalgebra(
        self,
        basis,
        grading=None,
        span_warning=False,
        simplify_basis=False,
        simplify_products_by_default=None,
        base_field=None,
        lazy_structure=False,
    ):
        elems = [
            (
                elem.ambient_rep
                if get_dgcv_category(elem) == "subalgebra_element"
                else elem
            )
            for elem in basis
        ]
        if not lazy_structure and basis:
            restricted = self._restricted_structure_data(basis, elems, simplify_basis)
            if restricted is not None:
                if simplify_products_by_default is None:
                    simplify_products_by_default = (
                        self.ambient.simplify_products_by_default
                    )
                known = None
                if self.ambient._parameters:
                    known = {"basis": list(self._singularities.get("basis", []))}
                return subalgebra_class(
                    elems,
                    self.ambient,
                    grading=grading,
                    _compressed_structure_data=restricted,
                    _internal_lock=retrieve_passkey(),
                    span_warning=False,
                    simplify_basis=False,
                    simplify_products_by_default=simplify_products_by_default,
                    _known_singularities=known,
                    base_field=base_field,
                )
        return self.ambient.subalgebra(
            elems,
            grading=grading,
            simplify_basis=simplify_basis,
            span_warning=span_warning,
            simplify_products_by_default=simplify_products_by_default,
            base_field=base_field,
            lazy_structure=lazy_structure,
        )

    def _restricted_structure_data(self, basis, elems, simplify_basis):
        pos = self.ambient._basis_index
        for elem in elems:
            try:
                idx = pos.get(elem)
            except TypeError:
                idx = None
            if idx is None:
                break
        else:
            return None
        index_map = {}
        for count, elem in enumerate(basis):
            if (
                get_dgcv_category(elem) != "subalgebra_element"
                or elem.algebra is not self
            ):
                return None
            hot = [
                (idx, v) for idx, v in elem.coeff_dict.items() if not _scalar_is_zero(v)
            ]
            if len(hot) != 1 or not _scalar_is_one(hot[0][1]) or hot[0][0] in index_map:
                return None
            index_map[hot[0][0]] = count
        if simplify_basis:
            for rep in elems:
                for c in rep.coeffs:
                    if not _scalar_is_zero(c):
                        if not get_free_symbols(c) and not _scalar_is_one(c):
                            return None
                        break
        sub_dim = len(basis)
        columns = {}
        for (i, j, k), v in self.structureDataDict.items():
            if i not in index_map or j not in index_map:
                continue
            if k in index_map:
                columns.setdefault((index_map[i], index_map[j]), {})[index_map[k]] = v
            elif v is not None and not _scalar_is_zero(v):
                return None
        skew = self.ambient.is_skew_symmetric()
        shape = (sub_dim, 1)
        new_data = {}
        for count in range(sub_dim):
            for j in range(count + 1 if skew else 0, sub_dim):
                col = columns.get((count, j), {})
                coeff_array = matrix_dgcv(
                    {idx: col[idx] for idx in sorted(col) if not _scalar_is_zero(col[idx])},
                    shape=shape,
                )
                new_data[(count, j)] = coeff_array
                if skew:
                    new_data[(j, count)] = -coeff_array
        return _structure_array(new_data, sub_dim)

    def subspace(self, basis: list | tuple = [], grading=None, span_warning=True):
        elems = [
            (
                elem.ambient_rep
                if get_dgcv_category(elem) == "subalgebra_element"
                else elem
            )
            for elem in basis
        ]
        return self.ambient.subspace(elems, grading=grading, span_warning=span_warning)

    def copy(
        self,
        label=None,
        basis_labels=None,
        register_in_vmf=False,
        initial_basis_index=None,
        simplify_products_by_default=None,
    ):
        if simplify_products_by_default is None:
            simplify_products_by_default = self.simplify_products_by_default
        if not isinstance(label, str) or label == "":
            label = "Alg_" + create_key()
        if isinstance(basis_labels, (tuple, list)):
            if (
                not all(isinstance(elem, str) for elem in basis_labels)
                or len(basis_labels) != self.dimension
            ):
                dgcv_warning(
                    "`basis_labels` is in an unsupported format and was ignored"
                )
                basis_labels = None
        if not isinstance(basis_labels, (tuple, list)):
            pref = (
                basis_labels
                if (isinstance(basis_labels, str) and basis_labels != "")
                else "_e"
            )
            IIdx = (
                initial_basis_index
                if isinstance(initial_basis_index, numbers.Integral)
                else 1
            )
            basis_labels = [f"{pref}{i + IIdx}" for i in range(self.dimension)]
        if not isinstance(self._grading, (list, tuple)) or len(self._grading) == 0:
            grad = None
        else:
            grad = self._grading
        _markers = {
            "parameters": set(self._parameters),
            "_educed_properties": dict(self._educed_properties),
            "semidirect_decomposition": getattr(self, "semidirect_decomposition", None),
            "tensor_decomposition": getattr(self, "tensor_decomposition", None),
            "base_field": self.base_field,
        }
        if register_in_vmf is True:
            return createAlgebra(
                self.structureData,
                label,
                basis_labels=basis_labels,
                grading=grad,
                return_created_object=True,
                simplify_products_by_default=simplify_products_by_default,
                _markers=_markers,
            )
        return algebra_class(
            self.structureData,
            grading=grad,
            simplify_products_by_default=simplify_products_by_default,
            _label=label,
            _basis_labels=basis_labels,
            _calledFromCreator=retrieve_passkey(),
            _markers=_markers,
        )

    def is_subspace_subalgebra(
        self, elements, return_structure_data=False, check_linear_independence=False
    ):
        """
        Checks if a set of elements is a subspace is a subalgebra. `check_linear_independence` will additional verify if provided spanning elements are a basis.

        Parameters
        ----------
        elements : list
            A list of algebra_element_class instances.
        return_structure_data : bool, optional
            If True, returns the structure constants for the subalgebra. Returned
            data becomes a dictionary
        check_linear_independence : bool, optional
            If True, a check of linear independence of basis elements is also performed

        Returns
        -------
        dict or bool
            - If return_structure_data=True, returns a dictionary with keys:
            - 'linearly_independent': True/False
            - 'closed_under_product': True/False
            - 'structure_data': 3D list of structure constants
            - Otherwise, returns True if the elements form a subspace subalgebra, False otherwise.
        """

        elems = [
            (
                elem.ambient_rep
                if get_dgcv_category(elem) == "subalgebra_element"
                else elem
            )
            for elem in elements
        ]
        return self.ambient.is_subspace_subalgebra(
            elems,
            return_structure_data=return_structure_data,
            check_linear_independence=check_linear_independence,
        )

    def filter_independent_elements(
        self,
        elements,
        apply_light_basis_simplification=False,
        return_indices: bool = False,
        surface_singularities: bool = False,
        simplify_singularities: bool = None,
        force_heavy_solve: bool = False,
    ):
        """
        Filters a set of elements to retain only a linearly independent subset.

        Parameters
        ----------
        elements : list of algebra_element_class or subalgebra_element_class
            The set of elements to filter.

        Returns
        -------
        list
            A subset of the input elements that are linearly independent and unique.
        """
        out = self.ambient.filter_independent_elements(
            [elem.ambient_rep for elem in elements],
            apply_light_basis_simplification=apply_light_basis_simplification,
            return_indices=True,
            surface_singularities=surface_singularities,
            simplify_singularities=simplify_singularities,
            force_heavy_solve=force_heavy_solve,
        )
        if surface_singularities:
            idx_list, sing = out
        else:
            idx_list = out
        if return_indices is True:
            return (idx_list, sing) if surface_singularities else idx_list
        filtered = [elements[idx] for idx in idx_list]
        return (filtered, sing) if surface_singularities else filtered

    def is_in_span(self, element, subspace_elements, assume_basis=False):
        return self.ambient.is_in_span(
            element, subspace_elements, assume_basis=assume_basis
        )

    def compute_center(self, for_associative_alg=False, assume_Lie_algebra=False):
        """
        Computes the center of the algebra as a subspace.

        Parameters
        ----------
        for_associative_alg : bool, optional
            If True, computes the center for an associative algebra. Defaults to False (assumes Lie algebra).

        Returns
        -------
        list
            A list of algebra_element_class instances that span the center of the algebra.

        Raises
        ------
        ValueError
            If `for_associative_alg` is False and the algebra is not a Lie algebra.

        Notes
        -----
        - For Lie algebras, the center is the set of elements `z` such that `z * x = 0` for all `x` in the algebra.
        - For associative algebras, the center is the set of elements `z` such that `z * x = x * z` for all `x` in the algebra.
        """

        if for_associative_alg is True:
            assume_Lie_algebra = False
        elif assume_Lie_algebra is False and not self.is_Lie_algebra():
            raise ValueError(
                "This algebra is not a Lie algebra. To compute the center for an associative algebra, set for_associative_alg=True."
            ) from None

        el, temp_vars = linear_combination(self.basis, _disposable=True)

        if for_associative_alg:
            eqns = [
                v
                for other in self.basis
                for v in (el * other - other * el).coeff_dict.values()
            ]
        else:
            eqns = [v for other in self.basis for v in (el * other).coeff_dict.values()]

        solutions = solve_dgcv(
            eqns, temp_vars, method="linsolve", simplify_result=False
        )
        if not solutions:
            dgcv_warning(
                "The internal solver (which depends on which symbolic engine is defaults in dgcv settings) returned no solutions, indicating that this computation of the center failed, as solutions do exist. An empty list is being returned."
            )
            return []

        el_sol = el.subs(solutions[0])

        free_variable_set = set()
        for j in el_sol.coeff_dict.values():
            free_variable_set |= set(get_free_symbols(j))
        if not free_variable_set:
            return []
        free_variables = tuple(sorted(free_variable_set, key=str))
        zeroing = {v: 0 for v in free_variables}

        return_list = []
        for var in free_variables:
            return_list.append(el_sol.subs({**zeroing, var: 1}))

        return return_list  ###!!! return subalgebra instead

    def killing_form_product(self, elem1, elem2, assume_Lie_algebra=False):
        coeffs1 = self.contains(elem1, return_basis_coeffs=True, strict_types=True)
        coeffs2 = self.contains(elem2, return_basis_coeffs=True, strict_types=True)
        if coeffs1 is False or coeffs2 is False:
            raise TypeError(
                "subalgebra_class.killing_form_product only operates on algebra elements from the dispatching subalgebra"
            )
        if self._killing_form is None:
            self._killing_form = freeze_matrix(
                killingForm(self, assume_Lie_algebra=assume_Lie_algebra)
            )
        kf = self._killing_form
        vec1 = matrix_dgcv(coeffs1, shape=(self.dimension, 1))
        vec2 = matrix_dgcv(coeffs2, shape=(1, self.dimension))
        return (vec2 * kf * vec1)[0]

    def direct_sum(
        self,
        other,
        grading=None,
        label=None,
        basis_labels=None,
        register_in_vmf=False,
        initial_basis_index=None,
        simplify_products_by_default=None,
        build_all_gradings=False,
    ):
        _sa_direct_sum(
            self,
            other,
            grading=grading,
            label=label,
            basis_labels=basis_labels,
            register_in_vmf=register_in_vmf,
            initial_basis_index=initial_basis_index,
            simplify_products_by_default=simplify_products_by_default,
            build_all_gradings=build_all_gradings,
        )

    def tensor_product(
        self,
        other,
        grading=None,
        label=None,
        basis_labels=None,
        register_in_vmf=False,
        initial_basis_index=None,
        simplify_products_by_default=None,
        build_all_gradings=False,
    ):
        numeric_types = expr_numeric_types()
        if get_dgcv_category(other) in {
            "algebra",
            "vectorspace",
            "subalgebra",
            "algebra_subspace",
            "vector_subspace",
        }:
            if simplify_products_by_default is None:
                simplify_products_by_default = getattr(
                    self, "simplify_products_by_default", False
                )
            if build_all_gradings is not True:
                grad1 = self.grading[:1] or [[0] * self.dimension]
                grad2 = other.grading[:1] or [[0] * other.dimension]
            else:
                grad1 = self.grading or [[0] * self.dimension]
                grad2 = other.grading or [[0] * other.dimension]
            builtG = []
            for gl1 in grad1:
                for gl2 in grad2:
                    builtG.append([w1 + w2 for w1 in gl1 for w2 in gl2])
            if not isinstance(grading, (list, tuple)):
                grading = []
            if isinstance(grading, (list, tuple)):
                if all(isinstance(elem, (list, tuple)) for elem in grading):
                    grading = [list(elem) for elem in grading] + builtG
                elif all(isinstance(elem, numeric_types) for elem in grading):
                    grading = [list(grading)] + builtG
                elif grading is not None:
                    dgcv_warning(
                        "The supplied grading data format is incompatible, and was ignored."
                    )
                    grading = builtG
                else:
                    grading = builtG

            if isinstance(basis_labels, (tuple, list)):
                if (
                    not all(isinstance(elem, str) for elem in basis_labels)
                    or len(basis_labels) != self.dimension * other.dimension
                ):
                    dgcv_warning(
                        f"`basis_labels` is in an unsupported format and was ignored. Recieved {basis_labels}, types: {[type(lab) for lab in basis_labels]}, target length {self.dimension}*{other.dimension}"
                    )
                    basis_labels = None
            _markers = {
                "prod": True,
                "lockKey": retrieve_passkey(),
                "tensor_decomposition": (self, other),
                "base_field": (
                    "real"
                    if self.base_field == "real"
                    and getattr(other, "base_field", "complex") == "real"
                    else "complex"
                ),
            }
            if label is None:
                label = f"{self.label}_tensor_{other.label}"
                _markers["_tex_label"] = (
                    f"{self._repr_latex_(raw=True, abbrev=True)}\\otimes {other._repr_latex_(raw=True, abbrev=True)}"
                )
            if basis_labels is None or not isinstance(basis_labels, str):
                basis_labels = [
                    f"{elem1.__repr__()}_tensor_{elem2.__repr__()}"
                    for elem1 in self.basis
                    for elem2 in other.basis
                ]
                _markers["_tex_basis_labels"] = [
                    f"{elem1._repr_latex_(raw=True)}\\otimes {elem2._repr_latex_(raw=True)}"
                    for elem1 in self.basis
                    for elem2 in other.basis
                ]
            if isinstance(basis_labels, str):
                pref = basis_labels
                IIdx = (
                    initial_basis_index
                    if isinstance(initial_basis_index, numbers.Integral)
                    else 1
                )
                basis_labels = [
                    f"{pref}{i + IIdx}" for i in range(self.dimension * other.dimension)
                ]
            if not isinstance(label, str) or label == "":
                label = "Alg_" + create_key()

            if register_in_vmf is True:
                from .subalgebras import createAlgebra

                return createAlgebra(
                    self.dimension * other.dimension,
                    label,
                    basis_labels=basis_labels,
                    grading=grading,
                    return_created_object=True,
                    simplify_products_by_default=simplify_products_by_default,
                    _markers=_markers,
                )
            else:
                _markers["registered"] = False
                return algebra_class(
                    self.dimension * other.dimension,
                    grading=grading,
                    simplify_products_by_default=simplify_products_by_default,
                    _label=label,
                    _basis_labels=basis_labels,
                    _calledFromCreator=retrieve_passkey(),
                    _markers=_markers,
                )
        else:
            return NotImplemented

    def __add__(self, other):
        if _scalar_is_zero(other):
            return self
        if get_dgcv_category(other) in {
            "algebra",
            "vectorspace",
            "subalgebra",
            "algebra_subspace",
            "vector_subspace",
        }:
            return self.direct_sum(other)
        return NotImplemented

    def __radd__(self, other):
        if _scalar_is_zero(other):
            return self
        return NotImplemented

    def __matmul__(self, other):
        if get_dgcv_category(other) in {
            "algebra",
            "vectorspace",
            "subalgebra",
            "algebra_subspace",
            "vector_subspace",
        }:
            return self.tensor_product(other)
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, expr_numeric_types()):
            return self._convert_to_tp().__rmatmul__(other)
        return NotImplemented
