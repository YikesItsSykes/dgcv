from __future__ import annotations

import numbers
from typing import List, Literal, Optional

from .._aux._backends._symbolic_router import _scalar_is_zero, get_free_symbols, subs
from .._aux._backends._types_and_constants import expr_numeric_types
from .._aux._utilities._config import (
    dgcv_warning,
    get_dgcv_settings_registry,
)
from .._aux._utilities._misc import linear_combination
from .._aux._vmf._safeguards import create_key, get_dgcv_category, retrieve_passkey
from .._aux.printing.printing import space_display
from ..core.arrays import freeze_matrix, matrix_dgcv
from ..core.base import dgcv_class
from ..core.solvers import solve_dgcv
from .aec import algebra_element_class
from .composition.sums import direct_sum
from .composition.tensor_products import tensor_product
from .display.multiplication_table import multiplication_table
from .dual import algebra_dual
from .format_support import _alg_init
from .interfacing import structure_equations
from .linear_algebra import _representation
from .subspaces import (
    is_ideal,
    is_subspace_subalgebra,
    new_alg_from_subalgebra,
    subalgebra,
)
from .threads import _algebra_methods, adjointRepresentation, killingForm


class algebra_class(_algebra_methods, dgcv_class):
    def __init__(
        self,
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
        _alg_init(
            self,
            structure_data,
            grading=grading,
            base_field=base_field,
            format_sparse=format_sparse,
            process_matrix_rep=process_matrix_rep,
            preferred_representation=preferred_representation,
            simplify_products_by_default=simplify_products_by_default,
            assume_skew=assume_skew,
            matrix_representation=matrix_representation,
            tensor_representation=tensor_representation,
            _basis_labels_parent=_basis_labels_parent,
            _label=_label,
            _basis_labels=_basis_labels,
            _calledFromCreator=_calledFromCreator,
            _callLock=_callLock,
            _print_warning=_print_warning,
            _child_print_warning=_child_print_warning,
            _exclude_from_VMF=_exclude_from_VMF,
            _markers=_markers,
        )

    def _class_builder(self, coeff_dict, valence):
        return algebra_element_class(self, coeff_dict, valence)

    @property
    def preferred_representation(self):
        if self._preferred_representation is None:
            if self._mat_rep is not None:
                dgcv_warning(
                    "A preferred representation format for this algebra was never set up, but a cached matrix representation was found and has been set as the default for `preferred_representation`."
                )
                self._preferred_rep_type = "matrix"
                self._preferred_representation = self._mat_rep
            elif self._tensor_rep is not None:
                dgcv_warning(
                    "A preferred representation format for this algebra was never set up, but a cached tensor product representation was found and has been set as the default for `preferred_representation`."
                )
                self._preferred_rep_type = "tensor"
                self._preferred_representation = self._tensor_rep
            else:
                dgcv_warning(
                    "A preferred representation format for this algebra was not specified, so it has been set to its adjoint representation."
                )
                self._preferred_rep_type = "matrix"
                self._preferred_representation = adjointRepresentation(self)
        return self._preferred_representation

    @property
    def tensor_representation(self):
        return self._tensor_rep

    @property
    def matrix_representation(self):
        return self._mat_rep

    @property
    def ambient(self):
        return self

    @property
    def basis_in_ambient_alg(self):
        return self.basis

    def _verbose_subject(self):
        return "The algebra" if self.label is None else str(self.label)

    @property
    def endomorphism_algebra(self):
        if self._endomorphisms is None:
            from .composition.composites import vector_space_endomorphisms

            self._endomorphisms = vector_space_endomorphisms(self)
        return self._endomorphisms

    def structure_equations(
        self,
        formatting: Optional[Literal["dict", "list"]] = "dict",
        new_basis_labels: Optional[str | List[str]] = None,
        abbreviate_for_skew_struct: bool = None,
        initial_index: int = 1,
        list_symbols_as_strings: bool = False,
    ):
        return structure_equations(
            self,
            formatting=formatting,
            new_basis_labels=new_basis_labels,
            abbreviate_for_skew_struct=abbreviate_for_skew_struct,
            initial_index=initial_index,
            list_symbols_as_strings=list_symbols_as_strings,
        )

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
                f"To access an algebra element or structure data component, provide one index for an element from the basis, two indices for a list of coefficients from the product  of two basis elements, or 3 indices for the corresponding entry in the structure array. Instead of an integer of list of integers, the following was given: {indices}"
            ) from None

    def __str__(self, VLP=None):
        if not self._registered:
            if (
                self._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self._callLock == retrieve_passkey() and isinstance(
                self._print_warning, str
            ):
                dgcv_warning(self._print_warning)
            else:
                dgcv_warning(
                    "This algebra instance was initialized without an assigned label. "
                    "It is recommended to initialize algebra objects with dgcv creator functions like `createFiniteAlg` instead -- or set `label` parameter if creating it via a dgcv class method."
                )

        reg = get_dgcv_settings_registry()
        VLP = (
            bool(VLP)
            if VLP is not None
            else bool(reg.get("verbose_label_printing", False))
        )
        if VLP is False:
            return str(self.label) if self.label else "Unnamed Algebra"

        return space_display(
            fmt="plain",
            basis_words=(lambda: [str(e) for e in self.basis]),
            dim=self.dimension,
            label=self.label,
            unlabeled_plain="Unnamed Algebra",
            max_dim=20,
            raw=True,
            abbrev=False,
            plain_wrapper="<{}>",
        )

    def _repr_latex_(self, verbose=False, abbrev=False, raw=False, **kwargs):
        if not self._registered:
            if (
                self._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self._callLock == retrieve_passkey() and isinstance(
                self._print_warning, str
            ):
                dgcv_warning(self._print_warning)
            else:
                dgcv_warning(
                    "This algebra instance was initialized without an assigned label. "
                    "It is recommended to initialize algebra objects with dgcv creator functions like `createFiniteAlg` instead -- or set `label` parameter if creating it via a dgcv class method."
                )

        reg = get_dgcv_settings_registry()
        if bool(reg.get("verbose_label_printing", False)) is False:
            return space_display(
                fmt="latex",
                basis_words=(),
                dim=self.dimension,
                label=self.label,
                label_tex=(
                    self._tex_label
                    if getattr(self, "_tex_label", None) is not None
                    else None
                ),
                mathfrak_label=True,
                unlabeled_tex=r"\text{Unnamed Algebra}",
                max_dim=20,
                raw=raw,
                abbrev=True,
                use_displaystyle=False,
                latex_wrapper=r"\langle {}\rangle",
            )

        if abbrev:
            return space_display(
                fmt="latex",
                basis_words=(),
                dim=self.dimension,
                label=self.label,
                label_tex=(
                    self._tex_label
                    if getattr(self, "_tex_label", None) is not None
                    else None
                ),
                mathfrak_label=True,
                unlabeled_tex=r"\text{Unnamed Algebra}",
                max_dim=20,
                raw=raw,
                abbrev=True,
                use_displaystyle=False,
                latex_wrapper=r"\langle {}\rangle",
            )

        return space_display(
            fmt="latex",
            basis_words=(lambda: [e._repr_latex_(raw=True) for e in self.basis]),
            dim=self.dimension,
            label=self.label,
            label_tex=(
                self._tex_label
                if getattr(self, "_tex_label", None) is not None
                else None
            ),
            mathfrak_label=True,
            unlabeled_tex=r"\text{Unnamed Algebra}",
            max_dim=20,
            raw=raw,
            abbrev=False,
            use_displaystyle=True,
            latex_wrapper=r"\langle {}\rangle",
        )

    def _latex(self, printer=None, raw=True, **kwargs):
        return self._repr_latex_(raw=raw)

    def is_subspace_subalgebra(
        self,
        elements,
        return_structure_data=False,
        check_linear_independence=False,
        surface_singularities=None,
    ):
        """
        Checks if a set of elements is a subspace is a subalgebra. `check_linear_independence` will additional verify if provided spanning elements are a basis.

        Parameters
        ----------
        elements : list
            A list of algebra_element_class instances.
        return_structure_data : bool, optional
            If True, returns the structure constants for the subalgebra

        check_linear_independence : bool, optional
            If True, a check of linear independence of basis elements is performed

        Returns
        -------
        dict or bool
        """

        return is_subspace_subalgebra(
            self,
            elements,
            return_structure_data=return_structure_data,
            check_linear_independence=check_linear_independence,
            surface_singularities=surface_singularities,
        )

    def check_grading_compatibility(
        self, verbose=False, test_weights=None, trust_test_weight_format=False
    ):
        """
        Checks if the algebra's structure constants are compatible with the assigned grading. Compatibility with alternate grading assignements can be tested by supplying alternate basis weights in the `test_weights` parameter

        Parameters
        ----------
        verbose : bool, optional (default=False)
            If True, prints detailed information about incompatibilities.
        test_weights : list of (lists of length self.dimension), optional
            elements in the inner lists represent weight values for basis elements to test compatibility against
        trust_test_weight_format : bool, optional (default=False)
            Set to True to forgo safeguard checks that test_weights is correctly formatted

        Returns
        -------
        bool
            True if the algebra is compatible with all assigned grading vectors or given test weights, False otherwise.

        Notes
        -----
        - The algebra's zero element (weights labeled as 'AllW') are treated as compatible with all grading vectors.
        - Non-homogeneous elements (weights labeled as 'NoW') are treated as incompatible.
        """
        default_check = False
        if test_weights is None:
            default_check = True
            test_weights = self.grading
        elif trust_test_weight_format is False:
            if not isinstance(test_weights, (list, tuple)) or not all(
                isinstance(j, (list, tuple)) and len(j) == self.dimension
                for j in test_weights
            ):
                raise TypeError(
                    "The `test_weights` parameter in `algebra_class.weighted_component` must be a list/tuple of lists/tuples that length matches `algebra_class.dimension` and whose elements are weight values representing weights elements in `algebra_class.basis`."
                )

        if default_check is True and not self._gradingNumber:
            raise ValueError(
                "No grading vectors are assigned to this algebra instance."
            ) from None
        if (
            default_check is True
            and isinstance(self._grading_compatible, bool)
            and self._grading_report
        ):
            compatible = self._grading_compatible
            failure_details = self._grading_report
        else:
            compatible = True
            failure_details = []

            for i, el1 in enumerate(self.basis):
                for j, el2 in enumerate(self.basis):
                    product = el1 * el2
                    product_weights = self.check_element_weight(
                        product,
                        test_weights=None if default_check else test_weights,
                    )

                    for g, grading_vector in enumerate(test_weights):
                        expected_weight = grading_vector[i] + grading_vector[j]

                        if product_weights[g] == "AllW":
                            continue  # Zero product is compatible with all weights

                        if (
                            product_weights[g] == "NoW"
                            or product_weights[g] != expected_weight
                        ):
                            compatible = False
                            failure_details.append(
                                {
                                    "grading_vector_index": g + 1,
                                    "basis_elements": (i + 1, j + 1),
                                    "weights": (grading_vector[i], grading_vector[j]),
                                    "expected_weight": expected_weight,
                                    "actual_weight": product_weights[g],
                                }
                            )
            if default_check is True:
                self._grading_compatible = compatible
                self._grading_report = failure_details

        if verbose:
            if not compatible:
                print("Grading Compatibility Check Failed:")
                for failure in failure_details:
                    print(
                        f"- Grading Vector {failure['grading_vector_index']}: "
                        f"Basis elements {failure['basis_elements'][0]} and {failure['basis_elements'][1]} "
                        f"(weights: {failure['weights'][0]}, {failure['weights'][1]}) "
                        f"produced weight {failure['actual_weight']}, expected {failure['expected_weight']}."
                    )
            else:
                if default_check:
                    ps = "all of its assigned Z-gradings."
                else:
                    ps = "the given weight system."
                print(f"The algebra structure of {self.label} is compatible with " + ps)
        return compatible

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
        """
        self._set_product_protocol()

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
                "The internal solver (determined by whichever symbolic engine set in defaults) returned no solutions, indicating that this computation of the center failed, as solutions do exist. An empty list is being returned."
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

    def get_structure_matrix(self, table_format=True, style=None):
        """
        Computes the structure equations matrix for the algebra.

        Parameters
        ----------
        table_format : bool, optional
            If True (default), returns a nicely formatted table
            If False, returns a raw list of lists
        style : str, optional
            dgcv theme name

        Returns
        -------
        list of lists
            structure equations matrix, whose (j, k)-entry is `basis[j] * basis[k]`.
        """
        if table_format is not True or style is not None:
            dgcv_warning(
                "`algebra_class.get_structure_matrix` ignores its `table_format` "
                "and `style` parameters and always returns a list of lists. Use "
                "`algebra_class.multiplication_table` for rendered output."
            )

        dimension = self.dimension
        structure_matrix = [
            [(self.basis[j] * self.basis[k]) for k in range(dimension)]
            for j in range(dimension)
        ]
        return structure_matrix

    def is_ideal(self, subspace_elements, assume_basis=False):
        """
        Checks if the given list of algebra elements spans an ideal.

        Parameters
        ----------
        subspace_elements : list
            A list of algebra_element_class instances representing the subspace
            they span.

        Returns
        -------
        bool
            True if the subspace is an ideal, False otherwise.
        """
        return is_ideal(self, subspace_elements, assume_basis=assume_basis)

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
        _called_from_subalgebra=None,
        **kwargs,
    ):
        return multiplication_table(
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
            _called_from_subalgebra=_called_from_subalgebra,
            **kwargs,
        )

    def subalgebra(
        self,
        basis,
        grading=None,
        span_warning=True,
        simplify_basis=False,
        simplify_products_by_default=None,
        surface_singularities=None,
        base_field=None,
        lazy_structure=False,
    ):
        return subalgebra(
            self,
            basis,
            grading=grading,
            span_warning=span_warning,
            simplify_basis=simplify_basis,
            simplify_products_by_default=simplify_products_by_default,
            surface_singularities=surface_singularities,
            base_field=base_field,
            lazy_structure=lazy_structure,
        )

    def new_alg_from_subalgebra(
        self,
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
        return new_alg_from_subalgebra(
            self,
            basis,
            grading=grading,
            span_warning=span_warning,
            simplify_basis=simplify_basis,
            label=label,
            basis_labels=basis_labels,
            register_in_vmf=register_in_vmf,
            initial_basis_index=initial_basis_index,
            simplify_products_by_default=simplify_products_by_default,
        )

    def killing_form_product(self, elem1, elem2, assume_Lie_algebra=False):
        if not self.contains(elem1, strict_types=True) or not self.contains(
            elem2, strict_types=True
        ):
            raise TypeError(
                "algebra_class.killing_form_product only operates on algebra elements from the dispatching algebra"
            )
        if get_dgcv_category(elem1) == "subalgebra_element":
            elem1 = elem1.ambient_rep
        if get_dgcv_category(elem2) == "subalgebra_element":
            elem2 = elem2.ambient_rep
        if self._killing_form is None:
            self._killing_form = freeze_matrix(
                killingForm(self, assume_Lie_algebra=assume_Lie_algebra)
            )
        kf = self._killing_form
        vec1 = matrix_dgcv(elem1.coeff_dict, shape=(self.dimension, 1))
        vec2 = matrix_dgcv(elem2.coeff_dict, shape=(1, self.dimension))
        return (vec2 * kf * vec1)[0]

    def subalgebra_from_grading_conditions(
        self,
        callable_bool_condition,
        sort_basis_by_grading_weights: bool = False,
        index_priority_for_lex_sort: list | tuple = [],
        reverse_sort_order: bool = False,
        label=None,
        basis_labels=None,
        register_in_vmf=False,
        initial_basis_index=None,
        simplify_products_by_default=None,
    ):

        weight_of = {}
        basis = []
        for elem in self.basis:
            weight = elem.check_element_weight()
            if callable_bool_condition(weight) is True:
                weight_of[id(elem)] = weight
                basis.append(elem)
        if sort_basis_by_grading_weights is True:
            grad_len = len(self.grading)
            idx_order = [j for j in index_priority_for_lex_sort if j < grad_len]
            if len(idx_order) < len(index_priority_for_lex_sort):
                dgcv_warning(
                    "Some indices provided in the `index_priority_for_lex_sort` parameter were out of range, and ignored."
                )
            if len(idx_order) == 0:
                idx_order = range(grad_len)
            basis = sorted(
                basis,
                key=lambda elem: [weight_of[id(elem)][idx] for idx in idx_order],
                reverse=reverse_sort_order,
            )
        grad = list(zip(*[weight_of[id(elem)] for elem in basis]))
        return self.new_alg_from_subalgebra(
            basis,
            grading=grad,
            span_warning=False,
            label=label,
            basis_labels=basis_labels,
            register_in_vmf=register_in_vmf,
            initial_basis_index=initial_basis_index,
            simplify_products_by_default=simplify_products_by_default,
        )

    def representation(
        self,
        rep_space=None,
        representation_basis=None,
        use_matrix_rep_instead_of_tensor=None,
    ):
        return _representation(
            self,
            rep_space=rep_space,
            representation_basis=representation_basis,
            use_matrix_rep_instead_of_tensor=use_matrix_rep_instead_of_tensor,
        )

    ###!!! candidate for vs methods mixin promotion
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
        return direct_sum(
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

    def copy(
        self,
        label: str = None,
        basis_labels: str | list[str] = None,
        register_in_vmf: bool = False,
        initial_basis_index: int = None,
        simplify_products_by_default: bool = None,
        parameter_sub_rules: dict = None,
    ) -> algebra_class:
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
        substituted = isinstance(parameter_sub_rules, dict) and bool(
            parameter_sub_rules
        )
        sd = (
            subs(self.structureData, parameter_sub_rules)
            if substituted
            else self.structureData
        )
        if substituted:
            educed = dict()
            params = set(get_free_symbols(sd))
        else:
            educed = dict(self._educed_properties)
            params = set(self._parameters)
        _markers = {
            "parameters": params,
            "_educed_properties": educed,
            "semidirect_decomposition": self.semidirect_decomposition,
            "tensor_decomposition": self.tensor_decomposition,
            "base_field": self.base_field,
        }
        if register_in_vmf is True:
            from .subspaces import createAlgebra

            return createAlgebra(
                sd,
                label,
                basis_labels=basis_labels,
                grading=grad,
                return_created_object=True,
                simplify_products_by_default=simplify_products_by_default,
                _markers=_markers,
            )
        return algebra_class(
            sd,
            grading=grad,
            simplify_products_by_default=simplify_products_by_default,
            _label=label,
            _basis_labels=basis_labels,
            _calledFromCreator=retrieve_passkey(),
            _markers=_markers,
        )

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
        tensor_product(
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

    def __matmul__(self, other):
        return self.tensor_product(other)

    def __rmatmul__(self, other):
        if _scalar_is_zero(other):
            return algebra_class({})
        if isinstance(other, expr_numeric_types()):
            return self._convert_to_tp().__rmatmul__(other)
        return NotImplemented

    def dual(self, invert_grad_weights=True):
        return algebra_dual(self, invert_grad_weights=invert_grad_weights)
