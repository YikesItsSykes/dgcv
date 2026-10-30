from __future__ import annotations

from ..._aux._backends._symbolic_router import _scalar_is_zero
from ..._aux._backends._types_and_constants import expr_numeric_types
from ..._aux._utilities._config import dgcv_warning
from ..._aux._utilities._misc import linear_combination
from ..._aux._vmf._safeguards import get_dgcv_category
from ..._aux.printing.printing._dgcv_display import show
from ...core.dgcv_core import wedge
from ...core.solvers import solve_dgcv
from .util import _extract_basis


class _vector_space_methods:
    """
    Shared basis and grading-level operations for `dgcv` vector-space-like
    classes.

    Notes
    -----
    Inheriting classes must provide `basis`, `dimension`, `grading`, `_gradingNumber`,
    `ambient`, `_class_builder` and `_verbose_subject`.
    """

    _basis_index_cache = None
    _graded_components = None

    @property
    def _basis_index(self):
        if self._basis_index_cache is None:
            index_map = dict()
            for idx, elem in enumerate(self.basis):
                if elem not in index_map:
                    index_map[elem] = idx
            self._basis_index_cache = index_map
        return self._basis_index_cache

    def __contains__(self, item):
        basis = self.basis
        if hasattr(basis, "materialize"):
            return item in basis
        return item in self._basis_index

    @property
    def zero_element(self):
        return self._class_builder({}, 1)

    def _materialize_full_basis(self):
        materialize = getattr(self.basis, "materialize", None)
        return self.basis if materialize is None else materialize()

    def _weight_coordinates(self, element):
        coeffs = self.contains(element, return_basis_coeffs=True)
        if coeffs is False:
            raise TypeError(
                f"Input to `check_element_weight` must belong to the {self._dgcv_category} instance whose `check_element_weight` is being called."
            ) from None
        return coeffs

    def check_element_weight(self, element, test_weights=None, flatten_weights=False):
        """
        Determines the weight vector of an element with respect to the grading vectors. Weight can instead be computed against another grading vector passed as a list of weights via the keyword `test_weights`.

        Parameters
        ----------
        element : (sub)algebra_element_class
            The element to analyze, given in the host's own basis coordinates.
        test_weights : list of scalars, optional (default: None)
        flatten_weights : bool, optional (default: False)
            If True, returns the contents of a length-1 list rather than the list.

        Returns
        -------
        tuple or weight value
            Weights corresponding to the grading vectors of this instance (or to `test_weights` if provided). Each entry is a scalar, the string 'AllW' if the element is zero, or 'NoW' if the element is not homogeneous.

        Notes
        -----
        - 'AllW' (All Weights) is returned for zero elements, which are compatible with all weights.
        - 'NoW' (No Weights) is returned for non-homogeneous elements.
        """
        if not test_weights and self._gradingNumber == 0:
            raise ValueError(
                f"This {self._dgcv_category} instance has no assigned grading vectors to test weighting w.r.t.."
            ) from None
        if element.is_zero:
            return tuple(["AllW"] * self._gradingNumber)
        if not test_weights and getattr(element, "_known_weight", None) is not None:
            if flatten_weights is True and len(element._known_weight) == 1:
                return element._known_weight[0]
            return element._known_weight
        if test_weights:
            numeric_types = expr_numeric_types()
            if not isinstance(test_weights, (list, tuple)):
                raise TypeError(
                    f"`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int, float, etc.). Received {test_weights}"
                ) from None
            for weight in test_weights:
                if not isinstance(weight, (list, tuple)):
                    raise TypeError(
                        f"`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int, float, etc.). Received {test_weights}"
                    ) from None
                if self.dimension != len(weight) or not all(
                    isinstance(j, numeric_types) for j in weight
                ):
                    raise TypeError(
                        f"`check_element_weight` expects `test_weights` to be None or a list/tuple of lists/tuples of weight values (int, float, etc.) of length {self.dimension}. Received {test_weights}"
                    ) from None
            GVs = test_weights
        else:
            GVs = self.grading
        coeff_indices = tuple(self._weight_coordinates(element))
        if not coeff_indices:
            weights = ["NoW"] * len(GVs)
        else:
            weights = []
            for grading_vector in GVs:
                first = grading_vector[coeff_indices[0]]
                homogeneous = True
                for i in coeff_indices:
                    if grading_vector[i] != first:
                        homogeneous = False
                        break
                weights.append(first if homogeneous else "NoW")
        if not test_weights:
            element._known_weight = tuple(weights)
        if flatten_weights and len(weights) == 1:
            return weights[0]
        return tuple(weights)

    def _verbose_subject(self):
        raise NotImplementedError

    def update_grading(self, new_weight_vectors_list, replace_instead_of_add=False):
        if isinstance(new_weight_vectors_list, (list, tuple)):
            if all(isinstance(elem, (list, tuple)) for elem in new_weight_vectors_list):
                if replace_instead_of_add is True:
                    self.grading = [tuple(elem) for elem in new_weight_vectors_list]
                else:
                    grad = list(self.grading) + [
                        tuple(elem) for elem in new_weight_vectors_list
                    ]
                    self.grading = grad
            else:
                raise TypeError(
                    f"update_grading expects first parameter to be a list of lists. The inner lists should have length {self.dimension}"
                )
        else:
            raise TypeError(
                f"update_grading expects first parameter to be a list of lists. The inner lists should have length {self.dimension}"
            )

    def compute_weight(self, element, test_weights=None, flatten_weights=False):
        return self.check_element_weight(
            element, test_weights=test_weights, flatten_weights=flatten_weights
        )

    def weighted_component(
        self,
        weights,
        test_weights=None,
        trust_test_weight_format=False,
    ):
        numeric_types = expr_numeric_types()
        if isinstance(weights, (set, dict)):
            weights = list(weights)
        if isinstance(weights, (list, tuple)):
            if all(isinstance(weight, numeric_types) for weight in weights):
                weights = [(weight,) for weight in weights]
            elif not all(isinstance(weight, (list, tuple)) for weight in weights):
                raise ValueError(
                    "The `weights` parameter in `algebra_class.weighted_component` must be a list/tuple of weights/multi-weights. If giving a single multi-weight, it should be a length-1 list/tuple of lists/tuples, as otherwise a bare mult-weight tuple will be interpreted as a list of singleton weights."
                ) from None
            else:
                weights = [tuple(weight) for weight in weights]
        else:
            raise ValueError(
                f"The `weights` parameter in `algebra_class.weighted_component` must be a list/tuple of weights/multi-weights. If giving a single multi-weight, it should be a length-1 list/tuple of lists/tuples, as otherwise a bare mult-weight tuple will be interpreted as a list of singleton weights. Instead received{weights}"
            ) from None
        try:
            weight_lookup = set(weights)
        except TypeError:  # In case weights list has un-hashable lists within it
            weight_lookup = weights
        if test_weights is None:
            test_weights = self.grading
        elif trust_test_weight_format is False:
            if not isinstance(test_weights, (list, tuple)) or not all(
                isinstance(j, (list, tuple)) and len(j) == self.dimension
                for j in test_weights
            ):
                raise TypeError(
                    "The `test_weights` parameter in `algebra_class.weighted_component` must be a list/tuple of lists/tuples that length matches `algebra_class.dimension` and whose elements are weight values representing weights elements in `algebra_class.basis`."
                )
        component = []
        for idx, elem in enumerate(self.basis):
            elem_weight = tuple(vector[idx] for vector in test_weights)
            if elem_weight in weight_lookup or all(w == "AllW" for w in elem_weight):
                component.append(elem)
        from ..subspaces import algebra_subspace_class

        return algebra_subspace_class(component, parent_algebra=self)

    @property
    def graded_components(self):
        if self._graded_components is None:
            basis = self.basis
            buckets = dict()
            for idx, weight in enumerate(zip(*self.grading)):
                key = tuple(weight)
                elem = basis[idx]
                elem._known_weight = key
                bucket = buckets.get(key)
                if bucket is None:
                    buckets[key] = [elem]
                else:
                    bucket.append(elem)
            try:
                keys = sorted(buckets)
            except TypeError:
                keys = sorted(buckets, key=lambda k: tuple(str(w) for w in k))
            from ..subspaces import algebra_subspace_class

            self._graded_components = {
                key: algebra_subspace_class(buckets[key], parent_algebra=self)
                for key in keys
            }
        return self._graded_components

    def compute_graded_component_wrt_weight_index(self, idx=0):
        if idx not in range(len(self.grading)):
            dgcv_warning(
                "The provided index is out of range. `compute_graded_component_wrt_weight_index` is using 0 instead."
            )
            idx = 0
        wc = dict()
        for idxs, comp in self.graded_components.items():
            key = idxs[idx]
            if key in wc:
                wc[key] = wc[key] + comp
            else:
                wc[key] = self.subspace([]) + comp
        return wc

    def grading_summary(self):

        gradingNumber = len(self.grading)
        graded_components = self.graded_components
        pref = self._repr_latex_(abbrev=True).replace("$", "")
        if "_" in pref:
            prefi = f"\\left({pref} \\right)_"
        else:
            prefi = f"{pref}_"
        strings = []
        for k, v in graded_components.items():
            inner = "".join(str(j) for j in k)
            latex = v._repr_latex_().replace("$", "").replace(r"\displaystyle", "")
            strings.append(f"$$ {prefi}{{{inner}}} = {latex},$$")
        if gradingNumber == 0 or not strings:
            show(f"The algebra ${pref}$ has no assigned grading.")
        else:
            if len(strings) > 1:
                strings.insert(-1, "and")
            strings[-1] = strings[-1][:-3] + ".$$"
            if gradingNumber == 1:
                gradPhrase = "graded"
            elif gradingNumber == 2:
                gradPhrase = "bi-graded"
            elif gradingNumber == 3:
                gradPhrase = "tri-graded"
            else:
                gradPhrase = f"{gradingNumber}-graded"
            show(
                f"The algebra ${pref}$ has {gradPhrase} components: {' '.join(strings)}"
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
        elements : list of algebra_element_class
            The set of elements to filter.

        Returns
        -------
        list of algebra_element_class
            A subset of the input elements that are linearly independent and unique.
        """
        warning_message = ""
        remain_subalg = True
        subalg = None
        if not isinstance(elements, (list, tuple)):
            warning_message += (
                "\n The given value for `elements` is not a list or tuple"
            )
        else:
            nonAE, wrongAlgebra, correct = [], [], []
            typeCheck = {"algebra_element", "subalgebra_element"}
            for elem in elements:
                if elem == 0:
                    continue
                if remain_subalg is True:
                    if get_dgcv_category(elem) == "algebra_element":
                        remain_subalg = False
                    elif get_dgcv_category(elem) == "subalgebra_element":
                        if subalg is None:
                            subalg = elem.algebra
                        elif subalg != elem.algebra:
                            remain_subalg = False
                if get_dgcv_category(elem) not in typeCheck:
                    nonAE.append(elem)
                elif (
                    get_dgcv_category(elem) == "algebra_element"
                    and elem.algebra != self
                ) or (
                    get_dgcv_category(elem) == "subalgebra_element"
                    and elem.algebra.ambient != self
                ):
                    wrongAlgebra.append(elem)
                else:
                    correct.append(elem)
            elements = correct
            if len(nonAE) > 0:
                warning_message += f"\n • These list elements are not `algebra_element` or `subalgebra_element` type: {nonAE}"
            if len(wrongAlgebra) > 0:
                warning_message += f"\n • These list elements are `algebra_element` or `subalgebra_element` type, but belong to a different, unrelated algebra: {wrongAlgebra}"
        if warning_message:
            raise ValueError(
                "The `algebra` method `filter_independent_elements` can only be applied to lists of elements belong to the parent algebra the method is called from or any its subalgebras. Given data has the following problems:"
                + warning_message
            ) from None

        if remain_subalg is False:
            elements = [
                (
                    elem.ambient_rep
                    if get_dgcv_category(elem) == "subalgebra_element"
                    else elem
                )
                for elem in elements
            ]
        else:
            elements = list(elements)

        if return_indices is True:
            out = _extract_basis(
                elements,
                ALBS=apply_light_basis_simplification,
                return_indices=True,
                surface_singularities=surface_singularities,
                simplify_singularities=simplify_singularities,
                force_heavy_solve=force_heavy_solve,
            )
            if surface_singularities:
                _, idxs, sing = out
            else:
                _, idxs = out
            return (idxs, sing) if surface_singularities else idxs
        return _extract_basis(
            elements,
            ALBS=apply_light_basis_simplification,
            surface_singularities=surface_singularities,
            simplify_singularities=simplify_singularities,
            force_heavy_solve=force_heavy_solve,
        )

    def is_in_span(self, element, subspace_elements, assume_basis=False):
        """
        Checks if a given algebra_element_class is in the span of subspace_elements.

        Parameters
        ----------
        element : algebra_element_class
            The element to check.
        subspace_elements : list
            A list of algebra_element_class instances representing the subspace they span.
        assume_basis : bool, optional
            If True, returns the wedge product rather than a bool. Note a *zero*
            wedge indicates the element is in the span.

        Returns
        -------
        bool
            True if the element is in the span of subspace_elements, False otherwise.
        """
        if (
            not isinstance(subspace_elements, (list, tuple))
            or len(subspace_elements) == 0
        ):
            return _scalar_is_zero(element)
        if assume_basis:
            return wedge(element, *subspace_elements)
        combo, variables = linear_combination(subspace_elements, _disposable=True)
        diff = element - combo
        eqns = list(diff.coeff_dict.values())
        sol2 = solve_dgcv(eqns, variables, method="linsolve", simplify_result=False)
        return bool(sol2)

    def subspace(self, basis: list | tuple = [], grading=None, span_warning=True):
        if grading is None:
            grading = self.grading
        from ..subspaces import algebra_subspace_class

        return algebra_subspace_class(
            basis, parent_algebra=self, test_weights=grading, span_warning=span_warning
        )

    def contains(self, items, return_basis_coeffs=False, strict_types=False):
        if isinstance(items, (list, tuple)):
            return [
                self.contains(item, return_basis_coeffs=return_basis_coeffs)
                for item in items
            ]

        if strict_types is False and items == 0:
            if return_basis_coeffs is True:
                return {}
            return True
        if strict_types is False and get_dgcv_category(items) == "tensorProduct":
            if next(iter(items._vs_spring)) == self.card and len(items.vs_id) == 1:
                k, v = next(iter(items.coeff_dict.items()))
                if len(k) == 1:
                    ne = v * (k[0][2].space.basis[k[0][0]])
                    if _scalar_is_zero(k[0][1]):
                        ne = ne.dual()
                    return self.contains(ne)
        if get_dgcv_category(items) == "subalgebra_element":
            items = items.ambient_rep
        if get_dgcv_category(items) == "algebra_element" and items.card == self.card:
            if return_basis_coeffs:
                return dict(items.coeff_dict)
            else:
                return True
        return False
