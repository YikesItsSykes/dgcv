from __future__ import annotations

from ..._aux._backends._symbolic_router import _scalar_is_zero
from ..._aux._backends._types_and_constants import symbol
from ..._aux._utilities._config import dgcv_warning
from ..._aux._utilities._misc import linear_combination
from ..._aux._vmf._safeguards import get_dgcv_category, retrieve_passkey
from ...core.arrays import matrix_dgcv
from ...core.base import dgcv_class
from ...core.solvers import solve_dgcv
from ..linear_algebra import _structure_array
from ..threads import _basis_builder, _indep_check, _vector_space_methods


class algebra_subspace_class(_vector_space_methods, dgcv_class):
    def __init__(
        self,
        basis,
        parent_algebra=None,
        test_weights=None,
        _grading=None,
        _internal_lock=None,
        span_warning=True,
        simplify_basis=False,
        **kwargs,
    ):
        # From former __new__: validate inputs and compute subspace attributes
        if not isinstance(basis, (list, tuple)):
            raise TypeError(
                "algebra_subspace_class expects first argument to a be a list or tuple of algebra_element_class instances"
            ) from None
        typeCheck = {"subalgebra_element", "algebra_element"}
        if not all(get_dgcv_category(j) in typeCheck or j == 0 for j in basis):
            raise TypeError(
                "algebra_subspace_class expects first argument to a be a list or tuple of algebra_element_class instances"
            ) from None
        if parent_algebra is None:
            if len(basis) > 0:
                if get_dgcv_category(basis[0].algebra) != "algebra":
                    if all(j.algebra == basis[0].algebra for j in basis[1:]):
                        parent_alg = basis[0].algebra.ambient
                    else:
                        parent_alg = basis[0].algebra.ambient
                else:
                    parent_alg = basis[0].algebra
            else:
                parent_alg = None
        elif get_dgcv_category(parent_algebra) in {"subalgebra", "algebra_subspace"}:
            parent_alg = parent_algebra.ambient
        elif get_dgcv_category(parent_algebra) == "algebra":
            parent_alg = parent_algebra
        else:
            raise TypeError(
                "algebra_subspace_class expects second argument to be an algebra instance or algebra subspace or subalgebra."
            ) from None

        filtered_basis = parent_alg.filter_independent_elements(
            basis,
            apply_light_basis_simplification=simplify_basis,
            surface_singularities=False,
        )
        if len(filtered_basis) < len(basis):
            basis = filtered_basis
            if span_warning:
                wmessage = (
                    " This can result in incorrect weighting assignements from the manual assignment provided to the `test_weights` parameter. To avoid this issue, provided a linearly independent spanning set instead."
                    if test_weights is None
                    else ""
                )
                dgcv_warning(
                    "The given list for `basis` was not linearly independent, so the algebra_subspace_class initializer computed a basis for its span to use instead."
                    + wmessage
                )

        self.filtered_basis = tuple(filtered_basis)
        self.basis = tuple(filtered_basis)
        self.dimension = len(filtered_basis)
        self.ambient = parent_alg
        grading_per_elem = []
        if (
            _internal_lock == retrieve_passkey()
            and test_weights is None
            and _grading is not None
        ):
            self.grading = _grading
        else:
            for elem in filtered_basis:
                weight = parent_alg.check_element_weight(
                    elem, test_weights=test_weights
                )
                grading_per_elem.append(weight)
            self.grading = [
                elem for elem in zip(*grading_per_elem) if "NoW" not in elem
            ]
        self.original_basis = basis
        self._parameters = self.ambient._parameters
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "algebra_subspace"

        # immutables
        self._grading = tuple(self.grading)
        self._gradingNumber = len(self._grading)
        self._singularities = {}

        self.card = getattr(parent_alg, "card", None)

        # attribute caches
        self._endomorphisms = None
        self._is_subalgebra = None

    def _class_builder(self, coeff_dict, valence):
        return self.ambient._class_builder(coeff_dict, valence)

    def contains(self, items, return_basis_coeffs=False, strict_types=False):
        if isinstance(items, (list, tuple)):
            return [
                self.contains(
                    item,
                    return_basis_coeffs=return_basis_coeffs,
                    strict_types=strict_types,
                )
                for item in items
            ]
        if strict_types is False and items == 0:
            if return_basis_coeffs is True:
                return {}
            return True
        item = items
        if get_dgcv_category(item) == "subalgebra_element":
            if item.algebra == self:
                if return_basis_coeffs is True:
                    return dict(item.coeff_dict)
                else:
                    return True
            item = item.ambient_rep
        if (
            not get_dgcv_category(item) == "algebra_element"
            or item.algebra != self.ambient
        ):
            return False

        pos = self._basis_index
        try:  ###!!! update to optimize try failure path
            found = pos.get(item)
        except TypeError:
            found = None
        if found is None:
            if self.dimension == 0:
                return False
            genElement, variables = linear_combination(self.basis, _disposable=True)
            sol = solve_dgcv(
                item - genElement, variables, method="linsolve", simplify_result=False
            )
            if len(sol) == 0:
                return False
        else:
            if return_basis_coeffs is True:
                return {found: 1}
        if return_basis_coeffs is True:
            s = sol[0]
            out = dict()
            for c, var in enumerate(variables):
                coeff = s.get(var, 0)
                if not _scalar_is_zero(coeff):
                    out[c] = coeff
            return out
        return True

    def __iter__(self):
        return iter(self.basis)

    def __getitem__(self, index):
        return self.basis[index]

    def is_subalgebra(self, return_structure_data=False, surface_singularities=None):
        """
        Test whether the subspace is closed under the ambient product.

        Parameters
        ----------
        return_structure_data : bool, default False
            Return the full report from `is_subspace_subalgebra` instead of a bool.
        surface_singularities : bool, optional
            Forwarded to `is_subspace_subalgebra`; defaults to True when the
            subspace carries free parameters.

        Returns
        -------
        bool or dict
            The closure verdict, or the full report when `return_structure_data`
            is True.
        """
        if surface_singularities is None and self._parameters:
            surface_singularities = True
        cached = self._is_subalgebra
        if isinstance(cached, dict):
            if return_structure_data is True:
                return cached
            return cached["closed_under_product"]
        if cached is not None and return_structure_data is not True:
            return cached
        out = self.ambient.is_subspace_subalgebra(
            self.filtered_basis,
            return_structure_data=return_structure_data,
            surface_singularities=surface_singularities,
        )
        if surface_singularities:
            out, _ = out
        self._is_subalgebra = out
        return out

    def __str__(self):
        b = getattr(self, "basis", None) or []
        return "span{" + ", ".join(str(e) for e in b) + "}"

    def _repr_latex_(self, raw: bool = False, **kwargs):
        """verbose=True keyword will override elision formatting in output."""
        b = getattr(self, "basis", None) or []
        if len(b) > 6 and not kwargs.pop("verbose", False):
            inner = [
                b[idx]._repr_latex_(raw=True)
                if idx < 2
                else b[-1]._repr_latex_(raw=True)
                if idx > 2
                else r"\ldots"
                for idx in range(4)
            ]
            inner = ", ".join(inner)
        else:
            inner = ", ".join(e._repr_latex_(raw=True) for e in b)
        inner = str(inner).replace("$", "").replace(r"\displaystyle", "")
        out = rf"\left\langle {inner} \right\rangle"
        return out if raw else rf"$\displaystyle {out}$"

    def _latex(self, printer=None, raw: bool = True, **kwargs):
        return self._repr_latex_(raw=raw)

    def __add__(self, other):
        if _scalar_is_zero(other):
            return self
        if get_dgcv_category(other) in {"algebra_subspace", "subalgebra"}:
            if other.dimension == 0:
                return self
            if self.dimension == 0:
                return other
            new_basis = list(self.basis)
            for elem in getattr(other, "basis_in_ambient_alg", other.basis):
                new_basis = _basis_builder(
                    new_basis, elem
                )  ###!!! optimize with wedge method
            return algebra_subspace_class(new_basis, self.ambient)
        return NotImplemented

    def __radd__(self, other):
        if _scalar_is_zero(other):
            return self
        return NotImplemented

    def generate_subalgebra(
        self, simplify_basis=False, simplify_products_by_default=None
    ):
        if get_dgcv_category(self) == "subalgebra":
            return self
        basis = list(self.basis)
        in_dim = len(basis)
        amb = self.ambient
        amb_dim = amb.dimension
        skew = amb.is_skew_symmetric()
        variables = [symbol(f"_indep_check_{idx}") for idx in range(amb_dim - 1)]
        sd_out = dict()
        previous_basis = []
        for _ in range(in_dim):  # it should never reach this bound
            prev_dim = len(previous_basis)
            previous_basis = list(basis)
            current_dim = len(previous_basis)
            if prev_dim == current_dim:
                break

            for idx1, e1 in enumerate(previous_basis):
                if idx1 < prev_dim:
                    start = prev_dim
                else:
                    start = idx1 + 1 if skew else 0
                for idx2 in range(start, current_dim):
                    e2 = previous_basis[idx2]
                    product = e1 * e2
                    result, cd = _indep_check(
                        basis,
                        product,
                        return_decomp_coeffs=True,
                        _solve_variables=variables,
                    )
                    if result is True:
                        out_int = len(basis)
                        sd_out[(idx1, idx2)] = {out_int: 1}
                        if skew:
                            sd_out[(idx2, idx1)] = {out_int: -1}
                        basis.append(product)
                    else:
                        out_coeffs = cd[0]
                        sd_out[(idx1, idx2)] = out_coeffs
                        if skew:
                            sd_out[(idx2, idx1)] = {
                                i: -coef for i, coef in out_coeffs.items()
                            }
        dim = len(basis)
        sd_out = _structure_array(
            {k: matrix_dgcv(v, shape=(dim, 1)) for k, v in sd_out.items()}, dim
        )
        from .subalgebras import subalgebra_class

        return subalgebra_class(
            basis,
            amb,
            _compressed_structure_data=sd_out,
            _internal_lock=retrieve_passkey(),
            span_warning=False,
            simplify_basis=simplify_basis,
            simplify_products_by_default=simplify_products_by_default,
        )

    @property
    def endomorphism_algebra(self):
        from ..composition.composites import vector_space_endomorphisms

        if self._endomorphisms is None:
            self._endomorphisms = vector_space_endomorphisms(self)
        return self._endomorphisms

    def append(self, item, recompute_gradings_and_return_new=False):
        if _scalar_is_zero(item):
            pass
        elif recompute_gradings_and_return_new:
            bas = list(self.filtered_basis)
            bas.append(item)
            return self.ambient.subspace(bas)
        elif not self.contains(item):
            self.original_basis = list(self.original_basis) + [item]
            self.basis = tuple(self.basis) + (item,)
            self.filtered_basis = tuple(self.filtered_basis) + (item,)
            self.dimension += 1
            self._basis_index_cache = None
            self.grading = [(0,) * self.dimension]
