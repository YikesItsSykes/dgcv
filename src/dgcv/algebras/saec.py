from __future__ import annotations

import numbers

from .._aux._backends._symbolic_router import (
    _scalar_is_zero,
    get_free_symbols,
    ratio,
    simplify,
    subs,
)
from .._aux._backends._types_and_constants import expr_numeric_types, expr_types
from .._aux._utilities._misc import zip_sum
from .._aux._vmf._safeguards import get_dgcv_category, retrieve_passkey
from ..core.base import dgcv_class
from ..core.tensors import tensorProduct


class subalgebra_element(dgcv_class):
    def __init__(self, alg, coeff_dict, valence, ambient_rep=None, _internalLock=None):
        self.algebra = alg
        self.vectorSpace = alg
        if valence not in (0, 1):
            raise ValueError(f"valence must be 0 or 1, got {valence!r}")
        self.valence = valence
        if isinstance(coeff_dict, dict):
            coeff_dict = {k: v for k, v in coeff_dict.items() if not _scalar_is_zero(v)}
        elif isinstance(coeff_dict, (list, tuple)):
            coeff_dict = {
                k: v for k, v in enumerate(coeff_dict) if not _scalar_is_zero(v)
            }
        elif get_dgcv_category(coeff_dict) == "array":
            coeff_dict = coeff_dict._data
        else:
            raise (
                "subalgebra_element recieved unsupports coeffs parameter format."
            ) from None
        self.coeff_dict = coeff_dict
        self._coeffs = None  # deprecated
        self._coeffs_hash_cache = None
        if _internalLock == retrieve_passkey():
            self._ambient_rep = ambient_rep
        else:
            self._ambient_rep = None
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "subalgebra_element"
        self.card = self.vectorSpace.card
        self._known_weight = None

    @property
    def coeffs(self):
        if self._coeffs is None:
            self._coeffs = tuple(
                self.coeff_dict.get(x, 0) for x in range(self.algebra.dimension)
            )
        return self._coeffs

    @property
    def _coeffs_hash(self):
        if self._coeffs_hash_cache is None:
            self._coeffs_hash_cache = frozenset(self.coeff_dict.items())
        return self._coeffs_hash_cache

    @property
    def ambient_rep(self):
        if self._ambient_rep is None:
            cd = self.coeff_dict
            if len(cd) == 0:
                self._ambient_rep = self.algebra.ambient.zero_element
            else:
                amb_basis = self.algebra.basis_in_ambient_alg
                self._ambient_rep = zip_sum(
                    list(cd.values()), [amb_basis[j] for j in cd]
                )
        return self._ambient_rep

    def __eq__(self, other):
        if not isinstance(other, subalgebra_element):
            return NotImplemented
        return (
            self.algebra == other.algebra
            and self._coeffs_hash == other._coeffs_hash
            and self.valence == other.valence
        )

    def __hash__(self):
        return hash((self.algebra, self._coeffs_hash, self.valence))

    def __str__(self):
        return self.ambient_rep.__str__()

    def _repr_latex_(self, verbose=False, raw=False, **kwargs):
        return self.ambient_rep._repr_latex_(verbose=verbose, raw=raw)

    def _latex(self, printer=None, raw=True, **kwargs):
        return self._repr_latex_(raw=raw)

    def _latex_verbose(self, printer=None):
        return self.ambient_rep._latex_verbose(printer=printer)

    @property
    def label(self):
        return self.__repr__()

    @property
    def is_zero(self):
        for j in self.coeff_dict.values():
            if not _scalar_is_zero(simplify(j)):
                return False
        return True

    @property
    def is_literal_zero(self):
        for j in self.coeff_dict.values():
            if not _scalar_is_zero(j):
                return False
        return True

    @property
    def __dgcv_zero_obstr__(self):
        cfs = []
        cfvars = set()
        for cf in self.coeff_dict.values():
            cfs.append(cf)
            cfvars |= get_free_symbols(cf)
        return cfs, cfvars

    def _si_wrap(self, obj):
        if self.algebra.simplify_products_by_default is True:
            return simplify(obj)
        else:
            return obj

    def __dgcv_simplify__(self, *args, **kwargs):
        newCoeffs = {k: simplify(v, **kwargs) for k, v in self.coeff_dict.items()}
        return subalgebra_element(self.algebra, newCoeffs, self.valence)

    def _eval_simplify(self, *args, **kwargs):
        newCoeffs = {k: simplify(v, **kwargs) for k, v in self.coeff_dict.items()}
        return subalgebra_element(self.algebra, newCoeffs, self.valence)

    def subs(self, subsData):
        newCoeffs = {k: subs(v, subsData) for k, v in self.coeff_dict.items()}
        return subalgebra_element(self.algebra, newCoeffs, self.valence)

    def dual(self):
        return subalgebra_element(self.algebra, self.coeff_dict, (self.valence + 1) % 2)

    def _convert_to_tp(self):
        return tensorProduct(
            {
                ((idx, self.valence, self.card),): c
                for idx, c in self.coeff_dict.items()
            },
            shape="all",
        )

    def _recursion_contract_hom(self, other):
        return self._convert_to_tp()._recursion_contract_hom(other)

    def _fast_add(self, other):
        """
        Internal-only: assumes `other` is a subalgebra_element class
        with the same algebra and valence. No type or safety checks etc.
        """
        new_dict = dict(self.coeff_dict)
        for k, v in other.coeff_dict.items():
            new_dict[k] = new_dict.get(k, 0) + v
        return subalgebra_element(
            self.algebra,
            new_dict,
            self.valence,
        )

    @classmethod
    def _dgcv_multiadd(cls, terms, start=0):
        if not isinstance(terms, (list, tuple)):
            terms = list(terms)
        if not terms:
            return start
        acc = {}
        alg = None
        valence = None
        residual = []
        if isinstance(start, cls):
            acc.update(start.coeff_dict)
            alg = start.algebra
            valence = start.valence
        elif not _scalar_is_zero(start):
            residual.append(start)
        for t in terms:
            if isinstance(t, cls):
                if alg is None:
                    alg = t.algebra
                    valence = t.valence
                if t.algebra == alg and t.valence == valence:
                    for k, v in t.coeff_dict.items():
                        acc[k] = acc.get(k, 0) + v
                    continue
            residual.append(t)
        if alg is None:
            return sum(terms, start)
        out = cls(alg, acc, valence)
        if residual:
            return sum(residual, out)
        return out

    @classmethod
    def _dgcv_multiadd_scaled(cls, pairs, start=0):
        if not isinstance(pairs, (list, tuple)):
            pairs = list(pairs)
        if not pairs:
            return start
        acc = {}
        alg = None
        valence = None
        spbd = False
        residual = []
        if isinstance(start, cls):
            acc.update(start.coeff_dict)
            alg = start.algebra
            valence = start.valence
            spbd = alg.simplify_products_by_default is True
        elif not _scalar_is_zero(start):
            residual.append(start)
        for c, t in pairs:
            if isinstance(t, cls):
                if alg is None:
                    alg = t.algebra
                    valence = t.valence
                    spbd = alg.simplify_products_by_default is True
                if t.algebra == alg and t.valence == valence:
                    if not _scalar_is_zero(c):
                        for k, v in t.coeff_dict.items():
                            acc[k] = acc.get(k, 0) + (
                                simplify(c * v) if spbd else c * v
                            )
                    continue
            residual.append(c * t)
        if alg is None:
            return sum([c * t for c, t in pairs], start)
        out = cls(alg, acc, valence)
        if residual:
            return sum(residual, out)
        return out

    def __add__(self, other):
        if _scalar_is_zero(other):
            return self
        if get_dgcv_category(other) == "subalgebra_element":
            if self.algebra == other.algebra and self.valence == other.valence:
                new_dict = dict(self.coeff_dict)
                for k, v in other.coeff_dict.items():
                    new_dict[k] = new_dict.get(k, 0) + v
                return subalgebra_element(self.algebra, new_dict, self.valence)
            elif other.algebra.ambient == self.algebra.ambient:
                return self.ambient_rep + other.ambient_rep
            else:
                other = other._convert_to_tp()
        if get_dgcv_category(other) in {
            "algebra_element",
            "tensorProduct",
        } or isinstance(other, expr_numeric_types()):
            if self.algebra.ambient == getattr(other, "algebra", None):
                return self.ambient_rep + other
            return self._convert_to_tp() + other
        if get_dgcv_category(other) == "fastTensorProduct":
            return other + self
        return self.ambient_rep.__add__(other)

    def __radd__(self, other):
        if _scalar_is_zero(other):
            return self
        if isinstance(other, expr_numeric_types()):
            return self._convert_to_tp().__radd__(other)
        return NotImplemented

    def __sub__(self, other):
        return (self).__add__(-other)

    def __rsub__(self, other):
        return (-self).__radd__(other)

    def __mul__(self, other):
        if get_dgcv_category(other) == "subalgebra_element":
            if self.algebra == other.algebra and self.valence == other.valence:
                sign = 1 if self.valence == 1 else -1
                alg = self.algebra
                spbd = self.algebra.simplify_products_by_default
                new_coeffs = dict()
                for idx1, c1 in self.coeff_dict.items():
                    for idx2, c2 in other.coeff_dict.items():
                        scalar = sign * c1 * c2
                        row = alg._structure_row(idx1, idx2)
                        for idx3, c3 in row._data.items():
                            new_coeffs[idx3] = new_coeffs.get(idx3, 0) + (
                                self._si_wrap(scalar * c3) if spbd else scalar * c3
                            )
                return subalgebra_element(self.algebra, new_coeffs, self.valence)
            elif other.algebra.ambient == self.algebra.ambient:
                return self.ambient_rep * other.ambient_rep
            else:
                return self._convert_to_tp().__mul__(other)
        elif isinstance(other, expr_numeric_types()):
            new_coeffs = {
                idx: self._si_wrap(coeff * other)
                for idx, coeff in self.coeff_dict.items()
            }
            return subalgebra_element(self.algebra, new_coeffs, self.valence)
        elif get_dgcv_category(other) == "algebra_element":
            return self.ambient_rep * other
        elif get_dgcv_category(other) == "tensorProduct":
            return self._convert_to_tp().__mul__(other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, expr_numeric_types()):
            return self * other
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return self._si_wrap(ratio(1, other) * self)
        elif isinstance(other, expr_types()):
            return self._si_wrap((1 / other) * self)
        else:
            raise TypeError(
                f"True division `/` of subalgebra elements by another object is only supported for scalars, not {type(other)}"
            ) from None

    def __matmul__(self, other):
        """Overload @ operator for tensor product."""
        if get_dgcv_category(other) == "tensorProduct":
            return self._convert_to_tp() @ other
        if isinstance(other, expr_numeric_types()):
            return other * self
        if get_dgcv_category(other) not in {
            "algebra_element",
            "subalgebra_element",
        }:
            return self._convert_to_tp().__matmul__(other)
        new_dict = {
            (
                (idx1, self.valence, self.card),
                (idx2, other.valence, other.card),
            ): c1 * c2
            for idx1, c1 in self.coeff_dict.items()
            for idx2, c2 in other.coeff_dict.items()
        }
        return self._si_wrap(tensorProduct(new_dict))

    def __rmatmul__(self, other):
        if isinstance(other, expr_numeric_types()):
            return other * self
        return self._convert_to_tp().__rmatmul__(other)

    def __xor__(self, other):
        if other == "":
            return self.dual()
        raise ValueError("Invalid operation. Use `^''` to denote the dual.") from None

    def __neg__(self):
        return -1 * self

    def __call__(self, other, **kwds):
        if (
            get_dgcv_category(other) == "subalgebra_element"
            and other.algebra == self.algebra
        ):
            cd = other.coeff_dict
            return sum(c * cd.get(idx, 0) for idx, c in self.coeff_dict.items())
        else:
            return self.ambient_rep(other)

    def compute_weight(self, test_weights=None, flatten_weights=False):
        return self.check_element_weight(
            test_weights=test_weights, flatten_weights=flatten_weights
        )

    def check_element_weight(self, test_weights=None, flatten_weights=False):
        """
        Determines the weight vector of this subalgebra_element with respect to its ambient algebra's grading vectors.

        Returns
        -------
        list
            A list of weights corresponding to the grading vectors of the parent algebra.
            Each entry is either an integer, sympy.Expr (weight), the string 'AllW' if the element is the zero element,
            or 'NoW' if the element is not homogeneous.

        Notes
        -----
        - This method calls the parent algebra' check_element_weight method.
        - 'AllW' is returned for zero elements, which are compaible with all weights.
        - 'NoW' is returned for non-homogeneous elements that do not satisfy the grading constraints.
        """
        return self.algebra.check_element_weight(
            self, test_weights=test_weights, flatten_weights=flatten_weights
        )

    def weighted_decomposition(self, test_weights=None, flatten_weights=False):
        weighted_components = {}
        for idx, coeff in self.coeff_dict.items():
            if not _scalar_is_zero(coeff):
                elem = self.algebra.basis[idx]
                w = elem.check_element_weight(
                    test_weights=test_weights, flatten_weights=flatten_weights
                )
                if isinstance(w, list):
                    w = tuple(w)
                weighted_components[w] = weighted_components.get(w, 0) + coeff * elem
        return weighted_components

    def terms(self):
        return [c * self.algebra.basis[idx] for idx, c in self.coeff_dict.items()]

    def dual_pairing(self, other):
        return self._convert_to_tp().dual_pairing(other)
