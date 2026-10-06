from __future__ import annotations

import numbers

from .._aux._backends._display import latex
from .._aux._backends._symbolic_router import (
    _scalar_is_zero,
    get_free_symbols,
    is_zero_knowing_zero_is_expected,
    ratio,
    simplify,
    subs,
)
from .._aux._backends._types_and_constants import expr_numeric_types
from .._aux._utilities._config import dgcv_warning, get_dgcv_settings_registry
from .._aux._vmf._safeguards import get_dgcv_category, retrieve_passkey
from .._aux.printing.printing import lincomb_latex, lincomb_plain
from ..core.base import dgcv_class
from ..core.tensors import tensorProduct


class algebra_element_class(dgcv_class):
    def __init__(self, alg, coeff_dict, valence, format_sparse=False):
        if not get_dgcv_category(alg) == "algebra":
            raise TypeError(
                "`algebra_element_class` expects the first argument to be an instance of the `algebra` class."
            ) from None
        if valence not in {0, 1}:
            raise TypeError(
                "algebra_element_class expects third argument to be 0 or 1."
            ) from None
        if isinstance(coeff_dict, dict):
            coeff_dict = {k: v for k, v in coeff_dict.items() if not _scalar_is_zero(v)}
        elif isinstance(coeff_dict, (list, tuple)):
            coeff_dict = {
                k: v for k, v in enumerate(coeff_dict) if not _scalar_is_zero(v)
            }
        elif get_dgcv_category(coeff_dict) == "array":
            coeff_dict = coeff_dict._data
        else:
            raise TypeError(
                "algebra_element_class recieved unsupports coeffs parameter format."
            ) from None
        self.algebra = alg
        self.vectorSpace = alg
        self.valence = valence
        self.is_sparse = format_sparse
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "algebra_element"
        self.card = self.vectorSpace.card
        self.coeff_dict = coeff_dict
        self._coeffs = None  # deprecated
        self._coeffs_hash_cache = None
        self._tensor_rep = None
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
    def tensor_representation(self):
        if self._tensor_rep is None and self.algebra.tensor_representation is not None:
            trep = self.algebra.tensor_representation
            self._tensor_rep = sum(c * trep[idx] for idx, c in self.coeff_dict.items())
        return self._tensor_rep

    def __eq__(self, other):
        if not isinstance(other, algebra_element_class):
            return NotImplemented
        return (
            self.algebra == other.algebra
            and self._coeffs_hash == other._coeffs_hash
            and self.valence == other.valence
            and self.is_sparse == other.is_sparse
        )

    def __hash__(self):
        return hash((self.algebra, self._coeffs_hash, self.valence, self.is_sparse))

    def _class_builder(self, coeff_dict, valence, format_sparse=False):
        return algebra_element_class(
            self.algebra, coeff_dict, valence, format_sparse=format_sparse
        )

    def __str__(self):
        if self.algebra.basis_labels is None:
            return "elem"

        if not self.algebra._registered:
            if (
                self.algebra._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self.algebra._callLock == retrieve_passkey() and isinstance(
                self.algebra._child_print_warning, str
            ):
                dgcv_warning(self.algebra._child_print_warning)
            else:
                dgcv_warning(
                    "This algebra_element_class's parent vector space (algebra_class) was initialized without an assigned label. "
                    "It is recommended to initialize `algebra_class` objects with dgcv creator functions like `createAlgebra` instead."
                )

        return lincomb_plain(
            self.coeff_dict,
            self.algebra.basis_labels,
            valence=self.valence,
            label_transform=None,
            fallback_label=self.algebra.basis_labels[0]
            if self.algebra.basis_labels
            else "e_1",
            include_zero_term=False,
        )

    def _repr_latex_(self, verbose=False, raw=False, **kwargs):
        if not self.vectorSpace._registered:
            if (
                self.vectorSpace._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self.vectorSpace._callLock == retrieve_passkey() and isinstance(
                self.vectorSpace._child_print_warning, str
            ):
                dgcv_warning(self.vectorSpace._child_print_warning)
            else:
                dgcv_warning(
                    "This algebra_element_class's parent vector space (algebra_class) was initialized without an assigned label. "
                    "It is recommended to initialize `algebra_class` objects with dgcv creator functions like `createAlgebra` instead."
                )

        return lincomb_latex(
            self.coeff_dict,
            vectorSpace=self.vectorSpace,
            valence=self.valence,
            verbose=verbose,
            raw=raw,
            apply_vlp_trim=True,
        )

    def _latex(self, printer=None, raw=True, **kwargs):
        return self._repr_latex_(raw=raw)

    def _latex_verbose(self, printer=None):
        """deprecated"""
        if not self.algebra._registered:
            if (
                self.algebra._exclude_from_VMF == retrieve_passkey()
                or get_dgcv_settings_registry()["forgo_warnings"] is True
            ):
                pass
            elif self.algebra._callLock == retrieve_passkey() and isinstance(
                self.algebra._child_print_warning, str
            ):
                dgcv_warning(self.algebra._child_print_warning)
            else:
                dgcv_warning(
                    "This algebra_element_class's parent vector space (an `algebra` class instance) was initialized without an assigned label. "
                    "It is recommended to initialize `algebra` class objects with dgcv creator functions like `createFiniteAlg` instead."
                )

        terms = []
        labels = self.algebra.basis_labels or [
            f"e_{i + 1}" for i in range(self.algebra.dimension)
        ]
        for idx, coeff in self.coeff_dict.items():
            basis_label = labels[idx]
            if _scalar_is_zero(coeff - 1):
                if self.valence == 1:
                    terms.append(rf"{basis_label}")
                else:
                    terms.append(rf"{basis_label}^*")
            elif _scalar_is_zero(coeff + 1):
                if self.valence == 1:
                    terms.append(rf"-{basis_label}")
                else:
                    terms.append(rf"-{basis_label}^*")
            else:
                if isinstance(coeff, expr_numeric_types()) and len(coeff.args) > 1:
                    if self.valence == 1:
                        terms.append(rf"({latex(coeff)}) {basis_label}")
                    else:
                        terms.append(rf"({latex(coeff)}) {basis_label}^*")
                else:
                    if self.valence == 1:
                        terms.append(rf"{latex(coeff)} {basis_label}")
                    else:
                        terms.append(rf"{latex(coeff)} {basis_label}^*")

        if not terms:
            return rf"0 {self.algebra.basis_labels[0] if self.algebra.basis_labels else 'e_1'}"

        result = " + ".join(terms).replace("+ -", "- ")

        def format_algebra_label(label):
            r"""
            Wrap the vector space label in \mathfrak{} if lowercase, and add subscripts for numeric suffixes or parts.
            """
            if "_" in label:
                main_part, subscript_part = label.split("_", 1)
                if main_part.islower():
                    return rf"\mathfrak{{{main_part}}}_{{{subscript_part}}}"
                return rf"{main_part}_{{{subscript_part}}}"
            elif label[-1].isdigit():
                label_text = "".join(filter(str.isalpha, label))
                label_number = "".join(filter(str.isdigit, label))
                if label_text.islower():
                    return rf"\mathfrak{{{label_text}}}_{{{label_number}}}"
                return rf"{label_text}_{{{label_number}}}"
            elif label.islower():
                return rf"\mathfrak{{{label}}}"
            return label

        return rf"\text{{Element of }} {format_algebra_label(self.algebra.label)}: {result}"

    @property
    def label(self):
        return self.__repr__()

    @property
    def is_zero(self):
        for j in self.coeff_dict.values():
            if not is_zero_knowing_zero_is_expected(j):
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

    def subs(self, subsData):
        newCoeffs = {idx: subs(j, subsData) for idx, j in self.coeff_dict.items()}
        return algebra_element_class(self.algebra, newCoeffs, self.valence)

    @property
    def ambient_rep(self):
        return self

    def __dgcv_simplify__(self, *args, **kwargs):
        return algebra_element_class(
            self.algebra,
            {idx: simplify(j, **kwargs) for idx, j in self.coeff_dict.items()},
            self.valence,
        )

    def _eval_simplify(self, *args, **kwargs):
        return algebra_element_class(
            self.algebra,
            {idx: simplify(j, **kwargs) for idx, j in self.coeff_dict.items()},
            self.valence,
        )

    def dual(self):
        return algebra_element_class(
            self.algebra,
            self.coeff_dict,
            (self.valence + 1) % 2,
        )

    def _convert_to_tp(self):
        return tensorProduct(
            {
                ((idx, self.valence, self.card),): j
                for idx, j in self.coeff_dict.items()
            },
            shape="all",
        )

    def _recursion_contract_hom(self, other):
        return self._convert_to_tp()._recursion_contract_hom(other)

    def _si_wrap(self, obj):
        if self.algebra.simplify_products_by_default is True:
            return simplify(obj)
        else:
            return obj

    def _fast_add(self, other):
        """
        Internal-only: assumes `other` is an algebra_element_class
        with the same algebra and valence. No type or safety checks etc.
        """
        new_dict = dict(self.coeff_dict)
        for k, v in other.coeff_dict.items():
            new_dict[k] = new_dict.get(k, 0) + v
        return algebra_element_class(
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
            if (
                other.algebra.ambient.card == self.card
                and self.valence == other.valence
            ):
                other = other.ambient_rep
            else:
                other = other._convert_to_tp()
        if get_dgcv_category(other) == "algebra_element":
            if self.algebra == other.algebra and self.valence == other.valence:
                new_dict = dict(self.coeff_dict)
                for k, v in other.coeff_dict.items():
                    new_dict[k] = new_dict.get(k, 0) + v
                return algebra_element_class(
                    self.algebra,
                    new_dict,
                    self.valence,
                )
            else:
                other = other._convert_to_tp()
        if isinstance(other, expr_numeric_types()):
            other = tensorProduct({tuple(): other})
        if isinstance(other, tensorProduct):
            return self._convert_to_tp() + other
        return NotImplemented

    def __radd__(self, other):
        if _scalar_is_zero(other):
            return self
        if isinstance(other, expr_numeric_types()):
            return tensorProduct({tuple(): other}) + self
        return NotImplemented

    def __sub__(self, other):
        if _scalar_is_zero(other):
            return self
        if get_dgcv_category(other) == "subalgebra_element":
            if (
                other.algebra.ambient.card == self.card
                and self.valence == other.valence
            ):
                other = other.ambient_rep
            else:
                other = other._convert_to_tp()
        if get_dgcv_category(other) == "algebra_element":
            if self.algebra == other.algebra and self.valence == other.valence:
                new_dict = dict(self.coeff_dict)
                for k, v in other.coeff_dict.items():
                    new_dict[k] = new_dict.get(k, 0) - v
                return algebra_element_class(
                    self.algebra,
                    new_dict,
                    self.valence,
                )
            else:
                other = other._convert_to_tp()
        if isinstance(other, expr_numeric_types()):
            other = tensorProduct({tuple(): other})
        if isinstance(other, tensorProduct):
            return self._convert_to_tp() - other
        return NotImplemented

    def __rsub__(self, other):
        if _scalar_is_zero(other):
            return -self
        if isinstance(other, expr_numeric_types()):
            return tensorProduct({tuple(): other}) - self
        return NotImplemented

    def __mul__(self, other):
        if get_dgcv_category(other) == "subalgebra_element":
            if (
                other.algebra.ambient.card == self.card
                and self.valence == other.valence
            ):
                other = other.ambient_rep
            else:
                other = other._convert_to_tp()
        if isinstance(other, algebra_element_class):
            if self.algebra == other.algebra and self.valence == other.valence:
                sign = 1 if self.valence == 1 else -1
                alg = self.algebra
                struct = alg.structureData
                spbd = self.algebra.simplify_products_by_default
                new_coeffs = dict()
                for idx1, c1 in self.coeff_dict.items():
                    for idx2, c2 in other.coeff_dict.items():
                        data = struct[idx1, idx2]._data
                        if not data:
                            continue
                        scalar = sign * c1 * c2
                        for idx3, c3 in data.items():
                            new_coeffs[idx3] = new_coeffs.get(idx3, 0) + (
                                self._si_wrap(scalar * c3) if spbd else scalar * c3
                            )

                return algebra_element_class(
                    self.algebra,
                    new_coeffs,
                    self.valence,
                )
            else:
                other = other._convert_to_tp()
        elif isinstance(other, tensorProduct):
            return self._si_wrap((self._convert_to_tp()) * other)
        elif isinstance(other, expr_numeric_types()):
            new_coeffs = {
                idx: self._si_wrap(j * other) for idx, j in self.coeff_dict.items()
            }
            return algebra_element_class(self.algebra, new_coeffs, self.valence)
        return NotImplemented

    def __rmul__(self, other):
        if get_dgcv_category(other) == "subalgebra_element":
            if (
                other.algebra.ambient.card == self.card
                and self.valence == other.valence
            ):
                return other.ambient_rep * self
        if isinstance(other, expr_numeric_types()) or get_dgcv_category(other) in {
            "subalgebra_element",
            "algebra_element",
            "tensorProduct",
        }:
            return self._si_wrap(self * other)
        return NotImplemented

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
            raise TypeError(
                f"unsuported operand types for `@`. Types {type(self)} and {type(other)}"
            ) from None
        new_dict = {
            (
                (idx1, self.valence, self.card),
                (idx2, other.valence, other.card),
            ): self._si_wrap(c1 * c2)
            for idx1, c1 in self.coeff_dict.items()
            for idx2, c2 in other.coeff_dict.items()
        }
        return tensorProduct(new_dict)

    def __rmatmul__(self, other):
        if isinstance(other, expr_numeric_types()):
            return other * self
        return self._convert_to_tp().__rmatmul__(other)

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return self._si_wrap(ratio(1, other) * self)
        elif isinstance(other, expr_numeric_types()):
            return self._si_wrap((1 / other) * self)
        else:
            raise TypeError(
                f"True division `/` of algebra elements by another object is only supported for scalars, not {type(other)}"
            ) from None

    def __neg__(self):
        return -1 * self

    def __xor__(self, other):
        if other == "":
            return self.dual()
        raise ValueError("Invalid operation. Use `^''` to denote the dual.") from None

    def __call__(self, other, **kwds):
        if get_dgcv_category(other) == "subalgebra_element":
            other = other.ambient_rep
        if get_dgcv_category(other) == "algebra_element":
            if other.algebra == self.algebra and other.valence != self.valence:
                cd = other.coeff_dict
                return sum(c * cd.get(idx, 0) for idx, c in self.coeff_dict.items())
            elif self.tensor_representation is not None:
                return self.tensor_representation(other)
            else:
                raise TypeError(
                    f"`algebra_element_class` call can only be applied to elements from the same algebra pairing one element with another of complementary valence, or applying elements from an endomorphism_space subclass. Recieved self: {self} and other: {other} belonging to {self.algebra} and {other.algebra} with valences {self.valence} and {other.valence}"
                )
        else:
            raise TypeError(
                f"`algebra_element_class` call cannot be applies objects of type {type(other)}"
            )

    def compute_weight(self, test_weights=None, flatten_weights=False):
        return self.check_element_weight(
            test_weights=test_weights, flatten_weights=flatten_weights
        )

    def check_element_weight(self, test_weights=None, flatten_weights=False):
        """
        Determines the weight vector of this algebra_element_class instance with respect to its algebra's grading vectors.

        Returns
        -------
        list
            A list of weights corresponding to the grading vectors of the parent algebra.

        Notes
        -----
        - 'AllW' is returned for zero elements, which are compaible with all weights.
        - 'NoW' is returned for non-homogeneous elements that do not satisfy the grading constraints.
        """

        return self.algebra.check_element_weight(
            self, test_weights=test_weights, flatten_weights=flatten_weights
        )

    def weighted_decomposition(self, test_weights=None, flatten_weights=False):
        weighted_components = {}
        for idx, coeff in self.coeff_dict.items():
            elem = self.algebra.basis[idx]
            w = elem.check_element_weight(
                test_weights=test_weights, flatten_weights=flatten_weights
            )
            if isinstance(w, list):
                w = tuple(w)
            weighted_components[w] = weighted_components.get(w, 0) + coeff * elem
        return weighted_components

    def coproduct(self):
        if self.valence != 0:
            return print(
                "The algebra co-product is only defined on dual Lie algebra elements as it is dual to the algebra product map."
            )
        terms = []
        for idx, c in self.coeff_dict.items():
            elem = self.algebra.basis[idx]
            if self.algebra._coproduct.get(elem, None) is None:
                tensor_terms = []
                for idx, e1 in enumerate(self.algebra.basis):
                    if self.algebra.is_skew_symmetric():
                        skew = True
                        start = idx + 1
                    else:
                        skew = False
                        start = 0
                    for e2 in self.algebra.basis[start:]:
                        if skew:
                            tensor_terms.append(
                                self(e1 * e2)
                                * (e1.dual() @ e2.dual() - e2.dual() @ e1.dual())
                            )
                        else:
                            tensor_terms.append(self(e1 * e2) * (e1.dual() @ e2.dual()))
                self.algebra._coproduct[elem] = sum(tensor_terms)
            terms.append(c * self.algebra._coproduct[elem])
        return sum(terms)

    @property
    def free_symbols(self):
        fs = set()
        for c in self.coeff_dict.values():
            fs |= get_free_symbols(c)
        return fs

    def dual_pairing(self, other):
        return self._convert_to_tp().dual_pairing(other)

    def decompose(self, format_as_list=True, return_basis=True):
        if self.valence == 1:
            out = self.coeffs if format_as_list else self.coeff_dict
            return (out, self.algebra.basis) if return_basis else out
        out = self.coeffs if format_as_list else self.coeff_dict
        return (out, [j.dual() for j in self.algebra.basis]) if return_basis else out

    def terms(self):
        return [c * self.algebra.basis[idx] for idx, c in self.coeff_dict.items()]
