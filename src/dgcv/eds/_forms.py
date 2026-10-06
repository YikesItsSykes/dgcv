"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

module: dgcv.eds._forms

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers

from .._aux._backends._cls_coercion import register_legacy_sympy_class
from .._aux._backends._display import latex as _routed_latex
from .._aux._backends._engine import engine_kind
from .._aux._backends._symbolic_router import (
    _scalar_is_minus_one,
    _scalar_is_one,
    _scalar_is_zero,
    get_free_symbols,
    scalars_equal,
)
from .._aux._backends._symbolic_router import simplify as _routed_simplify
from .._aux._backends._types_and_constants import (
    OP_ADD,
    as_engine_scalar,
    expr_numeric_types,
    expr_types,
    integer,
    op_expr,
    to_active_engine,
)
from .._aux._utilities._config import dgcv_warning
from .._aux._vmf._safeguards import retrieve_passkey
from .._aux.printing.printing._eds import (
    bar_label_latex,
    conjugation_prefix,
    ext_der_latex,
    ext_der_repr,
    ext_der_str,
)
from ..core.base import dgcv_class
from ..core.combinatorics.combinatorics import weightedPermSign
from ._atoms import (
    SortableObj,
    _custom_conj,
    _new_bridge,
    _zero_obstruction,
    barSortedStr,
    zero_form_atom,
)
from ._zero_forms import (
    _sympy_to_abstract_ZF,
    _zf_coeff_str,
    zero_form_class,
)


class _coefficient_mapped:
    def __dgcv_conjugate__(self, symbolic=False):
        return self._eval_conjugate()

    def __dgcv_apply__(self, func, **kwargs):
        return self._map_coefficients(lambda coeff: func(coeff, **kwargs))

    def simplify(self, **kwargs):
        return self._induce_method_from_descending("simplify", **kwargs)

    def factor(self, **kwargs):
        return self._induce_method_from_descending("factor", **kwargs)

    def expand(self, **kwargs):
        return self._induce_method_from_descending("expand", **kwargs)

    def trigsimp(self, **kwargs):
        return self._induce_method_from_descending("trigsimp", **kwargs)

    def cancel(self, **kwargs):
        return self._induce_method_from_descending("cancel", **kwargs)

    def together(self, **kwargs):
        return self._induce_method_from_descending("together", **kwargs)

    def apart(self, **kwargs):
        return self._induce_method_from_descending("apart", **kwargs)

    def ratsimp(self, **kwargs):
        return self._induce_method_from_descending("ratsimp", **kwargs)

    def powsimp(self, **kwargs):
        return self._induce_method_from_descending("powsimp", **kwargs)

    def logcombine(self, **kwargs):
        return self._induce_method_from_descending("logcombine", **kwargs)

    def expand_log(self, **kwargs):
        return self._induce_method_from_descending("expand_log", **kwargs)

    def expand_trig(self, **kwargs):
        return self._induce_method_from_descending("expand_trig", **kwargs)

    def expand_power_exp(self, **kwargs):
        return self._induce_method_from_descending("expand_power_exp", **kwargs)

    def expand_power_base(self, **kwargs):
        return self._induce_method_from_descending("expand_power_base", **kwargs)

    def numer(self, **kwargs):
        return self._induce_method_from_descending("numer", **kwargs)

    def denom(self, **kwargs):
        return self._induce_method_from_descending("denom", **kwargs)


class abstract_differential_form_atom(_coefficient_mapped, dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "abstDFAtom"
    _dgcv_categories = {"abstDFAtom", "eds_form"}

    def __dgcv_solve_bridge__(self, bridge):
        return bridge.lower(self._coeff)

    @property
    def __dgcv_zero_obstr__(self):
        return _zero_obstruction(self)

    def __init__(
        self, coeff, degree, label=None, ext_deriv_order=0, _markers=frozenset()
    ):
        coeff = to_active_engine(coeff)
        if _scalar_is_zero(coeff):
            coeff = 0
        elif _scalar_is_one(coeff):
            coeff = 1
        if isinstance(coeff, numbers.Integral):
            coeff = integer(coeff)
        self.label = label
        self.degree = degree
        self.coeff = coeff
        self._coeff = coeff
        self.ext_deriv_order = ext_deriv_order
        self._markers = _markers

        self.coeffs = [coeff]

    @property
    def is_zero(self):
        return _scalar_is_zero(self._coeff)

    def _sage_(self):
        raise AttributeError

    def __eq__(self, other):
        """
        Check equality of two abstract_DF_atom instances.
        """
        if not isinstance(other, abstract_differential_form_atom):
            return NotImplemented
        return (
            scalars_equal(self.coeff, other.coeff)
            and self.degree == other.degree
            and self.label == other.label
            and self.ext_deriv_order == other.ext_deriv_order
        )

    def __hash__(self):
        """
        Hash the abstract_DF_atom instance based on its attributes.
        """
        return hash((self.coeff, self.degree, self.label, self.ext_deriv_order))

    def _eval_conjugate(self):
        if self.label:
            pref = conjugation_prefix()
            prefL = len(pref)
            if "real" in self._markers:
                label = self.label
            elif self.label[0:prefL] == pref:
                label = self.label[prefL:]
            else:
                label = f"{pref}{self.label}"
        else:
            label = None

        def cMarkers(marker):
            if marker == "holomorphic":
                return "antiholomorphic"
            if marker == "antiholomorphic":
                return "holomorphic"
            return marker

        new_markers = frozenset([cMarkers(j) for j in self._markers])
        coeff = _custom_conj(self.coeff)
        return abstract_differential_form_atom(
            coeff,
            self.degree,
            label=label,
            ext_deriv_order=self.ext_deriv_order,
            _markers=new_markers,
        )

    def __repr__(self):
        """String representation for abstract_DF_atom."""

        if isinstance(self.coeff, (zero_form_atom, zero_form_class)):
            coeff_str = _zf_coeff_str(self.coeff, self.coeff.__repr__())
            return ext_der_repr(
                f"{coeff_str}{self.label}" if self.label else coeff_str,
                self.ext_deriv_order,
            )
        coeff_sympy = as_engine_scalar(self.coeff)
        if (
            not get_free_symbols(coeff_sympy)
            and self.ext_deriv_order is not None
            and self.ext_deriv_order > 0
        ):
            return "0"
        if _scalar_is_one(coeff_sympy):
            return str(self.label) if self.label else "1"
        elif _scalar_is_minus_one(coeff_sympy):
            return f"-{self.label}" if self.label else "-1"
        else:
            # Wrap in parentheses if there are multiple terms
            coeff_str = (
                f"({coeff_sympy})"
                if op_expr(coeff_sympy) == OP_ADD
                else str(coeff_sympy)
            )
            return (
                ext_der_repr(f"{coeff_str}{self.label}", self.ext_deriv_order)
                if self.label
                else ext_der_repr(coeff_str, self.ext_deriv_order)
            )

    def _latex(self, printer=None, raw=True, **kwargs):
        """LaTeX representation for abstract_DF_atom."""

        if isinstance(self.coeff, (zero_form_atom, zero_form_class)):
            coeff_latex = _routed_latex(self.coeff)
            return ext_der_latex(
                f"{coeff_latex}{bar_label_latex(self.label)}"
                if self.label
                else coeff_latex,
                self.ext_deriv_order,
            )
        coeff_sympy = as_engine_scalar(self.coeff)
        if (
            not get_free_symbols(coeff_sympy)
            and self.ext_deriv_order is not None
            and self.ext_deriv_order > 0
        ):
            return "0"
        if _scalar_is_one(coeff_sympy):
            return bar_label_latex(self.label) if self.label else "1"
        elif _scalar_is_minus_one(coeff_sympy):
            return f"-{bar_label_latex(self.label)}" if self.label else "-1"
        else:
            # Wrap in parentheses if there are multiple terms
            coeff_latex = (
                f"\\left({_routed_latex(coeff_sympy)}\\right)"
                if op_expr(coeff_sympy) == OP_ADD
                else _routed_latex(coeff_sympy)
            )
            return (
                ext_der_latex(
                    f"{coeff_latex}{bar_label_latex(self.label)}", self.ext_deriv_order
                )
                if self.label
                else coeff_latex
            )

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex() if raw else f"${self._latex()}$"

    def __str__(self):
        if isinstance(self.coeff, (zero_form_atom, zero_form_class)):
            coeff_str = _zf_coeff_str(self.coeff, self.coeff.__repr__())
            return ext_der_str(
                f"{coeff_str}{self.label}" if self.label else coeff_str,
                self.ext_deriv_order,
            )
        coeff_sympy = as_engine_scalar(self.coeff)
        if (
            not get_free_symbols(coeff_sympy)
            and self.ext_deriv_order is not None
            and self.ext_deriv_order > 0
        ):
            return "0"
        if _scalar_is_one(coeff_sympy):
            return str(self.label) if self.label else "1"
        elif _scalar_is_minus_one(coeff_sympy):
            return f"-{self.label}" if self.label else "-1"
        else:
            # Wrap in parentheses if there are multiple terms
            coeff_str = (
                f"({coeff_sympy})"
                if op_expr(coeff_sympy) == OP_ADD
                else str(coeff_sympy)
            )
            return (
                ext_der_str(f"{coeff_str}{self.label}", self.ext_deriv_order)
                if self.label
                else ext_der_str(coeff_str, self.ext_deriv_order)
            )

    def to_engine(self):
        if self.degree == 0:
            if self.label is not None:
                dgcv_warning(
                    "`abstract_DF_atom.to_sympy()` was called for an instance with nontrivial basis label, and that label is not encoded in the method's output."
                )
            return _new_bridge().lower(self.coeff)[0]
        else:
            dgcv_warning(
                "`abstract_DF_atom.to_sympy()` was called for an instance with positive degree, so `None` was returned."
            )

    def to_sympy(self):
        expr = self.to_engine()
        if engine_kind() == "sage" and hasattr(expr, "_sympy_"):
            return expr._sympy_()
        return expr

    def has_common_factor(self, other):
        if not isinstance(
            other,
            (abstract_differential_form_atom, abstract_differential_form_monomial),
        ):
            return False

        if isinstance(other, abstract_differential_form_atom):
            # Special case: label None and degree 0
            if self.label is None and self.degree == 0:
                return other.label is None and other.degree == 0
            # Otherwise, compare labels
            return self.label == other.label

        elif isinstance(other, abstract_differential_form_monomial):
            # Match against all factors in the monomial
            return any(
                self.has_common_factor(factor) for factor in other.factors_sorted
            )

        return False

    def _eval_simplify(
        self,
        ratio=None,
        measure=None,
        inverse=True,
        doit=True,
        rational=True,
        expand=False,
        **kwargs,
    ):
        return abstract_differential_form_atom(
            _routed_simplify(self.coeff),
            self.degree,
            self.label,
            _markers=self._markers,
        )

    def _eval_canonicalize(self, depth=1000):
        if hasattr(self.coeff, "_eval_canonicalize"):
            new_coeff = self.coeff._eval_canonicalize(depth=depth)
        else:
            new_coeff = self.coeff
        return abstract_differential_form_atom(
            new_coeff, self.degree, self.label, _markers=self._markers
        )

    def _map_coefficients(self, fn):
        return abstract_differential_form_atom(
            fn(self.coeff),
            self.degree,
            self.label,
            ext_deriv_order=self.ext_deriv_order,
            _markers=self._markers,
        )

    def _induce_method_from_descending(self, method_name, **kwargs):
        new_coeff = (
            getattr(self.coeff, method_name)(**kwargs)
            if hasattr(self.coeff, method_name)
            else self.coeff
        )
        return abstract_differential_form_atom(
            new_coeff,
            self.degree,
            self.label,
            ext_deriv_order=self.ext_deriv_order,
            _markers=self._markers,
        )

    def _subs_dgcv(self, data, with_diff_corollaries=False):
        # an alias for regular subs so that other functions can know the with_diff_corollaries keyword is available
        return self.subs(data, with_diff_corollaries=with_diff_corollaries)

    def subs(self, subs_data, with_diff_corollaries=False):
        if isinstance(subs_data, (list, tuple)) and all(
            isinstance(j, tuple) and len(j) == 2 for j in subs_data
        ):
            l1 = len(subs_data)
            subs_data = dict(subs_data)
            if len(subs_data) < l1:
                dgcv_warning(
                    "Provided substitution rules had repeat keys, and only one was used."
                )
        if self in subs_data:
            return subs_data[self]
        new_coeff = None
        if isinstance(self.coeff, (zero_form_atom, zero_form_class)):
            new_coeff = (self.coeff).subs(
                subs_data, with_diff_corollaries=with_diff_corollaries
            )
        elif isinstance(self.coeff, expr_types()):
            if not all(
                isinstance(k, expr_types()) and isinstance(v, expr_numeric_types())
                for k, v in subs_data.items()
            ):
                new_coeff = zero_form_class(
                    _sympy_to_abstract_ZF(self.coeff, subs_rules=subs_data)
                )
            else:
                new_coeff = (self.coeff).subs(subs_data)
        for k, v in subs_data.items():
            if isinstance(k, abstract_differential_form_atom):
                if (
                    self.degree == k.degree
                    and self.label == k.label
                    and self.ext_deriv_order == k.ext_deriv_order
                    and self._markers == k._markers
                ):
                    if new_coeff is None:
                        return (self.coeff / k.coeff) * v
                    else:
                        return (new_coeff / k.coeff) * v
        if new_coeff is None:
            return self
        else:
            return abstract_differential_form_atom(
                new_coeff,
                self.degree,
                self.label,
                self.ext_deriv_order,
                _markers=self._markers,
            )

    @property
    def free_symbols(self):
        if hasattr(self.coeff, "free_symbols"):
            return self.coeff.free_symbols
        return set()

    @property
    def _seperated_form(self):
        if _scalar_is_one(self.coeff):
            newAtom = self
        else:
            newAtom = abstract_differential_form_atom(
                1,
                self.degree,
                label=self.label,
                ext_deriv_order=self.ext_deriv_order,
                _markers=self._markers,
            )
        return newAtom, self.coeff

    def __mul__(self, other):
        """Handle left multiplication."""
        if isinstance(other, abstract_differential_form_atom):
            # Combine two atoms into a monomial product
            return abstract_differential_form_monomial([self, other])
        elif isinstance(other, abstract_differential_form_monomial):
            # Prepend this atom as a factor to the monomial's factors
            return abstract_differential_form_monomial([self] + other.factors_sorted)
        elif isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            return abstract_differential_form_atom(
                self.coeff * other, self.degree, self.label, _markers=self._markers
            )
        else:
            return NotImplemented

    def __rmul__(self, other):
        """Handle right multiplication."""
        if isinstance(other, abstract_differential_form_monomial):
            # Append this atom as a factor to the monomial
            return abstract_differential_form_monomial(other.factors_sorted + [self])
        elif isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            return abstract_differential_form_atom(
                self.coeff * other, self.degree, self.label, _markers=self._markers
            )
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, abstract_differential_form_atom) and other.degree == 0:
            if other.label is None or other.label == "":
                other = other.coeff
            else:
                other = other.coeff * zero_form_atom(
                    other.label, _markers=other._markers
                )
        if isinstance(other, (zero_form_class, zero_form_atom)) or isinstance(
            other, expr_numeric_types()
        ):
            if _scalar_is_zero(other):
                raise ZeroDivisionError("division by zero")
            return (1 / other) * self
        return NotImplemented

    def __neg__(self):
        return -1 * self

    def __add__(self, other):
        """Addition with another atom or monomial returns an abstract_DF."""
        if other is None:
            return self
        if isinstance(other, expr_numeric_types()):
            other = zero_form_class(other)
        if isinstance(
            other,
            (abstract_differential_form_monomial, abstract_differential_form_atom),
        ):
            return abstract_differential_form(
                [abstract_differential_form_monomial([self]), other]
            )
        elif isinstance(other, zero_form_atom):
            return abstract_differential_form(
                [
                    abstract_differential_form_monomial([self]),
                    abstract_differential_form_monomial(
                        [
                            abstract_differential_form_atom(
                                other, 0, _markers=other._markers
                            )
                        ]
                    ),
                ]
            )
        elif isinstance(other, zero_form_class):
            return abstract_differential_form(
                [
                    abstract_differential_form_monomial([self]),
                    abstract_differential_form_monomial(
                        [abstract_differential_form_atom(other, 0)]
                    ),
                ]
            )
        elif isinstance(other, abstract_differential_form):
            return abstract_differential_form(
                (abstract_differential_form_monomial([self]),) + tuple(other.terms)
            )
        else:
            raise TypeError(
                f"Unsupported operand type {type(other)} for + with `abstract_DF_atom`"
            )

    def __sub__(self, other):
        """Subtraction with another atom or monomial returns an abstract_DF."""
        if other is None:
            return self
        if isinstance(other, expr_numeric_types()):
            other = zero_form_class(other)
        elif isinstance(
            other,
            (abstract_differential_form_monomial, abstract_differential_form_atom),
        ):
            return abstract_differential_form(
                [abstract_differential_form_monomial([self]), -1 * other]
            )
        elif isinstance(other, abstract_differential_form):
            negated_terms = tuple([-1 * term for term in other.terms])
            return abstract_differential_form(
                [abstract_differential_form_monomial([self])] + list(negated_terms)
            )
        elif isinstance(other, (zero_form_atom, zero_form_class)):
            return abstract_differential_form(
                [
                    abstract_differential_form_monomial([self]),
                    abstract_differential_form_monomial(
                        [abstract_differential_form_atom(-1 * other, 0)]
                    ),
                ]
            )
        else:
            raise TypeError(
                f"Unsupported operand type for - with `abstract_DF_atom`: {type(other)}"
            )

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return other + (-self)

    def __lt__(self, other):
        """
        Lexicographic comparison: First by degree, then by label.
        """
        if not isinstance(other, abstract_differential_form_atom):
            return NotImplemented

        # Primary comparison: degree
        if self.degree != other.degree:
            return self.degree < other.degree

        # Secondary comparison: label (None precedes any string)
        if self.label is None:
            return True
        if other.label is None:
            return False
        return self.label < other.label


class abstract_differential_form_monomial(_coefficient_mapped, dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "abstDFMonom"
    _dgcv_categories = {"abstDFMonom", "eds_form"}

    def __dgcv_solve_bridge__(self, bridge):
        return bridge.lower(self._coeff)

    @property
    def __dgcv_zero_obstr__(self):
        return _zero_obstruction(self)

    def __init__(self, factors):
        if not isinstance(factors, (list, tuple)):
            raise TypeError(
                "`abstract_DF_monomial` expects `factors` to be a list or tuple"
            )
        if not all(
            isinstance(elem, abstract_differential_form_atom) for elem in factors
        ):
            raise TypeError(
                "`abstract_DF_monomial` expects `factors` to be a list of `abstract_DF_atom`"
            )

        class DegreeLabelSortable:
            def __init__(self, atom):
                self.degree = atom.degree
                self.label = atom.label

            def __lt__(self, other):
                if self.degree != other.degree:
                    return self.degree < other.degree
                if self.label is None:
                    return True
                if other.label is None:
                    return False
                return self.label < other.label

            def __eq__(self, other):
                return self.degree == other.degree and self.label == other.label

            def __le__(self, other):
                return self < other or self == other

        weighted_objs = [
            SortableObj((j, (j.degree, barSortedStr(j.label)))) for j in factors
        ]
        parity, objs_sorted, _ = weightedPermSign(
            weighted_objs,
            [DegreeLabelSortable(j) for j in factors],
            returnSorted=True,
            use_degree_attribute=True,
        )

        if parity == -1:
            objs_sorted = [
                SortableObj((abstract_differential_form_atom(-1, 0), (0, None)))
            ] + objs_sorted

        coeffFactor = 1
        new_objs = []

        for j in objs_sorted:
            if not _scalar_is_zero(coeffFactor):
                coeffFactor = (
                    coeffFactor * j.value.coeff
                    if not _scalar_is_zero(j.value.coeff)
                    else 0
                )
            if j.place[0] != 0:
                new_objs.append(
                    abstract_differential_form_atom(
                        1,
                        j.value.degree,
                        j.value.label,
                        j.value.ext_deriv_order,
                        j.value._markers,
                    )
                )

        consolidated_factor = abstract_differential_form_atom(
            coeffFactor, 0
        )  ### check !!!
        if _scalar_is_zero(coeffFactor):
            self.factors_sorted = [abstract_differential_form_atom(0, 0)]
            self._coeff = 0
        else:
            self.factors_sorted = [consolidated_factor] + new_objs
            self._coeff = coeffFactor
        if (
            len(self.factors_sorted) != len(set(self.factors_sorted))
            or len(self.factors_sorted) == 0
        ):
            self.factors_sorted = [abstract_differential_form_atom(0, 0)]
        self.factors = factors
        self.str_ids = tuple(
            "<Coeff>"
            if i == 0 and factor.label is None
            else (factor.label if factor.label is not None else "<None>")
            for i, factor in enumerate(self.factors_sorted)
        )
        self.degree = sum(
            factor.degree
            for factor in self.factors
            if not _scalar_is_zero(factor.coeff)
        )

    def _sage_(self):
        raise AttributeError

    def __eq__(self, other):
        """
        Check equality of two abstract_DF_monomial instances.
        """
        if not isinstance(other, abstract_differential_form_monomial):
            return NotImplemented
        return (
            self.factors_sorted == other.factors_sorted and self.degree == other.degree
        )

    def __hash__(self):
        """
        Hash the abstract_DF_monomial instance based on sorted factors.
        """
        return hash((tuple(self.factors_sorted), self.degree))

    def sort_key(self, order=None):  # for the sympy sorting.py default_sort_key
        return (
            4,
            self.degree,
            tuple(self.factors_sorted),
        )  # 4 is to group with function-like objects

    @property
    def is_zero(self):
        return _scalar_is_zero(self._coeff)

    @property
    def coeff(self):
        coeff = 1
        for factor in self.factors_sorted:
            if (
                isinstance(factor, abstract_differential_form_atom)
                and factor.degree == 0
            ):
                factor = factor.coeff
            if _scalar_is_zero(factor):
                coeff = 0
                break
            if isinstance(factor, (zero_form_class, zero_form_atom)) or isinstance(
                factor, expr_numeric_types()
            ):
                coeff *= factor
        return coeff

    @property
    def coeffs(self):
        return [self.coeff]

    def _eval_conjugate(self):
        return abstract_differential_form_monomial(
            [j._eval_conjugate() for j in self.factors]
        )

    def _eval_simplify(
        self,
        ratio=None,
        measure=None,
        inverse=True,
        doit=True,
        rational=True,
        expand=False,
        **kwargs,
    ):
        return abstract_differential_form_monomial(
            [j._eval_simplify() for j in self.factors]
        )

    def _eval_canonicalize(self, depth=1000):
        def _canon(obj):
            if hasattr(obj, "_eval_canonicalize"):
                return obj._eval_canonicalize(depth=depth)
            return obj

        return abstract_differential_form_monomial([_canon(j) for j in self.factors])

    def _subs_dgcv(self, data, with_diff_corollaries=False):
        # an alias for regular subs so that other functions can know the with_diff_corollaries keyword is available
        return self.subs(data, with_diff_corollaries=with_diff_corollaries)

    def subs(self, subs_data, with_diff_corollaries=False):
        return abstract_differential_form_monomial(
            [
                j._subs_dgcv(subs_data, with_diff_corollaries=with_diff_corollaries)
                for j in self.factors_sorted
            ]
        )

    def _map_coefficients(self, fn):
        return abstract_differential_form_monomial(
            [
                j._map_coefficients(fn) if hasattr(j, "_map_coefficients") else j
                for j in self.factors_sorted
            ]
        )

    def _induce_method_from_descending(self, method_name, **kwargs):
        new_factors = [
            getattr(j, method_name)(**kwargs) if hasattr(j, method_name) else j
            for j in self.factors_sorted
        ]
        return abstract_differential_form_monomial(new_factors)

    def _latex(self, printer=None, raw=True, **kwargs):
        """
        LaTeX representation
        """
        # Handle the leading degree 0 factor (coefficient)
        coeff0 = self.factors_sorted[0]
        if isinstance(coeff0, abstract_differential_form_atom):
            coeff_inner = coeff0.coeff
        else:
            coeff_inner = coeff0
        if isinstance(coeff_inner, zero_form_atom):
            if coeff_inner.is_one:
                coeff_latex = ""
            elif ((-1) * coeff_inner).is_one:
                coeff_latex = "-"
            elif coeff_inner.is_zero:
                coeff_latex = "0"
            else:
                coeff_latex = _routed_latex(coeff0)
        elif isinstance(coeff_inner, zero_form_class):
            if coeff_inner.is_one:
                coeff_latex = ""
            elif (-1 * coeff_inner).is_one:
                coeff_latex = "-"
            elif coeff_inner.is_zero:
                coeff_latex = "0"
            else:
                coeff_latex = _routed_latex(coeff0)
                if isinstance(coeff_inner.base, tuple) and coeff_inner.base[0] in {
                    "sub",
                    "add",
                }:
                    coeff_latex = f"\\left({coeff_latex}\\right)"
        else:
            if _scalar_is_one(coeff_inner):
                coeff_latex = ""
            elif _scalar_is_minus_one(coeff_inner):
                coeff_latex = "-"
            elif _scalar_is_zero(coeff_inner):
                coeff_latex = "0"
            else:
                coeff_latex = _routed_latex(coeff0)

        # Join the remaining factors using '\wedge'
        if len(self.factors_sorted) > 1:
            # Generate LaTeX for all non-coefficient factors
            factors_latex = " \\wedge ".join(
                _routed_latex(factor) for factor in self.factors_sorted[1:]
            )

            # Combine coefficient and factors, but omit '\cdot' if coeff_latex is empty or "-"
            if coeff_latex in ["", "-"]:
                return f"{coeff_latex}{factors_latex}"
            else:
                return f"{coeff_latex} \\cdot {factors_latex}"
        else:
            # Only the degree 0 factor (no other factors to join)
            if coeff_latex == "":
                return "1"
            elif coeff_latex == "-":
                return "-1"
            return coeff_latex

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex() if raw else f"${self._latex()}$"

    def __repr__(self):
        """
        String representation for abstract_DF_monomial.
        """
        # Handle the coefficient (first factor)
        coeff0 = self.factors_sorted[0]
        if isinstance(coeff0, abstract_differential_form_atom):
            coeff_inner = coeff0.coeff
        else:
            coeff_inner = coeff0
        if isinstance(coeff_inner, zero_form_atom):
            coeff_str = coeff0.__str__()
        elif isinstance(coeff_inner, zero_form_class):
            coeff_str = coeff0.__str__()
            if isinstance(coeff_inner.base, tuple) and coeff_inner.base[0] in {
                "sub",
                "add",
            }:
                coeff_str = f"({coeff_str})"
        else:
            coeff_str = str(coeff0)
        # Join the other factors using '*'
        if len(self.factors_sorted) > 1:
            factors_str = "*".join(str(factor) for factor in self.factors_sorted[1:])

            # Combine coefficient and factors, but omit '*' if coeff_str is empty or "-"
            if coeff_str in ["", "-"]:
                return f"{coeff_str}{factors_str}"
            else:
                return f"{coeff_str}*{factors_str}"
        else:
            # Only the degree 0 factor (no other factors to join)
            return coeff_str

    def __str__(self):
        """Fallback to the string representation."""
        return self.__repr__()

    def __mul__(self, other):
        """Handle left multiplication."""
        if isinstance(other, abstract_differential_form_monomial):
            # Combine factors of both monomials
            return abstract_differential_form_monomial(
                self.factors_sorted + other.factors_sorted
            )
        elif isinstance(other, abstract_differential_form_atom):
            # Append the atom as a factor to this monomial
            return abstract_differential_form_monomial(self.factors_sorted + [other])
        elif isinstance(other, (zero_form_class, zero_form_atom)) or isinstance(
            other, expr_numeric_types()
        ):
            # Scalar multiplication (prepend as an atom with degree 0)
            other_sympy = as_engine_scalar(other)
            return abstract_differential_form_monomial(
                [abstract_differential_form_atom(other_sympy, 0)] + self.factors_sorted
            )
        elif isinstance(other, (zero_form_class, zero_form_atom)):
            return abstract_differential_form_monomial(
                [abstract_differential_form_atom(other, 0)] + self.factors_sorted
            )
        else:
            # Allow Python to try __rmul__ of the other operand
            return NotImplemented

    def __rmul__(self, other):
        """Handle right multiplication (symmetrically supports scalar * abstract_DF_monomial)."""
        if isinstance(other, expr_numeric_types()):
            other_sympy = as_engine_scalar(other)
            return abstract_differential_form_monomial(
                [abstract_differential_form_atom(other_sympy, 0)] + self.factors_sorted
            )
        elif isinstance(other, (zero_form_class, zero_form_atom)):
            return abstract_differential_form_monomial(
                [abstract_differential_form_atom(other, 0)] + self.factors_sorted
            )
        else:
            raise TypeError(
                f"Unsupported operand type(s) for *: '{type(other).__name__}' and 'abstract_DF_monomial'"
            )

    def __truediv__(self, other):
        if isinstance(other, abstract_differential_form_atom) and other.degree == 0:
            if other.label is None or other.label == "":
                other = other.coeff
            else:
                other = other.coeff * zero_form_atom(
                    other.label, _markers=other._markers
                )
        if isinstance(other, (zero_form_class, zero_form_atom)) or isinstance(
            other, expr_numeric_types()
        ):
            if _scalar_is_zero(other):
                raise ZeroDivisionError("division by zero")
            return (1 / other) * self
        return NotImplemented

    def __neg__(self):
        return -1 * self

    def __add__(self, other):
        """Addition with another monomial or atom returns an abstract_DF."""
        if other is None:
            return self
        if isinstance(other, expr_numeric_types()):
            other = zero_form_class(other)
        if isinstance(
            other,
            (abstract_differential_form_monomial, abstract_differential_form_atom),
        ):
            return abstract_differential_form([self, other])
        elif isinstance(other, abstract_differential_form):
            return abstract_differential_form((self,) + tuple(other.terms))
        elif isinstance(other, (zero_form_class, zero_form_atom)):
            return abstract_differential_form(
                [self, abstract_differential_form_atom(other, 0)]
            )
        elif isinstance(other, expr_numeric_types()):
            other_sympy = as_engine_scalar(other)
            return abstract_differential_form(
                [self, abstract_differential_form_atom(other_sympy, 0)]
            )
        else:
            raise TypeError(
                "Unsupported operand type for + with `abstract_DF_monomial`"
            )

    def __sub__(self, other):
        """Subtraction with another monomial or atom returns an abstract_DF."""
        if other is None:
            return self
        if isinstance(other, expr_numeric_types()):
            other = zero_form_class(other)
        elif isinstance(
            other,
            (abstract_differential_form_monomial, abstract_differential_form_atom),
        ):
            return abstract_differential_form([self, -1 * other])
        elif isinstance(other, abstract_differential_form):
            negated_terms = tuple([-1 * term for term in other.terms])
            return abstract_differential_form([self] + negated_terms)
        elif isinstance(other, (zero_form_class, zero_form_atom)):
            return abstract_differential_form(
                [self, abstract_differential_form_atom(-1 * other, 0)]
            )
        elif isinstance(other, expr_numeric_types()):
            other_sympy = -1 * as_engine_scalar(other)
            return abstract_differential_form(
                [self, abstract_differential_form_atom(other_sympy, 0)]
            )
        else:
            raise TypeError(
                "Unsupported operand type for - with `abstract_DF_monomial`"
            )

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return other + (-self)

    @property
    def free_symbols(self):
        var_set = set()
        for factor in self.factors_sorted:
            if hasattr(factor, "free_symbols"):
                var_set |= factor.free_symbols
        return var_set


class abstract_differential_form(_coefficient_mapped, dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "abstract_DF"
    _dgcv_categories = {"abstract_DF", "eds_form"}

    def __dgcv_solve_bridge__(self, bridge):
        lowered = []
        for term in self.terms:
            lowered += bridge.lower(term)
        return lowered

    @property
    def __dgcv_zero_obstr__(self):
        return _zero_obstruction(self)

    def __init__(self, terms):
        # Validate terms input
        if not isinstance(terms, (list, tuple)):
            raise TypeError("`abstract_DF` expects `terms` to be a list or tuple")
        if not all(
            isinstance(
                elem,
                (abstract_differential_form_monomial, abstract_differential_form_atom),
            )
            for elem in terms
        ):
            raise TypeError(
                "`abstract_DF` expects `terms` to be a list of `abstract_DF_monomial` or `abstract_DF_atom`"
            )

        def process_abstDF(elem):
            """
            Check that all elements are abstract_DF_monomial/Atom instances.
            """
            if isinstance(elem, abstract_differential_form_monomial):
                return elem
            elif isinstance(elem, abstract_differential_form_atom):
                return abstract_differential_form_monomial([elem])
            else:
                raise TypeError(
                    "`abstract_DF` initializer expects `abstract_DF_monomial` or `abstract_DF_atom` instances"
                )

        # Process terms into abstract_DF_monomial instances
        processed_terms = [
            process_abstDF(term) for term in terms if not process_abstDF(term).is_zero
        ]
        # Handle empty terms: default to trivial zero form
        if not terms:
            terms = [abstract_differential_form_atom(0, 0)]

        # Simplify terms by combining like terms and removing zeros
        collected_terms = tuple(self.simplify_terms(processed_terms))
        if len(collected_terms) == 0:
            collected_terms = (
                abstract_differential_form_monomial(
                    [abstract_differential_form_atom(0, 0)]
                ),
            )
        self.terms = collected_terms
        self.degree = (
            self.terms[0].degree
            if all(j.degree == self.terms[0].degree for j in self.terms)
            else None
        )

    def _sage_(self):
        raise AttributeError

    def simplify_terms(self, terms):
        """
        Simplify a list of terms by combining like terms and removing zero terms.
        Terms with "<None>" in their `str_ids` will not be combined.
        """
        term_dict = {}

        for term in terms:
            # Use str_ids as the key for grouping like terms
            key = tuple(term.str_ids)

            # Skip combining terms that contain "<None>" in their key
            if "<None>" in key:
                # Treat these terms individually
                term_dict[id(term)] = {
                    "coeff": term.factors_sorted[0].coeff,
                    "tail": term.factors_sorted[1:],
                }
                continue

            # Extract the leading coefficient and trailing factors
            coeff = term.factors_sorted[0].coeff  # Leading coefficient
            tail = term.factors_sorted[
                1:
            ]  # Trailing factors (actual abstract_DF_atom instances)

            # Combine coefficients for like terms
            if key in term_dict:
                term_dict[key]["coeff"] = coeff + term_dict[key]["coeff"]
            else:
                term_dict[key] = {"coeff": coeff, "tail": tail}

        # Rebuild simplified terms list
        simplified_terms = [
            abstract_differential_form_monomial(
                [abstract_differential_form_atom(data["coeff"], 0)] + data["tail"]
            )
            for key, data in term_dict.items()
            if not _scalar_is_zero(data["coeff"])
        ]

        if not simplified_terms:
            return [
                abstract_differential_form_monomial(
                    [abstract_differential_form_atom(0, 0)]
                )
            ]

        # Return the simplified, sorted terms
        return tuple(sorted(simplified_terms, key=lambda t: t.factors_sorted))

    @property
    def coeffs(self):
        coeff_list = []
        for term in self.terms:
            if hasattr(term, "factors_sorted") and len(term.factors_sorted) > 0:
                coeff_list += [term.factors_sorted[0]]
        return coeff_list

    @property
    def is_zero(self):
        return all(term.is_zero for term in self.terms)

    def __eq__(self, other):
        """
        Check equality of two abstract_DF instances.
        """
        if not isinstance(other, abstract_differential_form):
            return NotImplemented
        return self.terms == other.terms and self.degree == other.degree

    def __hash__(self):
        """
        Hash the abstract_DF instance based on its terms.
        """
        return hash((self.terms, self.degree))

    def sort_key(self, order=None):  # for the sympy sorting.py default_sort_key
        return (4, self.degree, self.terms)  # 4 is to group with function-like objects

    def _eval_conjugate(self):
        return abstract_differential_form([j._eval_conjugate() for j in self.terms])

    def _eval_simplify(
        self,
        ratio=None,
        measure=None,
        inverse=True,
        doit=True,
        rational=True,
        expand=False,
        **kwargs,
    ):
        return abstract_differential_form([j._eval_simplify() for j in self.terms])

    def _eval_canonicalize(self, depth=1000):
        def _canon(obj):
            if hasattr(obj, "_eval_canonicalize"):
                return obj._eval_canonicalize(depth=depth)
            return obj

        return abstract_differential_form([_canon(j) for j in self.terms])

    def _subs_dgcv(self, data, with_diff_corollaries=False):
        # an alias for regular subs so that other functions can know the with_diff_corollaries keyword is available
        return self.subs(data, with_diff_corollaries=with_diff_corollaries)

    def subs(self, subs_data, with_diff_corollaries=False):
        return abstract_differential_form(
            [
                j._subs_dgcv(subs_data, with_diff_corollaries=with_diff_corollaries)
                for j in self.terms
            ]
        )

    def _map_coefficients(self, fn):
        return abstract_differential_form(
            [
                j._map_coefficients(fn) if hasattr(j, "_map_coefficients") else j
                for j in self.terms
            ]
        )

    def _induce_method_from_descending(self, method_name, **kwargs):
        new_terms = [
            getattr(j, method_name)(**kwargs) if hasattr(j, method_name) else j
            for j in self.terms
        ]
        return abstract_differential_form(new_terms)

    def __add__(self, other):
        if isinstance(other, expr_numeric_types()):
            other = zero_form_class(other)
        if isinstance(other, (zero_form_class, zero_form_atom)):
            other = abstract_differential_form_atom(other, 0)
        elif isinstance(other, expr_numeric_types()):
            other = abstract_differential_form_atom(as_engine_scalar(other), 0)
        if other is None:
            return self
        elif isinstance(other, abstract_differential_form):
            return abstract_differential_form(self.terms + other.terms)
        elif isinstance(
            other,
            (abstract_differential_form_monomial, abstract_differential_form_atom),
        ):
            return abstract_differential_form(self.terms + (other,))
        else:
            raise TypeError("Unsupported operand type for + with `abstract_DF`")

    def __sub__(self, other):
        if isinstance(other, expr_numeric_types()):
            other = zero_form_class(other)
        if isinstance(other, (zero_form_class, zero_form_atom)):
            other = abstract_differential_form_atom(other, 0)
        elif isinstance(other, expr_numeric_types()):
            other = abstract_differential_form_atom(as_engine_scalar(other), 0)
        if other is None:
            return self
        elif isinstance(other, abstract_differential_form):
            negated_terms = tuple([-1 * term for term in other.terms])
            return abstract_differential_form(self.terms + negated_terms)
        elif isinstance(
            other,
            (abstract_differential_form_monomial, abstract_differential_form_atom),
        ):
            return self + (-1 * other)
        else:
            raise TypeError("Unsupported operand type for - with `abstract_DF`")

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if isinstance(other, (zero_form_class, zero_form_atom)):
            other = abstract_differential_form_atom(other, 0)
        elif isinstance(other, expr_numeric_types()):
            other = abstract_differential_form_atom(as_engine_scalar(other), 0)
        if isinstance(other, expr_numeric_types()):
            # Scalar multiplication
            return abstract_differential_form([term * other for term in self.terms])
        if isinstance(other, abstract_differential_form):
            # Distribute over terms
            new_terms = [t1 * t2 for t1 in self.terms for t2 in other.terms]
            return abstract_differential_form(new_terms)
        if isinstance(other, abstract_differential_form_atom):
            other = abstract_differential_form_monomial([other])
        if isinstance(other, abstract_differential_form_monomial):
            # Multiply each term by the monomial
            return abstract_differential_form([term * other for term in self.terms])
        else:
            NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (zero_form_class, zero_form_atom)):
            other = abstract_differential_form_atom(other, 0)
        elif isinstance(other, expr_numeric_types()):
            other = abstract_differential_form_atom(as_engine_scalar(other), 0)
        if isinstance(other, expr_numeric_types()):
            return self * other
        if isinstance(other, abstract_differential_form_atom):
            other = abstract_differential_form_monomial([other])
        if isinstance(other, abstract_differential_form_monomial):
            other = abstract_differential_form([other])
        if isinstance(other, abstract_differential_form):
            return other.__mul__(self)
        else:
            NotImplemented

    def __neg__(self):
        return -1 * self

    def __repr__(self):
        """String representation for abstract_DF."""
        if len(self.terms) == 1:
            return repr(self.terms[0])

        # Build the string for terms
        terms_repr = [repr(term) for term in self.terms]

        result = terms_repr[0] if len(terms_repr) > 0 else ""
        for term in terms_repr[1:]:
            if term.startswith("-"):
                result += (
                    f" - {term[1:]}"  # Add space before "-" and strip the leading "-"
                )
            else:
                result += f" + {term}"  # Add "+" before positive terms
        return result

    def _latex(self, printer=None, raw=True, **kwargs):
        """LaTeX representation for SymPy's LaTeX printer."""
        if len(self.terms) == 1:
            return _routed_latex(self.terms[0])

        # Build the LaTeX string for terms
        terms_latex = [_routed_latex(term) for term in self.terms]

        result = terms_latex[0] if len(terms_latex) > 0 else ""
        for term in terms_latex[1:]:
            if term.startswith("-"):
                result += (
                    " - " + term[1:]
                )  # Add space before "-" and strip the leading "-"
            else:
                result += " + " + term  # Add "+" before positive terms
        return result

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex() if raw else f"${self._latex()}$"

    def __str__(self):
        return self.__repr__()

    @property
    def free_symbols(self):
        var_set = set()
        for term in self.terms:
            if hasattr(term, "free_symbols"):
                var_set |= term.free_symbols
        return var_set


# old dgcv version support stuff
for _cls in (
    abstract_differential_form_atom,
    abstract_differential_form_monomial,
    abstract_differential_form,
):
    register_legacy_sympy_class(_cls)

abstract_DF_atom = abstract_differential_form_atom
abstract_DF_monomial = abstract_differential_form_monomial
abstract_DF = abstract_differential_form
abstDFAtom = abstract_differential_form_atom
abstDFMonom = abstract_differential_form_monomial
