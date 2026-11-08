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
from .._aux._utilities._config import dgcv_warning, get_variable_registry
from .._aux._vmf._safeguards import (
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_passkey,
)
from .._aux.printing.printing._eds import (
    bar_label_latex,
    conjugation_prefix,
    ext_der_latex,
    ext_der_repr,
    ext_der_str,
)
from ..core.base import dgcv_class
from ..core.combinatorics.combinatorics import weightedPermSign
from ..core.tensors import tensorProduct
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
        return self._eval_simplify(**kwargs)

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


def _check_wedge_forms(forms):
    for form in forms:
        if not isinstance(form, abstract_differential_form_atom) or form.degree == 0:
            raise TypeError(
                "`coefficient` expects basis form atoms of positive degree as its arguments"
            )
        if form.label is not None and not _scalar_is_one(form.coeff):
            raise ValueError(
                f"`coefficient` expects unit basis forms; received the scaled form {form}"
            )


def _check_modulo_forms(forms):
    for form in forms:
        if (
            not isinstance(form, abstract_differential_form_atom)
            or form.degree != 1
            or not _scalar_is_one(form.coeff)
            or form.primary_coframe is None
        ):
            raise TypeError(f"`modulo` expects unit 1-forms of a coframe; received {form!r}")


def _wedge_coefficient(forms, tail, scalar):
    if len(forms) != len(tail) or len(set(forms)) != len(forms):
        return 0
    places = []
    for form in forms:
        for place, factor in enumerate(tail):
            if factor == form:
                places.append(place)
                break
        else:
            return 0
    sign = weightedPermSign(
        [SortableObj((form, place)) for form, place in zip(forms, places)],
        list(forms),
        use_degree_attribute=True,
    )
    return scalar if sign == 1 else -scalar


def _bare_scalar(value):
    if not isinstance(value, zero_form_class):
        return value
    base = value._base
    if base is None:
        symbols = value.free_symbols
        if len(symbols) == 1:
            atom = next(iter(symbols))
            if isinstance(atom, zero_form_atom) and value == atom:
                return atom
        return value
    return base if isinstance(base, zero_form_atom) else value


class abstract_differential_form_atom(_coefficient_mapped, dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "abstDFAtom"
    _dgcv_categories = {"abstDFAtom", "eds_form"}

    def __dgcv_solve_bridge__(self, bridge):
        bridge.note_eds_forms()
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
        self._hash = None

        self.coeffs = [coeff]

    @property
    def is_zero(self):
        return _scalar_is_zero(self._coeff) or self._is_formal_zero()

    def _is_formal_zero(self):
        if self.label is not None or not self.ext_deriv_order:
            return False
        coeff = self._coeff
        if isinstance(coeff, (zero_form_atom, zero_form_class)):
            return not coeff.free_symbols
        return not get_free_symbols(as_engine_scalar(coeff))

    @property
    def primary_coframe(self):
        if self.ext_deriv_order or not isinstance(self.label, str):
            return None
        for marker in self._markers:
            if not isinstance(marker, str):
                return marker if self in marker.forms else None
        registry = get_variable_registry()
        info = registry["paths"].get(self.label)
        if info is None:
            return None
        kind = info.get("kind")
        if kind not in (
            "differential_form",
            "differential_form_conjugate",
            "zero_form",
            "zero_form_conjugate",
        ):
            return None
        system = registry["eds"]["atoms"].get(info["path"][1])
        coframe = None if system is None else system.get("primary_coframe")
        if coframe is None:
            return None
        if kind.startswith("differential_form") and self not in coframe.forms:
            return None
        return coframe

    def _coframe_slot(self, name):
        coframe = self.primary_coframe
        if coframe is None:
            return None
        items = getattr(coframe, name, None)
        if items is None:
            return None
        return items[coframe.forms.index(self)]

    @property
    def dual_vector_field(self):
        return self._coframe_slot("dual_frame_in_coordinates")

    @property
    def form_in_coordinates(self):
        return self._coframe_slot("forms_in_coordinates")

    @property
    def parameters(self):
        coframe = self.primary_coframe
        return frozenset() if coframe is None else coframe.parameters

    @property
    def source_form(self):
        if not self.ext_deriv_order:
            return self
        if self.label is None:
            return abstract_differential_form_atom(self.coeff, 0, _markers=self._markers)
        return abstract_differential_form_atom(
            1, self.degree - self.ext_deriv_order, self.label, 0, self._markers
        )

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
        h = self._hash
        if h is None:
            h = hash((self.coeff, self.degree, self.label, self.ext_deriv_order))
            self._hash = h
        return h

    def __reduce__(self):
        coeff = self.coeff
        carried = not isinstance(coeff, (zero_form_class, zero_form_atom))
        state = {key: value for key, value in self.__dict__.items() if key != "_hash"}
        return (
            abstract_differential_form_atom._from_pickle,
            (coeff if carried else None, carried, self.degree, self.label, self.ext_deriv_order),
            state,
        )

    @classmethod
    def _from_pickle(cls, coeff, carried, degree, label, ext_deriv_order):
        obj = object.__new__(cls)
        obj._hash = hash((coeff, degree, label, ext_deriv_order)) if carried else None
        return obj

    def _paired_form(self):
        if self.ext_deriv_order or not isinstance(self.label, str):
            return None
        coframe = next((m for m in self._markers if not isinstance(m, str)), None)
        if coframe is None:
            return None
        rules = coframe.conj_rules
        for index, form in enumerate(coframe.forms):
            if form.label == self.label:
                target = rules.get(index, index)
                return None if target == index else coframe.forms[target]
        return None

    def _eval_conjugate(self):
        partner = self._paired_form()
        if partner is not None:
            return abstract_differential_form_atom(
                _custom_conj(self.coeff),
                self.degree,
                label=partner.label,
                ext_deriv_order=self.ext_deriv_order,
                _markers=partner._markers,
            )
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

        if self._is_formal_zero():
            return "0"
        if isinstance(self.coeff, (zero_form_atom, zero_form_class)):
            coeff_str = _zf_coeff_str(self.coeff, self.coeff.__repr__())
            return ext_der_repr(
                f"{coeff_str}*{self.label}" if self.label else coeff_str,
                self.ext_deriv_order,
            )
        coeff_sympy = as_engine_scalar(self.coeff)
        if _scalar_is_one(coeff_sympy):
            return ext_der_repr(str(self.label), self.ext_deriv_order) if self.label else "1"
        elif _scalar_is_minus_one(coeff_sympy):
            return ext_der_repr(f"-{self.label}", self.ext_deriv_order) if self.label else "-1"
        else:
            # Wrap in parentheses if there are multiple terms
            coeff_str = (
                f"({coeff_sympy})"
                if op_expr(coeff_sympy) == OP_ADD
                else str(coeff_sympy)
            )
            return (
                ext_der_repr(f"{coeff_str}*{self.label}", self.ext_deriv_order)
                if self.label
                else ext_der_repr(coeff_str, self.ext_deriv_order)
            )

    def _latex(self, printer=None, raw=True, **kwargs):
        """LaTeX representation for abstract_DF_atom."""

        if self._is_formal_zero():
            return "0"
        if isinstance(self.coeff, (zero_form_atom, zero_form_class)):
            coeff_latex = _routed_latex(self.coeff)
            return ext_der_latex(
                f"{coeff_latex}{bar_label_latex(self.label)}"
                if self.label
                else coeff_latex,
                self.ext_deriv_order,
            )
        coeff_sympy = as_engine_scalar(self.coeff)
        if _scalar_is_one(coeff_sympy):
            return (
                ext_der_latex(bar_label_latex(self.label), self.ext_deriv_order)
                if self.label
                else "1"
            )
        elif _scalar_is_minus_one(coeff_sympy):
            return (
                ext_der_latex(f"-{bar_label_latex(self.label)}", self.ext_deriv_order)
                if self.label
                else "-1"
            )
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
        if self._is_formal_zero():
            return "0"
        if isinstance(self.coeff, (zero_form_atom, zero_form_class)):
            coeff_str = _zf_coeff_str(self.coeff, self.coeff.__repr__())
            return ext_der_str(
                f"{coeff_str}*{self.label}" if self.label else coeff_str,
                self.ext_deriv_order,
            )
        coeff_sympy = as_engine_scalar(self.coeff)
        if _scalar_is_one(coeff_sympy):
            return ext_der_str(str(self.label), self.ext_deriv_order) if self.label else "1"
        elif _scalar_is_minus_one(coeff_sympy):
            return ext_der_str(f"-{self.label}", self.ext_deriv_order) if self.label else "-1"
        else:
            # Wrap in parentheses if there are multiple terms
            coeff_str = (
                f"({coeff_sympy})"
                if op_expr(coeff_sympy) == OP_ADD
                else str(coeff_sympy)
            )
            return (
                ext_der_str(f"{coeff_str}*{self.label}", self.ext_deriv_order)
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
        constraints = kwargs.pop("constraints", None)
        with_conjugates = kwargs.pop("with_conjugates", True)
        if constraints is not None:
            return _simplify_under_constraints(self, constraints, with_conjugates, **kwargs)
        return abstract_differential_form_atom(
            _routed_simplify(self.coeff, **kwargs),
            self.degree,
            self.label,
            ext_deriv_order=self.ext_deriv_order,
            _markers=self._markers,
        )

    def _eval_canonicalize(self, depth=1000):
        if hasattr(self.coeff, "_eval_canonicalize"):
            new_coeff = self.coeff._eval_canonicalize(depth=depth)
        else:
            new_coeff = self.coeff
        if new_coeff is self.coeff:
            return self
        return abstract_differential_form_atom(
            new_coeff,
            self.degree,
            self.label,
            ext_deriv_order=self.ext_deriv_order,
            _markers=self._markers,
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

    def _subs_dgcv(self, data, with_diff_corollaries=False, with_conjugates=None):
        # an alias for regular subs so that other functions can know the with_diff_corollaries keyword is available
        return self.subs(
            data, with_diff_corollaries=with_diff_corollaries, with_conjugates=with_conjugates
        )

    def subs(self, subs_data, with_diff_corollaries=False, with_conjugates=None):
        if isinstance(subs_data, (list, tuple)) and all(
            isinstance(j, tuple) and len(j) == 2 for j in subs_data
        ):
            l1 = len(subs_data)
            subs_data = dict(subs_data)
            if len(subs_data) < l1:
                dgcv_warning(
                    "Provided substitution rules had repeat keys, and only one was used."
                )
        close = with_diff_corollaries if with_conjugates is None else with_conjugates
        if close and isinstance(subs_data, dict):
            subs_data = _conjugate_closure(subs_data, "subs")
        if self in subs_data:
            return subs_data[self]
        new_coeff = None
        if isinstance(self.coeff, (zero_form_atom, zero_form_class)):
            new_coeff = (self.coeff).subs(
                subs_data,
                with_diff_corollaries=with_diff_corollaries,
                with_conjugates=False,
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
        return set(get_free_symbols(self.coeff))

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

    def coefficient(self, *forms):
        _check_wedge_forms(forms)
        if self.is_zero:
            return 0
        if self.degree == 0:
            return _bare_scalar(self.coeff) if not forms else 0
        if self.label is None and self.ext_deriv_order:
            unit, scalar = self, 1
        else:
            unit, scalar = self._seperated_form
        return _bare_scalar(_wedge_coefficient(forms, (unit,), scalar))

    def coefficients(self):
        if self.is_zero:
            return {}
        if self.degree == 0:
            return {(): _bare_scalar(self.coeff)}
        if self.label is None and self.ext_deriv_order:
            return {(self,): 1}
        unit, scalar = self._seperated_form
        return {(unit,): _bare_scalar(scalar)}

    def modulo(self, *one_forms):
        _check_modulo_forms(one_forms)
        if self.degree == 0 or (self.label is None and self.ext_deriv_order):
            return self
        if self._seperated_form[0] in one_forms:
            return abstract_differential_form([])
        return self

    def _convert_to_tp(self):
        return _form_tp(self)

    def __call__(self, *vectors):
        return _evaluate_on_vectors(self, vectors)

    def dual(self):
        return _form_dual(self)

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
                self.coeff * other,
                self.degree,
                self.label,
                ext_deriv_order=self.ext_deriv_order,
                _markers=self._markers,
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
                self.coeff * other,
                self.degree,
                self.label,
                ext_deriv_order=self.ext_deriv_order,
                _markers=self._markers,
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
        return NotImplemented

    def __sub__(self, other):
        """Subtraction with another atom or monomial returns an abstract_DF."""
        if other is None:
            return self
        if isinstance(other, expr_numeric_types()):
            other = zero_form_class(other)
        if isinstance(
            other,
            (abstract_differential_form_monomial, abstract_differential_form_atom),
        ):
            return abstract_differential_form(
                [abstract_differential_form_monomial([self]), -1 * other]
            )
        if isinstance(other, abstract_differential_form):
            negated_terms = [-1 * term for term in other.terms]
            return abstract_differential_form(
                [abstract_differential_form_monomial([self])] + negated_terms
            )
        if isinstance(other, (zero_form_atom, zero_form_class)):
            return abstract_differential_form(
                [
                    abstract_differential_form_monomial([self]),
                    abstract_differential_form_monomial(
                        [abstract_differential_form_atom(-1 * other, 0)]
                    ),
                ]
            )
        return NotImplemented

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
        bridge.note_eds_forms()
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
            if j.value.label is None and j.value.ext_deriv_order and j.value.degree:
                if j.value.is_zero:
                    coeffFactor = 0
                else:
                    new_objs.append(j.value)
                continue
            if not _scalar_is_zero(coeffFactor):
                coeffFactor = (
                    coeffFactor * j.value.coeff
                    if not _scalar_is_zero(j.value.coeff)
                    else 0
                )
            if j.place[0] != 0:
                if _scalar_is_one(j.value.coeff):
                    new_objs.append(j.value)
                else:
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

    @classmethod
    def _from_sorted(cls, coeff, tail):
        if _scalar_is_zero(coeff):
            return cls([abstract_differential_form_atom(0, 0)])
        obj = object.__new__(cls)
        factors = [abstract_differential_form_atom(coeff, 0)] + list(tail)
        obj.factors_sorted = factors
        obj._coeff = factors[0].coeff
        obj.factors = factors
        obj.str_ids = tuple(
            "<Coeff>"
            if i == 0 and factor.label is None
            else (factor.label if factor.label is not None else "<None>")
            for i, factor in enumerate(factors)
        )
        obj.degree = sum(factor.degree for factor in tail)
        return obj

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

    def coefficient(self, *forms):
        _check_wedge_forms(forms)
        if self.is_zero:
            return 0
        return _bare_scalar(
            _wedge_coefficient(
                forms, self.factors_sorted[1:], self.factors_sorted[0].coeff
            )
        )

    def coefficients(self):
        if self.is_zero:
            return {}
        return {
            tuple(self.factors_sorted[1:]): _bare_scalar(self.factors_sorted[0].coeff)
        }

    def modulo(self, *one_forms):
        _check_modulo_forms(one_forms)
        if any(factor in one_forms for factor in self.factors_sorted[1:]):
            return abstract_differential_form([])
        return self

    def _convert_to_tp(self):
        return _form_tp(self)

    def __call__(self, *vectors):
        return _evaluate_on_vectors(self, vectors)

    def dual(self):
        return _form_dual(self)

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
        constraints = kwargs.pop("constraints", None)
        with_conjugates = kwargs.pop("with_conjugates", True)
        if constraints is not None:
            return _simplify_under_constraints(self, constraints, with_conjugates, **kwargs)
        return abstract_differential_form_monomial(
            [j._eval_simplify(**kwargs) for j in self.factors]
        )

    def _eval_canonicalize(self, depth=1000):
        def _canon(obj):
            if hasattr(obj, "_eval_canonicalize"):
                return obj._eval_canonicalize(depth=depth)
            return obj

        new_factors = [_canon(j) for j in self.factors]
        if all(new is old for new, old in zip(new_factors, self.factors)):
            return self
        return abstract_differential_form_monomial(new_factors)

    def _subs_dgcv(self, data, with_diff_corollaries=False, with_conjugates=None):
        # an alias for regular subs so that other functions can know the with_diff_corollaries keyword is available
        return self.subs(
            data, with_diff_corollaries=with_diff_corollaries, with_conjugates=with_conjugates
        )

    def subs(self, subs_data, with_diff_corollaries=False, with_conjugates=None):
        close = with_diff_corollaries if with_conjugates is None else with_conjugates
        if close and isinstance(subs_data, dict):
            subs_data = _conjugate_closure(subs_data, "subs")
        return abstract_differential_form_monomial(
            [
                j._subs_dgcv(
                    subs_data,
                    with_diff_corollaries=with_diff_corollaries,
                    with_conjugates=False,
                )
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
        if isinstance(other, abstract_differential_form):
            return abstract_differential_form((self,) + tuple(other.terms))
        if isinstance(other, (zero_form_class, zero_form_atom)):
            return abstract_differential_form(
                [self, abstract_differential_form_atom(other, 0)]
            )
        return NotImplemented

    def __sub__(self, other):
        """Subtraction with another monomial or atom returns an abstract_DF."""
        if other is None:
            return self
        if isinstance(other, expr_numeric_types()):
            other = zero_form_class(other)
        if isinstance(
            other,
            (abstract_differential_form_monomial, abstract_differential_form_atom),
        ):
            return abstract_differential_form([self, -1 * other])
        if isinstance(other, abstract_differential_form):
            negated_terms = [-1 * term for term in other.terms]
            return abstract_differential_form([self] + negated_terms)
        if isinstance(other, (zero_form_class, zero_form_atom)):
            return abstract_differential_form(
                [self, abstract_differential_form_atom(-1 * other, 0)]
            )
        return NotImplemented

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
        bridge.note_eds_forms()
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
            term for term in map(process_abstDF, terms) if not term.is_zero
        ]
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
        self._hash = None
        self._coefficient_key = None

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
            abstract_differential_form_monomial._from_sorted(data["coeff"], data["tail"])
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

    def coefficient(self, *forms):
        _check_wedge_forms(forms)
        total = 0
        for term in self.terms:
            if term.is_zero:
                continue
            found = _wedge_coefficient(
                forms, term.factors_sorted[1:], term.factors_sorted[0].coeff
            )
            if not _scalar_is_zero(found):
                total = found if _scalar_is_zero(total) else total + found
        return _bare_scalar(total)

    def coefficients(self):
        out = {}
        for term in self.terms:
            if term.is_zero:
                continue
            key = tuple(term.factors_sorted[1:])
            scalar = term.factors_sorted[0].coeff
            out[key] = scalar if key not in out else out[key] + scalar
        return {key: _bare_scalar(value) for key, value in out.items()}

    def modulo(self, *one_forms):
        _check_modulo_forms(one_forms)
        kept = [
            term
            for term in self.terms
            if not any(factor in one_forms for factor in term.factors_sorted[1:])
        ]
        return self if len(kept) == len(self.terms) else abstract_differential_form(kept)

    def _convert_to_tp(self):
        return _form_tp(self)

    def __call__(self, *vectors):
        return _evaluate_on_vectors(self, vectors)

    def dual(self):
        return _form_dual(self)

    @property
    def is_zero(self):
        return all(term.is_zero for term in self.terms)

    def _comparison_key(self):
        key = self._coefficient_key
        if key is None:
            key = self.coefficients()
            self._coefficient_key = key
        return key

    def __eq__(self, other):
        """
        Check equality of two abstract_DF instances.
        """
        if not isinstance(other, abstract_differential_form):
            return NotImplemented
        if self is other or (self.terms == other.terms and self.degree == other.degree):
            return True
        if self.degree != other.degree:
            return False
        mine, theirs = self._comparison_key(), other._comparison_key()
        return mine.keys() == theirs.keys() and all(
            scalars_equal(mine[key], theirs[key]) for key in mine
        )

    def __hash__(self):
        """
        Hash the abstract_DF instance based on its coefficients.
        """
        h = self._hash
        if h is None:
            key = self._comparison_key()
            h = hash((self.degree, tuple(sorted(hash((k, v)) for k, v in key.items()))))
            self._hash = h
        return h

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
        constraints = kwargs.pop("constraints", None)
        with_conjugates = kwargs.pop("with_conjugates", True)
        if constraints is not None:
            return _simplify_under_constraints(self, constraints, with_conjugates, **kwargs)
        return abstract_differential_form(
            [j._eval_simplify(**kwargs) for j in self.terms]
        )

    def _eval_canonicalize(self, depth=1000):
        def _canon(obj):
            if hasattr(obj, "_eval_canonicalize"):
                return obj._eval_canonicalize(depth=depth)
            return obj

        return abstract_differential_form([_canon(j) for j in self.terms])

    def _subs_dgcv(self, data, with_diff_corollaries=False, with_conjugates=None):
        # an alias for regular subs so that other functions can know the with_diff_corollaries keyword is available
        return self.subs(
            data, with_diff_corollaries=with_diff_corollaries, with_conjugates=with_conjugates
        )

    def subs(self, subs_data, with_diff_corollaries=False, with_conjugates=None):
        close = with_diff_corollaries if with_conjugates is None else with_conjugates
        if close and isinstance(subs_data, dict):
            subs_data = _conjugate_closure(subs_data, "subs")
        return abstract_differential_form(
            [
                j._subs_dgcv(
                    subs_data,
                    with_diff_corollaries=with_diff_corollaries,
                    with_conjugates=False,
                )
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
        return NotImplemented

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
        return NotImplemented

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
        return NotImplemented

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
        return NotImplemented

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


def _unit_frame(units):
    coframes = set()
    for unit in units:
        coframe = unit.primary_coframe
        if coframe is None:
            raise TypeError(f"{unit} belongs to no coframe, so it has no dual frame")
        coframes.add(coframe)
    if len(coframes) > 1:
        raise TypeError(
            "the form mixes 1-forms of two coframes; a pairing with vector fields needs one coframe"
        )
    return next(iter(coframes)).dual() if coframes else None


def _form_from_entries(frame, entries):
    forms = frame.coframe.forms
    terms = []
    for key, value in entries.items():
        if _scalar_is_zero(value):
            continue
        factors = [abstract_differential_form_atom(value, 0)]
        factors.extend(forms[idx] for idx in key)
        terms.append(abstract_differential_form_monomial(factors))
    return abstract_differential_form(terms)


def _form_tp(form):
    entries = form.coefficients()
    frame = _unit_frame([unit for key in entries for unit in key])
    if frame is None:
        return tensorProduct({(): entries.get((), 0)})
    index = {unit: idx for idx, unit in enumerate(frame.coframe.forms)}
    card = frame.card
    return tensorProduct(
        {
            tuple((index[unit], 0, card) for unit in key): value
            for key, value in entries.items()
        },
        shape="skew",
    )


def _form_dual(form):
    if form.degree != 1:
        raise TypeError("`dual` of a form is defined for 1-forms; higher degrees pair through `form(X, ...)`")
    entries = form.coefficients()
    frame = _unit_frame([unit for key in entries for unit in key])
    if frame is None:
        raise TypeError("the 1-form belongs to no coframe, so it has no dual vector field")
    index = {unit: idx for idx, unit in enumerate(frame.coframe.forms)}
    return frame.element({index[key[0]]: value for key, value in entries.items()})


def _permutation_sign(order):
    sign = 1
    seen = [False] * len(order)
    for start in range(len(order)):
        if seen[start]:
            continue
        length = 0
        pos = start
        while not seen[pos]:
            seen[pos] = True
            pos = order[pos]
            length += 1
        if length % 2 == 0:
            sign = -sign
    return sign


def _contract_first_slots(current, steps):
    for step in steps:
        by_index = isinstance(step, numbers.Integral)
        nxt = {}
        for key, c in current.items():
            for pos, idx in enumerate(key):
                if by_index:
                    if idx != step:
                        continue
                    term = c
                else:
                    xi = step.get(idx)
                    if xi is None:
                        continue
                    term = c * xi
                if pos % 2:
                    term = -term
                tail = key[:pos] + key[pos + 1 :]
                nxt[tail] = nxt[tail] + term if tail in nxt else term
        current = nxt
    return current


def _multivector_frame(tensor):
    frames = set()
    for key in tensor.coeff_dict:
        for idx, valence, card in key:
            if valence != 1:
                raise TypeError("a form evaluates on multivectors, not on tensors with covariant factors")
            space = getattr(card, "space", None)
            if not query_dgcv_categories(space, "eds_frame"):
                raise TypeError("a form evaluates on multivectors over an abstract frame")
            frames.add(space)
    if len(frames) != 1:
        raise TypeError("the multivector mixes frames")
    return frames.pop()


def _evaluate_on_multivector(form, tensor):
    if tensor.shape not in ("skew", "all", "general"):
        raise TypeError("a form evaluates on skew or general tensors of vectors, not on a symmetric tensor")
    if tensor.min_degree != tensor.max_degree:
        raise TypeError("a multivector of mixed degree cannot be evaluated")
    frame = _multivector_frame(tensor)
    if tensor.max_degree == 1:
        return _evaluate_on_vectors(form, (frame.element(tensor),))
    if form.is_zero:
        return 0
    degree = form.degree
    if degree is None:
        raise TypeError("a form of mixed degree cannot be evaluated on multivectors")
    p = tensor.max_degree
    if degree == 0 or p == 0:
        return 0
    if p > degree:
        raise TypeError(f"a {degree}-form takes a multivector of degree at most {degree}; received {p}")
    entries = form.coefficients()
    if _unit_frame([unit for key in entries for unit in key]) is not frame:
        raise TypeError("the multivector belongs to another frame than the form's coframe")
    index = {unit: idx for idx, unit in enumerate(frame.coframe.forms)}
    current = {tuple(index[unit] for unit in key): value for key, value in entries.items()}
    total = {}
    for key, weight in tensor.coeff_dict.items():
        if _scalar_is_zero(weight):
            continue
        partial = _contract_first_slots(current, tuple(factor[0] for factor in key))
        for tail, value in partial.items():
            term = weight * value
            total[tail] = total[tail] + term if tail in total else term
    if p == degree:
        return total.get((), 0)
    return _form_from_entries(frame, total)


def _evaluate_on_vectors(form, vectors):
    if not vectors:
        return form
    if len(vectors) == 1 and get_dgcv_category(vectors[0]) == "tensorProduct":
        return _evaluate_on_multivector(form, vectors[0])
    frames = set()
    for vector in vectors:
        if not query_dgcv_categories(vector, "eds_vector"):
            raise TypeError(
                f"a form evaluates on abstract vector fields, not {type(vector).__name__}"
            )
        frames.add(vector.frame)
    if len(frames) != 1:
        raise TypeError("the vector fields belong to different frames")
    frame = frames.pop()
    if form.is_zero:
        return 0
    degree = form.degree
    if degree is None:
        raise TypeError("a form of mixed degree cannot be evaluated on vector fields")
    if len(vectors) > degree:
        raise TypeError(
            f"a {degree}-form takes at most {degree} vector field arguments; received {len(vectors)}"
        )
    if degree == 0:
        return 0
    entries = form.coefficients()
    if _unit_frame([unit for key in entries for unit in key]) is not frame:
        raise TypeError("the vector fields belong to another frame than the form's coframe")
    index = {unit: idx for idx, unit in enumerate(frame.coframe.forms)}
    current = {
        tuple(index[unit] for unit in key): value for key, value in entries.items()
    }
    current = _contract_first_slots(current, [vector.coeff_dict for vector in vectors])
    if len(vectors) == degree:
        return current.get((), 0)
    return _form_from_entries(frame, current)


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
