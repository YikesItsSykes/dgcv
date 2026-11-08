"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

module: dgcv.eds._vectors

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers
from math import factorial

from .._aux._backends._cls_coercion import register_legacy_sympy_class
from .._aux._backends._engine import engine_kind
from .._aux._backends._symbolic_router import (
    _scalar_is_one,
    _scalar_is_zero,
    get_free_symbols,
    is_zero_knowing_zero_is_expected,
)
from .._aux._backends._symbolic_router import simplify as _routed_simplify
from .._aux._backends._types_and_constants import (
    _disposable_symbol,
    _disposable_symbols,
    expr_numeric_types,
    rational,
)
from .._aux._utilities._misc import zip_sum
from .._aux._vmf._safeguards import (
    create_key,
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_passkey,
)
from .._aux.printing.printing import lincomb_latex, lincomb_plain
from .._aux.printing.printing._eds import bar_label_latex
from ..core.base import dgcv_class
from ..core.dgcv_core.spaces.spaces import _vs_card
from ..core.tensors import tensorProduct
from ._atoms import _custom_conj, _zero_obstruction, zero_form_atom
from ._forms import (
    _coefficient_mapped,
    _form_from_entries,
    _permutation_sign,
    _unit_frame,
    abstract_differential_form,
)
from ._zero_forms import zero_form_class


def _is_scalar(obj):
    return isinstance(obj, expr_numeric_types()) or bool(
        query_dgcv_categories(obj, "eds_scalar")
    )


def _is_vector(obj):
    return bool(query_dgcv_categories(obj, "eds_vector"))


def _constant_symbol(name):
    if engine_kind() == "builtin":
        return zero_form_atom(str(name), _markers=frozenset({"constant", "engine_symbol"}))
    return _disposable_symbol(name)


def _subs_scalar(value, data, with_diff_corollaries):
    if isinstance(value, numbers.Number):
        return value
    if not isinstance(value, (zero_form_atom, zero_form_class)):
        value = zero_form_class(value)
    return value.subs(
        data, with_diff_corollaries=with_diff_corollaries, with_conjugates=False
    )


def _vectors_frame(vectors, frame):
    for vector in vectors:
        if not _is_vector(vector):
            raise TypeError(
                f"expected abstract vector fields, received {type(vector).__name__}"
            )
        if vector.frame is not frame:
            raise TypeError("the vector fields must belong to the dual frame of the form's coframe")


class abstract_frame(dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "abst_frame"
    _dgcv_categories = {"abst_frame", "eds_frame"}

    def __init__(self, coframe, basis_labels=None, label=None):
        if get_dgcv_category(coframe) != "abst_coframe":
            raise TypeError("`abstract_frame` expects an abstract coframe")
        dimension = coframe.dimension
        if basis_labels is None:
            basis_labels = [
                f"e_{form.label}" if form.label is not None else f"e_{idx + 1}"
                for idx, form in enumerate(coframe.forms)
            ]
            tex_labels = [
                rf"e_{{{bar_label_latex(form.label)}}}"
                if form.label is not None
                else rf"e_{{{idx + 1}}}"
                for idx, form in enumerate(coframe.forms)
            ]
        else:
            basis_labels = [str(name) for name in basis_labels]
            if len(basis_labels) != dimension:
                raise ValueError(
                    f"a frame dual to a {dimension}-dimensional coframe needs {dimension} labels; received {len(basis_labels)}"
                )
            tex_labels = None
        self.coframe = coframe
        self.label = label
        self.dimension = dimension
        self.basis_labels = basis_labels
        self._tex_basis_labels = tex_labels
        self.grading = [tuple([0] * dimension)]
        self._gradingNumber = 1
        self.card = _vs_card(self)
        self._bracket_rows = {}
        self.basis = tuple(
            abstract_vector_field(self, {idx: 1}) for idx in range(dimension)
        )
        self._dual_labels = None
        self._tex_dual_labels = None

    @property
    def ambient(self):
        return self

    @property
    def dual_basis_labels(self):
        if self._dual_labels is None:
            self._dual_labels = [str(form) for form in self.coframe.forms]
        return self._dual_labels

    @property
    def _tex_dual_basis_labels(self):
        if self._tex_dual_labels is None:
            self._tex_dual_labels = [form._latex(raw=True) for form in self.coframe.forms]
        return self._tex_dual_labels

    @property
    def conj_rules(self):
        return self.coframe.conj_rules

    @property
    def zero_element(self):
        return abstract_vector_field(self, {})

    @property
    def is_zero(self):
        return False

    def _eval_conjugate(self):
        return self

    def __dgcv_conjugate__(self, symbolic=False):
        return self

    def _reset(self):
        self._bracket_rows.clear()

    def dual(self):
        return self.coframe

    def _index(self, item):
        if isinstance(item, numbers.Integral):
            if not -self.dimension <= item < self.dimension:
                raise IndexError(f"frame index {item} out of range for dimension {self.dimension}")
            return item % self.dimension
        if isinstance(item, abstract_vector_field) and item.frame is self:
            if len(item.coeff_dict) == 1:
                idx, value = next(iter(item.coeff_dict.items()))
                if value == 1:
                    return idx
        raise ValueError(f"{item} is not a basis element of this frame")

    def _row(self, i, j):
        key = (i, j)
        row = self._bracket_rows.get(key)
        if row is None:
            coframe = self.coframe
            row = tuple(
                -coframe.coefficient(i, j, of=k) for k in range(self.dimension)
            )
            self._bracket_rows[key] = row
        return row

    def __getitem__(self, indices):
        if isinstance(indices, numbers.Integral):
            return self.basis[indices]
        if isinstance(indices, tuple):
            if len(indices) == 1:
                return self.basis[indices[0]]
            if len(indices) == 2:
                return self._row(self._index(indices[0]), self._index(indices[1]))
            if len(indices) == 3:
                return self._row(self._index(indices[0]), self._index(indices[1]))[
                    self._index(indices[2])
                ]
        raise TypeError("frame indexing takes one, two or three indices")

    def coefficient(self, vector1, vector2, *, of=None):
        if of is None:
            raise TypeError("`coefficient` needs `of=` naming the basis element")
        return self[vector1, vector2, of]

    def __iter__(self):
        return iter(self.basis)

    def __len__(self):
        return self.dimension

    def __contains__(self, item):
        return isinstance(item, abstract_vector_field) and item.frame is self

    def element(self, coeffs):
        if isinstance(coeffs, abstract_vector_field):
            if coeffs.frame is not self:
                raise TypeError("the vector field belongs to another frame")
            return coeffs
        if get_dgcv_category(coeffs) == "tensorProduct":
            entries = {}
            for key, value in coeffs.coeff_dict.items():
                if _scalar_is_zero(value):
                    continue
                if len(key) != 1 or key[0][1] != 1 or key[0][2] is not self.card:
                    raise TypeError("`element` takes a degree-1 vector tensor over this frame")
                entries[key[0][0]] = entries.get(key[0][0], 0) + value
            return abstract_vector_field(self, entries)
        return abstract_vector_field(self, coeffs)

    def form(self, tp):
        if get_dgcv_category(tp) != "tensorProduct":
            raise TypeError("`form` takes a tensorProduct over this frame")
        entries = {}
        skew = tp.shape in ("skew", "all")
        for key, value in tp.coeff_dict.items():
            if _scalar_is_zero(value):
                continue
            for factor in key:
                if factor[1] != 0 or factor[2] is not self.card:
                    raise TypeError("`form` takes a covariant tensor over this frame")
            indices = tuple(factor[0] for factor in key)
            if len(set(indices)) != len(indices):
                continue
            if skew or len(indices) < 2:
                entries[indices] = entries.get(indices, 0) + value
                continue
            order = sorted(range(len(indices)), key=indices.__getitem__)
            sign = _permutation_sign(order)
            sorted_key = tuple(indices[pos] for pos in order)
            entries[sorted_key] = entries.get(sorted_key, 0) + sign * value
        if not skew:
            entries = {
                key: value * rational(1, factorial(len(key))) if len(key) > 1 else value
                for key, value in entries.items()
            }
        if not entries:
            return 0
        if set(entries) == {()}:
            return entries[()]
        return _form_from_entries(self, entries)

    def __repr__(self):
        return f"abstract_frame({', '.join(self.basis_labels)})"

    def __str__(self):
        return self.__repr__()

    def _latex(self, printer=None, raw=True, **kwargs):
        body = ", ".join(e._latex(raw=True) for e in self.basis)
        out = rf"\left\{{{body}\right\}}"
        return out if raw else rf"$\displaystyle {out}$"

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex(raw=raw, **kwargs)


class abstract_vector_field(_coefficient_mapped, dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "abstVF"
    _dgcv_categories = {"abstVF", "eds_vector", "vs_element"}
    valence = 1

    def __init__(self, frame, coeffs):
        if not isinstance(frame, abstract_frame):
            raise TypeError("`abstract_vector_field` expects an abstract frame")
        if isinstance(coeffs, dict):
            items = coeffs.items()
        elif isinstance(coeffs, (list, tuple)):
            items = enumerate(coeffs)
        else:
            raise TypeError("coefficients must be a dict keyed by basis index or a sequence")
        coeff_dict = {}
        for idx, value in items:
            if not isinstance(idx, numbers.Integral) or not 0 <= idx < frame.dimension:
                raise IndexError(f"frame index {idx} out of range for dimension {frame.dimension}")
            if _scalar_is_zero(value):
                continue
            coeff_dict[int(idx)] = value
        self.frame = frame
        self.vectorSpace = frame
        self.card = frame.card
        self.coeff_dict = coeff_dict
        self._hash = None

    @property
    def coeffs(self):
        get = self.coeff_dict.get
        return tuple(get(idx, 0) for idx in range(self.frame.dimension))

    @property
    def ambient_rep(self):
        return self

    @property
    def vector_field_in_coordinates(self):
        fields = self.frame.coframe.dual_frame_in_coordinates
        if fields is None:
            return None
        if not self.coeff_dict:
            return 0 * fields[0]
        return zip_sum(list(self.coeff_dict.values()), [fields[idx] for idx in self.coeff_dict])

    @property
    def is_zero(self):
        return all(is_zero_knowing_zero_is_expected(v) for v in self.coeff_dict.values())

    @property
    def is_literal_zero(self):
        return not self.coeff_dict

    @property
    def free_symbols(self):
        out = set()
        for value in self.coeff_dict.values():
            out |= set(get_free_symbols(value))
        return out

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self.frame, frozenset(self.coeff_dict.items())))
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, abstract_vector_field):
            return NotImplemented
        return self.frame is other.frame and self.coeff_dict == other.coeff_dict

    def __ne__(self, other):
        out = self.__eq__(other)
        return out if out is NotImplemented else not out

    def coefficient(self, vector):
        return self.coeff_dict.get(self.frame._index(vector), 0)

    def coefficients(self):
        basis = self.frame.basis
        return {basis[idx]: value for idx, value in self.coeff_dict.items()}

    def _with(self, coeff_dict):
        return abstract_vector_field(self.frame, coeff_dict)

    def _map_coefficients(self, fn):
        return self._with({k: fn(v) for k, v in self.coeff_dict.items()})

    def _induce_method_from_descending(self, name, **kwargs):
        def apply(value):
            method = getattr(value, name, None)
            return method(**kwargs) if callable(method) else value

        return self._map_coefficients(apply)

    def _scaled(self, scalar, left):
        if _scalar_is_one(scalar):
            return self

        def times(v):
            if _scalar_is_one(v):
                return scalar
            return scalar * v if left else v * scalar

        return self._with({k: times(v) for k, v in self.coeff_dict.items()})

    def _refuse(self, other):
        if _is_vector(other) or query_dgcv_categories(other, "eds_form"):
            raise TypeError(
                "`*` between abstract vector fields and forms is not defined; use `VF_bracket` for the bracket, "
                "`wedge` for the wedge product and `form(X)` or `interiorProduct` for the pairing"
            )

    def __add__(self, other):
        if isinstance(other, abstract_vector_field):
            if other.frame is not self.frame:
                raise TypeError("abstract vector fields over different frames cannot be added")
            out = dict(self.coeff_dict)
            for k, v in other.coeff_dict.items():
                out[k] = out[k] + v if k in out else v
            return self._with(out)
        if get_dgcv_category(other) == "tensorProduct":
            return self._convert_to_tp() + other
        if _is_scalar(other) and _scalar_is_zero(other):
            return self
        return NotImplemented

    def __radd__(self, other):
        if _is_scalar(other) and _scalar_is_zero(other):
            return self
        if get_dgcv_category(other) == "tensorProduct":
            return other + self._convert_to_tp()
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, abstract_vector_field):
            return self + (-other)
        if get_dgcv_category(other) == "tensorProduct":
            return self._convert_to_tp() - other
        if _is_scalar(other) and _scalar_is_zero(other):
            return self
        return NotImplemented

    def __rsub__(self, other):
        if _is_scalar(other) and _scalar_is_zero(other):
            return -self
        if get_dgcv_category(other) == "tensorProduct":
            return other - self._convert_to_tp()
        return NotImplemented

    def __neg__(self):
        return self._with({k: -v for k, v in self.coeff_dict.items()})

    def _wedge(self, other):
        if other.frame is not self.frame:
            raise TypeError("the wedge product takes abstract vector fields over the same frame")
        return self._convert_to_tp().shape_inferred_tensor_product(
            other._convert_to_tp(), impose_shape="skew"
        )

    def __mul__(self, other):
        if _is_scalar(other):
            return self._scaled(other, left=False)
        if isinstance(other, abstract_vector_field):
            return self._wedge(other)
        self._refuse(other)
        return NotImplemented

    def __rmul__(self, other):
        if _is_scalar(other):
            return self._scaled(other, left=True)
        if isinstance(other, abstract_vector_field):
            return other._wedge(self)
        self._refuse(other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, numbers.Integral):
            return self._scaled(rational(1, other), left=True)
        if _is_scalar(other):
            return self._scaled(1 / other, left=True)
        raise TypeError("an abstract vector field can only be divided by a scalar")

    def __matmul__(self, other):
        return self._convert_to_tp() @ other

    def __rmatmul__(self, other):
        return other @ self._convert_to_tp()

    def __call__(self, other):
        if query_dgcv_categories(other, "eds_form"):
            degree = getattr(other, "degree", None)
            if degree == 1:
                return other(self)
            if degree == 0:
                return self(other.coefficient())
            raise TypeError(
                "an abstract vector field applies to scalars and 1-forms; use `interiorProduct` for a form of higher degree"
            )
        if _is_vector(other) or get_dgcv_category(other) == "tensorProduct":
            raise TypeError("an abstract vector field applies to scalars and 1-forms")
        if isinstance(other, numbers.Number):
            return 0
        coframe = self.frame.coframe
        scalars = []
        derivatives = []
        for idx, value in self.coeff_dict.items():
            derivative = coframe_derivative(other, coframe, idx)
            if _scalar_is_zero(derivative):
                continue
            scalars.append(value)
            derivatives.append(derivative)
        if not derivatives:
            return 0
        return zip_sum(scalars, derivatives)

    def bracket(self, other):
        if not isinstance(other, abstract_vector_field) or other.frame is not self.frame:
            raise TypeError("the bracket takes two abstract vector fields over the same frame")
        frame = self.frame
        out = {}
        for i, c in self.coeff_dict.items():
            for j, d in other.coeff_dict.items():
                if i == j:
                    continue
                cd = c * d
                for k, s in enumerate(frame._row(i, j)):
                    if _scalar_is_zero(s):
                        continue
                    term = cd * s
                    out[k] = out[k] + term if k in out else term
        for k, d in other.coeff_dict.items():
            w = self(d)
            if not _scalar_is_zero(w):
                out[k] = out[k] + w if k in out else w
        for k, c in self.coeff_dict.items():
            w = other(c)
            if not _scalar_is_zero(w):
                out[k] = out[k] - w if k in out else -w
        return self._with(out)

    def dual(self):
        forms = self.frame.coframe.forms
        return abstract_differential_form(
            [value * forms[idx] for idx, value in self.coeff_dict.items()]
        )

    def _convert_to_tp(self):
        card = self.card
        return tensorProduct(
            {((idx, 1, card),): value for idx, value in self.coeff_dict.items()},
            shape="all",
        )

    def subs(self, data, with_diff_corollaries=False, with_conjugates=None):
        if isinstance(data, (list, tuple)):
            data = dict(data)
        close = with_diff_corollaries if with_conjugates is None else with_conjugates
        if close and isinstance(data, dict):
            data = _conjugate_closure(data, "subs")
        return self._map_coefficients(
            lambda value: _subs_scalar(value, data, with_diff_corollaries)
        )

    def _subs_dgcv(self, data, with_diff_corollaries=False, with_conjugates=None):
        return self.subs(
            data, with_diff_corollaries=with_diff_corollaries, with_conjugates=with_conjugates
        )

    def _eval_simplify(self, **kwargs):
        constraints = kwargs.pop("constraints", None)
        with_conjugates = kwargs.pop("with_conjugates", True)
        if constraints is not None:
            return _simplify_under_constraints(self, constraints, with_conjugates, **kwargs)
        return self._map_coefficients(
            lambda value: value if isinstance(value, numbers.Number) else _routed_simplify(value, **kwargs)
        )

    def _eval_canonicalize(self):
        def canon(value):
            method = getattr(value, "_eval_canonicalize", None)
            return method() if callable(method) else value

        return self._map_coefficients(canon)

    def _eval_conjugate(self):
        rules = self.frame.conj_rules
        return self._with({rules[k]: _custom_conj(v) for k, v in self.coeff_dict.items()})

    def __dgcv_solve_bridge__(self, bridge):
        return [
            lowered for value in self.coeff_dict.values() for lowered in bridge.lower(value)
        ]

    @property
    def __dgcv_zero_obstr__(self):
        return _zero_obstruction(self)

    @classmethod
    def _dgcv_multiadd(cls, terms, start=0):
        if not isinstance(terms, (list, tuple)):
            terms = list(terms)
        if not terms:
            return start
        acc = {}
        frame = None
        residual = []
        if isinstance(start, cls):
            acc.update(start.coeff_dict)
            frame = start.frame
        elif not _scalar_is_zero(start):
            residual.append(start)
        for t in terms:
            if isinstance(t, cls):
                if frame is None:
                    frame = t.frame
                if t.frame is frame:
                    for k, v in t.coeff_dict.items():
                        acc[k] = acc[k] + v if k in acc else v
                    continue
            residual.append(t)
        if frame is None:
            return sum(terms, start)
        out = cls(frame, acc)
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
        frame = None
        residual = []
        if isinstance(start, cls):
            acc.update(start.coeff_dict)
            frame = start.frame
        elif not _scalar_is_zero(start):
            residual.append(start)
        for c, t in pairs:
            if isinstance(t, cls):
                if frame is None:
                    frame = t.frame
                if t.frame is frame:
                    if not _scalar_is_zero(c):
                        for k, v in t.coeff_dict.items():
                            term = c * v
                            acc[k] = acc[k] + term if k in acc else term
                    continue
            residual.append(c * t)
        if frame is None:
            return sum([c * t for c, t in pairs], start)
        out = cls(frame, acc)
        if residual:
            return sum(residual, out)
        return out

    def __str__(self):
        return lincomb_plain(
            self.coeff_dict, self.frame.basis_labels, valence=1, include_zero_term=False
        )

    def __repr__(self):
        return self.__str__()

    def _latex(self, printer=None, raw=True, **kwargs):
        return lincomb_latex(self.coeff_dict, vectorSpace=self.frame, valence=1, raw=raw)

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex(raw=raw, **kwargs)

    def _sympystr(self, printer):
        return self.__repr__()


register_legacy_sympy_class(abstract_vector_field)


def annihilator(forms, frame=None, **kwargs):
    if query_dgcv_categories(forms, "eds_form"):
        forms = [forms]
    else:
        forms = list(forms)
    if frame is None and not all(query_dgcv_categories(form, "eds_form") for form in forms):
        from ..core.vector_fields_and_differential_forms.annihilators import (
            annihilator as _core,
        )

        return _core(forms, **kwargs)
    if frame is not None and not isinstance(frame, abstract_frame):
        raise TypeError("`frame` must be an abstract frame")
    entries = []
    for form in forms:
        if not query_dgcv_categories(form, "eds_form"):
            raise TypeError("`annihilator` over a frame takes abstract 1-forms")
        if form.is_zero:
            continue
        if form.degree != 1:
            raise TypeError("`annihilator` over a frame takes 1-forms")
        entries.append(form.coefficients())
    if frame is None:
        frame = _unit_frame([unit for coefficients in entries for key in coefficients for unit in key])
        if frame is None:
            raise TypeError("the forms belong to no coframe; pass `frame=`")
    if not entries:
        return list(frame.basis)
    index = {unit: idx for idx, unit in enumerate(frame.coframe.forms)}
    unknowns = _disposable_symbols(create_key(prefix="ann"), frame.dimension)
    eqns = []
    for coefficients in entries:
        scalars, variables = [], []
        for key, value in coefficients.items():
            if key[0] not in index:
                raise TypeError("the forms belong to another coframe than the frame's")
            scalars.append(value)
            variables.append(unknowns[index[key[0]]])
        eqns.append(zip_sum(scalars, variables))
    from ..core.solvers import solve_dgcv

    sols = solve_dgcv(
        eqns, unknowns, method="linsolve", pass_to_symbolic_engine=False, simplify_result=False
    )
    if not sols:
        return []
    sol = sols[0]
    free = [u for u in unknowns if str(sol.get(u, u)) == str(u)]
    if not free:
        return []
    general = abstract_vector_field(frame, {idx: sol.get(u, u) for idx, u in enumerate(unknowns)})
    out = []
    for u in free:
        assign = {v: 0 for v in free}
        assign[u] = 1
        vector = general.subs(assign)
        if not vector.is_literal_zero:
            out.append(vector)
    return out

abst_frame = abstract_frame
abstVF = abstract_vector_field
