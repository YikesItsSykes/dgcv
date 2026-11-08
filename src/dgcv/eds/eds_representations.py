"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

module: dgcv.eds.representations


Description: Classes and methods to handle EDS matrix representations.

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
    _scalar_is_zero,
    is_zero_knowing_zero_is_expected,
)
from .._aux._backends._symbolic_router import conjugate as _routed_conjugate
from .._aux._backends._symbolic_router import simplify as _routed_simplify
from .._aux._backends._types_and_constants import expr_numeric_types, rational
from .._aux._vmf._safeguards import get_dgcv_category, retrieve_passkey
from ..algebras.aec import algebra_element_class
from ..core.arrays import matrix_dgcv
from ..core.base import dgcv_class
from ._atoms import _zero_obstruction, zero_form_atom
from ._calculus import _conjugate_closure, _simplify_under_constraints, extDer
from ._forms import (
    abstract_differential_form,
    abstract_differential_form_atom,
    abstract_differential_form_monomial,
)
from ._zero_forms import zero_form_class

_EDS_CLASSES = (
    zero_form_class,
    zero_form_atom,
    abstract_differential_form_atom,
    abstract_differential_form_monomial,
    abstract_differential_form,
)


def _conjugate_entry(expr):
    if isinstance(expr, _EDS_CLASSES):
        return expr._eval_conjugate()
    return _routed_conjugate(expr)


def _subs_entry(expr, subs_data, with_diff_corollaries):
    if hasattr(expr, "_subs_dgcv"):
        return expr._subs_dgcv(
            subs_data,
            with_diff_corollaries=with_diff_corollaries,
            with_conjugates=False,
        )
    if hasattr(expr, "subs"):
        return expr.subs(subs_data)
    return expr


def _canonicalize_entry(expr, depth):
    if hasattr(expr, "_eval_canonicalize"):
        return expr._eval_canonicalize(depth=depth)
    return expr


def _form_coefficient(entry, form):
    if hasattr(entry, "coefficient") and hasattr(entry, "degree"):
        return entry.coefficient(form)
    return 0


def _unit_forms(entries):
    found = []
    for entry in entries:
        if not (hasattr(entry, "coefficients") and hasattr(entry, "degree")):
            continue
        for key in entry.coefficients():
            if len(key) == 1 and key[0] not in found:
                found.append(key[0])
    coframes = {getattr(form, "primary_coframe", None) for form in found}
    if len(coframes) == 1:
        cf = coframes.pop()
        if cf is not None:
            ordered = [form for form in cf.forms if form in found]
            return tuple(ordered + [form for form in found if form not in ordered])
    return tuple(found)


class DF_representation(dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "DF_representation"

    def __dgcv_conjugate__(self, symbolic=False):
        return self._eval_conjugate()

    def __dgcv_apply__(self, func, **kwargs):
        return self.applyfunc(lambda entry: func(entry, **kwargs))

    def __dgcv_solve_bridge__(self, bridge):
        bridge.note_eds_forms()
        lowered = []
        for entry in self:
            lowered += bridge.lower(entry)
        return lowered

    @property
    def __dgcv_zero_obstr__(self):
        return _zero_obstruction(self)

    def __init__(self, row_count=None, col_count=None, array_data=None):
        if all(arg is None for arg in [col_count, array_data]):
            if row_count is None:
                array_data = tuple()
            else:
                array_data = (
                    row_count  # for optional syntax where only array data is given
                )

        dgcv_classes = [
            zero_form_atom,
            abstract_differential_form_atom,
            abstract_differential_form_monomial,
            abstract_differential_form,
            zero_form_class,
        ]
        supported_classes = tuple(dgcv_classes) + expr_numeric_types()

        if callable(array_data):
            if row_count is None or col_count is None:
                raise ValueError(
                    "When using a lambda function for array_data, row_count and col_count must be specified."
                )
            array_data = tuple(
                tuple(array_data(j, k) for k in range(col_count))
                for j in range(row_count)
            )

        if row_count is None or col_count is None:
            if isinstance(array_data, dict):
                if array_data:
                    row_count = max(key[0] for key in array_data.keys()) + 1
                    col_count = max(key[1] for key in array_data.keys()) + 1
                else:
                    row_count, col_count = 0, 0
            elif isinstance(array_data, (list, tuple)):
                if all(isinstance(entry, (list, tuple)) for entry in array_data):
                    row_count = len(array_data)
                    col_count = (
                        max(len(row) for row in array_data) if row_count > 0 else 0
                    )
                else:
                    row_count = 1
                    col_count = len(array_data)
                    array_data = (tuple(array_data),)

        def validate_keys(key, r_count, c_count):
            if not (isinstance(key, tuple) and len(key) == 2):
                raise TypeError(
                    "Keys must be length-2 integers within specified row and column bounds"
                )
            if (
                any(entry < 0 for entry in key)
                or c_count <= key[1]
                or r_count <= key[0]
            ):
                raise TypeError("Keys must be within row and column bounds")
            return True

        sparse_data = tuple()
        if isinstance(array_data, dict) and all(
            validate_keys(key, row_count, col_count) for key in array_data.keys()
        ):
            if not all(
                isinstance(value, supported_classes) for value in array_data.values()
            ):
                raise ValueError(
                    f"DF_representation only supports entries from: {', '.join(str(cls) for cls in supported_classes)}"
                )
            sparse_data = tuple(
                {k: v for k, v in array_data.items() if not _scalar_is_zero(v)}.items()
            )
            array_data = tuple(
                tuple(array_data.get((j, k), 0) for k in range(col_count))
                for j in range(row_count)
            )
        elif isinstance(array_data, (list, tuple)) and all(
            isinstance(entry, (list, tuple)) for entry in array_data
        ):
            if len(array_data) != row_count:
                raise ValueError("Incompatible row count")
            for row in array_data:
                if len(row) != col_count:
                    raise ValueError("Incompatible column count")
                if not all(isinstance(entry, supported_classes) for entry in row):
                    raise ValueError(
                        f"Only supports entries from: {', '.join(str(cls) for cls in supported_classes)}"
                    )

            array_data = tuple(tuple(row) for row in array_data)
            sparse_data = {
                array_data[j][k] for j in range(row_count) for k in range(col_count)
            }
        self.row_count = row_count
        self.col_count = col_count
        self.array = array_data
        self.sparse_data = sparse_data

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            return self.array[key]  # Allow row-wise access

        if isinstance(row_key, numbers.Integral) and isinstance(
            col_key, numbers.Integral
        ):
            return self.array[row_key][col_key]
        else:
            new_data = [row[col_key] for row in self.array[row_key]]
            return DF_representation(
                len(new_data), len(new_data[0]) if new_data else 0, new_data
            )

    def __iter__(self):
        for row in self.array:
            for entry in row:
                yield entry

    def __hash__(self):
        return hash((self.row_count, self.col_count, self.array))

    def __eq__(self, other):
        if not isinstance(other, DF_representation):
            return False
        return self.array == other.array

    def __add__(self, other):
        if not isinstance(other, DF_representation):
            raise TypeError("Can only add DF_representation objects")
        if self.row_count != other.row_count or self.col_count != other.col_count:
            raise ValueError("Matrix dimensions must match for addition")
        new_data = [
            [self.array[i][j] + other.array[i][j] for j in range(self.col_count)]
            for i in range(self.row_count)
        ]
        return DF_representation(self.row_count, self.col_count, new_data)

    def __sub__(self, other):
        if not isinstance(other, DF_representation):
            raise TypeError("Can only subtract DF_representation objects")
        if self.row_count != other.row_count or self.col_count != other.col_count:
            raise ValueError("Matrix dimensions must match for subtraction")
        new_data = [
            [self.array[i][j] - other.array[i][j] for j in range(self.col_count)]
            for i in range(self.row_count)
        ]
        return DF_representation(self.row_count, self.col_count, new_data)

    def __mul__(self, scalar):
        if not (
            isinstance(scalar, (zero_form_class, zero_form_atom))
            or isinstance(scalar, expr_numeric_types())
        ):
            return NotImplemented
        new_data = [
            [scalar * self.array[i][j] for j in range(self.col_count)]
            for i in range(self.row_count)
        ]
        return DF_representation(self.row_count, self.col_count, new_data)

    __rmul__ = __mul__

    def __matmul__(self, other):
        if not isinstance(other, DF_representation):
            raise TypeError("Can only multiply with another DF_representation")
        if self.col_count != other.row_count:
            raise ValueError("Inner matrix dimensions must match for multiplication")
        new_data = [
            [
                sum(self.array[i][k] * other.array[k][j] for k in range(self.col_count))
                for j in range(other.col_count)
            ]
            for i in range(self.row_count)
        ]
        return DF_representation(self.row_count, other.col_count, new_data)

    def __repr__(self):
        return f"DF_representation(row_count={self.row_count}, col_count={self.col_count}, array={self.array})"

    def __str__(self):
        return "\n".join([" ".join(map(str, row)) for row in self.array])

    def flatten(self):
        return [entry for entry in self]

    @property
    def free_symbols(self):
        fs_set = set()
        for entry in self:
            if hasattr(entry, "free_symbols"):
                fs_set = fs_set | entry.free_symbols
        return fs_set

    @property
    def is_zero(self):
        return all(is_zero_knowing_zero_is_expected(entry) for entry in self)

    def _latex(self, printer=None, raw=True, **kwargs):
        rows = []
        for row in self.array:
            rendered_row = []
            for entry in row:
                if hasattr(entry, "_latex"):
                    entry_latex = _routed_latex(entry)
                else:
                    entry_latex = str(entry)
                rendered_row.append(entry_latex)
            rows.append(" & ".join(rendered_row))
        return r"\begin{pmatrix}" + " \\\\ ".join(rows) + r"\end{pmatrix}"

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex() if raw else f"${self._latex()}$"

    def applyfunc(self, func):
        new_data = [
            [func(self.array[i][j]) for j in range(self.col_count)]
            for i in range(self.row_count)
        ]
        return DF_representation(self.row_count, self.col_count, new_data)

    def _dgcv_eds_applyfunc(self, func):
        return self.applyfunc(func)

    def curvature(self, with_simplify=False):
        out = extDer(self) + self @ self
        return _routed_simplify(out) if with_simplify else out

    def bianchi(self, curvature=None, with_simplify=False):
        K = self.curvature() if curvature is None else curvature
        out = extDer(K) + self @ K - K @ self
        return _routed_simplify(out) if with_simplify else out

    def _eval_conjugate(self):
        return self.applyfunc(_conjugate_entry)

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
        opts = (
            dict(
                ratio=ratio,
                measure=measure,
                inverse=inverse,
                doit=doit,
                rational=rational,
                expand=expand,
                **kwargs,
            )
            if engine_kind() == "sympy"
            else dict(kwargs)
        )
        return self.applyfunc(lambda entry: _routed_simplify(entry, **opts))

    def _eval_canonicalize(self, depth=1000):
        return self.applyfunc(lambda entry: _canonicalize_entry(entry, depth))

    def _subs_dgcv(self, data, with_diff_corollaries=False, with_conjugates=None):
        # an alias for regular subs so that other functions can know the with_diff_corollaries keyword is available
        return self.subs(
            data, with_diff_corollaries=with_diff_corollaries, with_conjugates=with_conjugates
        )

    def subs(self, subs_data, with_diff_corollaries=False, with_conjugates=None):
        close = with_diff_corollaries if with_conjugates is None else with_conjugates
        if close and isinstance(subs_data, dict):
            subs_data = _conjugate_closure(subs_data, "subs")
        return self.applyfunc(
            lambda entry: _subs_entry(entry, subs_data, with_diff_corollaries)
        )


register_legacy_sympy_class(DF_representation)


class connection_form(dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "connection_form"

    def __init__(self, data, algebra=None):
        category = get_dgcv_category(data)
        if category == "abst_coframe":
            algebra = data.algebra if algebra is None else algebra
            if algebra is None:
                raise TypeError(
                    "`connection_form(CF)` needs the algebra the coframe takes values in: pass `algebra=g` "
                    "(`createAlgebra(CF, 'g')` for a coframe with constant structure coefficients, "
                    "`createAlgebra(CF.subs({...}), 'g')` with the curvature functions set to 0 otherwise)"
                )
            if get_dgcv_category(algebra) != "algebra":
                raise TypeError("`algebra` must be a dgcv algebra created with `createAlgebra`")
            if algebra.dimension != data.dimension:
                raise ValueError(
                    f"the coframe has {data.dimension} forms but the algebra has dimension {algebra.dimension}"
                )
            value = algebra_element_class(algebra, list(data.forms), 1)
        elif algebra is not None:
            raise TypeError("`algebra=` goes with a coframe; the other inputs carry their own algebra or none")
        elif category == "algebra_element":
            value, algebra = data, data.algebra
        elif isinstance(data, DF_representation):
            value = data
        elif isinstance(data, (list, tuple, dict)):
            value = DF_representation(data)
        else:
            raise TypeError(
                "`connection_form` takes a coframe (with `algebra=`), an algebra element with 1-form "
                f"coefficients, or a matrix of 1-forms; received {type(data).__name__}"
            )
        self.value = value
        self.algebra = algebra
        self.forms = _unit_forms(self._entries())

    def _entries(self):
        if self.algebra is None:
            return list(self.value)
        return list(self.value.coeff_dict.values())

    def _map(self, func):
        return connection_form(self.value._dgcv_eds_applyfunc(func))

    def _dgcv_eds_applyfunc(self, func):
        return self.value._dgcv_eds_applyfunc(func)

    def __dgcv_conjugate__(self, symbolic=False):
        return self._eval_conjugate()

    def __dgcv_apply__(self, func, **kwargs):
        return self._map(lambda entry: func(entry, **kwargs))

    def _eval_conjugate(self):
        return self._map(_conjugate_entry)

    def _eval_simplify(self, **kwargs):
        constraints = kwargs.pop("constraints", None)
        with_conjugates = kwargs.pop("with_conjugates", True)
        if constraints is not None:
            return _simplify_under_constraints(self, constraints, with_conjugates, **kwargs)
        return connection_form(self.value._eval_simplify(**kwargs))

    def simplify(self, **kwargs):
        return self._eval_simplify(**kwargs)

    def _eval_canonicalize(self, depth=1000):
        return self._map(lambda entry: _canonicalize_entry(entry, depth))

    def subs(self, subs_data, with_diff_corollaries=False, with_conjugates=None):
        close = with_diff_corollaries if with_conjugates is None else with_conjugates
        if close and isinstance(subs_data, dict):
            subs_data = _conjugate_closure(subs_data, "subs")
        return self._map(lambda entry: _subs_entry(entry, subs_data, with_diff_corollaries))

    def _subs_dgcv(self, data, with_diff_corollaries=False, with_conjugates=None):
        return self.subs(
            data, with_diff_corollaries=with_diff_corollaries, with_conjugates=with_conjugates
        )

    def curvature(self, with_simplify=False):
        v = self.value
        if self.algebra is None:
            out = extDer(v) + v @ v
        else:
            out = extDer(v) + rational(1, 2) * (v * v)
        return _routed_simplify(out) if with_simplify else out

    def bianchi(self, curvature=None, with_simplify=False):
        v = self.value
        K = self.curvature() if curvature is None else curvature
        if isinstance(K, connection_form):
            K = K.value
        if self.algebra is None:
            out = extDer(K) + v @ K - K @ v
        else:
            out = extDer(K) + v * K
        return _routed_simplify(out) if with_simplify else out

    def coefficient(self, form):
        if self.algebra is None:
            return matrix_dgcv(
                [[_form_coefficient(entry, form) for entry in row] for row in self.value.array]
            )
        return algebra_element_class(
            self.algebra,
            {idx: _form_coefficient(c, form) for idx, c in self.value.coeff_dict.items()},
            self.value.valence,
        )

    def coefficients(self):
        return {form: self.coefficient(form) for form in self.forms}

    def model_algebra(self, label, **kwargs):
        from ..algebras.creators.general import createAlgebra

        kwargs.setdefault("return_created_object", True)
        return createAlgebra([self.coefficient(form) for form in self.forms], label, **kwargs)

    @property
    def is_zero(self):
        return self.value.is_zero

    @property
    def free_symbols(self):
        return self.value.free_symbols

    def __iter__(self):
        return iter(self._entries())

    def __eq__(self, other):
        if not isinstance(other, connection_form):
            return NotImplemented
        return self.value == other.value

    def __hash__(self):
        return hash((connection_form, self.value))

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"connection_form({self.value!r})"

    def _latex(self, printer=None, raw=True, **kwargs):
        return self.value._latex(raw=True)

    def _repr_latex_(self, raw=False, **kwargs):
        tex = self._latex()
        return tex if raw else f"${tex}$"


register_legacy_sympy_class(connection_form)
