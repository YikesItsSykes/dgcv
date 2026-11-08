"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds.Pfaff_systems

module: dgcv.eds.Pfaff_systems._coordinates

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers

from ..._aux._backends._calculus import diff
from ..._aux._backends._symbolic_router import (
    _fast_simplify,
    _scalar_is_zero,
)
from ..._aux._backends._symbolic_router import (
    conjugate as _routed_conjugate,
)
from ..._aux._backends._types_and_constants import imag_unit, rational
from ..._aux._vmf._safeguards import create_key
from ...core.dgcv_core import differential_form_class, vector_field_class
from ...core.vector_fields_and_differential_forms import (
    coordinate_differential_form,
    coordinate_vector_field,
    get_frame,
)
from ...core.vector_fields_and_differential_forms.coframes import (
    _dual_combinations,
    _pairing_row,
)
from .. import _bridges
from ..pde._systems import _coordinate_kind, _equation_list
from ._util import _free, _subs, _zero_test


class _reality:
    def __init__(self, coordinates, pairs, lone, registered, real=(), extra=None):
        self.coordinates = tuple(coordinates)
        self.pairs = dict(pairs)
        self.lone = tuple(lone)
        self.real = tuple(real)
        self.extra = dict(extra or {})
        self.closed = not self.lone and bool(registered)
        self._rules = None

    def partner(self, x):
        return self.pairs.get(x, x)

    def is_real(self, x):
        return x in self.real

    @property
    def rules(self):
        if self._rules is None:
            rules = {}
            for x in self.coordinates:
                if x in self.lone:
                    continue
                target = self.pairs.get(x, x)
                conj = _routed_conjugate(x)
                if str(conj) != str(target):
                    rules[conj] = target
            for x, target in self.extra.items():
                conj = _routed_conjugate(x)
                if str(conj) != str(target):
                    rules[conj] = target
            self._rules = rules
        return self._rules


def _coordinate_reality(coordinates, conjugates=None, minted=()):
    coordinates = list(coordinates)
    conjugates = conjugates or {}
    minted_ids = {id(x) for x in minted}
    pairs, lone, real, registered = {}, [], [], 0
    for x in coordinates:
        if x in conjugates:
            partner = conjugates[x]
            if partner is None:
                lone.append(x)
            elif partner is x or str(partner) == str(x):
                real.append(x)
                registered += 1
            elif partner in coordinates:
                pairs[x] = partner
                registered += 1
            else:
                lone.append(x)
            continue
        if id(x) in minted_ids:
            continue
        kind, partner = _coordinate_kind(x)
        if kind in ("real", "imag"):
            real.append(x)
            registered += 1
            continue
        if kind not in ("holo", "anti"):
            continue
        if partner is not None and partner in coordinates:
            pairs[x] = partner
            registered += 1
        else:
            lone.append(x)
    if pairs and lone:
        x, partner = next(iter(pairs.items()))
        raise ValueError(
            f"the coordinates mix the conjugate pair ({x}, {partner}) with {lone[0]} alone, whose conjugate "
            "is not a coordinate; list both members of every pair or neither"
        )
    extra = {
        x: partner
        for x, partner in conjugates.items()
        if partner is not None and partner is not x and not any(x is c for c in coordinates)
    }
    return _reality(coordinates, pairs, lone, registered, real, extra)


def _conjugated_scalar(value, reality):
    if isinstance(value, numbers.Number):
        return value.conjugate() if isinstance(value, complex) else value
    out = _routed_conjugate(value)
    rules = reality.rules
    return _subs(out, rules) if rules else out


def _conjugated_rows(rows, reality, target):
    out = []
    for row in rows:
        new = [0] * len(row)
        for idx, entry in enumerate(row):
            if not _scalar_is_zero(entry):
                new[target(idx)] = _conjugated_scalar(entry, reality)
        out.append(new)
    return out


def _abstract_closed(source):
    return bool(source.min_conj_rules) or any(
        "real" in form._markers for form in source.forms
    )


def _row_present(row, rows):
    is_zero = _zero_test()
    return any(all(is_zero(a - b) for a, b in zip(row, other)) for other in rows)


def _appended_rows(rows, conjugated):
    extra, sources = [], []
    for idx, row in enumerate(conjugated):
        if not _row_present(row, list(rows) + extra):
            extra.append(row)
            sources.append(idx)
    return extra, sources


def _forms_from_rows(rows, columns, objects):
    out = []
    for row in rows:
        total = 0
        for x, entry in zip(columns, row):
            if not _scalar_is_zero(entry):
                total = total + entry * objects.form(x)
        out.append(total)
    return out


def _closed_equations(equations, reality):
    out = list(equations)
    if reality is None or not reality.closed:
        return out
    is_zero = _zero_test()
    if not reality.pairs:
        half = rational(1, 2)
        split = []
        for equation in equations:
            conj = _conjugated_scalar(equation, reality)
            if is_zero(conj - equation):
                split.append(equation)
                continue
            split.append(_fast_simplify(half * (equation + conj)))
            split.append(_fast_simplify(-imag_unit() * half * (equation - conj)))
        return _equation_list(split, is_zero)
    for equation in equations:
        conj = _conjugated_scalar(equation, reality)
        if not any(is_zero(conj - other) for other in out):
            out.append(conj)
    return out


class _registered_coordinates:
    conjugates = None
    minted = ()
    form = staticmethod(coordinate_differential_form)
    field = staticmethod(coordinate_vector_field)
    structure = staticmethod(_bridges.structure_coefficients)

    def __init__(self, conjugates=None):
        self.conjugates = conjugates

    @staticmethod
    def frame(images, coordinates):
        return get_frame(list(images), coordinate_space=coordinates)


class _local_coordinates:
    def __init__(self, coordinates, conjugates=None, minted=()):
        self.conjugates = conjugates
        self.minted = tuple(minted)
        self.key = create_key(prefix="jet")
        self.spaces = {self.key: tuple(coordinates)}
        self.index = {x: idx for idx, x in enumerate(coordinates)}
        self.forms = {}
        self.fields = {}

    def form(self, x):
        if x not in self.forms:
            self.forms[x] = differential_form_class(
                coeff_dict={(self.index[x], 0, self.key): 1},
                variable_spaces=self.spaces,
            )
        return self.forms[x]

    def field(self, x):
        if x not in self.fields:
            self.fields[x] = vector_field_class(
                coeff_dict={(self.index[x], 1, self.key): 1},
                variable_spaces=self.spaces,
            )
        return self.fields[x]

    def frame(self, images, coordinates):
        fields = [self.field(x) for x in coordinates]
        return _dual_combinations(
            [_pairing_row(form(field) for field in fields) for form in images],
            fields,
            "PfaffSystem",
            return_parameters=False,
            register_parameters=False,
            parameters_label=None,
        )

    def _exterior(self, form, parameters):
        coordinates = self.spaces[self.key]
        total = 0
        for key, value in form.coeff_dict.items():
            dx = self.form(coordinates[key[0]])
            for z in _free(value):
                if z in self.index and z not in parameters:
                    total = total + diff(value, z) * (self.form(z) * dx)
        return total

    def structure(self, images, frame, parameters):
        return _bridges.structure_coefficients(images, frame, parameters, exterior=self._exterior)
