"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

module: dgcv.eds._bridges

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers

from .._aux._backends._symbolic_router import (
    _scalar_is_zero,
    get_free_symbols,
    is_zero_knowing_zero_is_expected,
)
from .._aux._backends._symbolic_router import conjugate as _routed_conjugate
from .._aux._backends._symbolic_router import simplify as _routed_simplify
from .._aux._backends._types_and_constants import imag_unit, rational
from .._aux._utilities._config import dgcv_warning
from .._aux._vmf._safeguards import get_dgcv_category, query_dgcv_categories
from .._aux._vmf.vmf import vmf_lookup
from ..core.solvers import solve_dgcv
from ..core.vector_fields_and_differential_forms import (
    exteriorDerivative,
    get_coframe,
    get_frame,
)
from ..core.vector_fields_and_differential_forms._solver_support import _local_unknowns
from ..core.vector_fields_and_differential_forms._vmf_frames import (
    _infer_coordinate_space_from_objs,
)
from ._atoms import zero_form_atom


def _as_parameters(parameters):
    if parameters is None:
        return frozenset()
    if isinstance(parameters, (list, tuple, set, frozenset)):
        return frozenset(parameters)
    return frozenset([parameters])


def _as_sequence(items, name):
    if items is None:
        return None
    if not isinstance(items, (list, tuple)):
        raise TypeError(f"`{name}` must be a list or tuple")
    return list(items)


def _check_members(items, category, name):
    if items is None:
        return
    kind = "vector fields" if category == "vector_field" else "1-forms"
    for item in items:
        if not query_dgcv_categories(item, {category}) or (
            category == "differential_form" and getattr(item, "degree", None) != 1
        ):
            raise TypeError(f"`{name}` must contain coordinate {kind}; received {item!r}")


def _completed(items, name, init_dimension, dimension, min_conj_rules):
    if items is None or len(items) == dimension:
        return items
    if len(items) != init_dimension:
        raise TypeError(
            f"`{name}` must list one entry per coframe label ({init_dimension}) or one per form of the "
            f"completed coframe ({dimension}); received {len(items)}"
        )
    out = list(items) + [None] * (dimension - init_dimension)
    for source, target in min_conj_rules.items():
        out[target] = _routed_conjugate(items[source], symbolic=True)
    return out


def _coordinate_space(objs, parameters):
    return _infer_coordinate_space_from_objs(objs, parameters)


def validate_lists(forms, frame, init_dimension, dimension):
    forms = _as_sequence(forms, "forms")
    frame = _as_sequence(frame, "frame")
    for items, name in ((forms, "forms"), (frame, "frame")):
        if items is not None and len(items) not in (init_dimension, dimension):
            raise TypeError(
                f"`{name}` must list one entry per coframe label ({init_dimension}) or one per form of the "
                f"completed coframe ({dimension}); received {len(items)}"
            )
    _check_members(forms, "differential_form", "forms")
    _check_members(frame, "vector_field", "frame")


def coordinate_slots(forms, frame, parameters, dimension, init_dimension, min_conj_rules):
    forms = _as_sequence(forms, "forms")
    frame = _as_sequence(frame, "frame")
    parameters = _as_parameters(parameters)
    if forms is None and frame is None:
        raise TypeError(
            "a coordinate coframe needs `forms` (coordinate 1-forms), `frame` (their dual vector fields) or both"
        )
    forms = _completed(forms, "forms", init_dimension, dimension, min_conj_rules)
    frame = _completed(frame, "frame", init_dimension, dimension, min_conj_rules)
    _check_members(forms, "differential_form", "forms")
    _check_members(frame, "vector_field", "frame")
    if forms is None:
        forms = get_coframe(frame, coordinate_space=_coordinate_space(frame, parameters))
    elif frame is None:
        frame = get_frame(forms, coordinate_space=_coordinate_space(forms, parameters))
    return tuple(forms), tuple(frame), parameters


def _scalar(value):
    if get_dgcv_category(value) == "tensor_field":
        scalars = list(value.coeff_dict.values())
        return scalars[0] if len(scalars) == 1 else sum(scalars)
    return value


def structure_coefficients(forms, frame, parameters, exterior=None):
    out = {}
    n = len(frame)
    for k, form in enumerate(forms):
        derivative = (
            exteriorDerivative(form, parameters=parameters)
            if exterior is None
            else exterior(form, parameters)
        )
        if not query_dgcv_categories(derivative, {"differential_form"}):
            continue
        for i in range(n):
            for j in range(i + 1, n):
                value = _scalar(derivative(frame[i], frame[j]))
                if _scalar_is_zero(value):
                    continue
                value = _routed_simplify(value)
                if _scalar_is_zero(value):
                    continue
                out[(i, j, k)] = value
    return out


def _coordinate_coefficient(value):
    if isinstance(value, numbers.Number):
        return True
    try:
        symbols = get_free_symbols(value)
    except Exception:
        return False
    return all(vmf_lookup(symbol)["type"] == "coordinate" for symbol in symbols)


def _combination(coefficients, objs):
    total = 0
    for coefficient, obj in zip(coefficients, objs):
        if _scalar_is_zero(coefficient):
            continue
        total = total + coefficient * obj
    return total


def _inverse(matrix):
    m = len(matrix)
    unknowns = _local_unknowns(m * m)
    eqns = []
    for j in range(m):
        for k in range(m):
            s = 0
            for i in range(m):
                s = s + matrix[j][i] * unknowns[i * m + k]
            eqns.append(s - (1 if j == k else 0))
    sols = solve_dgcv(eqns, unknowns, method="linsolve", simplify_result=False)
    if len(sols) != 1:
        return None
    sol = sols[0]
    unknown_set = set(unknowns)
    inverse = []
    for i in range(m):
        row = []
        for k in range(m):
            u = unknowns[i * m + k]
            if u not in sol or set(get_free_symbols(sol[u])) & unknown_set:
                return None
            row.append(sol[u])
        inverse.append(row)
    return inverse


_DROPPED = (
    "`transform_coframe`: the coordinate frame and forms of the original coframe were not carried over because "
)


def transformed_slots(original, new_basis):
    frame = original.dual_frame_in_coordinates
    images = original.forms_in_coordinates
    parameters = original.parameters
    if frame is None and images is None:
        return None, None, parameters
    old = original.forms
    matrix = [[new.coefficient(form) for form in old] for new in new_basis]
    if not all(_coordinate_coefficient(entry) for row in matrix for entry in row):
        dgcv_warning(_DROPPED + "a transformation coefficient is not a coordinate function.")
        return None, None, parameters
    new_images = None
    if images is not None:
        new_images = tuple(_combination(row, images) for row in matrix)
    new_frame = None
    if frame is not None:
        inverse = _inverse(matrix)
        if inverse is None:
            dgcv_warning(_DROPPED + "the transformation is not invertible.")
        else:
            m = len(matrix)
            new_frame = tuple(
                _combination([inverse[i][k] for i in range(m)], frame) for k in range(m)
            )
    return new_frame, new_images, parameters


def _is_number(value):
    return isinstance(value, numbers.Number) or not get_free_symbols(value)


def _coordinate_potential(form, parameters):
    spaces = getattr(form, "_variable_spaces", None) or {}
    coefficients = []
    coordinates = []
    for key, value in form.coeff_dict.items():
        coordinate = spaces[key[-1]][key[0]]
        if coordinate in parameters or not _is_number(value):
            return None
        coefficients.append(value)
        coordinates.append(coordinate)
    if len(coordinates) == 1 and _scalar_is_zero(coefficients[0] - 1):
        return coordinates[0]
    return _combination(coefficients, coordinates) if coordinates else None


def _form_vanishes(form):
    if getattr(form, "is_zero", False):
        return True
    values = getattr(form, "coeff_dict", None)
    if values is None:
        return False
    return all(is_zero_knowing_zero_is_expected(value) for value in values.values())


def _is_holomorphic_coordinate(image, parameters):
    value = _coordinate_potential(image, parameters)
    if value is None or isinstance(value, numbers.Number):
        return False
    return vmf_lookup(value, relatives=False).get("sub_type") == "holo"


def conjugate_pairs(images, parameters, rules=None, holomorphic=()):
    taken = set(rules or ()) | set((rules or {}).values())
    out = {}
    for i, image in enumerate(images):
        if i in taken or image is None:
            continue
        conjugated = _routed_conjugate(image)
        for j in range(i + 1, len(images)):
            if j in taken or images[j] is None or not _form_vanishes(conjugated - images[j]):
                continue
            key, value = i, j
            if j in holomorphic or (
                i not in holomorphic
                and _is_holomorphic_coordinate(images[j], parameters)
                and not _is_holomorphic_coordinate(image, parameters)
            ):
                key, value = j, i
            out[key] = value
            taken |= {i, j}
            break
    return out


def coordinate_potentials(forms, images, parameters):
    out = {}
    for form, image in zip(forms, images):
        value = _coordinate_potential(image, parameters)
        if value is not None:
            out[form] = value
    return out


def transformed_potentials(original, forms, new_basis):
    if not original._potential_index:
        return {}, {}
    matrix = [[new.coefficient(form) for form in original.forms] for new in new_basis]
    inverse = _inverse(matrix)
    if inverse is None:
        dgcv_warning(
            "`transform_coframe`: the potentials of the original coframe were not carried over because "
            "the transformation could not be inverted."
        )
        return {}, {}
    potential_index = {}
    for atom, row in original._potential_index.items():
        new_row = {}
        for i in range(len(forms)):
            value = _combination(list(row.values()), [inverse[j][i] for j in row])
            if not _scalar_is_zero(value):
                new_row[i] = value.to_engine() if _is_number(value) and hasattr(value, "to_engine") else value
        potential_index[atom] = new_row
    potentials = {}
    for form, row in zip(forms, matrix):
        terms = [(c, original.forms[j]) for j, c in enumerate(row) if not _scalar_is_zero(c)]
        if terms and all(_is_number(c) and old in original.potentials for c, old in terms):
            if len(terms) == 1 and _scalar_is_zero(terms[0][0] - 1):
                potentials[form] = original.potentials[terms[0][1]]
            else:
                potentials[form] = _combination(
                    [c for c, _ in terms], [original.potentials[old] for _, old in terms]
                )
    return potentials, potential_index


def _independent_of(atom, forms):
    rules = getattr(atom, "coframe_independants", None) or {}
    for form in forms:
        if not any(
            form in cf.forms and cf.forms.index(form) in indices
            for cf, indices in rules.items()
        ):
            return False
    return True


def _is_parameter(symbol, source):
    if symbol in source.parameters:
        return True
    return isinstance(symbol, zero_form_atom) and _independent_of(symbol, source.forms)


def _wedge_names(coframe, i, j, k):
    return tuple(coframe.forms[idx].label for idx in (i, j, k))


def _constant_symbols(value, source, coframe, i, j, k):
    if isinstance(value, numbers.Number):
        return set()
    symbols = set(get_free_symbols(value))
    bad = [symbol for symbol in symbols if not _is_parameter(symbol, source)]
    if bad:
        f_i, f_j, f_k = _wedge_names(coframe, i, j, k)
        names = ", ".join(str(symbol) for symbol in bad)
        raise ValueError(
            f"the coefficient {value} of {f_i}^{f_j} in d{f_k} is not a constant ({names} is not a declared "
            "parameter); substitute values with `createAlgebra(CF.subs({...}), ...)` or declare parameters with "
            "`createCoframe(..., parameters=[...])`"
        )
    return symbols


def _is_real(value):
    if isinstance(value, numbers.Real):
        return True
    return _scalar_is_zero(_routed_simplify(value - _routed_conjugate(value)))


def algebra_data(coframe, source=None, real=False):
    source = coframe if source is None else source
    n = coframe.dimension
    out = {}
    parameters = set()
    for k in range(n):
        for i in range(n):
            for j in range(i + 1, n):
                value = coframe[i, j, k]
                if _scalar_is_zero(value):
                    continue
                parameters |= _constant_symbols(value, source, coframe, i, j, k)
                if real and not _is_real(value):
                    f_i, f_j, f_k = _wedge_names(coframe, i, j, k)
                    raise ValueError(
                        f"the coefficient {value} of {f_i}^{f_j} in d{f_k} is not real; `base_field='real'` needs "
                        "real structure constants"
                    )
                out[(i, j, k)] = -value
    return {
        "structure_data": out,
        "parameters": frozenset(parameters),
        "assume_skew": True,
        "dimension": n,
    }


def real_basis(coframe):
    if not coframe.min_conj_rules:
        return coframe
    half = rational(1, 2)
    imu = imag_unit()
    rules = {}
    labels = [form.label for form in coframe.forms]
    for source, target in coframe.min_conj_rules.items():
        theta, bar = coframe.forms[source], coframe.forms[target]
        rules[theta] = half * (theta + bar)
        rules[bar] = -(half * (imu * (theta - bar)))
        labels[source] = f"Re_{theta.label}"
        labels[target] = f"Im_{theta.label}"
    return transform_coframe(coframe, rules, new_coframe_labels=labels, label=None)


def maurer_cartan_coefficients(algebra):
    n = algebra.dimension
    out = {}
    for k in range(n):
        for i in range(n):
            for j in range(i + 1, n):
                value = algebra[i, j, k]
                if not _scalar_is_zero(value):
                    out[(i, j, k)] = -value
    return out


def is_Lie_algebra(coframe):
    try:
        algebra_data(coframe)
    except ValueError:
        return False
    return all(
        _scalar_is_zero(_routed_simplify(extDer(form, coframe, order=2)))
        for form in coframe.forms
    )
