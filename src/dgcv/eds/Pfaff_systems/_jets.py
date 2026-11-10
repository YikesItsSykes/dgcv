"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds.Pfaff_systems

module: dgcv.eds.Pfaff_systems._jets

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers
from dataclasses import dataclass
from itertools import combinations_with_replacement

from ..._aux._backends._calculus import diff
from ..._aux._backends._display import latex as _routed_latex
from ..._aux._backends._engine import engine_kind
from ..._aux._backends._types_and_constants import imag_unit, rational, symbol
from ..._aux._utilities._config import (
    dgcv_warning,
    get_variable_registry,
    update_working_namespace_k_v,
    working_namespace,
)
from ..._aux._vmf._safeguards import (
    creator_label_check_enabled,
    retrieve_passkey,
    validate_creator_labels,
    validate_label,
)
from ..._aux._vmf.vmf import clearVar, vmf_lookup
from ..._aux.printing.printing._eds import (
    conjugation_prefix,
    label_derivative_parts,
    stem_derivative_tex,
)
from ...core.base import dgcv_class
from ...core.dgcv_core.atom_factory.creator import createVariables
from .. import _bridges
from .._atoms import zero_form_atom
from ..pde._report import _Report
from ..pde._summary import _jet_dimension
from ..pde._systems import (
    _coordinate_kind,
    _directions,
    _equation_list,
    pde_system,
    pde_system_data,
)
from ._chart import _restrict_forms, _warn_on_divisors
from ._coordinates import (
    _closed_equations,
    _coordinate_reality,
    _local_coordinates,
    _registered_coordinates,
)
from ._pfaff import PfaffSystem
from ._util import _free, _zero_test

_passkey = retrieve_passkey()
_CONVERSIONS = ("holToReal", "symToReal", "realToSym", "realToHol", "symToHol")
_STEM_NAMES = ("independent-variable", "dependent-variable", "derivative")


def _index_name(stem, i, sigma, m):
    tail = "_".join(str(s) for s in sigma)
    if m == 1:
        return f"{stem}_{tail}" if tail else stem
    return f"{stem}__{i}_{tail}" if tail else f"{stem}__{i}"


def _stem(entry, name):
    if isinstance(entry, str) and entry:
        return (entry, None, None)
    if (
        isinstance(entry, (list, tuple))
        and len(entry) == 3
        and all(isinstance(s, str) and s for s in entry)
    ):
        return tuple(entry)
    raise TypeError(
        f"the {name} stem is a string, or a triple (stem, real_stem, imaginary_stem) for a complex stem"
    )


def _checked_sizes(n, m, k, labels, holomorphic=False, coordinates=None):
    for name, value in (("n", n), ("m", m), ("k", k)):
        if not isinstance(value, numbers.Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if labels is None:
        if coordinates is None:
            raise TypeError(
                "`labels` are the three stems (independent variables, dependent variables, derivatives); "
                "only a space built from `coordinates=` may leave them out"
            )
        return int(n), int(m), int(k), None
    labels = tuple(labels)
    if len(labels) != 3:
        raise TypeError("`labels` are the three stems (independent variables, dependent variables, derivatives)")
    stems = tuple(_stem(entry, name) for entry, name in zip(labels, _STEM_NAMES))
    if (stems[1][1] is None) != (stems[2][1] is None):
        raise TypeError(
            "a complex unknown has complex derivatives: the dependent and derivative stems are both "
            "standard or both complex"
        )
    if holomorphic and any(stem[1] is None for stem in stems):
        raise TypeError("`holomorphic=True` takes three complex stems")
    return int(n), int(m), int(k), stems


def _multi_indices(n, k):
    return {length: list(combinations_with_replacement(range(1, n + 1), length)) for length in range(k + 1)}


def _layout(n, m, k, stems, holomorphic):
    (stem_x, real_x, _), (stem_u, real_u, _), (stem_p, _, _) = stems
    complex_x = real_x is not None and not holomorphic
    complex_u = real_u is not None and not holomorphic
    directions = [f"{stem_x}{j}" for j in range(1, n + 1)]
    if complex_x:
        directions += [f"BAR{name}" for name in directions]
    dependents = [_index_name(stem_u, i, (), m) for i in range(1, m + 1)]
    if complex_u:
        dependents += [f"BAR{name}" for name in dependents]
    if holomorphic:
        direction_partner = {j: None for j in range(n)}
        unknown_partner = {i: None for i in range(m)}
    else:
        direction_partner = {j: (j + n) % (2 * n) for j in range(2 * n)} if complex_x else {}
        if complex_u:
            unknown_partner = {i: (i + m) % (2 * m) for i in range(2 * m)}
        else:
            unknown_partner = {i: i for i in range(m)} if complex_x else {}
    return _pde_layout(
        directions,
        direction_partner,
        dependents,
        unknown_partner,
        k,
        lambda i, sigma: _index_name(stem_p, i, sigma, m),
    )


def _real_name(name, stem, real_stem):
    return real_stem + name[len(stem):]


def _jet_stem(space, i):
    first = space.p.get((i, (1,)))
    label = None if first is None else str(first)
    if label is None or not label.endswith("_1"):
        return None
    return label_derivative_parts(label[:-2])


def _coordinate_parts(space):
    conjugates = space._conjugates or {}
    paired = any(partner is not None and partner is not key for key, partner in conjugates.items())
    if not paired and engine_kind() == "builtin":
        return {}, {}
    prefix = conjugation_prefix()

    def primary(symbol):
        partner = conjugates.get(symbol)
        return partner is None or partner is symbol or not str(symbol).startswith(prefix)

    position = {x: j + 1 for j, x in enumerate(space.x)}
    index_tex, out = {}, {}
    for x in space.x:
        if primary(x):
            index_tex[x] = str(position[x])
        else:
            index_tex[x] = r"\overline{" + str(position[conjugates[x]]) + "}"
            out[x] = r"\overline{" + _routed_latex(conjugates[x]) + "}"
    slot = {u: i for i, u in enumerate(space.u, 1)}
    stems = {}
    for u in space.u:
        if primary(u):
            stems[u] = (_jet_stem(space, slot[u]), False)
        else:
            stems[u] = (_jet_stem(space, slot[conjugates[u]]), True)
            out[u] = r"\overline{" + _routed_latex(conjugates[u]) + "}"
    parts = {}
    for (i, sigma), coordinate in space.p.items():
        stem, conjugated = stems[space.u[i - 1]]
        if sigma and stem is not None:
            parts[coordinate] = (stem, tuple(index_tex[space.x[j - 1]] for j in sigma), conjugated)
    return parts, out


def _coordinate_tex(parts, out):
    out = dict(out)
    for coordinate, (stem, indices, conjugated) in parts.items():
        out[coordinate] = stem_derivative_tex(stem, indices, conjugated)
    return {coordinate: tex for coordinate, tex in out.items() if tex != _routed_latex(coordinate)}


@dataclass(frozen=True, repr=False, eq=False)
class jet_restriction_data(_Report):
    forms: tuple
    coordinates: tuple
    solved: dict
    divisors: tuple

    _rows = (
        (r"\text{forms}", "forms"),
        (r"\text{coordinates}", "coordinates"),
        (r"\text{solved}", "solved"),
        (r"\text{divisors}", "divisors"),
    )


class jetSpace(dgcv_class):
    _dgcv_class_check = _passkey
    _dgcv_category = "jet_space"

    def __init__(
        self,
        n,
        m,
        k,
        labels=("x", "u", "p"),
        coordinates=None,
        conjugates=None,
        holomorphic=False,
        _objects=None,
        _minted=None,
    ):
        if conjugates is not None and coordinates is None:
            raise TypeError("`conjugates` describes the symbols of `coordinates`; pass both")
        n, m, k, stems = _checked_sizes(n, m, k, labels, holomorphic, coordinates)
        self.labels = None if labels is None else tuple(labels)
        self._stems = stems
        self._sizes = (n, m)
        self.holomorphic = bool(holomorphic)
        if stems is None:
            N, M, names, name_map = n, m, None, {}
        else:
            groups, name_map = _layout(n, m, k, stems, self.holomorphic)
            names = [name for group in groups for name in group]
            N, M = len(groups[0]), len(groups[1])
        self.n, self.m, self.k = N, M, k
        self.multi_indices = _multi_indices(N, k)
        count = _jet_dimension(N, M, k)
        if coordinates is None:
            minted = {name: symbol(name) for name in names}
            coordinates = [minted[name] for name in names]
            conjugates = {
                minted[name]: None if partner is None else minted[partner]
                for name, partner in name_map.items()
            }
            minted_symbols = tuple(coordinates)
        else:
            minted_symbols = () if _minted is None else tuple(_minted)
            coordinates = list(coordinates)
            if len(coordinates) != count:
                raise ValueError(
                    f"J^{k}({N}, {M}) has {count} coordinates, in the order x, u, p; "
                    f"received {len(coordinates)}"
                )
            conjugates = None if conjugates is None else dict(conjugates)
        self.coordinates = tuple(coordinates)
        self._conjugates = conjugates
        self._minted = minted_symbols
        self._reality = _coordinate_reality(self.coordinates, conjugates, self._minted)
        self._objects = (
            _local_coordinates(self.coordinates, conjugates, self._minted)
            if _objects is None
            else _objects
        )
        self.x = self.coordinates[: self.n]
        self.u = self.coordinates[self.n : self.n + self.m]
        self.p = {}
        self._key = {}
        remaining = iter(self.coordinates[self.n :])
        for length in range(self.k + 1):
            for i in range(1, self.m + 1):
                for sigma in self.multi_indices[length]:
                    coordinate = next(remaining)
                    self.p[(i, sigma)] = coordinate
                    self._key[coordinate] = (i, sigma)
        self.dx = tuple(self._objects.form(x) for x in self.x)
        self.contact_forms = tuple(
            self._contact_form(i, sigma)
            for length in range(self.k)
            for i in range(1, self.m + 1)
            for sigma in self.multi_indices[length]
        )
        self.dim = len(self.coordinates)
        self._coordinate_parts = _coordinate_parts(self)
        self.coordinate_tex = _coordinate_tex(*self._coordinate_parts)
        self.conversion = {}
        self._complex_source = None
        self._systems = ()
        self.pde = None
        self.equations = ()
        self.jet_coordinate_rules = {}
        self.parameter_symbols = ()

    @property
    def conjugates(self):
        return dict(self._conjugates or {})

    @classmethod
    def from_equations(
        cls, equations, *, dependent=None, parameters=None, coframe=None, labels=None, order=None
    ):
        data = _pde_data(equations, dependent, parameters, coframe)
        k = _jet_order(data, order)
        layout = _pde_coordinates(data, labels, k)
        return _space_from_data(data, k, layout, {name: symbol(name) for name in layout[1]}, False)

    def _plus(self, sigma, j):
        return tuple(sorted(sigma + (j,)))

    def _sigma(self, sigma):
        if isinstance(sigma, numbers.Integral):
            sigma = (sigma,)
        sigma = tuple(int(s) for s in sigma)
        if any(not 1 <= s <= self.n for s in sigma):
            raise ValueError(f"multi-index entries must lie in 1..{self.n}")
        if len(sigma) > self.k:
            raise ValueError(f"the multi-index {sigma} has order {len(sigma)} > k = {self.k}")
        return tuple(sorted(sigma))

    def jet_coordinate(self, i, sigma=None):
        if sigma is None:
            if self.m == 1 and not isinstance(i, numbers.Integral):
                i, sigma = 1, i
            else:
                sigma = ()
        if not isinstance(i, numbers.Integral) or not 1 <= i <= self.m:
            raise ValueError(f"the dependent-variable index must lie in 1..{self.m}")
        return self.p[(int(i), self._sigma(sigma))]

    def order(self, coordinate):
        if coordinate in self._key:
            return len(self._key[coordinate][1])
        if coordinate in self.x:
            return 0
        raise ValueError(f"{coordinate} is not a coordinate of {self}")

    def jet_coordinates(self, length):
        return tuple(coordinate for (i, sigma), coordinate in self.p.items() if len(sigma) == length)

    def _contact_form(self, i, sigma):
        form = self._objects.form(self.p[(i, sigma)])
        for j in range(1, self.n + 1):
            form = form - self.p[(i, self._plus(sigma, j))] * self.dx[j - 1]
        return form

    def total_derivative(self, F, j):
        if not isinstance(j, numbers.Integral) or not 1 <= j <= self.n:
            raise ValueError(f"the direction j must lie in 1..{self.n}")
        if _bridges._is_number(F):
            return 0
        free = _free(F)
        if free & set(self.jet_coordinates(self.k)):
            raise ValueError(
                f"the expression depends on coordinates of order {self.k}, so D_{j} of it lives on "
                f"J^{self.k + 1}: use a jet space with k = {self.k + 1}"
            )
        total = diff(F, self.x[j - 1])
        for (i, sigma), coordinate in self.p.items():
            if len(sigma) < self.k and coordinate in free:
                total = total + self.p[(i, self._plus(sigma, j))] * diff(F, coordinate)
        return total

    def prolong_equations(self, equations, order=1):
        is_zero = _zero_test()
        equations = _equation_list(equations, is_zero)
        prolonged = list(equations)
        level = [(1, F) for F in equations]
        for _ in range(order):
            level = [
                (j, self.total_derivative(F, j)) for start, F in level for j in range(start, self.n + 1)
            ]
            prolonged.extend(F for _, F in level)
        return _equation_list(prolonged, is_zero)

    def _differential(self, F):
        total = 0
        for x in self.coordinates:
            if x in _free(F):
                total = total + diff(F, x) * self._objects.form(x)
        return total

    def _restricted_inputs(self, equations):
        _, divisors, chart, (forms, horizontal) = _restrict_forms(
            equations,
            self.coordinates,
            self._objects,
            self._reality,
            self.contact_forms,
            self.dx,
            last=self.x,
        )
        _warn_on_divisors(divisors)
        return list(forms), list(chart), list(horizontal), (), list(divisors)

    def _full_space_system(self, forms, equations, labels, filter_horizontals):
        system = PfaffSystem(
            list(forms),
            labels=labels,
            parameters=self.parameter_symbols,
            coordinates=list(self.coordinates),
            filter_horizontals_for_indep_mod_Pfaff=filter_horizontals,
            _objects=self._objects,
        )
        system.defining_equations = tuple(equations)
        system._independent_variables = tuple(self.x)
        return system

    @staticmethod
    def _regular_on_a_sample(system):
        try:
            point, _, _ = system._sample_point(seed=0)
        except ValueError:
            return True
        return point is None or system.is_regular_point(point)

    def _pfaff_inputs(self, equations, restrict, labels="omega", filter_horizontals=True):
        equations = _closed_equations(
            _equation_list(self.equations if equations is None else equations, _zero_test()),
            self._reality,
        )
        if restrict or not equations:
            return (*self._restricted_inputs(equations), None, (tuple(equations), True))
        forms = list(self.contact_forms) + [self._differential(F) for F in equations]
        system = self._full_space_system(forms, equations, labels, filter_horizontals)
        if restrict is None and not self._regular_on_a_sample(system):
            dgcv_warning(
                "the Pfaff system on the whole jet space drops rank along the PDE (an equation's differential "
                "is a combination of the others' there, as after a prolongation), so the system restricted to "
                "the PDE is used; pass restrict=True to choose it silently or restrict=False to keep the whole space"
            )
            return (*self._restricted_inputs(equations), None, (tuple(equations), True))
        return (
            forms,
            list(self.coordinates),
            list(self.dx),
            tuple(equations),
            [],
            system,
            (tuple(equations), restrict),
        )

    def restrict(self, equations=None):
        if equations is None:
            equations = self.equations
        solved, divisors, chart, (forms,) = _restrict_forms(
            _closed_equations(_equation_list(equations, _zero_test()), self._reality),
            self.coordinates,
            self._objects,
            self._reality,
            self.contact_forms,
            last=self.x,
        )
        return jet_restriction_data(tuple(forms), tuple(chart), solved, tuple(divisors))

    def pfaff_system(
        self, equations=None, *, restrict=None, labels="omega", filter_horizontals_for_indep_mod_Pfaff=True
    ):
        forms, chart, horizontal, defining, divisors, system, source = self._pfaff_inputs(
            equations, restrict, labels, filter_horizontals_for_indep_mod_Pfaff
        )
        if system is None:
            system = PfaffSystem(
                forms,
                labels=labels,
                parameters=self.parameter_symbols,
                coordinates=chart,
                filter_horizontals_for_indep_mod_Pfaff=filter_horizontals_for_indep_mod_Pfaff,
                _objects=self._objects,
            )
        system._set_horizontal_forms(horizontal)
        system._attach_equations(defining, divisors, self, source)
        return system

    def __dgcv_converter__(
        self,
        conv,
        skipVar=None,
        variables_scope=None,
        convert_everything=False,
        conversion_dict=None,
    ):
        if conv not in _CONVERSIONS:
            return None
        if conv in ("realToSym", "realToHol"):
            return self if self._complex_source is None else self._complex_source
        if conv == "symToHol" or self._stems is None or all(stem[1] is None for stem in self._stems):
            return self
        if self.holomorphic:
            raise TypeError(
                "a holomorphic jet space has no real spelling: the real jets of a holomorphic function "
                "are cut out of the real jet space by the Cauchy-Riemann equations; build the space "
                "with holomorphic=False"
            )
        return self._real_jet_space()

    def _real_parts(self, symbols, stem, real_stem, imag_stem):
        real, imag = [], []
        registered = isinstance(self._objects, _registered_coordinates)
        for x in symbols:
            relatives = (
                (vmf_lookup(x, relatives=True).get("relatives") or {}) if registered else {}
            )
            if relatives.get("real") is not None:
                real.append(relatives["real"])
                imag.append(relatives["imag"])
            else:
                suffix = str(x)[len(stem):]
                real.append(symbol(real_stem + suffix))
                imag.append(symbol(imag_stem + suffix))
        return real, imag

    def _real_jet_space(self):
        n, m = self._sizes
        (stem_x, real_x, imag_x), (stem_u, real_u, imag_u), _ = self._stems
        i_unit, half = imag_unit(), rational(1, 2)
        if real_x is None:
            directions = list(self.x)
            operators = {j: {j: 1} for j in range(1, n + 1)}
        else:
            xs, ys = self._real_parts(self.x[:n], stem_x, real_x, imag_x)
            directions = xs + ys
            operators = {}
            for j in range(1, n + 1):
                operators[j] = {j: half, n + j: -i_unit * half}
                operators[n + j] = {j: half, n + j: i_unit * half}
        if real_u is None:
            dependents = list(self.u)
            parts = {i: {i: 1} for i in range(1, m + 1)}
        else:
            us, vs = self._real_parts(self.u[:m], stem_u, real_u, imag_u)
            dependents = us + vs
            parts = {}
            for i in range(1, m + 1):
                parts[i] = {i: 1, m + i: i_unit}
                parts[m + i] = {i: 1, m + i: -i_unit}
        N, M = len(directions), len(dependents)
        multi_indices = _multi_indices(N, self.k)
        keys = [
            (i, sigma)
            for length in range(1, self.k + 1)
            for i in range(1, M + 1)
            for sigma in multi_indices[length]
        ]
        jets = {
            (i, sigma): symbol(f"{dependents[i - 1]}_{'_'.join(str(s) for s in sigma)}")
            for i, sigma in keys
        }
        coordinates = directions + dependents + [jets[key] for key in keys]
        real = jetSpace(
            N, M, self.k, labels=None, coordinates=coordinates, conjugates={c: c for c in coordinates}
        )
        conversion = {}
        if real_x is not None:
            for j in range(n):
                conversion[self.x[j]] = xs[j] + i_unit * ys[j]
                conversion[self.x[n + j]] = xs[j] - i_unit * ys[j]
        if real_u is not None:
            for i in range(m):
                conversion[self.u[i]] = us[i] + i_unit * vs[i]
                conversion[self.u[m + i]] = us[i] - i_unit * vs[i]
        for (i, sigma), coordinate in self.p.items():
            if not sigma:
                continue
            weights = {(): 1}
            for j in sigma:
                spread = {}
                for tau, weight in weights.items():
                    for direction, scale in operators[j].items():
                        key = tuple(sorted(tau + (direction,)))
                        spread[key] = spread.get(key, 0) + weight * scale
                weights = spread
            total = 0
            for dependent, scale in parts[i].items():
                for tau, weight in weights.items():
                    total = total + scale * weight * jets[(dependent, tau)]
            conversion[coordinate] = total
        real.conversion = conversion
        real._complex_source = self
        return real

    def clear(self):
        registered = vmf_lookup(self, path=True).get("path")
        if registered is not None:
            clearVar(registered[1], report=False)

    def __repr__(self):
        return f"jetSpace({self.n}, {self.m}, {self.k})"

    def __str__(self):
        return self.__repr__()

    def _latex(self, printer=None, raw=True, **kwargs):
        coords = ", ".join(self.coordinate_tex.get(c) or _routed_latex(c) for c in self.coordinates)
        return rf"J^{{{self.k}}}({self.n}, {self.m}):\ \left({coords}\right)"

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex() if raw else f"${self._latex()}$"


def _pde_data(equations, dependent, parameters, coframe):
    if isinstance(equations, pde_system_data):
        if dependent is not None or parameters is not None or coframe is not None:
            raise TypeError("a `pde_system_data` record already fixes the unknowns, parameters and coframe")
        return equations
    return pde_system(equations, dependent=dependent, parameters=parameters, coframe=coframe)


def _jet_order(data, order):
    if order is None:
        return data.order
    if not isinstance(order, numbers.Integral) or order < data.order:
        raise ValueError(f"`order` is the order of the jet space, at least {data.order} for these equations")
    return int(order)


def _pde_layout(directions, direction_partner, unknowns, unknown_partner, k, jet_name):
    N, M = len(directions), len(unknowns)
    multi_indices = _multi_indices(N, k)

    def conjugate_sigma(sigma):
        return tuple(
            sorted(
                direction_partner[j - 1] + 1 if direction_partner.get(j - 1) is not None else j
                for j in sigma
            )
        )

    jets = {}
    for length in range(1, k + 1):
        for i in range(1, M + 1):
            for sigma in multi_indices[length]:
                p = unknown_partner.get(i - 1)
                if p is not None and p < i - 1:
                    jets[(i, sigma)] = "BAR" + jet_name(p + 1, conjugate_sigma(sigma))
                else:
                    jets[(i, sigma)] = jet_name(i, sigma)
    groups = [list(directions), list(unknowns)] + [
        [jets[(i, sigma)] for i in range(1, M + 1) for sigma in multi_indices[length]]
        for length in range(1, k + 1)
    ]
    conjugates = {}
    for i, p in direction_partner.items():
        conjugates[directions[i]] = None if p is None else directions[p]
    for i, p in unknown_partner.items():
        conjugates[unknowns[i]] = None if p is None else unknowns[p]
    for (i, sigma), name in jets.items():
        if i - 1 in unknown_partner:
            p = unknown_partner[i - 1]
            lone = any(direction_partner.get(j - 1, j - 1) is None for j in sigma)
            conjugates[name] = None if p is None or lone else jets[(p + 1, conjugate_sigma(sigma))]
    return groups, conjugates


def _pde_coordinates(data, labels, k):
    coframe = data.coframe
    forms = coframe.forms
    n, m, _ = data.signature
    directions = _directions(coframe)
    conj = data.conjugates
    direction_partner = {}
    for i, form in enumerate(forms):
        if i in directions.pairs:
            direction_partner[i] = directions.pairs[i]
        elif i in directions.real:
            direction_partner[i] = i
        elif i in directions.lone:
            direction_partner[i] = None
    unknowns = list(data.dependent_variables)
    position = {atom.label: i for i, atom in enumerate(unknowns)}
    unknown_partner = {}
    for i, atom in enumerate(unknowns):
        if atom in conj:
            value = conj[atom]
            unknown_partner[i] = None if value is None else position[value.label]
    partner_slots = {i for i, p in unknown_partner.items() if p is not None and p < i}
    slots, direction_atoms = [], {}
    for i, form in enumerate(forms):
        value = coframe.potentials.get(form)
        if value is not None and _is_registered_coordinate(value):
            slots.append(value)
        elif isinstance(value, zero_form_atom) and value.differential_order == 0:
            direction_atoms[i] = value
            slots.append(value.label)
        elif value is not None and not isinstance(value, numbers.Number) and _free(value) == {value}:
            slots.append(value)
        elif i in directions.pairs and i not in directions.primary:
            slots.append(f"BARx{directions.pairs[i] + 1}")
        else:
            slots.append(f"x{i + 1}")
    if labels is not None:
        stems = _checked_sizes(n, m, k, labels)[3]
        (stem_x, real_x, _), (stem_u, real_u, _), (stem_p, _, _) = stems
        complex_directions = bool(directions.pairs or directions.lone)
        if (real_x is not None) != complex_directions or (
            complex_directions and len(directions.pairs) + len(directions.lone) != n
        ):
            raise TypeError(
                "the independent-variable stem is a triple (stem, real_stem, imaginary_stem) when every "
                "direction is complex and a string when none is; these directions need the other kind, "
                "or no `labels=`"
            )
        complex_unknowns = {i for i in range(m) if i in unknown_partner and unknown_partner[i] != i}
        if complex_unknowns and len(complex_unknowns) != m:
            raise TypeError(
                "the unknowns are not all of one kind (complex pairs or real), so `labels=` cannot name "
                "them with one stem; leave `labels=` out to use their own labels"
            )
        if (real_u is not None) != bool(complex_unknowns):
            raise TypeError(
                "the dependent-variable stem is a triple (stem, real_stem, imaginary_stem) when the "
                "unknowns are complex and a string when they are real; these unknowns need the other kind"
            )
        counter, direction_names = 0, [None] * n
        for i in range(n):
            if complex_directions:
                if i in directions.primary:
                    counter += 1
                    direction_names[i] = f"{stem_x}{counter}"
            else:
                direction_names[i] = f"{stem_x}{i + 1}"
        for i in range(n):
            if direction_names[i] is None:
                direction_names[i] = "BAR" + direction_names[directions.pairs[i]]
        slots = direction_names
        count = m - len(partner_slots)
        unknown_names = [None] * m
        for i in range(m):
            if i not in partner_slots:
                unknown_names[i] = _index_name(stem_u, i + 1, (), count)
        for i in sorted(partner_slots):
            unknown_names[i] = "BAR" + unknown_names[unknown_partner[i]]

        def jet_name(i, sigma):
            return _index_name(stem_p, i, sigma, count)

    else:
        stems = None
        unknown_names = [atom.label for atom in unknowns]

        def jet_name(i, sigma):
            return _index_name(unknown_names[i - 1], 1, sigma, 1)

    groups, name_map = _pde_layout(slots, direction_partner, unknown_names, unknown_partner, k, jet_name)
    names = [name for group in groups for name in group if isinstance(name, str)]
    others = {atom.label for atom in data.parameters}
    clashes = sorted({name for name in names if names.count(name) > 1 or name in others})
    if clashes:
        raise ValueError(
            "the jet coordinate names "
            + ", ".join(clashes)
            + " collide with labels in the equations; pass `labels=` with three stems"
        )
    return slots, names, direction_atoms, name_map, groups, stems


def _is_registered_coordinate(value):
    return _coordinate_kind(value)[0] is not None


def _conjugate_symbols(name_map, minted):
    def resolve(value):
        return minted[value] if isinstance(value, str) else value

    return {resolve(key): None if value is None else resolve(value) for key, value in name_map.items()}


def _parameter_map(data, parameter_symbols):
    out = {}
    for atom in data.parameters:
        partner = data.conjugates.get(atom)
        if partner is not None and partner.label != atom.label and partner in parameter_symbols:
            out[parameter_symbols[atom]] = parameter_symbols[partner]
    return out


def _jet_coordinates(slots, names, minted):
    return [minted[slot] if isinstance(slot, str) else slot for slot in slots] + [
        minted[name] for name in names if name not in {slot for slot in slots if isinstance(slot, str)}
    ]


def _space_from_data(data, k, layout, symbols, registered):
    slots, names, direction_atoms, name_map, _, _ = layout
    parameter_symbols = {atom: symbol(atom.label) for atom in data.parameters}
    conjugates = _conjugate_symbols(name_map, symbols) | _parameter_map(data, parameter_symbols)
    n, m, _ = data.signature
    space = jetSpace(
        n,
        m,
        k,
        labels=None,
        coordinates=_jet_coordinates(slots, names, symbols),
        conjugates=conjugates,
        holomorphic=data.holomorphic,
        _objects=_registered_coordinates(conjugates) if registered else None,
        _minted=None if registered else tuple(symbols.values()),
    )
    _attach_pde(space, data, direction_atoms, parameter_symbols)
    return space


def _claim_label(label):
    label = validate_label(label)
    validate_creator_labels("eds_jet_spaces", (label,))
    if creator_label_check_enabled():
        clearVar(label, report=False)
    return label


def _attach_pde(space, data, direction_atoms, parameter_symbols):
    direction_symbols = {atom: space.x[i] for i, atom in direction_atoms.items()}
    equations, rules = data.in_jet_coordinates(space.p, parameter_symbols, direction_symbols)
    if space.k > data.order:
        equations = space.prolong_equations(equations, order=space.k - data.order)
    space.pde = data
    space.equations = tuple(equations)
    space.jet_coordinate_rules = rules
    space.parameter_symbols = tuple(parameter_symbols[atom] for atom in data.parameters)


def createJetSpace(
    label,
    n,
    m=None,
    k=None,
    labels=None,
    holomorphic=False,
    return_created_object=None,
    *,
    dependent=None,
    parameters=None,
    coframe=None,
    order=None,
):
    if not isinstance(n, numbers.Integral):
        if m is not None or k is not None or holomorphic:
            raise TypeError(
                "a jet space built from equations takes `labels=`, `dependent=`, `parameters=`, "
                "`coframe=` and `order=` only"
            )
        return _create_jet_space_from_equations(
            label, n, labels, dependent, parameters, coframe, order, return_created_object
        )
    if order is not None:
        raise TypeError("`order` belongs to a jet space built from equations; here it is `k`")
    if m is None or k is None:
        raise TypeError("`createJetSpace(label, n, m, k)` needs the three sizes")
    if labels is None:
        labels = ("x", "u", "p")
    n, m, k, stems = _checked_sizes(n, m, k, labels, holomorphic)
    label = _claim_label(label)
    groups, name_map = _layout(n, m, k, stems, holomorphic)
    conjugate_names = {name for group in groups[1:] for name in group if name.startswith("BAR")}
    stem_x, real_x, imag_x = stems[0]
    if real_x is None:
        createVariables(stem_x, n)
    else:
        createVariables(stem_x, real_x, imag_x, n)
    systems = [stem_x]
    for group, (stem, real, imag) in zip(groups[1:], [stems[1]] + [stems[2]] * (len(groups) - 2)):
        primitives = [name for name in group if name not in conjugate_names]
        if real is None:
            createVariables(" ".join(primitives))
        else:
            createVariables(
                " ".join(primitives),
                " ".join(_real_name(name, stem, real) for name in primitives),
                " ".join(_real_name(name, stem, imag) for name in primitives),
            )
        systems.extend(primitives)
    namespace = working_namespace()
    coordinates = [namespace[name] for group in groups for name in group]
    conjugates = {
        namespace[name]: None if partner is None else namespace[partner]
        for name, partner in name_map.items()
    }
    space = jetSpace(
        n,
        m,
        k,
        labels=labels,
        coordinates=coordinates,
        conjugates=conjugates,
        holomorphic=holomorphic,
        _objects=_registered_coordinates(conjugates),
    )
    _register_jet_space(label, space, systems, (n, m, k), labels, holomorphic)
    if return_created_object:
        return space


def _real_imaginary_names(name, stem):
    if stem is None or stem[1] is None:
        return name + "r", name + "i"
    return _real_name(name, stem[0], stem[1]), _real_name(name, stem[0], stem[2])


def _register_names(groups, name_map, stems):
    systems = []
    for index, group in enumerate(groups):
        stem = None if stems is None else stems[min(index, 2)]
        complex_names, plain = [], []
        for name in group:
            if not isinstance(name, str):
                continue
            partner = name_map.get(name, name)
            if partner is None or (partner != name and not name.startswith("BAR")):
                complex_names.append(name)
            elif partner == name:
                plain.append(name)
        if complex_names:
            parts = [_real_imaginary_names(name, stem) for name in complex_names]
            createVariables(
                " ".join(complex_names),
                " ".join(real for real, _ in parts),
                " ".join(imag for _, imag in parts),
            )
            systems.extend(complex_names)
        if plain:
            createVariables(" ".join(plain))
            systems.extend(plain)
    return systems


def _create_jet_space_from_equations(
    label, equations, labels, dependent, parameters, coframe, order, return_created_object
):
    data = _pde_data(equations, dependent, parameters, coframe)
    k = _jet_order(data, order)
    layout = _pde_coordinates(data, labels, k)
    _, names, _, name_map, groups, stems = layout
    label = _claim_label(label)
    systems = _register_names(groups, name_map, stems)
    namespace = working_namespace()
    space = _space_from_data(data, k, layout, {name: namespace[name] for name in names}, True)
    n, m, _ = data.signature
    _register_jet_space(label, space, systems, (n, m, k), labels, data.holomorphic)
    if return_created_object:
        return space


def _register_jet_space(label, space, systems, signature, labels, holomorphic):
    space._systems = tuple(systems)
    vr = get_variable_registry()
    vr["eds"].setdefault("jet_spaces", {})[label] = {
        "space": space,
        "coordinate_systems": list(systems),
        "signature": tuple(signature),
        "labels": None if labels is None else tuple(labels),
        "holomorphic": bool(holomorphic),
    }
    vr["_labels"][label] = {"path": ("eds", label, "jet_spaces"), "children": set(systems)}
    vr["paths"][label] = {"kind": "jet_space", "path": ("eds", label, "jet_spaces")}
    update_working_namespace_k_v(label, space)
