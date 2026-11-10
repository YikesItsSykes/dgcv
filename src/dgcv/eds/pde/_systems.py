"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds.pde

module: dgcv.eds.pde._systems

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers
from dataclasses import dataclass, field

from ..._aux._backends._engine import engine_kind
from ..._aux._backends._symbolic_router import conjugate as _routed_conjugate
from ..._aux._backends._symbolic_router import is_zero_knowing_zero_is_expected
from ..._aux._utilities._config import dgcv_warning
from ..._aux._vmf._safeguards import get_dgcv_category
from ..._aux._vmf.vmf import vmf_lookup
from ..._aux.printing.printing._eds import conjugation_prefix
from .._atoms import zero_form_atom
from .._bridges import _is_number
from .._calculus import _atoms_in
from .._coframes import abstract_coframe
from .._zero_forms import zero_form_class
from ._functions import function_name, read_functions
from ._report import _Report
from ._summary import equations_tex, equations_text, pde_summary


def _as_list(items):
    if items is None:
        return []
    if isinstance(items, (list, tuple, set, frozenset)):
        return list(items)
    return [items]


def _is_zero(value, scalar_zero):
    if isinstance(value, numbers.Number):
        return value == 0
    if isinstance(value, (zero_form_class, zero_form_atom)):
        return bool(getattr(value, "is_zero", False))
    return scalar_zero(value)


def _is_scalar(value):
    if isinstance(value, (numbers.Number, zero_form_class, zero_form_atom)):
        return True
    if isinstance(value, abstract_coframe) or hasattr(value, "degree"):
        return False
    return get_dgcv_category(value) not in ("differential_form", "distribution")


def is_pde_input(items):
    if isinstance(items, pde_system_data):
        return True
    items = _as_list(items)
    if not items:
        return False
    found = set()
    for item in items:
        if not _is_scalar(item):
            return False
        _atoms_in(item, found)
    return any(atom.differential_order > 0 for atom in found)


def _equation_list(equations, is_zero=is_zero_knowing_zero_is_expected):
    if equations is None:
        return []
    if isinstance(equations, dict):
        raise TypeError(
            "equations are expressions F with F = 0: write a solved equation a = b as a - b"
        )
    out = []
    for value in _as_list(equations):
        if _is_zero(value, is_zero):
            continue
        if _is_number(value):
            raise ValueError(f"the equations arrive at a contradiction: {value} = 0")
        out.append(value)
    return out


def _atom(value, name):
    if isinstance(value, zero_form_class) and isinstance(value.base, zero_form_atom):
        value = value.base
    if not isinstance(value, zero_form_atom):
        raise TypeError(f"`{name}` takes zero-form atoms; received {value!r}")
    return value


def _primitive(atom, found):
    for other in found:
        if other.label == atom.label and other.differential_order == 0:
            return other
    return zero_form_atom(atom.label, _markers=atom._markers)


def _derivative_coframe(derivatives, coframe):
    if coframe is not None and not isinstance(coframe, abstract_coframe):
        raise TypeError("`coframe` must be an abstract coframe")
    owners = set()
    for atom in derivatives:
        entries = [entry for entry in atom.coframe_derivatives if len(entry) > 1]
        if len(entries) != 1:
            raise TypeError(
                f"{atom} is differentiated along more than one coframe; a system of "
                "differential equations is written over one coframe"
            )
        owners.add(entries[0][0])
    if coframe is not None:
        if owners - {coframe}:
            raise TypeError(
                "the derivatives are taken along a coframe other than the `coframe` given"
            )
        return coframe
    if len(owners) != 1:
        if not owners:
            raise TypeError(
                "no derivative appears in the equations; pass `coframe=` and `dependent=` to "
                "name the coframe and the unknown functions"
            )
        raise TypeError(
            "the derivatives are taken along several coframes; a system of differential "
            "equations is written over one coframe"
        )
    return owners.pop()


def _check_holonomic(coframe):
    if not coframe.is_holonomic:
        raise TypeError(
            f"{coframe} is not holonomic (its structure equations are not all zero), so its "
            "coframe derivatives are not partial derivatives of a jet space"
        )


def _string_markers(form):
    return {marker for marker in form._markers if isinstance(marker, str)}


def _coordinate_kind(value):
    if value is None or isinstance(value, numbers.Number):
        return None, None
    info = vmf_lookup(value, relatives=True)
    if info.get("type") != "coordinate":
        return None, None
    kind = info.get("sub_type")
    if kind in ("holo", "anti"):
        return kind, (info.get("relatives") or {}).get("anti" if kind == "holo" else "holo")
    return kind, None


def _registered_partner(value):
    kind, partner = _coordinate_kind(value)
    if kind in ("real", "imag"):
        return "real", value
    if kind in ("holo", "anti"):
        return kind, partner
    if isinstance(value, zero_form_atom) and value.is_real:
        return "real", value
    return None, None


class _directions:
    def __init__(self, coframe):
        forms = coframe.forms
        potentials = [coframe.potentials.get(form) for form in forms]
        self.pairs, self.lone, self.real, self.primary = {}, set(), set(), set()
        for i, form in enumerate(forms):
            j = coframe.conj_rules.get(i, i)
            markers = _string_markers(form)
            value = potentials[i]
            if j != i:
                self.pairs[i] = j
                if i in coframe.min_conj_rules or "holomorphic" in markers:
                    self.primary.add(i)
                continue
            if "real" in markers:
                self.real.add(i)
                continue
            kind, mate = _registered_partner(value)
            if kind == "real":
                self.real.add(i)
            elif kind in ("holo", "anti"):
                if mate is not None and mate in potentials:
                    self.pairs[i] = potentials.index(mate)
                    if kind == "holo":
                        self.primary.add(i)
                else:
                    self.lone.add(i)
                    if kind == "holo":
                        self.primary.add(i)
            elif "holomorphic" in markers or "antiholomorphic" in markers:
                self.lone.add(i)
                if "holomorphic" in markers:
                    self.primary.add(i)
        for i, j in list(self.pairs.items()):
            if i not in self.primary and j not in self.primary:
                self.primary.add(min(i, j))
        if self.pairs and self.lone:
            i, j = next(iter(self.pairs.items()))
            k = min(self.lone)
            raise TypeError(
                f"the coframe mixes the conjugate pair ({forms[i]}, {forms[j]}) with {forms[k]}, "
                "whose conjugate is not a direction; give both members of every pair or neither"
            )
        self.forms = forms
        self.closure = bool(self.pairs or self.real)
        self.holomorphic = bool(self.lone)

    def conjugates(self):
        out = {}
        for i, j in self.pairs.items():
            out[self.forms[i]] = self.forms[j]
        for i in self.real:
            out[self.forms[i]] = self.forms[i]
        for i in self.lone:
            out[self.forms[i]] = None
        return out


def _families(primitives, order, occurring):
    prefix = conjugation_prefix()
    out, seen = [], set()
    for label in order:
        if label in seen:
            continue
        atom = primitives[label]
        conj = _routed_conjugate(atom)
        partner = conj if isinstance(conj, zero_form_atom) and conj.label != atom.label else None
        if partner is not None and atom.label.startswith(prefix) and not partner.label.startswith(prefix):
            atom, partner = partner, atom
        seen.add(atom.label)
        if partner is not None:
            seen.add(partner.label)
        out.append((atom, partner, partner is not None and partner.label in occurring))
    return out


def _unknowns(families, directions):
    dependents, extras, conjugates = [], [], {}
    for atom, partner, occurs in families:
        if partner is None or atom.is_real:
            dependents.append(atom)
            if directions.closure or directions.holomorphic:
                conjugates[atom] = atom
        elif directions.holomorphic:
            dependents.append(atom)
            conjugates[atom] = None
            if occurs:
                extras.append(partner)
                conjugates[partner] = None
                lone = ", ".join(str(directions.forms[i]) for i in sorted(directions.lone))
                dgcv_warning(
                    f"{partner} occurs with {atom} over the holomorphic direction(s) {lone}, which have no "
                    f"conjugate direction, so {partner} is treated as an unknown independent of {atom}; add "
                    "the conjugate direction to the coframe for the conjugate-pair reading"
                )
        elif directions.closure or occurs:
            dependents.append(atom)
            extras.append(partner)
            conjugates[atom] = partner
            conjugates[partner] = atom
        else:
            dependents.append(atom)
    return tuple(dependents + extras), conjugates


def _multi_index(atom):
    entry = next(entry for entry in atom.coframe_derivatives if len(entry) > 1)
    return tuple(sorted(index + 1 for index in entry[1:]))


@dataclass(frozen=True)
class pde_system_data(_Report):
    equations: tuple
    coframe: object
    dependent_variables: tuple
    parameters: tuple
    order: int
    derivatives: dict
    conjugates: dict = field(default_factory=dict)
    fixed_functions: tuple = ()
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    def singularities(self):
        held = self._cache.get("singularities", {})
        return {source: [list(locus) for locus in loci] for source, loci in held.items() if loci}

    def __repr__(self):
        return f"pde_system_data({equations_text(self.equations)})"

    def _latex(self, printer=None, raw=True, **kwargs):
        return equations_tex(self.equations)

    def summary(self, theme=None, use_latex=None, *, plain_text=None, return_displayable=False, **kwargs):
        return pde_summary(
            self,
            theme=theme,
            use_latex=use_latex,
            plain_text=plain_text,
            return_displayable=return_displayable,
            **kwargs,
        )

    @property
    def signature(self):
        return (self.coframe.dimension, len(self.dependent_variables), self.order)

    @property
    def holomorphic(self):
        return any(value is None for value in self.conjugates.values())

    def multi_index(self, atom):
        atom = _atom(atom, "atom")
        try:
            return self.derivatives[atom]
        except KeyError:
            raise KeyError(f"{atom} is not an unknown function of the system or a derivative of one")

    def in_jet_coordinates(self, jet_coordinates, parameter_symbols=None, direction_symbols=None):
        rules = {atom: jet_coordinates[key] for atom, key in self.derivatives.items()}
        for atom in self.parameters:
            if parameter_symbols is not None and atom in parameter_symbols:
                rules[atom] = parameter_symbols[atom]
        for atom, value in (direction_symbols or {}).items():
            rules[atom] = value
        equations = []
        for equation in self.equations:
            value = equation.subs(rules) if isinstance(equation, (zero_form_class, zero_form_atom)) else equation
            if isinstance(value, zero_form_class) and engine_kind() != "builtin":
                left = set()
                _atoms_in(value, left)
                if left:
                    raise ValueError(
                        "these zero forms have no jet coordinate: "
                        + ", ".join(str(atom) for atom in sorted(left, key=str))
                    )
                value = value.to_engine()
            equations.append(value)
        return equations, rules


def pde_system(equations, *, dependent=None, parameters=None, coframe=None):
    equations = _equation_list(equations)
    found = set()
    _atoms_in(equations, found)
    derivatives = sorted(
        (atom for atom in found if atom.differential_order > 0),
        key=lambda atom: (atom.differential_order, atom.label, _multi_index(atom)),
    )
    coframe = _derivative_coframe(derivatives, coframe)
    _check_holonomic(coframe)
    directions = _directions(coframe)
    declared = frozenset(
        _atom(value, "parameters")
        for value in _as_list(parameters)
        if function_name(value) is None
    )
    primitives, forced = {}, []
    for atom in derivatives:
        primitives.setdefault(atom.label, _primitive(atom, found))
    for value in _as_list(dependent):
        name = function_name(value)
        if name is not None:
            raise TypeError(
                f"{name} is a function_dgcv, which a system reads as a fixed function; an unknown "
                "function is a zero-form atom (createZeroForm)"
            )
        atom = _atom(value, "dependent")
        if atom.differential_order:
            raise TypeError("`dependent` names the unknown functions themselves, not their derivatives")
        primitives.setdefault(atom.label, atom)
        forced.append(atom.label)
    if not primitives or not derivatives:
        raise TypeError(
            "no derivative appears in the equations, so there is no system of differential equations"
        )
    for atom in primitives.values():
        if atom.is_constant or atom in coframe.parameters or atom in declared:
            raise TypeError(f"{atom} is a constant or parameter, so it is not an unknown function")
    rest = sorted(label for label in primitives if label not in forced)
    occurring = {atom.label for atom in found} | set(forced)
    dependents, conjugates = _unknowns(_families(primitives, forced + rest, occurring), directions)
    unknown_labels = {atom.label for atom in dependents}
    coordinates = {
        value
        for value in (coframe.potentials.get(form) for form in coframe.forms)
        if isinstance(value, zero_form_atom)
    }
    found_parameters = tuple(
        sorted(
            (
                atom
                for atom in found
                if atom.differential_order == 0 and atom.label not in unknown_labels and atom not in coordinates
            ),
            key=lambda atom: atom.label,
        )
    )
    parameter_labels = {atom.label: atom for atom in found_parameters}
    for atom in found_parameters:
        conj = _routed_conjugate(atom)
        if isinstance(conj, zero_form_atom) and conj.label != atom.label and conj.label in parameter_labels:
            conjugates[atom] = parameter_labels[conj.label]
    conjugates = directions.conjugates() | conjugates
    potentials = [coframe.potentials.get(form) for form in coframe.forms]
    functions, _ = read_functions(
        equations,
        [
            value
            for i, value in enumerate(potentials)
            if value is not None and i not in directions.pairs and i not in directions.lone
        ],
        lambda s: isinstance(s, zero_form_atom)
        and (s.differential_order > 0 or s.label in unknown_labels),
    )
    position = {atom.label: i for i, atom in enumerate(dependents, 1)}
    index = {atom: (position[atom.label], ()) for atom in dependents}
    for atom in derivatives:
        index[atom] = (position[atom.label], _multi_index(atom))
    return pde_system_data(
        tuple(equations),
        coframe,
        dependents,
        found_parameters,
        max(atom.differential_order for atom in derivatives),
        index,
        conjugates,
        tuple(entry[0] for entry in functions.values()),
    )
