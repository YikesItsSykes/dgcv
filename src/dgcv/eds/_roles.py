"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

module: dgcv.eds._roles

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from .._aux._utilities._config import get_variable_registry

ROLES = ("standard", "holomorphic", "antiholomorphic", "real", "imaginary")
FAMILY_ROLES = ("holomorphic", "antiholomorphic", "real", "imaginary")
ROLE_MARKERS = frozenset({"holomorphic", "antiholomorphic", "imaginary"})
ASSUMPTION_NAMES = frozenset(
    {
        "real",
        "positive",
        "negative",
        "nonnegative",
        "nonpositive",
        "nonzero",
        "integer",
        "constant",
    }
)
CONJUGATE_ROLE = {
    "holomorphic": "antiholomorphic",
    "antiholomorphic": "holomorphic",
    "real": "real",
    "imaginary": "imaginary",
    "standard": "standard",
}


class ComplexFamily:
    __slots__ = ("label", "holomorphic", "antiholomorphic", "real", "imaginary")

    def __init__(self, label, holomorphic, antiholomorphic, real, imaginary):
        self.label = label
        self.holomorphic = holomorphic
        self.antiholomorphic = antiholomorphic
        self.real = real
        self.imaginary = imaginary

    @property
    def members(self):
        return (self.holomorphic, self.antiholomorphic, self.real, self.imaginary)

    def role_of(self, atom):
        for role in FAMILY_ROLES:
            member = getattr(self, role)
            if member == atom or getattr(member, "label", None) == getattr(
                atom, "label", None
            ):
                return role
        return "standard"

    def partner(self, role):
        if role not in FAMILY_ROLES:
            raise ValueError(f"unknown complex role {role!r}")
        return getattr(self, role)

    def __eq__(self, other):
        return isinstance(other, ComplexFamily) and other.members == self.members

    def __hash__(self):
        return hash(("dgcv.builtin.family", self.label))

    def __repr__(self):
        return (
            f"ComplexFamily({self.label!r}, {', '.join(str(m) for m in self.members)})"
        )


def _relatives_entry(label):
    registry = get_variable_registry()
    entry = registry.get("paths", {}).get(label)
    if not isinstance(entry, dict):
        return None, None
    path = entry.get("path")
    if not (
        isinstance(path, tuple)
        and len(path) >= 4
        and path[0] == "complex_variable_systems"
    ):
        return None, None
    system = registry.get("complex_variable_systems", {}).get(path[1])
    if not isinstance(system, dict):
        return None, None
    rel = system.get("variable_relatives", {}).get(label)
    if not isinstance(rel, dict):
        return None, None
    return path[1], rel


def role_of_label(label):
    _, rel = _relatives_entry(label)
    if rel is None:
        return "standard"
    role = rel.get("complex_positioning")
    return role if role in FAMILY_ROLES else "standard"


def family_of_label(label, atom_type):
    system_label, rel = _relatives_entry(label)
    if rel is None:
        return None
    members = rel.get("complex_family")
    if not (isinstance(members, tuple) and len(members) == 4):
        return None
    if not all(isinstance(m, atom_type) for m in members):
        return None
    return ComplexFamily(system_label, *members)
