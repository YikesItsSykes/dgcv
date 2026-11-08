"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

module: dgcv.eds._coframes

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers
from collections import Counter

from .._aux._backends._cls_coercion import register_legacy_sympy_class
from .._aux._backends._display import latex as _routed_latex
from .._aux._backends._symbolic_router import _scalar_is_zero, get_free_symbols
from .._aux._backends._symbolic_router import conjugate as _routed_conjugate
from .._aux._backends._symbolic_router import simplify as _routed_simplify
from .._aux._backends._types_and_constants import expr_numeric_types
from .._aux._utilities._config import dgcv_warning, get_variable_registry
from .._aux._vmf._safeguards import (
    create_key,
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_passkey,
)
from .._aux.printing.printing._string_processing import label_indices
from ..core.base import dgcv_class
from . import _bridges
from ._atoms import _empty_coframe, zero_form_atom
from ._forms import (
    _bare_scalar,
    _wedge_coefficient,
    abstract_differential_form,
    abstract_differential_form_atom,
    abstract_differential_form_monomial,
)
from ._zero_forms import zero_form_class


def _as_sequence(obj):
    return list(obj) if isinstance(obj, (list, tuple, set, frozenset)) else [obj]


def _is_structure_coefficient_label(label):
    base, superscripts, subscripts = label_indices(label)
    return bool(superscripts) and bool(subscripts)


def _validated_slot(items, dimension, category, name):
    if items is None:
        return None
    items = tuple(items)
    ok = len(items) == dimension and all(
        query_dgcv_categories(item, {category}) for item in items
    )
    if ok and category == "differential_form":
        ok = all(getattr(item, "degree", None) == 1 for item in items)
    if not ok:
        kind = "vector field" if category == "vector_field" else "1-form"
        raise TypeError(
            f"`{name}` must list one coordinate {kind} per basis 1-form of the coframe ({dimension} expected)"
        )
    return items


class structure_equation_system(dict):
    _dgcv_eqn_left_prefix = r"\operatorname{d}"
    _owner = None

    def _touch(self):
        owner = self._owner
        if owner is not None:
            owner._invalidate_caches()

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._touch()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._touch()

    def __ior__(self, other):
        super().__ior__(other)
        self._touch()
        return self

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._touch()

    def pop(self, *args):
        out = super().pop(*args)
        self._touch()
        return out

    def popitem(self):
        out = super().popitem()
        self._touch()
        return out

    def clear(self):
        super().clear()
        self._touch()

    def setdefault(self, key, default=None):
        out = super().setdefault(key, default)
        self._touch()
        return out

    def of(self, *forms):
        out = structure_equation_system()
        for form in forms:
            if form not in self:
                raise KeyError(
                    f"{form} is not a 1-form of this structure-equation system"
                )
            out[form] = self[form]
        return out

    def copy(self):
        return structure_equation_system(self)

    @property
    def free_symbols(self):
        found = set()
        for value in self.values():
            found |= getattr(value, "free_symbols", set())
        return found

    def _latex(self, printer=None, raw=True, **kwargs):
        if not self:
            return r"\varnothing"
        rows = r" \\ ".join(
            f"{self._dgcv_eqn_left_prefix}{_routed_latex(form)} &= "
            f"{_routed_latex(0 if eqn is None else eqn)}"
            for form, eqn in self.items()
        )
        return r"\begin{aligned} " + rows + r" \end{aligned}"

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex() if raw else f"${self._latex()}$"


def _potential_name(value):
    return value.label if isinstance(value, zero_form_atom) else str(value)


def _record_potentials(coframe):
    for entry in get_variable_registry()["eds"]["coframes"].values():
        if entry["coframe"] is coframe:
            entry["potentials"] = {
                form.label: _potential_name(value) for form, value in coframe.potentials.items()
            }


class abstract_coframe(dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "abst_coframe"

    def _eval_conjugate(self):
        return self

    def __dgcv_conjugate__(self, symbolic=False):
        return self._eval_conjugate()

    def __init__(
        self,
        coframe_basis,
        structure_equations,
        min_conj_rules=None,
        frame=None,
        coordinate_forms=None,
        parameters=None,
        algebra=None,
        potentials=None,
        potential_index=None,
    ):
        """
        Create a new abstract_coframe instance.

        Parameters:
        ==========
            - structure_equations (dict): A dictionary where keys are abstract_DF_atom instances (representing 1-forms), and values are either abstract_DF instances (representing the differential) or None.
        """
        if min_conj_rules is None:
            min_conj_rules = {}
        if not isinstance(coframe_basis, (list, tuple)):
            raise TypeError(
                f"Given `coframe_basis` must be a list/tuple. Instead recieved type {type(coframe_basis)}"
            )
        if len(coframe_basis) != len(set(coframe_basis)):
            counter = Counter(coframe_basis)
            repeated_elements = [item for item, count in counter.items() if count > 1]
            raise TypeError(
                f"`coframe_basis` should not have repeated elements\nRepeated elements: {repeated_elements}"
            )
        # Validate basis elements
        for df in coframe_basis:
            if not isinstance(df, abstract_differential_form_atom):
                raise TypeError(
                    f"Given `coframe_basis` should contain only `abstract_DF_atom` instances. Instead recieved {type(df)}"
                )

        # Validate structure_equations input
        if not isinstance(structure_equations, dict):
            raise TypeError("structure_equations must be a dictionary.")

        # Validate keys and values
        for key, value in structure_equations.items():
            if key not in coframe_basis:
                raise TypeError(
                    f"Keys in `structure_equations` dict must be also be in `coframe_basis`.\nErroneus key recieved{key}"
                )
            if value is None:
                structure_equations[key] = abstract_differential_form([])
            elif not isinstance(value, abstract_differential_form):
                raise TypeError(
                    "Values in structure_equations dict must be abstract_DF instances or None."
                )

        forms_tuple = tuple(coframe_basis)
        for df in forms_tuple:
            structure_equations.setdefault(df, abstract_differential_form([]))
        #### the following may be a useful alternative for resolving ordering ambiguity
        # forms_tuple = tuple(sorted(structure_equations.keys(), key=lambda x: (x.degree, x.label)))

        conj_atoms = list(min_conj_rules.keys()) + list(min_conj_rules.values())
        if (
            len(conj_atoms) != len(set(conj_atoms))
            or not all(isinstance(j, numbers.Integral) for j in conj_atoms)
            or not all(j in range(len(forms_tuple)) for j in conj_atoms)
        ):
            raise ValueError(
                f"`conj_rules` should be an invertible dict containing `int` indices in the range 0 to {len(forms_tuple)}"
            )
        # Create the instance
        self.forms = forms_tuple
        self.structure_equations = structure_equation_system(structure_equations)
        self.min_conj_rules = min_conj_rules
        self.inverted_conj_rules = {v: k for k, v in min_conj_rules.items()}
        self.conj_rules = (
            min_conj_rules
            | {v: k for k, v in min_conj_rules.items()}
            | {j: j for j in range(len(forms_tuple)) if j not in conj_atoms}
        )
        self.hash_key = create_key("hash_key", key_length=16)
        self._hash = hash(self.hash_key)
        self.dimension = len(forms_tuple)
        self._commute_cache = {}
        self._abstract_frame = None
        self.structure_equations._owner = self
        self.dual_frame_in_coordinates = _validated_slot(frame, self.dimension, "vector_field", "frame")
        self.forms_in_coordinates = _validated_slot(
            coordinate_forms, self.dimension, "differential_form", "coordinate_forms"
        )
        self.parameters = frozenset() if parameters is None else frozenset(parameters)
        if algebra is not None and get_dgcv_category(algebra) not in ("algebra", "subalgebra"):
            raise TypeError("`algebra` must be a dgcv algebra or subalgebra")
        self.algebra = algebra
        self.potentials = dict(potentials or {})
        for form in self.potentials:
            if form not in self.forms:
                raise ValueError(f"{form} is not a basis 1-form of this coframe")
        if potential_index is None:
            potential_index = {
                value: {self.forms.index(form): 1}
                for form, value in self.potentials.items()
                if isinstance(value, zero_form_atom)
            }
        self._potential_index = dict(potential_index)

    def _check_differential(self, form, name, real=False, constant=False, independent=False):
        if constant or name in {_potential_name(p) for p in self.parameters}:
            raise ValueError(f"{name} is constant, so {form} cannot be its differential")
        if independent:
            raise ValueError(f"{name} is independent of {form}, so {form} cannot be its differential")
        index = self.forms.index(form)
        if real and self.conj_rules[index] != index:
            raise ValueError(
                f"{name} is real, so {form} cannot be its differential: "
                f"{form} and its conjugate {self.forms[self.conj_rules[index]]} are independent"
            )

    def _potential_writes(self, form, atom):
        if form not in self.forms:
            raise ValueError(f"{form} is not a basis 1-form of this coframe")
        if not isinstance(atom, zero_form_atom) or atom.coframe_derivatives:
            raise TypeError("a potential must be a zero-form atom, not a derivative")
        writes = [(form, atom)]
        partner = self.forms[self.conj_rules[self.forms.index(form)]]
        if partner != form:
            conj = _routed_conjugate(atom)
            if isinstance(conj, zero_form_atom) and conj != atom:
                writes.append((partner, conj))
        for target, value in writes:
            if target in self.potentials:
                raise ValueError(f"{target} already has the potential {self.potentials[target]}")
            if value in self._potential_index:
                raise ValueError(f"{value} already has a recorded differential on this coframe")
        return writes

    def _write_potentials(self, writes):
        for form, atom in writes:
            self.potentials[form] = atom
            self._potential_index[atom] = {self.forms.index(form): 1}
        _record_potentials(self)

    def _set_potential(self, form, atom):
        self._write_potentials(self._potential_writes(form, atom))

    def _drop_potentials(self, names):
        names = set(names)
        kept = {
            form: value
            for form, value in self.potentials.items()
            if not names & {_potential_name(atom) for atom in get_free_symbols(value)}
        }
        rows = {atom: row for atom, row in self._potential_index.items() if atom.label not in names}
        if len(kept) == len(self.potentials) and len(rows) == len(self._potential_index):
            return
        self.potentials = kept
        self._potential_index = rows
        _record_potentials(self)

    @property
    def is_exact(self):
        return all(form in self.potentials for form in self.forms)

    def _invalidate_caches(self):
        self._commute_cache.clear()
        if self._abstract_frame is not None:
            self._abstract_frame._reset()

    def dual(self):
        if self._abstract_frame is None:
            self._abstract_frame = abstract_frame(self)
        return self._abstract_frame

    def __str__(self):
        return self.__repr__()

    def _sage_(self):
        raise AttributeError

    def __iter__(self):
        return iter(self.forms)

    def __getitem__(self, indices):
        if isinstance(indices, numbers.Integral):
            return self.forms[indices]
        if isinstance(indices, (list, tuple)):
            if len(indices) == 1:
                return self.forms[indices[0]]
            if len(indices) == 2:
                return tuple(
                    self.coefficient(indices[0], indices[1], of=k)
                    for k in range(self.dimension)
                )
            if len(indices) == 3:
                return self.coefficient(indices[0], indices[1], of=indices[2])
            raise TypeError(
                f"Expected one, two, or three indices. Received {len(indices)}: {indices}"
            )
        raise TypeError(
            f"Index a coframe with one integer for a basis 1-form, two for the tuple of coefficients of that wedge pair over the structure equations, or three for one structure coefficient. Received: {indices}"
        )

    def _form_index(self, form):
        if isinstance(form, numbers.Integral):
            if not -self.dimension <= form < self.dimension:
                raise IndexError(
                    f"index {form} out of bounds; expected in range 0 to {self.dimension - 1}"
                )
            return int(form) % self.dimension
        if form in self.forms:
            return self.forms.index(form)
        raise ValueError(f"{form} is not a basis 1-form of {self}")

    def _equation_coefficient(self, i, j, k):
        eqn = self.structure_equations.get(self.forms[k])
        if not hasattr(eqn, "coefficient"):
            return 0
        return eqn.coefficient(self.forms[i], self.forms[j])

    def coefficient(self, form1, form2, *, of=None):
        if of is None:
            raise TypeError(
                "`coefficient` needs `of=<1-form or index>` naming the structure equation to read"
            )
        i, j, k = (self._form_index(f) for f in (form1, form2, of))
        return self._equation_coefficient(i, j, k)

    def coefficients(self, of=None, containing=None):
        targets = (
            range(self.dimension)
            if of is None
            else [self._form_index(f) for f in _as_sequence(of)]
        )
        keep = (
            None
            if containing is None
            else {self._form_index(f) for f in _as_sequence(containing)}
        )
        out = {}
        for k in targets:
            eqn = self.structure_equations.get(self.forms[k])
            if not hasattr(eqn, "coefficients"):
                continue
            found = []
            for tail, scalar in eqn.coefficients().items():
                if len(tail) != 2 or any(f not in self.forms for f in tail):
                    continue
                i, j = sorted(self.forms.index(f) for f in tail)
                if keep is not None and i not in keep and j not in keep:
                    continue
                value = _wedge_coefficient((self.forms[i], self.forms[j]), tail, scalar)
                if not _scalar_is_zero(value):
                    found.append(((i, j), _bare_scalar(value)))
            for (i, j), value in sorted(found, key=lambda item: item[0]):
                out[(self.forms[i], self.forms[j], self.forms[k])] = value
        return out

    def coefficient_atoms(self, of=None):
        targets = (
            range(self.dimension)
            if of is None
            else [self._form_index(f) for f in _as_sequence(of)]
        )
        found = set()
        for k in targets:
            eqn = self.structure_equations.get(self.forms[k])
            for atom in getattr(eqn, "free_symbols", ()):
                if isinstance(atom, zero_form_atom) and _is_structure_coefficient_label(
                    atom.label
                ):
                    found.add(atom)
        return sorted(
            found, key=lambda atom: (atom.differential_order, atom.sort_key())
        )

    @property
    def free_symbols(self):
        return self.structure_equations.free_symbols

    def export_algebra_data(self, base_field=None):
        if base_field == "real":
            return _bridges.algebra_data(_bridges.real_basis(self), source=self, real=True)
        return _bridges.algebra_data(self)

    def is_Lie_algebra(self):
        return _bridges.is_Lie_algebra(self)

    def structure_coeff(self, lo1, lo2, hi):
        if not (
            0 <= lo1 < self.dimension
            and 0 <= lo2 < self.dimension
            and 0 <= hi < self.dimension
        ):
            raise IndexError(
                f"indices out of bounds: lo1={lo1}, lo2={lo2}, hi={hi}; "
                f"expected in range 0 to {self.dimension - 1}"
            )
        df = self.structure_equations[self.forms[hi]]
        form1 = self.forms[lo1]
        form2 = self.forms[lo2]
        coeff = 0

        if isinstance(
            df, (abstract_differential_form_atom, abstract_differential_form_monomial)
        ):
            df = abstract_differential_form([df])

        if isinstance(df, abstract_differential_form):
            for term in df.terms:
                scale = 1
                termCoeff = term.coeff
                if (
                    isinstance(termCoeff, abstract_differential_form_atom)
                    and termCoeff.degree == 0
                ):
                    scale = termCoeff
                elif isinstance(
                    termCoeff, (zero_form_atom, zero_form_class)
                ) or isinstance(termCoeff, expr_numeric_types()):
                    scale = termCoeff

                for count, factor in enumerate(term.factors_sorted):
                    if factor == form1:
                        if form2 in term.factors_sorted[count + 1 :]:
                            return scale
                    elif factor == form2:
                        if form1 in term.factors_sorted[count + 1 :]:
                            return -scale

        return coeff

    def _directions_commute(self, i, j):
        if i == j:
            return True
        key = (i, j) if i < j else (j, i)
        cache = self._commute_cache
        value = cache.get(key)
        if value is None:
            value = all(
                _scalar_is_zero(self.structure_coeff(key[0], key[1], k))
                for k in range(self.dimension)
            )
            cache[key] = value
        return value

    def _all_directions_commute(self):
        value = self._commute_cache.get("all")
        if value is None:
            value = all(
                self._directions_commute(i, j)
                for i in range(self.dimension)
                for j in range(i + 1, self.dimension)
            )
            self._commute_cache["all"] = value
        return value

    @property
    def is_zero(self):
        return False

    @property
    def is_holonomic(self):
        return self._all_directions_commute()

    def _slots(self):
        return {
            "frame": self.dual_frame_in_coordinates,
            "coordinate_forms": self.forms_in_coordinates,
            "parameters": self.parameters,
            "algebra": self.algebra,
            "potentials": dict(self.potentials),
            "potential_index": dict(self._potential_index),
        }

    def _rebuilt(self, equations):
        atoms = tuple(
            abstract_differential_form_atom(
                1,
                1,
                label=form.label,
                _markers=frozenset(m for m in form._markers if isinstance(m, str)),
            )
            for form in self.forms
        )
        mapping = dict(zip(self.forms, atoms))
        rebuilt = {}
        for form, eqn in equations.items():
            if eqn is None or not hasattr(eqn, "coefficients"):
                rebuilt[mapping[form]] = abstract_differential_form([])
                continue
            terms = []
            for tail, scalar in eqn.coefficients().items():
                term = scalar
                for atom in tail:
                    term = term * mapping.get(atom, atom)
                terms.append(term)
            rebuilt[mapping[form]] = abstract_differential_form(terms)
        coframe = abstract_coframe(atoms, rebuilt, self.min_conj_rules, **self._slots())
        coframe._own()
        return coframe

    def _own(self, forms=None):
        for form in self.forms if forms is None else forms:
            form._markers = frozenset(form._markers) | {self}

    def _map_structure_equations(self, fn):
        return self._rebuilt(
            {key: fn(value) for key, value in self.structure_equations.items()}
        )

    def subs(self, data, with_diff_corollaries=False, with_conjugates=None):
        close = with_diff_corollaries if with_conjugates is None else with_conjugates
        if close and isinstance(data, dict):
            data = _conjugate_closure(data, "subs")
        return self._map_structure_equations(
            lambda eqn: eqn.subs(
                data, with_diff_corollaries=with_diff_corollaries, with_conjugates=False
            )
        )

    def _subs_dgcv(self, data, with_diff_corollaries=False, with_conjugates=None):
        return self.subs(
            data, with_diff_corollaries=with_diff_corollaries, with_conjugates=with_conjugates
        )

    def _conjugate_equations(self, replace_eqns):
        out = dict(replace_eqns)
        for key, value in replace_eqns.items():
            if key not in self.forms:
                continue
            partner = self.forms[self.conj_rules[self.forms.index(key)]]
            if partner == key or partner in out:
                continue
            if value is None:
                out[partner] = None
            elif hasattr(value, "_eval_conjugate"):
                out[partner] = value._eval_conjugate()
            else:
                out[partner] = _routed_conjugate(value)
        return out

    def _eval_simplify(self, **kwargs):
        constraints = kwargs.pop("constraints", None)
        with_conjugates = kwargs.pop("with_conjugates", True)
        if constraints is not None:
            return _simplify_under_constraints(self, constraints, with_conjugates, **kwargs)
        return self._map_structure_equations(lambda e: _routed_simplify(e, **kwargs))

    def copy(self):
        """
        Return another abstract_coframe instance with the same coframe_basis and current structure equations, but with a new hash key. Useful for modifying structure equations of the copy without changing the original.
        """
        return self._rebuilt(dict(self.structure_equations))

    def __eq__(self, other):
        if not isinstance(other, abstract_coframe):
            return NotImplemented
        return self.forms == other.forms and self.hash_key == other.hash_key

    def __hash__(self):
        return self._hash

    def __reduce__(self):
        if self is _empty_coframe():
            return (_empty_coframe, ())
        state = {key: value for key, value in self.__dict__.items() if key != "_hash"}
        return (abstract_coframe._from_pickle, (self.hash_key,), state)

    @classmethod
    def _from_pickle(cls, hash_key):
        obj = object.__new__(cls)
        obj.hash_key = hash_key
        obj._hash = hash(hash_key)
        return obj

    def sort_key(self, order=None):  # for the sympy sorting.py default_sort_key
        return (10, self.forms)  # 10 is to group with misc objects

    def __lt__(self, other):
        if not isinstance(other, abstract_coframe):
            return NotImplemented
        return self.hash_key < other.hash_key

    def __repr__(self):
        """
        String representation of the coframe as a list of 1-forms.
        """
        return f"abstract_coframe({', '.join(map(str, self.forms))})"

    def _latex(self, printer=None, raw=True, **kwargs):
        """
        LaTeX representation of the coframe as a list of 1-forms.
        """
        return r"\{" + r", ".join(_routed_latex(form) for form in self.forms) + r"\}"

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex() if raw else f"${self._latex()}$"

    def update_structure_equations(
        self, replace_symbols=None, replace_eqns=None, simplify=True, with_conjugates=True
    ):
        """
        Update the structure equation for a specific form.

        Parameters:
        - form: An abstract_DF_atom instance representing the 1-form to update.
        - equation: An abstract_DF instance or None representing the new structure equation.
        """
        if replace_symbols is None:
            replace_symbols = {}
        if replace_eqns is None:
            replace_eqns = {}
        if not (isinstance(replace_symbols, dict) and isinstance(replace_eqns, dict)):
            raise TypeError(
                "If specified, `replace_symbols` and `replace_eqns` should be `dict` type"
            )
        if with_conjugates:
            replace_symbols = _conjugate_closure(replace_symbols, "update")
            replace_eqns = self._conjugate_equations(replace_eqns)
        self._invalidate_caches()
        for key in replace_symbols.keys():
            if key in self.structure_equations.keys():
                dgcv_warning(
                    "It appears `replace_symbols` dictionary passed to `update_structure_equations` contains a 1-form in the coframe. Probably this dictionary key-value pair should be assigned to the `replace_eqns` dictionary instead, i.e. use `update_structure_equations(replace_eqns=...)` instead "
                )
        for key, value in self.structure_equations.items():
            if hasattr(value, "subs") and callable(getattr(value, "subs")):
                if simplify:
                    if isinstance(value, abstract_differential_form_atom):
                        self.structure_equations[key] = (
                            value.subs(replace_symbols)
                        )._eval_simplify()
                    else:
                        self.structure_equations[key] = _routed_simplify(
                            value.subs(replace_symbols)
                        )
                else:
                    self.structure_equations[key] = value.subs(replace_symbols)
        for key, value in replace_eqns.items():
            if key in self.forms:
                self.structure_equations[key] = value
        if simplify:
            for key, value in self.structure_equations.items():
                if value is None:
                    self.structure_equations[key] = abstract_differential_form([])
                elif isinstance(value, abstract_differential_form_atom):
                    self.structure_equations[key] = value._eval_simplify()
                else:
                    self.structure_equations[key] = _routed_simplify(value)
        self._invalidate_caches()


# old version compatibility stuff
register_legacy_sympy_class(abstract_coframe)

abst_coframe = abstract_coframe
