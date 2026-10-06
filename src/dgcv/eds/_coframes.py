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
from .._aux._backends._symbolic_router import simplify as _routed_simplify
from .._aux._backends._types_and_constants import expr_numeric_types
from .._aux._utilities._config import dgcv_warning
from .._aux._vmf._safeguards import create_key, retrieve_passkey
from ..core.base import dgcv_class
from ._atoms import zero_form_atom
from ._forms import (
    abstract_differential_form,
    abstract_differential_form_atom,
    abstract_differential_form_monomial,
)
from ._zero_forms import zero_form_class


class abstract_coframe(dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "abst_coframe"

    def _eval_conjugate(self):
        return self

    def __dgcv_conjugate__(self, symbolic=False):
        return self._eval_conjugate()

    def __init__(self, coframe_basis, structure_equations, min_conj_rules={}):
        """
        Create a new abstract_coframe instance.

        Parameters:
        ==========
            - structure_equations (dict): A dictionary where keys are abstract_DF_atom instances (representing 1-forms), and values are either abstract_DF instances (representing the differential) or None.
        """
        if not isinstance(coframe_basis, (list, tuple)) and len(coframe_basis) > 0:
            raise TypeError(
                f"Given `coframe_basis` must be a non-empty list/tuple. Instead recieved type {type(coframe_basis)}"
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
        self.structure_equations = structure_equations  # dictionary (mutable)
        self.min_conj_rules = min_conj_rules
        self.inverted_conj_rules = {v: k for k, v in min_conj_rules.items()}
        self.conj_rules = (
            min_conj_rules
            | {v: k for k, v in min_conj_rules.items()}
            | {j: j for j in range(len(forms_tuple)) if j not in conj_atoms}
        )
        self.hash_key = create_key("hash_key", key_length=16)
        self.dimension = len(forms_tuple)

    def __str__(self):
        return self.__repr__()

    def _sage_(self):
        raise AttributeError

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
            df = abstract_differential_form(df)

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

    @property
    def is_zero(self):
        return False

    def _map_structure_equations(self, fn):
        return abstract_coframe(
            self.forms,
            {key: fn(value) for key, value in self.structure_equations.items()},
            self.min_conj_rules,
        )

    def subs(self, data, with_diff_corollaries=False):
        return self._map_structure_equations(
            lambda eqn: eqn.subs(data, with_diff_corollaries=with_diff_corollaries)
        )

    def _subs_dgcv(self, data, with_diff_corollaries=False):
        return self.subs(data, with_diff_corollaries=with_diff_corollaries)

    def _eval_simplify(self, **kwargs):
        return self._map_structure_equations(lambda e: _routed_simplify(e, **kwargs))

    def copy(self):
        """
        Return another abstract_coframe instance with the same coframe_basis and current structure equations, but with a new hash key. Useful for modifying structure equations of the copy without changing the original.
        """
        return abstract_coframe(
            self.forms, self.structure_equations, self.min_conj_rules
        )

    def __eq__(self, other):
        if not isinstance(other, abstract_coframe):
            return NotImplemented
        return self.forms == other.forms and self.hash_key == other.hash_key

    def __hash__(self):
        return hash((self.forms, self.hash_key))

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
        self, replace_symbols={}, replace_eqns={}, simplify=True
    ):
        """
        Update the structure equation for a specific form.

        Parameters:
        - form: An abstract_DF_atom instance representing the 1-form to update.
        - equation: An abstract_DF instance or None representing the new structure equation.
        """
        if not (isinstance(replace_symbols, dict) and isinstance(replace_eqns, dict)):
            raise TypeError(
                "If specified, `replace_symbols` and `replace_eqns` should be `dict` type"
            )
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


# old version compatibility stuff
register_legacy_sympy_class(abstract_coframe)

abst_coframe = abstract_coframe
