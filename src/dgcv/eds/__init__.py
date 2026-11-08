"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

Description:
------------
This sub-package provides functionality for abstract exterior differential systems (EDS).

Modules included:
    - _atoms: zero-form atoms (the builtin engine's symbols).
    - _zero_forms, _zf_ops, _roles, _constants: zero forms, their operator tags, complex roles and the I / e / pi leaves.
    - _forms: abstract differential form atoms, monomials and sums.
    - _coframes: abstract coframes and their structure equations.
    - _vectors: abstract frames and vector fields.
    - _calculus: exterior and coframe derivatives, conjugate closure, simplification under constraints.
    - _creators: createCoframe, createDiffForm, createZeroForm, createFrame.
    - _bridges: coordinate coframes and frames as abstract ones.
    - eds_operations: transform_coframe and related operations.
    - eds_representations: DF_representation, connection_form.
    - Pfaff_systems: Pfaff systems, jet spaces and the Cartan-Kahler invariants (`from dgcv.eds.Pfaff_systems import *`).
    - eds, ast: re-export and deprecation shims.

---
Author (of this sub-package): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from . import _atoms as _atoms_impl
from . import (
    _bridges,  # noqa: F401
    _calculus,
)
from . import _coframes as _coframes_impl
from . import _forms as _forms_impl
from . import _vectors as _vectors_impl
from . import _zero_forms as _expr_impl
from ._atoms import zero_form_atom, zeroFormAtom
from ._calculus import (
    coframe_derivative,
    conjugate_closure,
    derivative_atoms,
    extDer,
    simplify_with_PDEs,
)
from ._coframes import abst_coframe, abstract_coframe
from ._creators import createCoframe, createDiffForm, createFrame, createZeroForm
from ._forms import (
    abstDFAtom,
    abstDFMonom,
    abstract_DF,
    abstract_DF_atom,
    abstract_DF_monomial,
    abstract_differential_form,
    abstract_differential_form_atom,
    abstract_differential_form_monomial,
)
from ._vectors import (
    abst_frame,
    abstract_frame,
    abstract_vector_field,
    abstVF,
    annihilator,
)
from ._zero_forms import abstract_ZF, zero_form_class
from .eds_operations import transform_coframe
from .eds_representations import DF_representation, connection_form
from .Pfaff_systems import createJetSpace

_atoms_impl.zero_form_class = zero_form_class
_atoms_impl.abstract_differential_form_atom = abstract_differential_form_atom
_atoms_impl.abstract_differential_form_monomial = abstract_differential_form_monomial
_atoms_impl.abstract_differential_form = abstract_differential_form
_atoms_impl.abstract_coframe = abstract_coframe
_atoms_impl.coframe_derivative = coframe_derivative
_atoms_impl._swap_CFD_order = _calculus._swap_CFD_order
_atoms_impl._eds_lift = _expr_impl._eds_lift
_atoms_impl._conjugate_closure = _calculus._conjugate_closure
_expr_impl._conjugate_closure = _calculus._conjugate_closure
_forms_impl._conjugate_closure = _calculus._conjugate_closure
_coframes_impl._conjugate_closure = _calculus._conjugate_closure
_atoms_impl._simplify_under_constraints = _calculus._simplify_under_constraints
_expr_impl._simplify_under_constraints = _calculus._simplify_under_constraints
_forms_impl._simplify_under_constraints = _calculus._simplify_under_constraints
_coframes_impl._simplify_under_constraints = _calculus._simplify_under_constraints
_bridges.transform_coframe = transform_coframe
_bridges.extDer = extDer
_expr_impl.abstract_differential_form_atom = abstract_differential_form_atom
_expr_impl.abstract_differential_form_monomial = abstract_differential_form_monomial
_expr_impl.abstract_differential_form = abstract_differential_form
_coframes_impl.abstract_frame = abstract_frame
_forms_impl.abstract_frame = abstract_frame
_vectors_impl.coframe_derivative = coframe_derivative
_vectors_impl._conjugate_closure = _calculus._conjugate_closure
_vectors_impl._simplify_under_constraints = _calculus._simplify_under_constraints

__all__ = [
    "annihilator",
    "zero_form_atom",
    "zero_form_class",
    "abstract_differential_form",
    "abstract_differential_form_atom",
    "abstract_differential_form_monomial",
    "zeroFormAtom",
    "createZeroForm",
    "createJetSpace",
    "createDiffForm",
    "abst_coframe",
    "abstract_coframe",
    "abst_frame",
    "abstract_frame",
    "abstract_vector_field",
    "abstVF",
    "createFrame",
    "abstDFAtom",
    "abstDFMonom",
    "abstract_DF_atom",
    "abstract_DF_monomial",
    "createCoframe",
    "abstract_DF",
    "abstract_ZF",
    "extDer",
    "simplify_with_PDEs",
    "coframe_derivative",
    "conjugate_closure",
    "derivative_atoms",
    "DF_representation",
    "connection_form",
    "transform_coframe",
]
