"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

Description:
------------
This sub-package provides functionality for abstract exterior differential systems (EDS).

Modules included:
    - eds: Core EDS definitions and functions.
    - eds_representations: Classes and methods to handle EDS matrix representations.
    - eds_operations: Additional operations for EDS objects.

---
Author (of this sub-package): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from . import _atoms as _atoms_impl
from . import _zero_forms as _expr_impl
from . import _calculus
from ._atoms import zero_form_atom, zeroFormAtom
from ._calculus import coframe_derivative, extDer, simplify_with_PDEs
from ._coframes import abst_coframe, abstract_coframe
from ._creators import createCoframe, createDiffForm, createZeroForm
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
from ._zero_forms import abstract_ZF, zero_form_class
from .eds_operations import transform_coframe
from .eds_representations import DF_representation

_atoms_impl.zero_form_class = zero_form_class
_atoms_impl.abstract_differential_form_atom = abstract_differential_form_atom
_atoms_impl.abstract_differential_form_monomial = abstract_differential_form_monomial
_atoms_impl.abstract_differential_form = abstract_differential_form
_atoms_impl.abstract_coframe = abstract_coframe
_atoms_impl.coframe_derivative = coframe_derivative
_atoms_impl._swap_CFD_order = _calculus._swap_CFD_order
_atoms_impl._eds_lift = _expr_impl._eds_lift
_expr_impl.abstract_differential_form_atom = abstract_differential_form_atom
_expr_impl.abstract_differential_form_monomial = abstract_differential_form_monomial
_expr_impl.abstract_differential_form = abstract_differential_form

__all__ = [
    "zero_form_atom",
    "zero_form_class",
    "abstract_differential_form",
    "abstract_differential_form_atom",
    "abstract_differential_form_monomial",
    "zeroFormAtom",
    "createZeroForm",
    "createDiffForm",
    "abst_coframe",
    "abstract_coframe",
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
    "DF_representation",
    "transform_coframe",
]
