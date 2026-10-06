"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

module: dgcv.eds.eds

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from ._atoms import (
    SortableObj,
    _custom_conj,
    barSortedStr,
    zero_form_atom,
    zeroFormAtom,
)
from ._calculus import (
    _cofrDer_abstract_ZF,
    _cofrDer_zeroFormAtom,
    _extDer_abstDFAtom,
    _extDer_abstDFMonom,
    _extDer_abstract_DF,
    _extDer_abstract_ZF,
    _extDer_zeroFormAtom,
    _swap_CFD_order,
    coframe_derivative,
    extDer,
    simplify_with_PDEs,
)
from ._coframes import abst_coframe, abstract_coframe
from ._creators import (
    _DFFactory,
    _zeroFormFactory,
    createCoframe,
    createDiffForm,
    createZeroForm,
)
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
from ._zero_forms import (
    _engine_to_abstract_ZF,
    _generate_str_id,
    _loop_ZF_format_conversions,
    _lower_zf_base,
    _zf_coeff_str,
    abstract_ZF,
    zero_form_class,
)
from ._zf_ops import OP_REGISTRY, OpSpec, op_spec, register_op

__all__ = [
    "zero_form_atom",
    "zero_form_class",
    "abstract_differential_form",
    "abstract_differential_form_atom",
    "abstract_differential_form_monomial",
    "OP_REGISTRY",
    "OpSpec",
    "SortableObj",
    "_DFFactory",
    "_cofrDer_abstract_ZF",
    "_cofrDer_zeroFormAtom",
    "_custom_conj",
    "_engine_to_abstract_ZF",
    "_extDer_abstDFAtom",
    "_extDer_abstDFMonom",
    "_extDer_abstract_DF",
    "_extDer_abstract_ZF",
    "_extDer_zeroFormAtom",
    "_generate_str_id",
    "_loop_ZF_format_conversions",
    "_lower_zf_base",
    "_swap_CFD_order",
    "_zeroFormFactory",
    "_zf_coeff_str",
    "abstDFAtom",
    "abstDFMonom",
    "abst_coframe",
    "abstract_DF",
    "abstract_DF_atom",
    "abstract_DF_monomial",
    "abstract_ZF",
    "abstract_coframe",
    "barSortedStr",
    "coframe_derivative",
    "createCoframe",
    "createDiffForm",
    "createZeroForm",
    "extDer",
    "op_spec",
    "register_op",
    "simplify_with_PDEs",
    "zeroFormAtom",
]

_DEPRECATED_PRIVATE = {
    "_equation_formatting": "dgcv.core.solvers._bridge.SolveBridge.lower",
    "_sympify_abst_ZF": "abstract_ZF.to_engine",
    "_sympy_to_abstract_ZF": "dgcv.eds._zero_forms._engine_to_abstract_ZF",
}


def __getattr__(name):
    if name in _DEPRECATED_PRIVATE:
        from .._aux._utilities._config import dgcv_warning, dgcvDeprecationWarning
        from . import _zero_forms

        dgcv_warning(
            f"`dgcv.eds.eds.{name}` is deprecated.",
            dgcvDeprecationWarning,
            stacklevel=2,
            old_kw=name,
            new_kw=_DEPRECATED_PRIVATE[name],
            sunset="2027",
        )
        return getattr(_zero_forms, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
