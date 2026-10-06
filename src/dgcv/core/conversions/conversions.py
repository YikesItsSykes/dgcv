"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.core

module: dgcv.core.conversions


---
Author (of this sub-package): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from __future__ import annotations

import numbers
from typing import Any, Dict, Optional, Set, Tuple

from ..._aux._backends._exact_arith import exact_reciprocal
from ..._aux._backends._symbolic_router import get_free_symbols, simplify, subs
from ..._aux._backends._types_and_constants import (
    expr_types,
    imag_unit,
    is_atomic,
    to_active_engine,
)
from ..._aux._utilities._config import get_variable_registry
from ..._aux._vmf._safeguards import check_dgcv_category
from ..._aux._vmf.vmf import vmf_lookup
from ..combinatorics.combinatorics import carProd_with_weights_without_R, permSign

__all__ = [
    "allToHol",
    "allToReal",
    "allToSym",
    "holToReal",
    "holToSym",
    "symToHol",
    "symToReal",
    "realToHol",
    "realToSym",
]


# -----------------------------------------------------------------------------
# converters
# -----------------------------------------------------------------------------
def holToReal(
    expr,
    skipVar=None,
    convert_everything: bool = True,
    *,
    _conversion_dict=None,
    variables_scope=None,
):
    vr = get_variable_registry()
    conv = _get_conv_map(vr, "holToReal", skipVar, _conversion_dict, variables_scope)

    hooked = _dgcv_convert_hook(
        expr,
        "holToReal",
        skipVar=skipVar,
        variables_scope=variables_scope,
        convert_everything=convert_everything,
        conversion_dict=conv,
    )
    if hooked is not None:
        return hooked

    if check_dgcv_category(expr):
        new = _converter_hook(
            expr,
            to_kind="real",
            converter=holToReal,
            skipVar=skipVar,
            variables_scope=variables_scope,
            convert_everything=convert_everything,
        )
        if new is not None:
            return new

    return _convert_expr(expr, conv, convert_everything=convert_everything)


def realToSym(
    expr,
    skipVar=None,
    convert_everything: bool = True,
    *,
    _conversion_dict=None,
    variables_scope=None,
):
    vr = get_variable_registry()
    conv = _get_conv_map(vr, "realToSym", skipVar, _conversion_dict, variables_scope)

    hooked = _dgcv_convert_hook(
        expr,
        "realToSym",
        skipVar=skipVar,
        variables_scope=variables_scope,
        convert_everything=convert_everything,
        conversion_dict=conv,
    )
    if hooked is not None:
        return hooked

    if check_dgcv_category(expr):
        new = _converter_hook(
            expr,
            to_kind="complex",
            converter=realToSym,
            skipVar=skipVar,
            variables_scope=variables_scope,
            convert_everything=convert_everything,
        )
        if new is not None:
            return new

    return _convert_expr(expr, conv, convert_everything=convert_everything)


def symToHol(
    expr,
    skipVar=None,
    convert_everything: bool = True,
    *,
    _conversion_dict=None,
    variables_scope=None,
):
    vr = get_variable_registry()
    conv = _get_conv_map(vr, "symToHol", skipVar, _conversion_dict, variables_scope)

    hooked = _dgcv_convert_hook(
        expr,
        "symToHol",
        skipVar=skipVar,
        variables_scope=variables_scope,
        convert_everything=convert_everything,
        conversion_dict=conv,
    )
    if hooked is not None:
        return hooked

    if check_dgcv_category(expr):
        new = _converter_hook(
            expr,
            to_kind="complex",
            converter=symToHol,
            skipVar=skipVar,
            variables_scope=variables_scope,
            convert_everything=convert_everything,
        )
        if new is not None:
            return new

    return _convert_expr(expr, conv, convert_everything=convert_everything)


def realToHol(
    expr,
    skipVar=None,
    convert_everything: bool = True,
    *,
    _conversion_dict=None,
    variables_scope=None,
):
    vr = get_variable_registry()
    conv = _get_conv_map(vr, "realToHol", skipVar, _conversion_dict, variables_scope)

    hooked = _dgcv_convert_hook(
        expr,
        "realToHol",
        skipVar=skipVar,
        variables_scope=variables_scope,
        convert_everything=convert_everything,
        conversion_dict=conv,
    )
    if hooked is not None:
        return hooked

    if check_dgcv_category(expr):
        new = _converter_hook(
            expr,
            to_kind="complex",
            converter=realToHol,
            skipVar=skipVar,
            variables_scope=variables_scope,
            convert_everything=convert_everything,
        )
        if new is not None:
            return new

    return _convert_expr(expr, conv, convert_everything=convert_everything)


def symToReal(
    expr,
    skipVar=None,
    convert_everything: bool = True,
    *,
    _conversion_dict=None,
    variables_scope=None,
):
    vr = get_variable_registry()
    conv = _get_conv_map(vr, "symToReal", skipVar, _conversion_dict, variables_scope)

    hooked = _dgcv_convert_hook(
        expr,
        "symToReal",
        skipVar=skipVar,
        variables_scope=variables_scope,
        convert_everything=convert_everything,
        conversion_dict=conv,
    )
    if hooked is not None:
        return hooked

    if check_dgcv_category(expr):
        new = _converter_hook(
            expr,
            to_kind="real",
            converter=symToReal,
            skipVar=skipVar,
            variables_scope=variables_scope,
            convert_everything=convert_everything,
        )
        if new is not None:
            return new

    return _convert_expr(expr, conv, convert_everything=convert_everything)


def holToSym(expr, skipVar=None, convert_everything: bool = True, variables_scope=None):
    expr = holToReal(
        expr,
        skipVar=skipVar,
        convert_everything=convert_everything,
        variables_scope=variables_scope,
    )
    expr = realToSym(
        expr,
        skipVar=skipVar,
        convert_everything=convert_everything,
        variables_scope=variables_scope,
    )
    return expr


def allToReal(
    expr, skipVar=None, convert_everything: bool = True, variables_scope=None
):
    return symToReal(
        expr,
        skipVar=skipVar,
        convert_everything=convert_everything,
        variables_scope=variables_scope,
    )


def allToHol(expr, skipVar=None, convert_everything: bool = True, variables_scope=None):
    ###!!! refine later
    expr = symToHol(
        expr,
        skipVar=skipVar,
        convert_everything=convert_everything,
        variables_scope=variables_scope,
    )
    return realToHol(
        expr,
        skipVar=skipVar,
        convert_everything=convert_everything,
        variables_scope=variables_scope,
    )


def allToSym(expr, skipVar=None, convert_everything: bool = True, variables_scope=None):
    return holToSym(
        expr,
        skipVar=skipVar,
        convert_everything=convert_everything,
        variables_scope=variables_scope,
    )


def cleanUpConjugation(arg1, skipVar=None, variables_scope=None):
    return realToSym(
        symToReal(
            arg1,
            skipVar=skipVar,
            convert_everything=False,
            variables_scope=variables_scope,
        ),
        skipVar=skipVar,
        convert_everything=False,
        variables_scope=variables_scope,
    )


# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------
categories = {"vector_field", "differential_form"}


def _maybe_to_real(expr, *, skipVar, variables_scope, convert_everything: bool):
    if not convert_everything:
        return None
    f = getattr(expr, "_to_real", None)
    if not callable(f):
        return None
    plan = {"skipVar": skipVar, "variables_scope": variables_scope}
    try:
        return f(plan=plan)
    except TypeError:
        return f(plan)


def _maybe_to_complex(expr, *, skipVar, variables_scope, convert_everything: bool):
    if not convert_everything:
        return None
    f = getattr(expr, "_to_complex", None)
    if not callable(f):
        return None
    plan = {"skipVar": skipVar, "variables_scope": variables_scope}
    try:
        return f(plan=plan)
    except TypeError:
        return f(plan)


def _maybe_apply(
    expr, converter, *, skipVar, variables_scope, convert_everything: bool
):
    ap = getattr(expr, "apply", None)
    if not callable(ap):
        return None

    def conv(e):
        return converter(
            e,
            skipVar=skipVar,
            variables_scope=variables_scope,
            convert_everything=convert_everything,
        )

    return ap(conv)


def _converter_hook(
    expr,
    *,
    to_kind: str,  # "real" or "complex"
    converter,
    skipVar=None,
    variables_scope=None,
    convert_everything: bool = True,
):
    out = None

    if to_kind == "real":
        out = _maybe_to_real(
            expr,
            skipVar=skipVar,
            variables_scope=variables_scope,
            convert_everything=convert_everything,
        )
    else:
        out = _maybe_to_complex(
            expr,
            skipVar=skipVar,
            variables_scope=variables_scope,
            convert_everything=convert_everything,
        )

    base = out if out is not None else expr

    applied = _maybe_apply(
        base,
        converter,
        skipVar=skipVar,
        variables_scope=variables_scope,
        convert_everything=convert_everything,
    )
    if applied is not None:
        return applied

    if out is not None:
        return out

    return None


def _iter_skip_items(x) -> Tuple[Any, ...]:
    if x is None:
        return ()
    if isinstance(x, (str, bytes)):
        return (x,)
    try:
        return tuple(x)
    except TypeError:
        return (x,)


def _vmf_relatives(obj) -> Optional[dict]:
    try:
        info = vmf_lookup(obj, relatives=True)
    except Exception:
        return None
    if not isinstance(info, dict):
        return None
    rel = info.get("relatives", None)
    return rel if isinstance(rel, dict) else None


def _vmf_system_label(obj) -> Optional[str]:
    rel = _vmf_relatives(obj)
    if not rel:
        return None
    lab = rel.get("system_label", None)
    return lab if isinstance(lab, str) and lab else None


def _vmf_relative_atoms(obj) -> Optional[Tuple[Any, ...]]:
    rel = _vmf_relatives(obj)
    if not rel:
        return None
    atoms = []
    for k in ("standard", "holo", "anti", "real", "imag"):
        v = rel.get(k, None)
        if v is not None:
            atoms.append(v)
    return tuple(atoms) or None


def _prune_conversion_dict_for_skipVar(conv, vr, skipVar):
    items = _iter_skip_items(skipVar)
    if not items:
        return conv

    system_labels: Set[str] = set()
    atomic_items = []

    for it in items:
        if isinstance(it, str):
            system_labels.add(it)
        else:
            atomic_items.append(it)

    if system_labels:
        cvs = vr.get("complex_variable_systems", {})
        for lab in system_labels:
            cs = cvs.get(lab, {})
            family_values = cs.get("family_values", ((), (), (), ()))

            if cs.get("family_type") == "single":
                family_values = tuple((j,) for j in family_values)

            for block in family_values:
                for sym in block or ():
                    conv.pop(sym, None)

    for it in atomic_items:
        lab = _vmf_system_label(it)
        if lab and lab in system_labels:
            continue

        atoms = _vmf_relative_atoms(it)
        if not atoms:
            continue

        for sym in atoms:
            conv.pop(sym, None)

    return conv


def _scope_atoms(vr, variables_scope) -> Optional[Set[Any]]:
    items = _iter_skip_items(variables_scope)
    if not items:
        return None

    system_labels: Set[str] = set()
    atomic_items = []

    for it in items:
        if isinstance(it, str):
            system_labels.add(it)
        else:
            atomic_items.append(it)

    out: Set[Any] = set()

    if system_labels:
        cvs = vr.get("complex_variable_systems", {})
        for lab in system_labels:
            cs = cvs.get(lab, {})
            family_values = cs.get("family_values", ((), (), (), ()))

            if cs.get("family_type") == "single":
                family_values = tuple((j,) for j in family_values)

            for block in family_values:
                for sym in block or ():
                    out.add(sym)

    for it in atomic_items:
        lab = _vmf_system_label(it)
        if lab and lab in system_labels:
            continue

        atoms = _vmf_relative_atoms(it)
        if not atoms:
            continue

        out.update(atoms)

    return out


def _filter_conv_to_scope(base, allowed: Optional[Set[Any]]):
    if allowed is None:
        return base

    out = {}
    for k in allowed:
        try:
            v = base.get(k, None)
        except Exception:
            try:
                v = base[k]
            except Exception:
                v = None
        if v is not None:
            out[k] = v
    return out


def _dgcv_convert_hook(
    expr,
    conv,
    *,
    skipVar=None,
    variables_scope=None,
    convert_everything=True,
    conversion_dict=None,
):
    f = getattr(expr, "__dgcv_converter__", None)
    if not callable(f):
        return None

    try:
        return f(
            conv,
            skipVar=skipVar,
            variables_scope=variables_scope,
            convert_everything=convert_everything,
            conversion_dict=conversion_dict,
        )
    except TypeError:
        return f(
            conv,
            skipVar=skipVar,
            convert_everything=convert_everything,
            conversion_dict=conversion_dict,
        )


def _get_conv_map(vr, kind: str, skipVar, _conversion_dict, variables_scope):
    base = _conversion_dict
    if base is None:
        base = vr.get("conversion_dictionaries", {}).get(kind, {})

    allowed = _scope_atoms(vr, variables_scope)
    if allowed is not None:
        base = _filter_conv_to_scope(base, allowed)

    if not skipVar:
        return base

    try:
        conv = base.copy()
    except Exception:
        conv = dict(base)

    _prune_conversion_dict_for_skipVar(conv, vr, skipVar)
    return conv


def _filtered_subs(expr, conv) -> Optional[Dict[Any, Any]]:
    syms = get_free_symbols(expr)
    if not syms:
        return None

    out = {}
    for s in syms:
        try:
            v = conv[s]
        except Exception:
            try:
                v = conv.get(s, None)
            except Exception:
                v = None
        if v is not None:
            out[s] = v

    return out or None


def _convert_expr(expr, conv, *, convert_everything: bool):
    expr = to_active_engine(expr)
    if isinstance(expr, numbers.Number):
        return expr

    if is_atomic(expr):
        try:
            return conv.get(expr, expr)
        except Exception:
            return expr

    if isinstance(expr, expr_types()):
        fs = _filtered_subs(expr, conv)
        if not fs:
            return expr
        res = subs(expr, fs)
        return simplify(res) if convert_everything else res

    ap = getattr(expr, "applyfunc", None)
    if callable(ap):
        return ap(
            lambda e: _convert_expr(e, conv, convert_everything=convert_everything)
        )

    return expr


# -----------------------------------------------------------------------------
# for tensors
# -----------------------------------------------------------------------------
def _coeff_dict_formatter(
    varSpace, coeff_dict, valence, total_degree, _varSpace_type, data_shape
):
    """
    Helper function to populate conversion dicts for tensor field classes
    """
    variable_registry = get_variable_registry()
    CVS = variable_registry["complex_variable_systems"]

    exhaust1 = list(varSpace)
    populate = {
        "compCoeffDataDict": dict(),
        "realCoeffDataDict": dict(),
        "holVarDict": dict(),
        "antiholVarDict": dict(),
        "realVarDict": dict(),
        "imVarDict": dict(),
        "preProcessMinDataToHol": dict(),
        "preProcessMinDataToReal": dict(),
    }
    if _varSpace_type == "real":
        for var in varSpace:
            varStr = str(var)
            if var in exhaust1:
                for parent in CVS.values():
                    if varStr in parent["variable_relatives"]:
                        cousin = (
                            set(
                                parent["variable_relatives"][varStr]["complex_family"][
                                    2:
                                ]
                            )
                            - {var}
                        ).pop()
                        if cousin in exhaust1:
                            exhaust1.remove(cousin)
                        if (
                            parent["variable_relatives"][varStr]["complex_positioning"]
                            == "real"
                        ):
                            realVar = var
                            exhaust1.remove(var)
                            imVar = cousin
                        else:
                            realVar = cousin
                            exhaust1.remove(var)
                            imVar = var
                        holVar = parent["variable_relatives"][varStr]["complex_family"][
                            0
                        ]
                        antiholVar = parent["variable_relatives"][varStr][
                            "complex_family"
                        ][1]
                        populate["holVarDict"][holVar] = [realVar, imVar]
                        populate["antiholVarDict"][antiholVar] = [
                            realVar,
                            imVar,
                        ]
                        populate["realVarDict"][realVar] = [holVar, antiholVar]
                        populate["imVarDict"][imVar] = [holVar, antiholVar]
    else:  # _varSpace_type == 'complex'
        for var in varSpace:
            varStr = str(var)
            if var in exhaust1:
                for parent in CVS.values():
                    if varStr in parent["variable_relatives"]:
                        cousin = (
                            set(
                                parent["variable_relatives"][varStr]["complex_family"][
                                    :2
                                ]
                            )
                            - {var}
                        ).pop()
                        if cousin in exhaust1:
                            exhaust1.remove(cousin)
                        if (
                            parent["variable_relatives"][varStr]["complex_positioning"]
                            == "holomorphic"
                        ):
                            holVar = var
                            exhaust1.remove(var)
                            antiholVar = cousin
                        else:
                            holVar = cousin
                            exhaust1.remove(var)
                            antiholVar = var
                        realVar = parent["variable_relatives"][varStr][
                            "complex_family"
                        ][2]
                        imVar = parent["variable_relatives"][varStr]["complex_family"][
                            3
                        ]
                        populate["holVarDict"][holVar] = [realVar, imVar]
                        populate["antiholVarDict"][antiholVar] = [
                            realVar,
                            imVar,
                        ]
                        populate["realVarDict"][realVar] = [holVar, antiholVar]
                        populate["imVarDict"][imVar] = [holVar, antiholVar]
    new_realVarSpace = tuple(populate["realVarDict"].keys())
    new_holVarSpace = tuple(populate["holVarDict"].keys())
    new_antiholVarSpace = tuple(populate["antiholVarDict"].keys())
    new_imVarSpace = tuple(populate["imVarDict"].keys())

    if len(valence) == 0:
        if _varSpace_type == "real":
            populate["realCoeffDataDict"] = [
                varSpace,
                coeff_dict,
            ]
            populate["compCoeffDataDict"] = [
                new_holVarSpace + new_antiholVarSpace,
                {(0,) * total_degree: coeff_dict[(0,) * total_degree]},
            ]
        else:
            populate["compCoeffDataDict"] = [
                varSpace,
                coeff_dict,
            ]
            populate["realCoeffDataDict"] = [
                new_realVarSpace + new_imVarSpace,
                {(0,) * total_degree: coeff_dict[(0,) * total_degree]},
            ]
    else:

        def _retrieve_indices(term, typeSet=None):
            if typeSet == "symb":
                dictLoc = populate["realVarDict"] | populate["imVarDict"]
                refTuple = new_holVarSpace + new_antiholVarSpace
                termList = dictLoc[term]
            elif typeSet == "real":
                dictLoc = populate["holVarDict"] | populate["antiholVarDict"]
                refTuple = new_realVarSpace + new_imVarSpace
                termList = dictLoc[term]
            index_a = refTuple.index(termList[0])
            index_b = refTuple.index(termList[1], index_a + 1)
            return [index_a, index_b]

        # set up the conversion dicts for index conversion
        if _varSpace_type == "real":
            populate["preProcessMinDataToHol"] = {
                j: _retrieve_indices(varSpace[j], "symb") for j in range(len(varSpace))
            }

        else:  # if _varSpace_type == 'complex'
            populate["preProcessMinDataToReal"] = {
                j: _retrieve_indices(varSpace[j], "real") for j in range(len(varSpace))
            }

        # coordinate VF and DF conversion
        def decorateWithWeights(index, variance_rule, target="symb"):
            if variance_rule == 0:  # covariant case
                covariance = True
            else:  # contravariant case
                covariance = False

            if target == "symb":
                if (
                    varSpace[index]
                    in variable_registry["conversion_dictionaries"][
                        "real_part"
                    ].values()
                ):
                    holScale = (
                        exact_reciprocal(2) if covariance else 1
                    )  # D_z (d_z) coeff of D_x (d_x)
                    antiholScale = (
                        exact_reciprocal(2) if covariance else 1
                    )  # D_BARz (d_BARz) coeff of D_x (d_x)
                else:
                    holScale = (
                        -imag_unit() / 2 if covariance else imag_unit()
                    )  # D_z (d_z) coeff of D_y (d_y)
                    antiholScale = (
                        imag_unit() / 2 if covariance else -imag_unit()
                    )  # d_BARz (D_BARz) coeff of d_y (D_y)
                return [
                    [populate["preProcessMinDataToHol"][index][0], holScale],
                    [
                        populate["preProcessMinDataToHol"][index][1],
                        antiholScale,
                    ],
                ]
            else:  # converting from hol to real
                if (
                    varSpace[index]
                    in variable_registry["conversion_dictionaries"]["holToReal"]
                ):
                    realScale = (
                        1 if covariance else exact_reciprocal(2)
                    )  # D_x (d_x) coeff in D_z (d_z)
                    imScale = (
                        imag_unit()
                        if covariance
                        else -imag_unit() * exact_reciprocal(2)
                    )  # D_y (d_y) coeff in D_z (d_z)
                else:
                    realScale = (
                        1 if covariance else exact_reciprocal(2)
                    )  # D_x (d_x) coeff of D_BARz (d_BARz)
                    imScale = (
                        -imag_unit()
                        if covariance
                        else imag_unit() * exact_reciprocal(2)
                    )  # D_y (d_y) coeff of D_BARz (d_BARz)
                return [
                    [populate["preProcessMinDataToReal"][index][0], realScale],
                    [populate["preProcessMinDataToReal"][index][1], imScale],
                ]

        otherDict = dict()
        for term_index, term_coeff in coeff_dict.items():
            if _varSpace_type == "real":
                reformatTarget = "symb"
            else:
                reformatTarget = "real"
            termIndices = [
                decorateWithWeights(k, valence[j], target=reformatTarget)
                for j, k in enumerate(term_index)
            ]
            prodWithWeights = carProd_with_weights_without_R(*termIndices)
            prodWWRescaled = [[tuple(k[0]), term_coeff * k[1]] for k in prodWithWeights]
            minimal_term_set = _shape_basis(prodWWRescaled, data_shape)
            for term in minimal_term_set:
                if term[0] in otherDict:
                    oldVal = otherDict[term[0]]
                    otherDict[term[0]] = allToSym(oldVal + term[1])
                else:
                    otherDict[term[0]] = allToSym(term[1])

        if _varSpace_type == "real":
            populate["realCoeffDataDict"] = [
                varSpace,
                coeff_dict,
            ]
            populate["compCoeffDataDict"] = [
                new_holVarSpace + new_antiholVarSpace,
                otherDict,
            ]
        else:
            populate["compCoeffDataDict"] = [
                varSpace,
                coeff_dict,
            ]
            populate["realCoeffDataDict"] = [
                new_realVarSpace + new_imVarSpace,
                otherDict,
            ]

    return (
        populate,
        new_realVarSpace,
        new_holVarSpace,
        new_antiholVarSpace,
        new_imVarSpace,
    )


def _shape_basis(basis, shape):
    if shape == "symmetric":
        old_basis = dict(basis)
        new_basis = dict()
        for index, value in old_basis.items():
            new_index = tuple(sorted(index))
            if new_index in new_basis:
                new_basis[new_index] += value
            else:
                new_basis[new_index] = value
        return list(new_basis.items())
    if shape == "skew":
        old_basis = dict(basis)
        new_basis = dict()
        for index, value in old_basis.items():
            permS, new_index = permSign(index, returnSorted=True)
            new_index = tuple(new_index)
            if new_index in new_basis:
                new_basis[new_index] += permS * value
            else:
                new_basis[new_index] = permS * value
        return list(new_basis.items())
    return basis
