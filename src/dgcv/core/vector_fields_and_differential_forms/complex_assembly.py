from __future__ import annotations

from typing import Any, Sequence

from ..._aux._utilities._config import dgcv_warning
from ..._aux._vmf.vmf import vmf_lookup
from ..dgcv_core import vector_field_class
from .retrieval import get_VF


def assembleFromHolVFC(
    coeffs: Sequence[Any],
    hol_vars: Sequence[Any],
    *,
    _warn_on_nonholo: bool = True,
) -> "vector_field_class":
    if not isinstance(hol_vars, (list, tuple)):
        raise TypeError("hol_vars must be a list or tuple.")
    if len(coeffs) != len(hol_vars):
        raise ValueError("coeffs and hol_vars must have the same length.")
    if _warn_on_nonholo:
        _warnForNonholos(hol_vars, dispatcher="assembleFromHolVFC")

    picked = [
        (_find_family_member(z, coorType="holo", dispatcher="assembleFromHolVFC")[0])
        for z in hol_vars
    ]
    Dz_list = get_VF(*picked)

    out = None
    for c, Dz in zip(coeffs, Dz_list):
        if not c:
            continue
        term = c * Dz
        out = term if out is None else (out + term)

    if out is None:
        out = vector_field_class(coeff_dict={tuple(): 0}, data_shape="all")

    return out


def assembleFromAntiholVFC(
    coeffs: Sequence[Any],
    hol_vars: Sequence[Any],
    *,
    _warn_on_nonholo: bool = True,
) -> "vector_field_class":
    if not isinstance(hol_vars, (list, tuple)):
        raise TypeError("hol_vars must be a list or tuple.")
    if len(coeffs) != len(hol_vars):
        raise ValueError("coeffs and hol_vars must have the same length.")
    if _warn_on_nonholo:
        _warnForNonholos(hol_vars, dispatcher="assembleFromAntiholVFC")

    picked = [
        (
            _find_family_member(
                z, coorType="anti", dispatcher="assembleFromAntiholVFC"
            )[0]
        )
        for z in hol_vars
    ]
    Dzb_list = get_VF(*picked)

    out = None
    for c, Dzb in zip(coeffs, Dzb_list):
        if not c:
            continue
        term = c * Dzb
        out = term if out is None else (out + term)

    if out is None:
        out = vector_field_class(coeff_dict={tuple(): 0}, data_shape="all")

    return out


def assembleFromCompVFC(
    holomorphic_coeffs: Sequence[Any],
    antiholomorphic_coeffs: Sequence[Any],
    hol_vars: Sequence[Any],
    *,
    _warn_on_nonholo: bool = True,
) -> "vector_field_class":
    if not isinstance(hol_vars, (list, tuple)):
        raise TypeError("hol_vars must be a list or tuple.")
    if len(holomorphic_coeffs) != len(hol_vars) or len(antiholomorphic_coeffs) != len(
        hol_vars
    ):
        raise ValueError("Coefficient lengths must match hol_vars length.")

    vf_h = assembleFromHolVFC(
        holomorphic_coeffs, hol_vars, _warn_on_nonholo=_warn_on_nonholo
    )
    vf_a = assembleFromAntiholVFC(
        antiholomorphic_coeffs, hol_vars, _warn_on_nonholo=_warn_on_nonholo
    )
    return vf_h + vf_a


def _find_family_member(coordinate: Any, *, coorType: str, dispatcher: str):
    info = vmf_lookup(coordinate, path=True, relatives=True)
    if info.get("type") != "coordinate":
        raise TypeError(
            f"{dispatcher} requires variables registered in the VMF as coordinates. "
            "Suggestion: initialize complex coordinate systems with dgcv.createVariables(...)."
        )
    st = info.get("sub_type")
    if st == "standard":
        raise TypeError(
            f"{dispatcher} requires complex coordinates (holo/anti/real/imag), "
            "but received a standard coordinate. Suggestion: use dgcv.createVariables(...) "
            "to register a complex coordinate system in the VMF."
        )

    st = info.get("sub_type")

    rel = info.get("relatives") or {}
    if coorType == st:
        out = coordinate
    else:
        out = rel.get(coorType, None)

    if out is None:
        raise TypeError(
            f"{dispatcher} requires the variable to belong to a complex coordinate system "
            "registered in the dgcv VMF."
            "And `coorType` must be one of: 'holo', 'anti', 'real', 'imag'"
        )

    return out, st


def _warnForNonholos(hol_vars: Sequence[Any], *, dispatcher: str):
    inadmissibles = []
    for v in hol_vars:
        info = vmf_lookup(v, relatives=False)
        if info.get("type") != "coordinate":
            continue
        st = info.get("sub_type")
        if st not in (None, "holo"):
            inadmissibles.append((v, st))

    if inadmissibles:
        dgcv_warning(
            f"{dispatcher}: holomorphic variables are recommended for hol_vars. "
            f"Received non-holo coordinates: {inadmissibles}. "
            "Proceeding by infering holomorphic relatives from the dgcv VMF."
        )
