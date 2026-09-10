"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: special_fields

module: dgcv.special_fields.complex_structures


Description: This module provides tools uniquely relevant for complex differential geometry within the dgcv package. It includes Dolbeault operators (Del and DelBar) and a class for constructing and analyzing Kähler structures.

Key Functions:
    - Del(): Applies the holomorphic Dolbeault operator ∂ to a differential form or scalar.
    - DelBar(): Applies the antiholomorphic Dolbeault operator ∂̅ to a differential form or scalar.

Key Classes:
    - KahlerStructure: Represents a Kähler structure, with properties and attributes to compute many of their invariants.

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import Any, Dict, List, Literal, Sequence, Tuple

from .._aux._backends._symbolic_router import (
    get_free_symbols,
    is_zero_knowing_zero_is_expected,
    simplify,
)
from .._aux._backends._types_and_constants import expr_numeric_types, rational
from .._aux._vmf._safeguards import (
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_passkey,
)
from .._aux._vmf.vmf import vmf_lookup
from ..core.base import dgcv_class
from ..core.conversions.conversions import allToReal, allToSym
from ..core.dgcv_core import (
    complex_struct_op,
    differential_form_class,
    tensor_field_class,
)
from ..core.vector_fields_and_differential_forms import (
    _prep_symb_set_for_ext_der,
    exteriorDerivative,
    makeZeroForm,
)
from .Riemannian_geometry import metricClass


# -----------------------------------------------------------------------------
# Dolbeault operators
# -----------------------------------------------------------------------------
def _dolbeault_relevant_atoms(form, *, want: str):
    rv = get_free_symbols(form)
    if rv:
        params = getattr(form, "parameters", None)
        if params:
            rv = set(rv) - set(params)
    else:
        rv = set()

    rel = _prep_symb_set_for_ext_der(rv)

    picked = {}
    for syslbl, sysreg in rel.items():
        for atom, pair in sysreg.items():
            if not isinstance(pair, tuple) or len(pair) < 2:
                continue
            h, a = pair[0], pair[1]
            if want == "holo":
                target = h
            elif want == "anti":
                target = a
            else:
                raise ValueError("want must be 'holo' or 'anti'")
            if target is None:
                continue
            picked[target] = True

    if not picked:
        return tuple()

    bad = []
    out = []
    for atom in picked.keys():
        info = vmf_lookup(atom, relatives=False)
        if info.get("type") != "coordinate":
            continue
        st = info.get("sub_type")
        if st == "standard":
            bad.append(atom)
            continue
        out.append(atom)

    if bad:
        raise TypeError(
            "`Del`/`DelBar` require complex coordinates registered in the VMF (holo/anti/real/imag). "
            "Suggestion: initialize complex coordinate systems with `createVariables(..., complex=True)`."
        )

    return tuple(out)


def Del(obj):
    if get_dgcv_category(obj) == "array":
        return obj.apply(Del)
    if get_dgcv_category(obj) != "tensor_field":
        if isinstance(obj, expr_numeric_types()):
            obj = makeZeroForm(obj, dgcvType="complex")
        else:
            raise TypeError("`Del` expects a differential form or scalar expression.")

    if not query_dgcv_categories(obj, {"differential_form"}):
        raise TypeError("`Del` expects a differential form or scalar expression.")

    form = obj
    atoms = _dolbeault_relevant_atoms(form, want="holo")

    accumulation = 0
    for atom in atoms:
        ds = vmf_lookup(atom, differential_system=True).get("differential_system")
        if ds is None:
            continue
        vf = ds.get("vf")
        df = ds.get("df")
        if vf is None or df is None:
            continue
        accumulation = accumulation + df * (form.apply(vf))

    return accumulation


def DelBar(obj):
    if get_dgcv_category(obj) == "array":
        return obj.apply(DelBar)
    if get_dgcv_category(obj) != "tensor_field":
        if isinstance(obj, expr_numeric_types()):
            obj = makeZeroForm(obj, dgcvType="complex")
        else:
            raise TypeError(
                "`DelBar` expects a differential form or scalar expression."
            )

    if not query_dgcv_categories(obj, {"differential_form"}):
        raise TypeError("`DelBar` expects a differential form or scalar expression.")

    form = obj
    atoms = _dolbeault_relevant_atoms(form, want="anti")

    accumulation = 0
    for atom in atoms:
        ds = vmf_lookup(atom, differential_system=True).get("differential_system")
        if ds is None:
            continue
        vf = ds.get("vf")
        df = ds.get("df")
        if vf is None or df is None:
            continue
        accumulation = accumulation + df * (form.apply(vf))

    return accumulation


# -----------------------------------------------------------------------------
# Kahler geometry
# -----------------------------------------------------------------------------
def _as_tuple(x: Any) -> Tuple[Any, ...]:
    if x is None:
        return tuple()
    if isinstance(x, tuple):
        return x
    if isinstance(x, (list, set)):
        return tuple(x)
    return (x,)


def _coerce_manual_varSpace_holo_only(varSpace: Sequence[Any]) -> Tuple[Any, ...]:
    vs = tuple(varSpace)
    if len(vs) != len(set(vs)):
        raise TypeError("varSpace must have distinct variables (no duplicates).")

    out: List[Any] = []
    for v in vs:
        info = vmf_lookup(v, relatives=True)
        if info.get("type") != "coordinate":
            raise TypeError("varSpace entries must be VMF-registered coordinates.")
        rel = info.get("relatives") or {}
        holo = _as_tuple(rel.get("holo"))
        if not (len(holo) == 1 and holo[0] == v):
            raise TypeError(
                "Manual varSpace must consist only of holomorphic coordinates from dgcv complex systems."
            )
        out.append(v)
    return tuple(out)


def _expand_holo_varSpace(
    holo_vs: Sequence[Any],
    *,
    formatting: Literal["complex", "real"],
) -> Tuple[Any, ...]:
    first: List[Any] = []
    second: List[Any] = []

    for v in holo_vs:
        info = vmf_lookup(v, relatives=True)
        if info.get("type") != "coordinate":
            raise TypeError("varSpace entries must be VMF-registered coordinates.")
        rel = info.get("relatives") or {}

        holo = _as_tuple(rel.get("holo"))
        anti = _as_tuple(rel.get("anti"))
        real = _as_tuple(rel.get("real"))
        imag = _as_tuple(rel.get("imag"))

        if not (len(holo) == 1 and holo[0] == v):
            raise TypeError(
                "Manual varSpace must consist only of holomorphic coordinates from dgcv complex systems."
            )

        if formatting == "complex":
            if len(anti) != 1 or anti[0] is None:
                raise TypeError("Could not resolve antiholomorphic partner from VMF.")
            first.append(holo[0])
            second.append(anti[0])
        else:
            if len(real) != 1 or len(imag) != 1 or real[0] is None or imag[0] is None:
                raise TypeError("Could not resolve real/imag partners from VMF.")
            first.append(real[0])
            second.append(imag[0])

    out = tuple(first + second)
    if len(out) != len(set(out)):
        raise TypeError("Manual varSpace expansion produced duplicates.")
    return out


class KahlerStructure(dgcv_class):
    def __init__(
        self,
        Kahler_form: differential_form_class,
        *,
        varSpace: None | Sequence[Any] = None,
        variable_inference_behavior: Literal["max", "min"] = "min",
        formatting: Literal["complex", "real"] = "complex",
    ):
        if not query_dgcv_categories(Kahler_form, {"differential_form"}):
            raise TypeError(
                "KahlerStructure expects a dgcv differential_form_class object."
            )

        if formatting == "real":
            omega = allToReal(Kahler_form)
        elif formatting == "complex":
            omega = allToSym(Kahler_form)
        else:
            raise TypeError("formatting must be 'complex' or 'real'.")

        if variable_inference_behavior not in ("max", "min"):
            raise TypeError("variable_inference_behavior must be 'min' or 'max'.")

        if varSpace is None:
            if variable_inference_behavior == "min":
                min_any = tuple(omega.infer_minimal_varSpace())
            else:
                min_any = tuple(omega.infer_varSpace(formatting="any"))

            if not min_any:
                raise TypeError(
                    "KahlerStructure could not determine a nonempty varSpace."
                )

            vs_complex = tuple(omega.infer_varSpace(formatting="complex"))
            vs_real = tuple(omega.infer_varSpace(formatting="real"))
        else:
            holo_only = _coerce_manual_varSpace_holo_only(varSpace)
            vs_complex = _expand_holo_varSpace(holo_only, formatting="complex")
            vs_real = _expand_holo_varSpace(holo_only, formatting="real")
            min_any = tuple(omega.infer_minimal_varSpace())

        if formatting == "complex":
            self.varSpace = tuple(vs_complex)
            want_fmt = "complex"
        else:
            self.varSpace = tuple(vs_real)
            want_fmt = "real"

        if not self.varSpace:
            raise TypeError(
                "KahlerStructure could not determine a nonempty expanded varSpace."
            )

        min_any_set = set(min_any)
        bad = [v for v in self.varSpace if v not in min_any_set]
        if bad:
            raise TypeError(
                "KahlerStructure varSpace contains variables not present in the (converted) Kahler form."
            )

        self.kahlerForm = omega
        self.formatting = formatting
        self.variable_inference_behavior = variable_inference_behavior
        self._varSpace_type = want_fmt

        vs_sorted_r, vs_map_r = omega.infer_varSpace(
            formatting="real", return_dict=True
        )
        vs_sorted_c, vs_map_c = omega.infer_varSpace(
            formatting="complex", return_dict=True
        )

        self.varSpace_real = tuple(vs_sorted_r)
        self.varSpace_complex = tuple(vs_sorted_c)
        self._vs_map_real = dict(vs_map_r)
        self._vs_map_complex = dict(vs_map_c)

        self._vf_cache: Dict[Any, Any] = {}
        for v in set(self.varSpace_real) | set(self.varSpace_complex):
            self._vf_cache[v] = self._vf_from_coord(v)

        self.coor_frame = tuple(self._vf_cache[v] for v in self.varSpace)
        self.coor_frame_complex = tuple(
            self._vf_cache[v] for v in self.varSpace_complex
        )

        use_map = self._vs_map_complex if formatting == "complex" else self._vs_map_real

        metric_cd: Dict[Tuple[Any, ...], Any] = {}
        half = rational(1, 2)
        for vi, vf_i in zip(self.varSpace, self.coor_frame):
            sys_i, idx_i = use_map[vi]
            for vj, vf_j in zip(self.varSpace, self.coor_frame):
                sys_j, idx_j = use_map[vj]
                val = half * self.kahlerForm(vf_i, complex_struct_op(vf_j))
                if not val:
                    continue

                k = (idx_i, idx_j, 0, 0, sys_i, sys_j)
                ks = (idx_j, idx_i, 0, 0, sys_j, sys_i)

                if ks in metric_cd:
                    metric_cd[ks] = metric_cd.get(ks, 0) + val
                else:
                    metric_cd[k] = metric_cd.get(k, 0) + val

        metric_tf = tensor_field_class(
            coeff_dict=metric_cd if metric_cd else {tuple(): 0},
            data_shape="symmetric",
            dgcvType="complex" if formatting == "complex" else "standard",
            variable_spaces=getattr(self.kahlerForm, "_variable_spaces", None),
            parameters=getattr(self.kahlerForm, "parameters", set()),
        )

        self.metric = metricClass(
            metric_tf,
            varSpace=self.varSpace,
            variable_inference_behavior=variable_inference_behavior,
            formatting=formatting,
        )

        self._is_closed = None
        self._holRiemann = None
        self._holRicci = None
        self._Bochner = None

        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "kahler_structure"

    @staticmethod
    def _vf_from_coord(a: Any):
        info = vmf_lookup(a, differential_system=True)
        ds = info.get("differential_system", None)
        if not isinstance(ds, dict):
            raise TypeError(
                "KahlerStructure requires coordinates registered in VMF with differential objects."
            )
        vf = ds.get("vf", None)
        if vf is None or not query_dgcv_categories(vf, {"vector_field"}):
            raise TypeError(
                "KahlerStructure requires coordinates registered in VMF with vector fields."
            )
        return vf

    @property
    def is_closed(self):
        if self._is_closed is None:
            self._is_closed = is_zero_knowing_zero_is_expected(
                exteriorDerivative(self.kahlerForm)
            )
        return self._is_closed

    @property
    def holRiemann(self):
        if self._holRiemann is None:
            VFBasis = self.coor_frame_complex
            dim = len(VFBasis)
            R = self.metric.RiemannCurvature

            cd: Dict[Tuple[Any, ...], Any] = {}
            for j in range(dim):
                vj = self.varSpace_complex[j]
                sj, ij = self._vs_map_complex[vj]
                for k in range(dim):
                    vk = self.varSpace_complex[k]
                    sk, ik = self._vs_map_complex[vk]
                    for L in range(dim):
                        vL = self.varSpace_complex[L]
                        sL, iL = self._vs_map_complex[vL]
                        for m in range(dim):
                            vm = self.varSpace_complex[m]
                            sm, im = self._vs_map_complex[vm]
                            val = simplify(
                                R(
                                    VFBasis[j],
                                    complex_struct_op(VFBasis[k]),
                                    VFBasis[L],
                                    complex_struct_op(VFBasis[m]),
                                )
                            )
                            if not val:
                                continue
                            key = (ij, ik, iL, im, 0, 0, 0, 0, sj, sk, sL, sm)
                            cd[key] = cd.get(key, 0) + val

            self._holRiemann = tensor_field_class(
                coeff_dict=cd if cd else {tuple(): 0},
                data_shape="general",
                dgcvType="complex",
                variable_spaces=getattr(
                    self.metric.SymTensorField, "_variable_spaces", None
                ),
                parameters=getattr(self.metric.SymTensorField, "parameters", set()),
            )
        return self._holRiemann

    @property
    def holRicci(self):
        if self._holRicci is None:
            VFBasis = self.coor_frame_complex
            dim = len(VFBasis)
            Ric = self.metric.RicciTensor

            cd: Dict[Tuple[Any, ...], Any] = {}
            for j in range(dim):
                vj = self.varSpace_complex[j]
                sj, ij = self._vs_map_complex[vj]
                for k in range(dim):
                    vk = self.varSpace_complex[k]
                    sk, ik = self._vs_map_complex[vk]
                    val = simplify(Ric(VFBasis[j], complex_struct_op(VFBasis[k])))
                    if not val:
                        continue
                    key = (ij, ik, 0, 0, sj, sk)
                    cd[key] = cd.get(key, 0) + val

            self._holRicci = tensor_field_class(
                coeff_dict=cd if cd else {tuple(): 0},
                data_shape="general",
                dgcvType="complex",
                variable_spaces=getattr(
                    self.metric.SymTensorField, "_variable_spaces", None
                ),
                parameters=getattr(self.metric.SymTensorField, "parameters", set()),
            )
        return self._holRicci

    @property
    def Bochner(self):
        if self._Bochner is None:
            VFBasis = self.coor_frame_complex
            dim = len(VFBasis)
            compDim = int(rational(dim, 2))

            g = self.metric.SymTensorField
            R = self.metric.RiemannCurvature
            Ric = self.metric.RicciTensor
            S = self.metric.scalarCurvature

            def entry_rule(j, h, L, k):
                term1 = R(j, h, L, k)
                term2 = rational(1, compDim + 2) * (
                    (g(j, k)) * (Ric(L, h))
                    + (g(L, k)) * (Ric(j, h))
                    + (g(L, h)) * (Ric(j, k))
                    + (g(j, h)) * (Ric(L, k))
                )
                term3 = (
                    S
                    * rational(1, 2 * (compDim + 1) * (compDim + 2))
                    * (g(L, h) * g(j, k) + g(j, h) * g(L, k))
                )
                return simplify(term1 + term2 - term3)

            cd: Dict[Tuple[Any, ...], Any] = {}
            for j in range(compDim):
                vj = self.varSpace_complex[j]
                sj, ij = self._vs_map_complex[vj]  # system name, index from vj
                for k in range(compDim, dim):
                    vk = self.varSpace_complex[k]
                    sk, ik = self._vs_map_complex[vk]  # system name, index from vk
                    for L in range(compDim):
                        vL = self.varSpace_complex[L]
                        sL, iL = self._vs_map_complex[vL]  # system name, index from vL
                        for m in range(compDim, dim):
                            vm = self.varSpace_complex[m]
                            sm, im = self._vs_map_complex[
                                vm
                            ]  # system name, index from vm

                            val = entry_rule(
                                VFBasis[j], VFBasis[k], VFBasis[L], VFBasis[m]
                            )
                            if val == 0:
                                continue
                            key = (ij, ik, iL, im, 0, 0, 0, 0, sj, sk, sL, sm)
                            cd[key] = cd.get(key, 0) + val

            self._Bochner = tensor_field_class(
                coeff_dict=cd if cd else {tuple(): 0},
                data_shape="general",
                dgcvType="complex",
                _simplifyKW={
                    "simplify_rule": None,
                    "simplify_ignore_list": None,
                    "preferred_basis_element": (0, compDim, 0, compDim),
                },
                variable_spaces=getattr(
                    self.metric.SymTensorField, "_variable_spaces", None
                ),
                parameters=getattr(self.metric.SymTensorField, "parameters", set()),
            )
        return self._Bochner
