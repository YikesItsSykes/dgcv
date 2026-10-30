"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: special_fields

module: dgcv.special_fields.Riemannian_geometry

Description:
This module provides tools for Riemannian geometry within the dgcv package. It includes
functions and classes for defining and manipulating metrics, Christoffel symbols, curvature tensors,
and Levi-Civita connections.

Key Classes:
    - metricClass: Represents a Riemannian metric and provides methods to compute Christoffel symbols,
      Riemann curvature, Ricci curvature, scalar curvature, and Weyl curvature.
    - LeviCivitaConnectionClass: Defines a Levi-Civita connection based on a set of Christoffel symbols
      of the second kind.

Key Functions:
    - metric_from_matrix(): Creates a metricClass object from a given coordinate space and matrix representation
      of the metric.

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

from typing import Any, Dict, Literal, Optional, Sequence, Tuple

from .._aux._backends import simplify_dgcv as simplify
from .._aux._backends._calculus import diff
from .._aux._backends._symbolic_router import _scalar_is_zero
from .._aux._backends._types_and_constants import rational
from .._aux._vmf._safeguards import get_dgcv_category, retrieve_passkey
from .._aux._vmf.vmf import vmf_lookup
from ..core.arrays import _spool, array_dgcv, matrix_dgcv
from ..core.base import dgcv_class
from ..core.conversions.conversions import allToReal, allToSym, symToHol
from ..core.dgcv_core import (
    assemble_tensor_field,
    tensor_field_class,
)

__all__ = ["metricClass", "metric_from_matrix"]


# -----------------------------------------------------------------------------
# body
# -----------------------------------------------------------------------------
def _convert_varSpace(varSpace, formatting):
    out = []
    for v in varSpace:
        info = vmf_lookup(v, relatives=True)
        rel = (info.get("relatives") or {}) if info.get("type") == "coordinate" else {}

        def _one(key):
            t = rel.get(key)
            if t is None:
                return None
            if isinstance(t, tuple):
                return t[0] if len(t) == 1 else None
            return t

        holo, anti, real, imag = (_one(k) for k in ("holo", "anti", "real", "imag"))
        if formatting == "real":
            if v == holo and real is not None:
                v = real
            elif v == anti and imag is not None:
                v = imag
        else:
            if v == real and holo is not None:
                v = holo
            elif v == imag and anti is not None:
                v = anti
        if v not in out:
            out.append(v)
    return tuple(out)


class metricClass(dgcv_class):
    def __init__(
        self,
        STF: tensor_field_class,
        *,
        varSpace: None | Sequence[Any] = None,
        variable_inference_behavior: Literal["max", "min"] = "min",
        formatting: Literal["complex", "real"] = "complex",
    ):
        if get_dgcv_category(STF) != "tensor_field":
            raise TypeError("metricClass expects a tensor_field_class instance.")

        if formatting not in ("complex", "real"):
            raise ValueError("formatting must be 'complex' or 'real'.")

        tf = allToSym(STF) if formatting == "complex" else allToReal(STF)

        deg = getattr(tf, "total_degree", None)
        cd = getattr(tf, "coeff_dict", None)
        shape = getattr(tf, "data_shape", None)

        if deg != 2:
            raise TypeError(
                f"metricClass expects a degree-2 tensor field. recieved: {tf}"
            )
        if not isinstance(cd, dict) or not all(k[2] == 0 and k[3] == 0 for k in cd):
            raise TypeError("metricClass expects a pure covariant (0,2) tensor field.")
        if shape != "symmetric":
            raise TypeError("metricClass expects data_shape='symmetric'.")

        if variable_inference_behavior not in ("max", "min"):
            raise ValueError("variable_inference_behavior must be 'max' or 'min'.")

        inferred = (
            tf.infer_minimal_varSpace()
            if variable_inference_behavior == "min"
            else tf.infer_varSpace(formatting="any")
        )

        if varSpace is not None:
            vs = tuple(varSpace)
            missing = [v for v in inferred if v not in vs]
            if missing:
                raise ValueError(
                    "Provided varSpace does not contain all inferred variables: "
                    + ", ".join(str(x) for x in missing)
                )
        else:
            vs = tuple(inferred)

        self.SymTensorField = tf
        self.varSpace = vs
        _, vs_map = tf.infer_varSpace(
            formatting="real" if formatting == "real" else "complex",
            return_dict=True,
        )

        self._sysidx_to_pos = {}
        for pos, atom in enumerate(self.varSpace):
            syslbl, local_idx = vs_map[atom]
            self._sysidx_to_pos[(syslbl, local_idx)] = pos
        self.formatting = formatting
        self.variable_inference_behavior = variable_inference_behavior

        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "metric"

        self._matrixRep: Optional[matrix_dgcv] = None
        self._matrixRepInv: Optional[matrix_dgcv] = None
        self._coeffArray: Optional[array_dgcv] = None

        self._Gamma2_cache: Optional[Dict[Tuple[int, int, int], Any]] = None
        self._Gamma1_cache: Optional[Dict[Tuple[int, int, int], Any]] = None
        self._R13_cache: Optional[Dict[Tuple[int, int, int, int], Any]] = None
        self._R04_cache: Optional[Dict[Tuple[int, int, int, int], Any]] = None
        self._Ricci_cache: Optional[Dict[Tuple[int, int], Any]] = None

        self._CFFK = None
        self._CFSK = None
        self._RCT13 = None
        self._RCT04 = None
        self._Ricci = None
        self._SCT = None
        self._tracelessRicci = None
        self._Weyl = None
        self._Einstein = None
        self._raw_latex = None

    @property
    def coeffArray(self) -> array_dgcv:
        if self._coeffArray is None:
            dim = len(self.varSpace)
            shp = (dim, dim)
            flat = {}
            data = self.SymTensorField.coeff_dict
            for k, v in data.items():
                if not v or not isinstance(k, tuple) or len(k) != 6:
                    continue
                i, j, _, _, sys_i, sys_j = k
                try:
                    ii = self._sysidx_to_pos[(sys_i, i)]
                    jj = self._sysidx_to_pos[(sys_j, j)]
                except KeyError:
                    continue
                flat[(ii, jj)] = flat.get((ii, jj), 0) + v
                if ii != jj:
                    flat[(jj, ii)] = flat.get((jj, ii), 0) + v
            self._coeffArray = array_dgcv(flat, shape=shp)
        return self._coeffArray

    @property
    def matrixRep(self) -> matrix_dgcv:
        if self._matrixRep is None:
            dim = len(self.varSpace)
            M = matrix_dgcv.zeros(dim, dim)
            data = self.SymTensorField.coeff_dict
            for k, v in data.items():
                if not v or not isinstance(k, tuple) or len(k) != 6:
                    continue
                i, j, _, _, sys_i, sys_j = k
                try:
                    ii = self._sysidx_to_pos[(sys_i, i)]
                    jj = self._sysidx_to_pos[(sys_j, j)]
                except KeyError:
                    continue
                cur = M[ii, jj]
                M[ii, jj] = cur + v if cur else v
                if ii != jj:
                    cur2 = M[jj, ii]
                    M[jj, ii] = cur2 + v if cur2 else v
            self._matrixRep = M
        return self._matrixRep

    @property
    def matrixRepInv(self) -> matrix_dgcv:
        if self._matrixRepInv is None:
            self._matrixRepInv = self.matrixRep.inverse()
        return self._matrixRepInv

    def _Gamma2(self, j: int, k: int, i: int):
        """
        Gamma_{j,k}^i indexing convention:
        - last index is the raised one, cached by (j,k,i)
        """
        cache = self._Gamma2_cache
        if cache is None:
            cache = self._Gamma2_cache = {}
        key = (j, k, i)
        hit = cache.get(key, None)
        if hit is not None:
            return hit

        g = self.matrixRep
        ginv = self.matrixRepInv
        vs = self.varSpace
        dim = len(vs)
        half = rational(1, 2)

        s = 0
        for m in range(dim):
            s += ginv[i, m] * (
                diff(g[m, k], vs[j]) + diff(g[j, m], vs[k]) - diff(g[j, k], vs[m])
            )
        val = simplify(half * s)
        cache[key] = val
        return val

    def _Gamma1(self, a: int, b: int, c: int):
        """
        Γ_{a,b,c} symbols of the first kind:
        1/2(D_b g_{c,a} + D_a g_{c,b} - D_c g_{a,b})
        cached by (a,b,c)
        """
        cache = self._Gamma1_cache
        if cache is None:
            cache = self._Gamma1_cache = {}
        key = (a, b, c)
        hit = cache.get(key, None)
        if hit is not None:
            return hit

        g = self.matrixRep
        vs = self.varSpace
        half = rational(1, 2)
        val = simplify(
            half * (diff(g[c, a], vs[b]) + diff(g[c, b], vs[a]) - diff(g[a, b], vs[c]))
        )
        cache[key] = val
        return val

    def __str__(self):
        return str(self.SymTensorField)

    def _repr_latex_(self, raw: bool = False, **kwargs):
        if self._raw_latex is None:
            f = getattr(
                symToHol(self.SymTensorField, convert_everything=False),
                "_repr_latex_",
                None,
            )
            self._raw_latex = f(raw=True, **kwargs)
        return self._raw_latex if raw else f"$\\displaystyle {self._raw_latex}$"

    def _to_real(self, plan=None):
        return self._to_format("real", plan)

    def _to_complex(self, plan=None):
        return self._to_format("complex", plan)

    def _to_format(self, formatting, plan=None):
        if self.formatting == formatting:
            return self
        if isinstance(plan, dict) and (
            plan.get("skipVar") or plan.get("variables_scope")
        ):
            raise ValueError(
                "metric conversion does not support skipVar or variables_scope"
            )
        return metricClass(
            self.SymTensorField,
            varSpace=_convert_varSpace(self.varSpace, formatting),
            variable_inference_behavior=self.variable_inference_behavior,
            formatting=formatting,
        )

    def _local_info_for_index(self, i: int) -> str:
        cache = getattr(self, "_local_systems", None)
        if cache is None:
            cache = self._local_systems = {}
        hit = cache.get(i, None)
        if hit is not None:
            return hit

        atom = self.varSpace[i]
        info = vmf_lookup(atom, path=True, system_index=True)
        p = info.get("path")
        idx = info.get("system_index")
        if not (isinstance(p, tuple) and len(p) >= 3):
            raise KeyError(f"metricClass: could not resolve system label for {atom}.")
        lab = p[1]
        cache[i] = (lab, idx)
        return lab, idx

    @property
    def Christoffel_symbols_of_the_second_kind(self) -> array_dgcv:
        if self._CFSK is None:
            dim = len(self.varSpace)
            shp = (dim, dim, dim)
            flat = {}
            for j in range(dim):
                for k in range(dim):
                    for i in range(dim):
                        v = self._Gamma2(j, k, i)
                        if v:
                            flat[_spool((j, k, i), shp)] = v
            self._CFSK = array_dgcv(flat, shape=shp)
        return self._CFSK

    @property
    def Christoffel_symbols_of_the_first_kind(self) -> array_dgcv:
        if self._CFFK is None:
            dim = len(self.varSpace)
            shp = (dim, dim, dim)
            flat = {}
            for a in range(dim):
                for b in range(dim):
                    for c in range(dim):
                        v = self._Gamma1(a, b, c)
                        if v:
                            flat[_spool((a, b, c), shp)] = v
            self._CFFK = array_dgcv(flat, shape=shp)
        return self._CFFK

    def _R13(self, r: int, j: int, k: int, i: int):
        cache = self._R13_cache
        if cache is None:
            cache = self._R13_cache = {}
        key = (r, j, k, i)
        hit = cache.get(key, None)
        if hit is not None:
            return hit

        vs = self.varSpace
        dim = len(vs)

        term = diff(self._Gamma2(j, r, i), vs[k]) - diff(self._Gamma2(k, r, i), vs[j])

        s1 = 0
        s2 = 0
        for s in range(dim):
            s1 += self._Gamma2(k, s, i) * self._Gamma2(j, r, s)
            s2 += self._Gamma2(j, s, i) * self._Gamma2(k, r, s)

        val = simplify(term + s1 - s2)
        cache[key] = val
        return val

    @property
    def RiemannCurvature_1_3_type(self) -> array_dgcv:
        if self._RCT13 is None:
            dim = len(self.varSpace)
            shp = (dim, dim, dim, dim)
            flat = {}
            for r in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        for i in range(dim):
                            v = self._R13(r, j, k, i)
                            if v:
                                flat[_spool((r, j, k, i), shp)] = v
            self._RCT13 = array_dgcv(flat, shape=shp)
        return self._RCT13

    def _R04(self, a: int, b: int, c: int, d: int):
        cache = self._R04_cache
        if cache is None:
            cache = self._R04_cache = {}
        key = (a, b, c, d)
        hit = cache.get(key, None)
        if hit is not None:
            return hit

        g = self.matrixRep
        dim = len(self.varSpace)

        s = 0
        for p in range(dim):
            s += g[a, p] * self._R13(b, c, d, p)

        val = simplify(-s)
        cache[key] = val
        return val

    def _Ricci_entry(self, i: int, j: int):
        cache = self._Ricci_cache
        if cache is None:
            cache = self._Ricci_cache = {}
        key = (i, j)
        hit = cache.get(key, None)
        if hit is not None:
            return hit

        ginv = self.matrixRepInv
        dim = len(self.varSpace)

        s = 0
        for p in range(dim):
            for q in range(dim):
                s += ginv[p, q] * self._R04(i, p, j, q)

        val = simplify(s)
        cache[key] = val
        return val

    @property
    def RiemannCurvature(self) -> tensor_field_class:
        if self._RCT04 is None:
            dim = len(self.varSpace)
            cd = {}
            for a in range(dim):
                sa, ia = self._local_info_for_index(a)
                for b in range(dim):
                    sb, ib = self._local_info_for_index(b)
                    for c in range(dim):
                        sc, ic = self._local_info_for_index(c)
                        for d in range(dim):
                            sd, idd = self._local_info_for_index(d)
                            v = self._R04(a, b, c, d)
                            if v:
                                cd[(ia, ib, ic, idd, 0, 0, 0, 0, sa, sb, sc, sd)] = v

            self._RCT04 = tensor_field_class(
                coeff_dict=cd,
                data_shape="general",
                dgcvType=self.SymTensorField.dgcvType,
                _simplifyKW=self.SymTensorField._simplifyKW,
                variable_spaces=getattr(self.SymTensorField, "_variable_spaces", None),
                parameters=getattr(self.SymTensorField, "parameters", set()),
            )
        return self._RCT04

    @property
    def RicciTensor(self) -> tensor_field_class:
        if self._Ricci is None:
            dim = len(self.varSpace)
            cd = {}
            for i in range(dim):
                si, ii = self._local_info_for_index(i)
                for j in range(i, dim):
                    sj, ij = self._local_info_for_index(j)
                    v = self._Ricci_entry(i, j)
                    if v:
                        cd[(ii, ij, 0, 0, si, sj)] = v

            self._Ricci = tensor_field_class(
                coeff_dict=cd,
                data_shape="symmetric",
                dgcvType=self.SymTensorField.dgcvType,
                _simplifyKW=self.SymTensorField._simplifyKW,
                variable_spaces=getattr(self.SymTensorField, "_variable_spaces", None),
                parameters=getattr(self.SymTensorField, "parameters", set()),
            )
        return self._Ricci

    @property
    def scalarCurvature(self):
        if self._SCT is None:
            ginv = self.matrixRepInv
            Ric = self.RicciTensor
            dim = len(self.varSpace)

            s = 0
            for p in range(dim):
                sp, ip = self._local_info_for_index(p)
                for q in range(dim):
                    sq, iq = self._local_info_for_index(q)
                    v = Ric.coeff_dict.get(
                        (ip, iq, 0, 0, sp, sq),
                        Ric.coeff_dict.get((iq, ip, 0, 0, sq, sp), 0),
                    )
                    if v:
                        s += ginv[p, q] * v
            self._SCT = simplify(s)
        return self._SCT

    @property
    def Einstein_tensor(self) -> tensor_field_class:
        if self._Einstein is None:
            Ric = self.RicciTensor
            SC = self.scalarCurvature
            self._Einstein = Ric - rational(1, 2) * SC * self.SymTensorField
        return self._Einstein

    def sectionalCurvature(self, vf1, vf2):
        num = self.RiemannCurvature(vf1, vf2, vf1, vf2)
        den = (
            self.SymTensorField(vf1, vf1) * self.SymTensorField(vf2, vf2)
            - (self.SymTensorField(vf1, vf2)) ** 2
        )
        return num / den


# class LeviCivitaConnectionClass(dgcv_class):
#     def __init__(
#         self,
#         varSpace,
#         Christoffel_symbols_of_the_second_kind,
#         variable_handling_default="standard",
#     ):
#         self.varSpace = tuple(varSpace)
#         dim = len(self.varSpace)

#         Gamma = Christoffel_symbols_of_the_second_kind
#         if not isinstance(Gamma, array_dgcv):
#             try:
#                 Gamma = array_dgcv(Gamma, shape=(dim, dim, dim))
#             except Exception as e:
#                 raise TypeError(
#                     "LeviCivitaConnectionClass expects Christoffel symbols as array_dgcv "
#                     "or array-like data convertible to array_dgcv. "
#                     f"Failed to convert input: {e}"
#                 )

#         if Gamma.shape != (dim, dim, dim):
#             raise TypeError(
#                 "Christoffel_symbols_of_the_second_kind must have shape (dim,dim,dim) matching varSpace."
#             )

#         self.Christoffel_symbols = Gamma

#         if variable_handling_default == "complex":
#             variable_registry = get_variable_registry()
#             cd = variable_registry["conversion_dictionaries"]
#             if all(v in cd["realToSym"] for v in self.varSpace):
#                 self._varSpace_type = "real"
#             elif all(v in cd["symToReal"] for v in self.varSpace):
#                 self._varSpace_type = "complex"
#             else:
#                 raise KeyError(
#                     "For variable_handling_default='complex', varSpace must come from one dgcv complex-variable family."
#                 )
#         else:
#             self._varSpace_type = "standard"

#         self._dgcv_class_check = retrieve_passkey()
#         self._dgcv_category = "LeviCivitaConnection"

#     def __call__(self, vf1, vf2):
#         if not (isinstance(vf1, VFClass) and isinstance(vf2, VFClass)):
#             raise TypeError(
#                 "LeviCivitaConnectionClass only operates on pairs of VFClass objects."
#             )

#         dim = len(self.varSpace)
#         shpG = (dim, dim, dim)
#         Gdata = self.Christoffel_symbols._data

#         def Gamma(j, k, L):
#             return Gdata.get(_spool((j, k, L), shpG), 0)

#         def _coeff(VF1, VF2, L):
#             term1 = 0
#             for j in range(dim):
#                 cj = VF1.coeffs[j]
#                 if cj != 0:
#                     term1 += diff(VF2.coeffs[L], self.varSpace[j]) * cj

#             term2 = 0
#             for j in range(dim):
#                 cj = VF1.coeffs[j]
#                 if cj == 0:
#                     continue
#                 for k in range(dim):
#                     ck = VF2.coeffs[k]
#                     if ck == 0:
#                         continue
#                     g = Gamma(j, k, L)
#                     if g != 0:
#                         term2 += g * cj * ck
#             return term1 + term2

#         if self._varSpace_type == "standard":
#             vf1 = changeVFBasis(vf1, self.varSpace)
#             vf2 = changeVFBasis(vf2, self.varSpace)
#             newCoeffs = [_coeff(vf1, vf2, L) for L in range(dim)]
#             return VFClass(self.varSpace, newCoeffs)

#         if self._varSpace_type == "real":
#             vf1 = changeVFBasis(allToReal(vf1), self.varSpace)
#             vf2 = changeVFBasis(allToReal(vf2), self.varSpace)
#             newCoeffs = [_coeff(vf1, vf2, L) for L in range(dim)]
#             return VFClass(self.varSpace, newCoeffs, dgcvType="complex")

#         # complex
#         vf1 = changeVFBasis(allToSym(vf1), self.varSpace)
#         vf2 = changeVFBasis(allToSym(vf2), self.varSpace)
#         newCoeffs = [_coeff(vf1, vf2, L) for L in range(dim)]
#         return VFClass(self.varSpace, newCoeffs, dgcvType="complex")

#     def __dgcv_simplify__(self, method=None, **kwargs):
#         return self._eval_simplify(**kwargs)

#     def _eval_simplify(self, **kwargs):
#         return self


def metric_from_matrix(coordinates, matrix):
    coords = tuple(coordinates)
    n = len(coords)

    if isinstance(matrix, array_dgcv):
        if matrix.shape != (n, n):
            raise TypeError(
                "metric_from_matrix: array_dgcv must have shape (n,n) matching len(coordinates)."
            )

        def entry(i, j):
            return matrix[i, j]

    else:
        M = matrix if isinstance(matrix, matrix_dgcv) else matrix_dgcv(matrix)
        if M.shape != (n, n):
            raise TypeError(
                "metric_from_matrix: matrix must be coercible to a square (n,n) matrix matching len(coordinates)."
            )

        def entry(i, j):
            return M[i, j]

    sparse_data = {}
    for i in range(n):
        for j in range(i, n):
            v = entry(i, j)
            if not _scalar_is_zero(v):
                sparse_data[(i, j)] = v

    return metricClass(assemble_tensor_field(coords, sparse_data, shape="symmetric"))
