from __future__ import annotations

from numbers import Integral
from typing import Any, Dict, Literal, Optional, Tuple

from ....._aux._backends._symbolic_router import _scalar_is_zero
from ....._aux._utilities._config import get_dgcv_settings_registry
from ....._aux._vmf._safeguards import retrieve_passkey
from ....._aux._vmf.vmf import order_coordinates, vmf_lookup
from .workers import (
    _infer_variable_spaces_from_coeff_dict,
    _is_scalar_coeff_dict,
    _missing_system_msg,
    _process_coeffs_dict_new,
    _to_complex_algo,
    _to_real_algo,
    _variable_spaces_types_algo,
)


class _tensor_field_construction:
    def __init__(
        self,
        varSpace=None,
        coeff_dict=None,
        valence=None,
        data_shape: str = "general",
        dgcvType: str = "standard",
        _simplifyKW=None,
        variable_spaces: Optional[dict] = None,
        parameters=set(),
        _inheritance=None,
    ):
        if data_shape not in ("general", "symmetric", "skew", "all"):
            data_shape = "general"
        if coeff_dict is None:
            coeff_dict = {}
        if not isinstance(coeff_dict, dict):
            raise TypeError("`coeff_dict` must be a dictionary.")

        if _simplifyKW is None:
            _simplifyKW = {
                "simplify_rule": None,
                "simplify_ignore_list": None,
                "preferred_basis_element": None,
            }

        self.dgcvType = dgcvType
        self._simplifyKW = _simplifyKW
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "tensor_field"
        self.parameters = set(parameters)

        self._variable_spaces = (
            dict(variable_spaces) if isinstance(variable_spaces, dict) else {}
        )
        self._system_index_cache: Dict[str, Dict[Any, int]] = {}
        self._shape_checked = False

        self._preferred_basis_element = None
        self._expanded_coeff_dict = None
        self._coeffArray = None
        self._coeffs = None

        self._cd_formats = None
        self._realVarSpace = None
        self._holVarSpace = None
        self._antiholVarSpace = None
        self._imVarSpace = None
        self._varSpace_type = "standard"
        self._variable_spaces_types = None
        self._key_profiles = {}  # str values "s", "r", "c" tuples for standard, real, complex
        self._conj_key_profiles = {}  # numeric value index shif for conjugation
        self._coordinate_format = None
        self._free_symbols = None
        self._coeff_free_symbols = None
        self._coef_profile = None
        self._max_degree = None
        self._min_degree = None
        self._validated_format: Literal[
            "open", "standard", "complex", "real", "mixed"
        ] = "open"
        self._num_den = None
        self._denom_cm = None
        self._num_cd = None
        self._minimal_coordinate_space = None

        if coeff_dict == {}:
            coeff_dict = {tuple(): 0}

        old_mode = (varSpace is not None) or (valence is not None)

        if (not old_mode) and _is_scalar_coeff_dict(coeff_dict):
            self.coordinates = tuple() if varSpace is None else tuple(varSpace)
            self.valence = tuple()
            self.coeff_dict = {tuple(): coeff_dict.get(tuple(), 0)}
            self.data_shape = "all"
            self._shape_checked = True
            self.varSpace = self.coordinates
        else:
            if old_mode:
                self.coordinates, self.valence, self.coeff_dict, self.data_shape = (
                    self._init_from_old_format(
                        varSpace, coeff_dict, valence, data_shape
                    )
                )
            elif _inheritance and _inheritance.get("_canonical"):
                first_key = next(iter(coeff_dict))
                deg = len(first_key) // 3
                self.coordinates = tuple() if varSpace is None else tuple(varSpace)
                self.valence = tuple(first_key[deg : 2 * deg])
                self.coeff_dict = coeff_dict
                self.data_shape = data_shape
                self._validated_format = _inheritance["_validated_format"]
            else:
                self._variable_spaces = _infer_variable_spaces_from_coeff_dict(
                    coeff_dict, self._variable_spaces
                )
                (
                    self.coordinates,
                    self.valence,
                    self.coeff_dict,
                    self.data_shape,
                    self._validated_format,
                ) = self._init_from_standard_data(
                    varSpace, coeff_dict, data_shape, self._variable_spaces
                )
                if _inheritance:
                    self._validated_format = _inheritance.get(
                        "_validated_format", self._validated_format
                    )

                if self.coeff_dict and self._validated_format == "mixed":
                    if (
                        get_dgcv_settings_registry().get(
                            "preferred_variable_format", None
                        )
                        == "real"
                    ):
                        new_cd = _to_real_algo(
                            cd=self.coeff_dict,
                            vst=_variable_spaces_types_algo(self._variable_spaces),
                        )
                    else:
                        new_cd = _to_complex_algo(
                            cd=self.coeff_dict,
                            vst=_variable_spaces_types_algo(self._variable_spaces),
                        )

                    self.coeff_dict, self._validated_format = new_cd, "complex"
            self.coordinates = order_coordinates(self.coordinates)
            self.varSpace = self.coordinates
            self._shape_checked = True

    def _set_degrees(self):
        m, mm = None, None
        for key in self.coeff_dict:
            lk = len(key) // 3
            if m is None:
                m = lk
                mm = lk
            else:
                m = max(m, lk)
                mm = min(mm, lk)
        self._max_degree, self._min_degree = m, mm

    def _init_from_old_format(self, varSpace, coeff_dict, valence, data_shape):
        if varSpace is None:
            raise TypeError("`varSpace` is required when `valence` is provided.")
        varSpace_t = tuple(varSpace)

        if valence is None:
            if not coeff_dict:
                valence_t = tuple()
            else:
                first_key = next(iter(coeff_dict))
                if not isinstance(first_key, tuple):
                    raise TypeError("Keys in `coeff_dict` must be tuples.")
                valence_t = (0,) * len(first_key)
        else:
            valence_t = tuple(valence)

        if not all(v in (0, 1) for v in valence_t):
            raise ValueError("`valence` must contain only 0s and 1s.")

        if len(set(valence_t)) > 1 and data_shape in ("symmetric", "skew"):
            raise ValueError(
                "Symmetry constraints require pure covariant or pure contravariant valence."
            )

        new_cd, inferred_type = self._convert_old_coeff_dict(
            varSpace_t, coeff_dict, valence_t
        )

        if self.dgcvType == "complex" and inferred_type != "complex":
            self.dgcvType = "standard"

        processed_cd, eff_shape, formatting = _process_coeffs_dict_new(
            new_cd, data_shape, formatting=True
        )
        return varSpace_t, valence_t, processed_cd, eff_shape

    def _init_from_standard_data(
        self, varSpace, coeff_dict, data_shape, variable_spaces
    ):
        varSpace_t = tuple() if varSpace is None else tuple(varSpace)

        if not coeff_dict:
            return varSpace_t, tuple(), {tuple(): 0}, "all"

        first_key = next(iter(coeff_dict))
        deg = len(first_key) // 3
        valence_t = tuple(first_key[deg : 2 * deg])

        processed_cd, eff_shape, formatting = _process_coeffs_dict_new(
            coeff_dict,
            data_shape,
            variable_spaces,
            formatting=True,
        )
        return varSpace_t, valence_t, processed_cd, eff_shape, formatting

    def _convert_old_coeff_dict(
        self,
        varSpace_t: Tuple[Any, ...],
        old_cd: Dict[Tuple[int, ...], Any],
        valence_t: Tuple[int, ...],
    ) -> Tuple[Dict[Tuple[Any, ...], Any], str]:
        inferred = "standard"
        sys_for_var = {}
        for v in varSpace_t:
            info = vmf_lookup(v, path=True, relatives=False)
            p = info.get("path")
            if not (isinstance(p, tuple) and len(p) >= 2):
                raise KeyError(
                    f"tensor_field_class: variable '{v}' is not registered in the VMF."
                )
            branch, system_label = p[0], p[1]
            if branch == "complex_variable_systems":
                inferred = "complex"
            sys_for_var[v] = system_label

        for system_label in set(sys_for_var.values()):
            if system_label not in self._variable_spaces:
                info = vmf_lookup(
                    system_label, path=True, relatives=True, flattened_relatives=True
                )
                flat = info.get("flattened_relatives", None)
                if not isinstance(flat, tuple) or len(flat) == 0:
                    raise KeyError(_missing_system_msg(system_label))
                self._variable_spaces[system_label] = flat
            self._require_cached_system(system_label)

        new_cd: Dict[Tuple[Any, ...], Any] = {}
        for key, value in old_cd.items():
            if _scalar_is_zero(value):
                continue
            if not isinstance(key, tuple):
                raise TypeError("Keys in `coeff_dict` must be tuples.")
            if len(key) != len(valence_t):
                raise ValueError("`coeff_dict` keys must match tensor degree.")

            idxs = []
            syslbls = []

            for i in key:
                if not isinstance(i, Integral):
                    raise TypeError("Old-style indices must be integers.")
                ii = int(i)
                if ii < 0 or ii >= len(varSpace_t):
                    raise ValueError("Old-style index out of range.")
                var = varSpace_t[ii]
                sys_label = sys_for_var[var]
                syslbls.append(sys_label)

                idx_map = self._system_index_cache.get(sys_label)
                if idx_map is None:
                    self._require_cached_system(sys_label)
                    idx_map = self._system_index_cache[sys_label]

                j = idx_map.get(var)
                if j is None:
                    raise KeyError(
                        f"tensor_field_class: variable '{var}' not found in cached system '{sys_label}'."
                    )
                idxs.append(j)

            nk = tuple(idxs + list(valence_t) + syslbls)
            new_cd[nk] = new_cd.get(nk, 0) + value

        new_cd = {k: v for k, v in new_cd.items() if not _scalar_is_zero(v)}
        if not new_cd:
            return {tuple(): 0}, inferred

        return new_cd, inferred
