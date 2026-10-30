from __future__ import annotations

from ....._aux._backends._parallel import parallel_map
from ....._aux._backends._symbolic_router import _scalar_is_zero, conjugate
from ....._aux._backends._types_and_constants import imag_unit, rational
from ....._aux._utilities._config import dgcv_warning, get_dgcv_settings_registry
from ....conversions.conversions import _coeff_dict_formatter, allToSym
from ...coordinate_formats import conj_with_hol_coor
from .workers import _profile_key_full_inference, _to_complex_algo, _to_real_algo


class _tensor_field_formats:
    def _to_real(self, plan=None):
        new_dict = _to_real_algo(
            plan=plan, cd=self.coeff_dict, vst=self.variable_spaces_types
        )
        return self.__class__(
            coeff_dict=new_dict,
            data_shape=self.data_shape,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces,
        )

    def _to_complex(self, plan=None):
        new_dict = _to_complex_algo(
            plan=plan, cd=self.coeff_dict, vst=self.variable_spaces_types
        )
        return self.__class__(
            coeff_dict=new_dict,
            data_shape=self.data_shape,
            dgcvType="complex",
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces,
        )

    def _get_parts(self, *, type: str):
        t = str(type).lower()

        if t in {"holo", "holomorphic"}:
            want = "ch"
            do_all_to_sym = True
            mode = "pure"
        elif t in {"anti", "antiholo", "antiholomorphic"}:
            want = "ca"
            do_all_to_sym = True
            mode = "pure"
        elif t == "mixed":
            want = None
            do_all_to_sym = True
            mode = "mixed"
        elif t == "standard":
            want = "ss"
            do_all_to_sym = False
            mode = "pure"
        elif t in {"real", "imag", "imaginary"}:
            cfg = get_dgcv_settings_registry()
            if not cfg.get("forgo_warnings", False):
                info_f = getattr(self, "_coordinate_format_info", None)
                info = info_f() if callable(info_f) else None
                if not (isinstance(info, dict) and info.get("dgcv_type") == "complex"):
                    dgcv_warning(
                        "Requested real/imaginary part of a tensor field whose variables are not registered as dgcv complex coordinate systems. "
                        "The result is only well-defined if such variables are assumed real.",
                        UserWarning,
                        stacklevel=2,
                    )

            half = rational(1, 2)
            if t == "real":
                return half * (self + conjugate(self))
            return imag_unit() * half * (-self + conjugate(self))

        else:
            raise ValueError(
                "type must be one of: 'holomorphic', 'antiholomorphic', 'real', 'imaginary', 'mixed', 'standard'."
            )

        obj = self
        if do_all_to_sym:
            obj = allToSym(obj)

        new_cd = {}
        for k, v in obj.coeff_dict.items():
            if _scalar_is_zero(v):
                continue
            prof = obj._profile_key(k)

            if mode == "pure":
                if all(tag == want for tag in prof):
                    new_cd[k] = new_cd.get(k, 0) + v
            else:  # "mixed"
                if len(set(prof)) > 1:
                    new_cd[k] = new_cd.get(k, 0) + v

        if not new_cd:
            new_cd = {tuple(): 0}

        return self.__class__(
            coeff_dict=new_cd,
            data_shape=getattr(self, "data_shape", "general"),
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces,
        )

    def _profile_key(self, k):
        prof = self._key_profiles.get(k)
        if prof is not None:
            return prof
        prof = _profile_key_full_inference(k, _variable_dict=self.variable_spaces_types)
        self._key_profiles[k] = prof
        return prof

    def _flip_format(self, slot: int, k, v, *, to_kind: str):
        degree = len(k) // 3
        idxs = list(k[:degree])
        valences = k[degree : 2 * degree]
        systems = k[2 * degree :]

        sys = systems[slot]
        sys_data = self.variable_spaces_types.get(sys)
        if sys_data is None or sys_data.get("type") != "complex":
            return {k: v}

        b0, b1, b2 = sys_data["breaks"]
        idx = idxs[slot]
        vb = valences[slot]

        half = rational(1, 2)
        imU = imag_unit()

        def _mk(new_idx, coeff):
            new_idxs = list(idxs)
            new_idxs[slot] = new_idx
            nk = tuple(new_idxs) + tuple(valences) + tuple(systems)
            return nk, coeff

        if to_kind == "r":
            if idx >= b1:
                return {k: v}

            if idx >= b0:
                real_idx = idx + b0
                imag_idx = idx + b1
                if vb == 1:
                    nk1, c1 = _mk(real_idx, half * v)
                    nk2, c2 = _mk(imag_idx, half * imU * v)
                else:
                    nk1, c1 = _mk(real_idx, v)
                    nk2, c2 = _mk(imag_idx, -imU * v)
                return {nk1: c1, nk2: c2}

            real_idx = idx + b1
            imag_idx = idx + b2
            if vb == 1:
                nk1, c1 = _mk(real_idx, half * v)
                nk2, c2 = _mk(imag_idx, -half * imU * v)
            else:
                nk1, c1 = _mk(real_idx, v)
                nk2, c2 = _mk(imag_idx, imU * v)
            return {nk1: c1, nk2: c2}

        if to_kind == "c":
            if idx < b1:
                return {k: v}

            if idx < b2:
                holo_idx = idx - b1
                anti_idx = idx - b1
                if vb == 1:
                    nk1, c1 = _mk(holo_idx, v)
                    nk2, c2 = _mk(anti_idx, v)
                else:
                    nk1, c1 = _mk(holo_idx, half * v)
                    nk2, c2 = _mk(anti_idx, half * v)
                return {nk1: c1, nk2: c2}

            holo_idx = idx - b2
            anti_idx = idx - b2
            if vb == 1:
                nk1, c1 = _mk(holo_idx, imU * v)
                nk2, c2 = _mk(anti_idx, -imU * v)
            else:
                nk1, c1 = _mk(holo_idx, -imU * half * v)
                nk2, c2 = _mk(anti_idx, imU * half * v)
            return {nk1: c1, nk2: c2}

        raise ValueError("to_kind must be 'r' or 'c'")

    def __dgcv_apply__(self, fun, **kwargs):
        return self.apply(fun, **kwargs)

    def apply(
        self,
        func,
        *,
        drop_zeros: bool = True,
        data_shape: str | None = None,
        dgcvType: str | None = None,
        _simplifyKW=None,
        variable_spaces=None,
        **func_kwargs,
    ):
        if not callable(func):
            raise TypeError("apply(func): `func` must be callable.")

        items = list(self.coeff_dict.items())
        values = parallel_map(
            "coefficient_map",
            lambda v: func(v, **func_kwargs),
            [v for _, v in items],
            min_items=16,
        )
        out = {}
        for (k, _), vv in zip(items, values):
            if drop_zeros and _scalar_is_zero(vv):
                continue
            out[k] = vv

        if not out:
            out = {tuple(): 0}

        return self.__class__(
            coeff_dict=out,
            data_shape=self.data_shape if data_shape is None else data_shape,
            dgcvType=self.dgcvType if dgcvType is None else dgcvType,
            _simplifyKW=self._simplifyKW if _simplifyKW is None else _simplifyKW,
            variable_spaces=self._variable_spaces
            if variable_spaces is None
            else variable_spaces,
        )

    def __dgcv_conjugate__(self, symbolic=False):
        conj = conjugate if symbolic is False else conj_with_hol_coor
        new_cd = {}
        vst = self.variable_spaces_types
        cache = self._conj_key_profiles

        for k, v in self.coeff_dict.items():
            if k == tuple():
                nk = tuple()
                nv = conj(v)
                new_cd[nk] = new_cd.get(nk, 0) + nv
                continue

            shifts = cache.get(k)
            if shifts is None:
                L = len(k)
                d = L // 3
                idxs = k[:d]
                systems = k[2 * d :]

                out = []
                for idx, sys in zip(idxs, systems):
                    sys_data = vst.get(sys)
                    if sys_data is None or sys_data.get("type") != "complex":
                        out.append(0)
                        continue

                    b0, b1, b2 = sys_data["breaks"]

                    if idx < b0:
                        out.append(b0)
                    elif idx < b1:
                        out.append(-b0)
                    else:
                        out.append(0)

                shifts = tuple(out)
                cache[k] = shifts

            d = len(shifts)
            idxs = k[:d]
            tail = k[d:]  # valences + systems

            new_idxs = tuple(i + s for i, s in zip(idxs, shifts))
            nk = new_idxs + tail
            nv = conj(v)
            new_cd[nk] = new_cd.get(nk, 0) + nv

        return self.__class__(
            coeff_dict=new_cd,
            data_shape=self.data_shape,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces,
        )

    def holomorphic_part(self):
        return self._get_parts(type="holomorphic")

    def antiholomorphic_part(self):
        return self._get_parts(type="antiholomorphic")

    def mixed_term_component(self):
        return self._get_parts(type="mixed")

    def pure_standard_coordinate_terms(self):
        return self._get_parts(type="standard")

    def real_part(self):
        return self._get_parts(type="real")

    def imaginary_part(self):
        return self._get_parts(type="imaginary")

    @property
    def cd_formats(self):
        if self._cd_formats is not None:
            return self._cd_formats
        if self.dgcvType == "standard":
            return self._cd_formats
        vs = self._legacy_varSpace()
        cd = self._legacy_coeff_dict()
        if vs is None or cd is None:
            return self._cd_formats
        (
            populate,
            self._realVarSpace,
            self._holVarSpace,
            self._antiholVarSpace,
            self._imVarSpace,
        ) = _coeff_dict_formatter(
            vs,
            cd,
            self.valence,
            self.total_degree,
            getattr(self, "_varSpace_type", "standard"),
            self.data_shape if self.data_shape != "all" else "general",
        )
        self._cd_formats = populate
        return populate

    def __dgcv_re__(self):
        return self.real_part()

    def __dgcv_im__(self):
        return self.imaginary_part()

    def _coordinate_format_info(self) -> dict:
        cached = getattr(self, "_coordinate_format", None)
        if cached is not None:
            return cached

        vst = self.variable_spaces_types

        saw_standard = False
        saw_complex = False

        saw_holo = False
        saw_anti = False
        saw_real = False
        saw_imag = False

        for k in self.coeff_dict:
            d = len(k) // 3
            if d == 0:
                continue

            idxs = k[:d]
            systems = k[2 * d :]

            for idx, sys in zip(idxs, systems):
                sys_data = vst[sys]
                if sys_data["type"] != "complex":
                    saw_standard = True
                    continue

                saw_complex = True
                b0, b1, b2 = sys_data["breaks"]

                if idx < b0:
                    saw_holo = True
                elif idx < b1:
                    saw_anti = True
                elif idx < b2:
                    saw_real = True
                else:
                    saw_imag = True

        if not saw_complex:
            out = {"dgcv_type": "standard", "sub_type": "standard", "role": "standard"}
            self._coordinate_format = out
            return out

        if saw_standard:
            out = {"dgcv_type": "mixed", "sub_type": "mixed", "role": "mixed"}
            self._coordinate_format = out
            return out

        has_complex_block = saw_holo or saw_anti
        has_real_block = saw_real or saw_imag

        if has_complex_block and not has_real_block:
            role = "mixed"
            if saw_holo and not saw_anti:
                role = "holo"
            elif saw_anti and not saw_holo:
                role = "anti"
            out = {"dgcv_type": "complex", "sub_type": "complex", "role": role}
            self._coordinate_format = out
            return out

        if has_real_block and not has_complex_block:
            role = "mixed"
            if saw_real and not saw_imag:
                role = "real"
            elif saw_imag and not saw_real:
                role = "imag"
            out = {"dgcv_type": "complex", "sub_type": "real", "role": role}
            self._coordinate_format = out
            return out

        out = {"dgcv_type": "complex", "sub_type": "mixed", "role": "mixed"}
        self._coordinate_format = out
        return out
