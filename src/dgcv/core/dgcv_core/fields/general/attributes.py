from __future__ import annotations

import itertools

from ....._aux._backends._symbolic_router import (
    _scalar_is_zero,
    as_numer_denom,
    expand,
    gcd_routed,
    get_free_symbols,
    lcm_routed,
)
from ....._aux._backends._symbolic_router import cancel as cancel_dgcv
from ....._aux._backends._symbolic_router import factor as factor_dgcv
from ....._aux._vmf._safeguards import retrieve_passkey
from ....._aux._vmf.vmf import vmf_lookup
from ....arrays import _spool, array_dgcv
from .workers import _expand_special_to_general


class _tensor_field_attributes:
    @property
    def max_degree(self):
        if self._max_degree is None:
            self._set_degrees()
        return self._max_degree

    @property
    def min_degree(self):
        if self._min_degree is None:
            self._set_degrees()
        return self._min_degree

    @property
    def total_degree(self):
        return self.max_degree

    @property
    def degree(self):
        return self.total_degree

    @property
    def free_symbols(self):
        if self._free_symbols is None:
            fs = set()
            for atoms in self._variable_spaces.values():
                fs.update(atoms)

            for v in self.coeff_dict.values():
                fs |= get_free_symbols(v)

            self._free_symbols = fs
        return self._free_symbols

    @property
    def coeff_free_symbols(self):
        if self._coeff_free_symbols is None:
            fs = set()
            for v in self.coeff_dict.values():
                fs |= get_free_symbols(v)
            self._coeff_free_symbols = fs
        return self._coeff_free_symbols

    @property
    def coef_profile(self):
        """Returns a set of variable types present in the coeffs (among, standard, real, hol, sym, mixed). "mixed" is included if and only if there are mixed complex coordinate types (e.g., real+holo or holo+sym, etc.)"""
        if self._coef_profile is None:
            atoms = self.coeff_free_symbols
            out = set()
            type_dict = {"imag": "real", "anti": "sym"}
            standard = set()
            for atom in atoms:
                new_type = vmf_lookup(atom)["sub_type"]
                if new_type == "standard":
                    standard = {"standard"}
                    continue
                out.add(type_dict.get(new_type, new_type))
                if len(out) > 1:
                    out |= {"mixed"}
            self._coef_profile = standard | out
        return self._coef_profile

    @property
    def _compute_nd_decomp(self):

        if self._num_den is None:
            self._num_den = tuple(
                zip(
                    *[as_numer_denom(expand(coef)) for coef in self.coeff_dict.values()]
                )
            )
        return self._num_den

    @property
    def denominators(self):
        return self._compute_nd_decomp[1]

    @property
    def numerators(self):
        return self._compute_nd_decomp[0]

    def scale_to_polynomial_attempt(self, factor=True, return_scale=False, try_hard=False):

        if self._denom_cm is None:
            self._denom_cm = lcm_routed(*self.denominators)
        if self._num_cd is None:
            self._num_cd = gcd_routed(*self.numerators)
        scaled = self._denom_cm * self / self._num_cd
        if factor:
            try:
                scaled = factor_dgcv(scaled, try_hard=try_hard)
            except NotImplementedError:
                scaled = cancel_dgcv(scaled)
        if return_scale:
            return scaled, self._denom_cm
        return scaled

    @property
    def homogeneous_parts(self):
        new_dicts = dict()
        for k, v in self.coeff_dict.items():
            deg = len(k) // 3
            valence = tuple(k[deg : 2 * deg])
            new_dicts[valence] = new_dicts.get(valence, dict())
            new_dicts[valence][k] = v
        return [
            self.__class__(
                coeff_dict=cd,
                data_shape=self.data_shape,
                _simplifyKW=self._simplifyKW,
                variable_spaces=self._variable_spaces,
                parameters=self.parameters,
            )
            for cd in new_dicts.values()
        ]

    @property
    def coeffArray(self):
        if self._coeffArray is not None:
            return self._coeffArray

        self._require_legacy_view("coeffArray")

        vs = self._legacy_varSpace()
        deg = self.total_degree
        n = len(vs)
        shape = (n,) * deg

        cd = self.expanded_coeff_dict
        if self.data_shape == "symmetric":
            sparse_data = {
                idx: cd.get(tuple(sorted(idx)), 0)
                for idx in itertools.product(range(n), repeat=deg)
            }
        else:
            sparse_data = {
                idx: cd.get(idx, 0) for idx in itertools.product(range(n), repeat=deg)
            }
        flat = {_spool(idx_tup, shape): v for idx_tup, v in sparse_data.items()}

        arr = array_dgcv.__new__(array_dgcv)
        arr._data = flat
        arr.shape = shape
        arr.ndim = len(shape)
        arr._dgcv_class_check = retrieve_passkey()
        arr._dgcv_category = "array"

        self._coeffArray = arr
        return self._coeffArray

    @property
    def expanded_coeff_dict(self):
        if self._expanded_coeff_dict is not None:
            return self._expanded_coeff_dict

        if self.data_shape in ("general", "all"):
            if self.data_shape == "all":
                self._expanded_coeff_dict = {tuple(): self.coeff_dict.get(tuple(), 0)}
            else:
                self._expanded_coeff_dict = {
                    k: v for k, v in self.coeff_dict.items() if not _scalar_is_zero(v)
                }
            return self._expanded_coeff_dict

        self._expanded_coeff_dict = _expand_special_to_general(
            self.coeff_dict, self.data_shape
        )
        return self._expanded_coeff_dict

    def _is_scalar(self) -> bool:
        return self.data_shape == "all" and self.total_degree == 0

    def _scalar_value(self):
        return self.coeff_dict.get(tuple(), 0)
