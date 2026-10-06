from __future__ import annotations

from ..._aux._backends._symbolic_router import (
    _resolve_subs_keys,
    _scalar_is_zero,
    get_free_symbols,
    subs,
)
from ..._aux._backends._types_and_constants import expr_numeric_types
from ..._aux._utilities._config import dgcv_warning
from ..._aux._vmf._safeguards import get_dgcv_category, retrieve_passkey


def _hom_id_from_decomp(decomp, label, pref, ngla):
    hidd = dict()
    for key, v in decomp.items():
        jidx, kidx, jdeg, kdeg = key
        try:
            jfac = ngla[jdeg][jidx]
            if kdeg < 0:
                kfac = ngla[kdeg][kidx]
            else:
                kfac = f"{pref}_{kidx + 1}__{{[{kdeg}]}}"
            hidd[(jfac, kfac)] = v
        except Exception:
            return None
    return [hidd, label]


def _dict_to_algebra(coeff_dict, alg):
    basis = getattr(alg, "basis", ())
    dim = len(basis)
    new = {}
    for k, v in coeff_dict.items():
        if len(k) != 1:
            return False
        idx = k[0]
        if idx < 0 or idx >= dim:
            return False
        new[idx] = v
    if not new:
        return 0
    return basis[0]._class_builder(new, 1)


def _one_shot_algebra(alg):
    return (
        get_dgcv_category(alg) == "algebra"
        and alg.simplify_products_by_default is not True
    )


class _fast_tensor_products:
    def __init__(self, coeff_dict, alg=None, _validated=None, _atomic_index=-1):
        if isinstance(coeff_dict, _fast_tensor_products):
            coeff_dict, alg, _validated = (
                coeff_dict.coeff_dict,
                coeff_dict.algebra,
                coeff_dict.degree,
            )
        self._atomic_index = _atomic_index
        self.algebra = alg
        if _validated is None:
            if get_dgcv_category(coeff_dict) == "tensorProduct":
                if alg is None:
                    self.algebra = coeff_dict.vector_space
                self.coeff_dict = dict()
                self.degree = 0
                for k, v in coeff_dict.coeff_dict.items():
                    newkey = tuple(factor[0] for factor in k)
                    self.coeff_dict[newkey] = v
                    self.degree = max(self.degree, len(newkey))
            elif isinstance(coeff_dict, dict):
                self.coeff_dict = dict()
                self.degree = 0
                for k, v in coeff_dict.items():
                    if not _scalar_is_zero(v) or k == tuple():
                        self.coeff_dict[k] = v
                        self.degree = max(self.degree, len(k))
            elif get_dgcv_category(coeff_dict) in {
                "algebra_element",
                "subalgebra_element",
            }:
                if alg is None:
                    self.algebra = coeff_dict.algebra
                self.degree = 1
                if get_dgcv_category(coeff_dict) == "algebra_element":
                    self.coeff_dict = {
                        (k,): v
                        for k, v in sorted(coeff_dict.coeff_dict.items())
                        if not _scalar_is_zero(v)
                    }
                else:
                    self.coeff_dict = {
                        (k,): v
                        for k, v in enumerate(coeff_dict.coeffs)
                        if not _scalar_is_zero(v)
                    }
            else:
                self.coeff_dict = dict()
        else:
            self.degree = _validated
            self.coeff_dict = coeff_dict
        if len(self.coeff_dict) == 0:
            self.coeff_dict = {tuple(): 0}
            self.degree = 0
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "fastTensorProduct"
        self._is_zero = None
        self._coeffs = None
        self._alg_split_cache = None
        self._to_algebra_cache = None
        if self.degree < max(len(k) for k in self.coeff_dict):
            raise TypeError("ftp init fail")

    @property
    def is_zero(self):
        if self._is_zero is None:
            self._is_zero = (
                False
                if any(not _scalar_is_zero(v) for v in self.coeff_dict.values())
                else True
            )
        return self._is_zero

    @property
    def coeffs(self):
        if self._coeffs is None:
            self._coeffs = list(self.coeff_dict.values())
        return self._coeffs

    @property
    def __dgcv_zero_obstr__(self):
        cfs = []
        cfvars = set()
        for cf in self.coeff_dict.values():
            cfs.append(cf)
            cfvars |= get_free_symbols(cf)
        return cfs, cfvars

    def _to_algebra(self, alg=None):
        own = alg is None or alg is self.algebra
        if own and self._to_algebra_cache is not None:
            return self._to_algebra_cache
        if alg is None:
            alg = self.algebra
        if _one_shot_algebra(alg):
            ae = _dict_to_algebra(self.coeff_dict, alg)
            if own:
                self._to_algebra_cache = ae
            return ae
        ae = 0
        basis = getattr(alg, "basis", [])
        dim = len(basis)
        for k, v in self.coeff_dict.items():
            if len(k) != 1:
                ae = False
                break
            idx = k[0]
            if idx < 0 or idx >= dim:
                ae = False
                break
            ae += v * basis[idx]
        if own:
            self._to_algebra_cache = ae
        return ae

    def _alg_split(self):
        cached = self._alg_split_cache
        if cached is not None:
            return cached
        alg_basis = getattr(self.algebra, "basis", [])
        alg_dim = len(alg_basis)
        ae = 0
        high = {}
        deg = 0
        for k, v in self.coeff_dict.items():
            if len(k) == 1:
                idx = k[0]
                if 0 <= idx < alg_dim:
                    ae += v * alg_basis[idx]
                    continue
                dgcv_warning(
                    "fast_tensor_products non-aliased mul is being tried for aliased elements",
                    wc_label="debug_log",
                )
                continue
            high[k] = v
            deg = max(deg, len(k))
        by_first = {}
        by_last = {}
        by_tail = {}
        for k, v in high.items():
            by_first.setdefault(k[0], []).append((k[:-1], v))
            by_last.setdefault(k[-1], []).append((k[1:], v))
            by_tail.setdefault(k[-1], []).append((k[:-1], v))
        cached = (ae, high, deg, by_first, by_last, by_tail)
        self._alg_split_cache = cached
        return cached

    def _convert_to_tp(
        self,
        _hom_id_map=None,
        _hom_id_label=None,
        _hom_id=None,
        _decomp_complete=True,
    ):
        """
        _hom_id_map should be a pair of (str pref, map from neg weights to components)
        """
        from ...core.tensors import tensorProduct

        new_dict = dict()
        card = self.algebra.card
        for k, v in self.coeff_dict.items():
            newkey = tuple(
                (idx, 1 if pos == 0 else 0, card) for pos, idx in enumerate(k)
            )
            new_dict[newkey] = v
        homid = None
        homdecomp = None
        hom_source = _hom_id
        if hom_source:
            decomp, label = hom_source
            if _decomp_complete:
                homdecomp = dict(decomp)
            if _hom_id_map:
                if _hom_id_label:
                    label = _hom_id_label
                pref, ngla = _hom_id_map
                homid = _hom_id_from_decomp(decomp, label, pref, ngla)
        return tensorProduct(new_dict, _hom_id=homid, _hom_decomp=homdecomp)

    def __add__(self, other):
        if _scalar_is_zero(other):
            return self
        if isinstance(other, _fast_tensor_products):
            new_dict = dict(self.coeff_dict)
            deg = self.degree
            for k, v in other.coeff_dict.items():
                deg = max(len(k), deg)
                new_dict[k] = self.coeff_dict.get(k, 0) + v

            return _fast_tensor_products(new_dict, self.algebra, _validated=deg)
        if get_dgcv_category(other) in {
            "algebra_element",
            "subalgebra_element",
        }:
            return self + _fast_tensor_products(other)
        return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return (self).__add__(-other)

    def __rsub__(self, other):
        return (-self) + other

    @classmethod
    def _dgcv_multiadd(cls, terms, start=0):
        if not isinstance(terms, (list, tuple)):
            terms = list(terms)
        if not terms:
            return start
        acc = {}
        alg = None
        alg_set = False
        deg = 0
        residual = []
        if isinstance(start, cls):
            acc.update(start.coeff_dict)
            alg = start.algebra
            alg_set = True
            deg = start.degree
        elif not (isinstance(start, int) and start == 0):
            residual.append(start)
        for t in terms:
            if not isinstance(t, cls):
                residual.append(t)
                continue
            if not alg_set:
                alg = t.algebra
                alg_set = True
            if t.is_zero:
                continue
            for k, v in t.coeff_dict.items():
                deg = max(deg, len(k))
                if not _scalar_is_zero(v):
                    acc[k] = acc.get(k, 0) + v
        out = cls(
            {k: v for k, v in acc.items() if not _scalar_is_zero(v)},
            alg,
            _validated=deg,
        )
        if residual:
            return sum(residual, out)
        return out

    @classmethod
    def _dgcv_multiadd_scaled(cls, pairs, start=0):
        if not isinstance(pairs, (list, tuple)):
            pairs = list(pairs)
        if not pairs:
            return start
        acc = {}
        alg = None
        alg_set = False
        deg = 0
        residual = []
        if isinstance(start, cls):
            acc.update(start.coeff_dict)
            alg = start.algebra
            alg_set = True
            deg = start.degree
        elif not (isinstance(start, int) and start == 0):
            residual.append(start)
        pending = {}
        for c, t in pairs:
            if not isinstance(t, cls):
                residual.append(c * t)
                continue
            if not alg_set:
                alg = t.algebra
                alg_set = True
            if _scalar_is_zero(c) or t.is_zero:
                continue
            for k, v in t.coeff_dict.items():
                deg = max(deg, len(k))
                lst = pending.get(k)
                if lst is None:
                    pending[k] = [(c, v)]
                else:
                    lst.append((c, v))
        for k, lst in pending.items():
            hook = getattr(type(lst[0][1]), "_dgcv_multiadd_scaled", None)
            if hook is not None:
                total = hook(lst, acc.get(k, 0))
            else:
                total = acc.get(k, 0)
                for c, v in lst:
                    total = total + c * v
            if _scalar_is_zero(total):
                acc.pop(k, None)
            else:
                acc[k] = total
        out = cls(
            {k: v for k, v in acc.items() if not _scalar_is_zero(v)},
            alg,
            _validated=deg,
        )
        if residual:
            return sum(residual, out)
        return out

    def __mul__(self, other):
        if type(other) is not _fast_tensor_products and isinstance(
            other, expr_numeric_types()
        ):
            if _scalar_is_zero(other):
                return _fast_tensor_products({tuple(): 0}, self.algebra, _validated=0)
            return _fast_tensor_products(
                {k: other * v for k, v in self.coeff_dict.items()},
                self.algebra,
                _validated=self.degree,
            )
        if isinstance(other, _fast_tensor_products):
            if other.degree == 0:
                return sum(v * self for v in other.coeff_dict.values())
            if self.degree == 1:
                algebraized = self._to_algebra()
                if algebraized is not False:
                    return other._mul_alg(algebraized, -1)
            if self.degree == 0:
                return sum(v * other for v in self.coeff_dict.values())
            if other.degree == 1:
                algebraized = self._to_algebra()
                if algebraized is not False:
                    return self * algebraized
            new_dict = dict()
            deg = 0
            ae1, self_high, self_deg = self._alg_split()[:3]
            ae2, other_high, _other_deg, by_first, by_last, _bt = other._alg_split()
            for k1, v1 in self_high.items():
                k1L, k1A, k1B, k1T = k1[0], k1[:-1], k1[1:], k1[-1]
                for k2A, v2 in by_first.get(k1T, ()):
                    newkey = k1B + k2A
                    newval = new_dict.get(newkey, 0) + v1 * v2
                    if not _scalar_is_zero(newval):
                        deg = max(len(newkey), deg)
                        new_dict[newkey] = newval
                    else:
                        new_dict.pop(newkey, None)
                for k2B, v2 in by_last.get(k1L, ()):
                    newkey = k2B + k1A
                    newval = new_dict.get(newkey, 0) - v1 * v2
                    if not _scalar_is_zero(newval):
                        deg = max(len(newkey), deg)
                        new_dict[newkey] = newval
                    else:
                        new_dict.pop(newkey, None)
            out = _fast_tensor_products(new_dict, self.algebra, _validated=deg)
            if not isinstance(ae1, int):
                out = out + ae1 * other
            if self_high and not isinstance(ae2, int):
                out = out + (
                    _fast_tensor_products(self_high, self.algebra, _validated=self_deg)
                    * ae2
                )
            return out
        if get_dgcv_category(other) in {
            "algebra_element",
            "subalgebra_element",
        }:
            return self._mul_alg(other, 1)
        return NotImplemented

    def _mul_alg(self, other, sign):
        if self.degree == 0:
            if sign < 0:
                other = -other
            return sum(v * other for v in self.coeff_dict.values())
        if self.degree == 1:
            algebraized = self._to_algebra()
            if algebraized is not False:
                if sign < 0:
                    return other * algebraized
                return algebraized * other
            dgcv_warning(
                "fast_tensor_products non-aliased mul is being tried for aliased elements",
                wc_label="debug_log",
            )
            return self * _fast_tensor_products(-other if sign < 0 else other)
        new_dict = dict()
        ae1, _high, _deg, _bf, _bl, by_tail = self._alg_split()
        for idx, c in other.coeff_dict.items():
            if _scalar_is_zero(c):
                continue
            if sign < 0:
                c = -c
            for k1A, v1 in by_tail.get(idx, ()):
                newval = new_dict.get(k1A, 0) + c * v1
                if not _scalar_is_zero(newval):
                    new_dict[k1A] = newval
                else:
                    new_dict.pop(k1A, None)
        algebraized = False
        if self.degree == 2 and new_dict and _one_shot_algebra(self.algebra):
            algebraized = _dict_to_algebra(new_dict, self.algebra)
        if algebraized is False:
            new_tensor = _fast_tensor_products(
                new_dict, self.algebra, _validated=self.degree - 1
            )
            if self.degree == 2:
                algebraized = new_tensor._to_algebra()
        if isinstance(ae1, int):
            return new_tensor if algebraized is False else algebraized
        tail = other * ae1 if sign < 0 else ae1 * other
        if algebraized is not False:
            return algebraized + tail
        return new_tensor + tail

    def __rmul__(self, other):
        if isinstance(other, expr_numeric_types()):
            if _scalar_is_zero(other):
                return _fast_tensor_products(dict(), self.algebra, _validated=0)
            return _fast_tensor_products(
                {k: other * v for k, v in self.coeff_dict.items()},
                self.algebra,
                _validated=self.degree,
            )
        if self.degree == 0:
            return sum(v * other for v in self.coeff_dict.values())
        if get_dgcv_category(other) in {"algebra_element", "subalgebra_element"}:
            return self._mul_alg(other, -1)
        return self * (-other)

    def __neg__(self):
        return _fast_tensor_products(
            {k: -v for k, v in self.coeff_dict.items()},
            self.algebra,
            _validated=self.degree,
        )

    def __matmul__(self, other):
        if isinstance(other, expr_numeric_types()):
            return self * other
        if get_dgcv_category(other) in {
            "algebra_element",
            "subalgebra_element",
        }:
            ac = other.coeff_dict
            new_dict = dict()
            for k, v in self.coeff_dict.items():
                for idx, c in sorted(ac.items()):
                    if not _scalar_is_zero(c):
                        newkey = k + (idx,)
                        newval = new_dict.get(newkey, 0) + c * v
                        if not _scalar_is_zero(newval):
                            new_dict[newkey] = newval
                        else:
                            new_dict.pop(newkey, None)
            return _fast_tensor_products(
                new_dict, self.algebra, _validated=self.degree + 1
            )
        if isinstance(other, _fast_tensor_products):
            ac = other.coeff_dict
            new_dict = dict()
            for k, v in self.coeff_dict.items():
                for idx, c in other.coeff_dict.items():
                    newkey = k + idx
                    newval = new_dict.get(newkey, 0) + c * v
                    if not _scalar_is_zero(newval):
                        new_dict[newkey] = newval
                    else:
                        new_dict.pop(newkey, None)
            return _fast_tensor_products(
                new_dict, self.algebra, _validated=self.degree + other.degree
            )
        return NotImplemented

    def __rmatmul__(self, other):
        return self.__matmul__(other)

    def subs(self, subs_data):
        data = _resolve_subs_keys(self, subs_data)
        direct = isinstance(data, dict) and not any(type(k) is str for k in data)
        new_dict = {}
        for k, v in self.coeff_dict.items():
            if direct and getattr(v, "_dgcv_category", None) == "abstract_ZF":
                new_dict[k] = v.subs(data)
            else:
                new_dict[k] = subs(v, data)
        return _fast_tensor_products(new_dict, self.algebra, _validated=self.degree)
