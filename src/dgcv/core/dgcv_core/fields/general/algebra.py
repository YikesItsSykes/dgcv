from __future__ import annotations

from bisect import bisect_left, bisect_right

from ....._aux._backends._exact_arith import exact_reciprocal
from ....._aux._backends._symbolic_router import _scalar_is_zero
from ....._aux._backends._types_and_constants import check_dgcv_scalar
from ....._aux._vmf._safeguards import (
    check_dgcv_category,
    get_dgcv_category,
    query_dgcv_categories,
)
from .workers import (
    _process_coeffs_dict_new,
    _slot_format,
    _sort_slot_key,
    _variable_spaces_types_algo,
)


def _merge_variable_spaces(out, other):
    for k, v in other.items():
        if k in out:
            if k is None:
                out[k] = out[None] | v
            elif out[k] != v:
                raise ValueError(
                    f"Incompatible cached variable spaces {out[k]} and {v} for system '{k}'."
                )
            continue
        out[k] = v
    return out


class _tensor_field_algebra:
    def swap_tensor_valence(self):
        def key_change(key):
            deg = len(key) // 3
            new_k = tuple(
                j if c < deg or 2 * deg <= c else 1 - j for c, j in enumerate(key)
            )
            return new_k

        cd = {key_change(key): value for key, value in self.coeff_dict.items()}
        if query_dgcv_categories(self, "differential_form"):
            from ..vector_fields import vector_field_class

            return vector_field_class(
                coeff_dict=cd, _simplifyKW=self._simplifyKW, parameters=self.parameters
            )
        if query_dgcv_categories(self, "vector_field"):
            from ..differential_forms import differential_form_class

            return differential_form_class(
                coeff_dict=cd, _simplifyKW=self._simplifyKW, parameters=self.parameters
            )
        from . import tensor_field_class

        return tensor_field_class(
            coeff_dict=cd, _simplifyKW=self._simplifyKW, parameters=self.parameters
        )

    def _with_same_meta(self, *, coeff_dict, data_shape=None, variable_spaces=None):
        return self.__class__(
            coeff_dict=coeff_dict,
            data_shape=self.data_shape if data_shape is None else data_shape,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces
            if variable_spaces is None
            else variable_spaces,
        )

    @classmethod
    def _dgcv_multiadd(cls, terms, start=0):
        if not isinstance(terms, (list, tuple)):
            terms = list(terms)
        if not terms:
            return start
        acc = {}
        meta = None
        vs = {}
        residual = []
        if isinstance(start, cls) and not start._is_scalar():
            meta = (start.dgcvType, start._simplifyKW, start.data_shape)
            vs = dict(start._variable_spaces)
            for k, v in start.coeff_dict.items():
                if not _scalar_is_zero(v):
                    acc[k] = v
        elif not _scalar_is_zero(start):
            residual.append(start)
        for t in terms:
            if not isinstance(t, cls) or t._is_scalar():
                residual.append(t)
                continue
            if meta is None:
                meta = (t.dgcvType, t._simplifyKW, t.data_shape)
            elif t.dgcvType != meta[0] or t.data_shape != meta[2]:
                residual.append(t)
                continue
            _merge_variable_spaces(vs, t._variable_spaces)
            for k, v in t.coeff_dict.items():
                if _scalar_is_zero(v):
                    continue
                acc[k] = acc[k] + v if k in acc else v
        if meta is None:
            return sum(terms, start)
        acc = {k: v for k, v in acc.items() if not _scalar_is_zero(v)} or {tuple(): 0}
        out = cls(
            coeff_dict=acc,
            dgcvType=meta[0],
            _simplifyKW=meta[1],
            variable_spaces=vs,
            data_shape=meta[2],
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
        meta = None
        vs = {}
        residual = []
        if isinstance(start, cls) and not start._is_scalar():
            meta = (start.dgcvType, start._simplifyKW, start.data_shape)
            vs = dict(start._variable_spaces)
            for k, v in start.coeff_dict.items():
                if not _scalar_is_zero(v):
                    acc[k] = v
        elif not _scalar_is_zero(start):
            residual.append(start)
        for c, t in pairs:
            if not isinstance(t, cls) or t._is_scalar() or not check_dgcv_scalar(c):
                residual.append(c * t)
                continue
            if meta is None:
                meta = (t.dgcvType, t._simplifyKW, t.data_shape)
            elif t.dgcvType != meta[0] or t.data_shape != meta[2]:
                residual.append(c * t)
                continue
            _merge_variable_spaces(vs, t._variable_spaces)
            if _scalar_is_zero(c):
                continue
            for k, v in t.coeff_dict.items():
                nv = c * v
                if _scalar_is_zero(nv):
                    continue
                acc[k] = acc[k] + nv if k in acc else nv
        if meta is None:
            return sum([c * t for c, t in pairs], start)
        acc = {k: v for k, v in acc.items() if not _scalar_is_zero(v)} or {tuple(): 0}
        out = cls(
            coeff_dict=acc,
            dgcvType=meta[0],
            _simplifyKW=meta[1],
            variable_spaces=vs,
            data_shape=meta[2],
        )
        if residual:
            return sum(residual, out)
        return out

    def _coerce_to_general(self):
        if self.data_shape == "general":
            return self
        if self.data_shape == "all":
            return self
        g = self.expanded_coeff_dict
        out = self._with_same_meta(coeff_dict=g, data_shape="general")
        out._shape_checked = True
        return out

    def _maybe_promote_general_to(self, target_shape: str):
        if self._shape_checked:
            return
        self._shape_checked = True
        if self.data_shape != "general":
            return
        if target_shape not in ("symmetric", "skew"):
            return
        if not self.valence or len(set(self.valence)) != 1:
            return
        new_cd, eff_shape = _process_coeffs_dict_new(self.coeff_dict, target_shape)
        if eff_shape == target_shape:
            self.coeff_dict = new_cd
            self.data_shape = eff_shape
            self._expanded_coeff_dict = None
            self._coeffArray = None
            self._cd_formats = None
            self._hash = None
            self._minimal_coordinate_space = None

    def __add__(self, other):
        if _scalar_is_zero(other):
            return self
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self._is_scalar() or other._is_scalar():
            raise TypeError("Cannot add tensors of different degrees.")

        return self._add_tensor(other, coerce_shapes=True)

    def __radd__(self, other):
        if _scalar_is_zero(other):
            return self
        return NotImplemented

    def __neg__(self):
        if self._is_scalar():
            return self.__class__(
                coeff_dict={tuple(): -self._scalar_value()},
                data_shape="all",
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=self._variable_spaces,
            )
        return self._with_same_meta(
            coeff_dict={k: -v for k, v in self.coeff_dict.items()},
            variable_spaces=self._variable_spaces,
        )

    def __sub__(self, other):
        if _scalar_is_zero(other):
            return self
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self + (-other)

    def __matmul__(self, other):
        tf = other
        if check_dgcv_category(tf):
            coerce = getattr(tf, "as_tensor_field", None)
            if callable(coerce):
                tf = coerce()
        if not isinstance(tf, self.__class__):
            return NotImplemented
        return self._shape_product(tf, kind="general")

    def __mul__(self, other):
        if check_dgcv_scalar(other):
            if self._is_scalar():
                return self.__class__(
                    coeff_dict={tuple(): other * self._scalar_value()},
                    data_shape="all",
                    dgcvType=self.dgcvType,
                    _simplifyKW=self._simplifyKW,
                    variable_spaces=self._variable_spaces,
                )
            return self._with_same_meta(
                coeff_dict={k: other * v for k, v in self.coeff_dict.items()},
                variable_spaces=self._variable_spaces,
            )

        tf = other
        if check_dgcv_category(tf):
            coerce = getattr(tf, "as_tensor_field", None)
            if callable(coerce):
                tf = coerce()
        if not isinstance(tf, self.__class__):
            return NotImplemented

        return self._shape_product(tf, kind="general")

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if check_dgcv_scalar(scalar):
            return self * exact_reciprocal(scalar)
        return NotImplemented

    def _add_tensor(self, other, *, coerce_shapes: bool, _return_raw: bool = False):
        a = self
        b = other

        a_shape = a.data_shape
        b_shape = b.data_shape

        if coerce_shapes:
            if a_shape == "general" and b_shape in ("symmetric", "skew"):
                a._maybe_promote_general_to(b_shape)
                a_shape = a.data_shape
            if b_shape == "general" and a_shape in ("symmetric", "skew"):
                b._maybe_promote_general_to(a_shape)
                b_shape = b.data_shape

        if a_shape == "all":
            out_shape = b_shape
        elif b_shape == "all":
            out_shape = a_shape
        elif a_shape == b_shape:
            out_shape = a_shape
        else:
            out_shape = "general"

        aa = a if a_shape == out_shape or a_shape == "all" else a._coerce_to_general()
        bb = b if b_shape == out_shape or b_shape == "all" else b._coerce_to_general()

        new_cd = {}
        for k, v in aa.coeff_dict.items():
            if not _scalar_is_zero(v):
                new_cd[k] = v
        for k, v in bb.coeff_dict.items():
            if not _scalar_is_zero(v):
                new_cd[k] = new_cd.get(k, 0) + v

        new_cd, eff_shape = _process_coeffs_dict_new(new_cd, out_shape)
        merged_vs = a._merged_variable_spaces(b)

        if _return_raw:
            return new_cd, eff_shape, merged_vs

        return self.__class__(
            coeff_dict=new_cd,
            data_shape=eff_shape,
            dgcvType=a.dgcvType,
            _simplifyKW=a._simplifyKW,
            variable_spaces=merged_vs,
        )

    def _tp_concat_cd_fast(self, other, shape=None):
        a = self
        b = other
        out = {}

        def _parity_sign(order):
            n = len(order)
            sign = 1
            seen = [False] * n
            for i in range(n):
                if seen[i]:
                    continue
                j = i
                cycle_len = 0
                while not seen[j]:
                    seen[j] = True
                    j = order[j]
                    cycle_len += 1
                if cycle_len and (cycle_len % 2 == 0):
                    sign = -sign
            return sign

        skew = shape == "skew"
        b_items = []
        for kb, vb in b.coeff_dict.items():
            if _scalar_is_zero(vb):
                continue
            if kb:
                db = len(kb) // 3
                ib = kb[:db]
                vb_bits = kb[db : 2 * db]
                sb = kb[2 * db :]
            else:
                ib = vb_bits = sb = tuple()
            slots_b = frozenset(zip(ib, vb_bits, sb)) if skew else None
            b_items.append((kb, vb, ib, vb_bits, sb, slots_b))

        for ka, va in a.coeff_dict.items():
            if _scalar_is_zero(va):
                continue
            if ka:
                da = len(ka) // 3
                ia = ka[:da]
                va_bits = ka[da : 2 * da]
                sa = ka[2 * da :]
            else:
                ia = va_bits = sa = tuple()
            slots_a = frozenset(zip(ia, va_bits, sa)) if skew else None

            for kb, vb, ib, vb_bits, sb, slots_b in b_items:
                inds = ia + ib
                bits = va_bits + vb_bits
                sys = sa + sb
                n = len(inds)

                if n == 0:
                    nk = tuple()
                    out[nk] = out.get(nk, 0) + va * vb
                    continue

                if skew:
                    if slots_a & slots_b:
                        continue
                    nk = inds + bits + sys
                    out[nk] = out.get(nk, 0) + va * vb
                    continue

                if shape == "symmetric":
                    if shape == "skew":
                        order = sorted(
                            range(n), key=lambda k: (str(inds[k]), bits[k], sys[k])
                        )
                        sign = _parity_sign(order)
                    else:
                        order = sorted(
                            range(n), key=lambda k: (inds[k], bits[k], sys[k])
                        )
                        sign = 1

                    inds2 = tuple(inds[k] for k in order)
                    bits2 = tuple(bits[k] for k in order)
                    sys2 = tuple(sys[k] for k in order)

                    if shape == "skew":
                        seen = set()
                        for t in zip(inds2, bits2, sys2):
                            if t in seen:
                                sign = 0
                                break
                            seen.add(t)
                        if sign == 0:
                            continue

                    nk = inds2 + bits2 + sys2
                    out[nk] = out.get(nk, 0) + sign * va * vb
                    continue

                nk = inds + bits + sys
                out[nk] = out.get(nk, 0) + va * vb

        return out

    def _skew_concat_canonical(self, other, variable_spaces):
        vst = _variable_spaces_types_algo(variable_spaces)
        key_cache = {}
        fmt_cache = {}

        def slot_key(slot):
            out = key_cache.get(slot)
            if out is None:
                out = _sort_slot_key(slot)
                key_cache[slot] = out
            return out

        def slot_fmt(slot):
            out = fmt_cache.get(slot)
            if out is None:
                out = _slot_format(slot[0], slot[2], vst)
                fmt_cache[slot] = out
            return out

        def items_of(tf):
            out = []
            canonical = tf.data_shape == "skew"
            for k, v in tf.coeff_dict.items():
                if _scalar_is_zero(v):
                    continue
                d = len(k) // 3
                if d > 1 and not canonical:
                    return None
                slots = tuple(zip(k[:d], k[d : 2 * d], k[2 * d :]))
                keys = tuple(slot_key(s) for s in slots)
                fmts = set()
                for s in slots:
                    f = slot_fmt(s)
                    if f is None:
                        return None
                    fmts.add(f)
                out.append((slots, keys, v, fmts))
            return out

        a_items = items_of(self)
        if a_items is None:
            return None
        b_items = items_of(other)
        if b_items is None:
            return None
        out = {}
        used_a = [False] * len(a_items)
        used_b = [False] * len(b_items)
        for ia, (sa, ka, va, fa) in enumerate(a_items):
            la = len(sa)
            for ib, (sb, kb, vb, fb) in enumerate(b_items):
                lb = len(sb)
                if lb == 1 and la:
                    kb0 = kb[0]
                    lo = bisect_left(ka, kb0)
                    hi = bisect_right(ka, kb0)
                    if lo < hi:
                        if sb[0] in sa[lo:hi]:
                            continue
                        return None
                    merged = sa[:lo] + sb + sa[lo:]
                    inv = la - lo
                elif la == 1 and lb:
                    ka0 = ka[0]
                    lo = bisect_left(kb, ka0)
                    hi = bisect_right(kb, ka0)
                    if lo < hi:
                        if sa[0] in sb[lo:hi]:
                            continue
                        return None
                    merged = sb[:lo] + sa + sb[lo:]
                    inv = lo
                else:
                    i = 0
                    j = 0
                    inv = 0
                    merged = []
                    dup = False
                    while i < la and j < lb:
                        if ka[i] < kb[j]:
                            merged.append(sa[i])
                            i += 1
                        elif kb[j] < ka[i]:
                            merged.append(sb[j])
                            j += 1
                            inv += la - i
                        elif sa[i] == sb[j]:
                            dup = True
                            break
                        else:
                            return None
                    if dup:
                        continue
                    if i < la:
                        merged.extend(sa[i:])
                    elif j < lb:
                        merged.extend(sb[j:])
                if merged:
                    idxs, bits, sys = zip(*merged)
                    nk = tuple(idxs) + tuple(bits) + tuple(sys)
                else:
                    nk = tuple()
                v = va * vb
                if inv & 1:
                    v = -v
                out[nk] = out[nk] + v if nk in out else v
                used_a[ia] = True
                used_b[ib] = True
        out = {k: v for k, v in out.items() if not _scalar_is_zero(v)}
        if not out or tuple() in out:
            return None
        formats = set()
        for used, items in ((used_a, a_items), (used_b, b_items)):
            for flag, item in zip(used, items):
                if flag:
                    formats |= item[3]
        if "complex" in formats:
            fmt = "mixed" if "real" in formats else "complex"
        elif "real" in formats:
            fmt = "real"
        elif formats:
            fmt = "standard"
        else:
            fmt = "open"
        return out, fmt

    def _shape_product(self, other, *, kind: str):
        a = self
        b = other

        if kind == "general":
            aa = a if a.data_shape in ("general", "all") else a._coerce_to_general()
            bb = b if b.data_shape in ("general", "all") else b._coerce_to_general()
            cd = aa._tp_concat_cd_fast(bb)
            cd, eff_shape = _process_coeffs_dict_new(cd, "general")
            return self.__class__(
                coeff_dict=cd,
                data_shape=eff_shape,
                dgcvType=a.dgcvType,
                _simplifyKW=a._simplifyKW,
                variable_spaces=a._merged_variable_spaces(b),
            )

        if kind == "skew":
            if a.data_shape in ("skew", "all") and b.data_shape in ("skew", "all"):
                merged_vs = a._merged_variable_spaces(b)
                fast = a._skew_concat_canonical(b, merged_vs)
                if fast is not None:
                    cd, fmt = fast
                    return self.__class__(
                        coeff_dict=cd,
                        data_shape="skew",
                        dgcvType=a.dgcvType,
                        _simplifyKW=a._simplifyKW,
                        variable_spaces=merged_vs,
                        _inheritance={"_validated_format": fmt, "_canonical": True},
                    )
                cd = a._tp_concat_cd_fast(b, shape="skew")
                cd, eff_shape = _process_coeffs_dict_new(cd, "skew")
                return self.__class__(
                    coeff_dict=cd,
                    data_shape=eff_shape,
                    dgcvType=a.dgcvType,
                    _simplifyKW=a._simplifyKW,
                    variable_spaces=a._merged_variable_spaces(b),
                )
            ab = a._shape_product(b, kind="general")
            ba = b._shape_product(a, kind="general")
            cd = dict(ab.coeff_dict)
            for k, v in ba.coeff_dict.items():
                if not _scalar_is_zero(v):
                    cd[k] = cd.get(k, 0) - v
            cd, eff_shape = _process_coeffs_dict_new(cd, "skew")
            return self.__class__(
                coeff_dict=cd,
                data_shape=eff_shape,
                dgcvType=a.dgcvType,
                _simplifyKW=a._simplifyKW,
                variable_spaces=a._merged_variable_spaces(b),
            )

        if kind == "symmetric":
            if a.data_shape in ("symmetric", "all") and b.data_shape in (
                "symmetric",
                "all",
            ):
                cd = a._tp_concat_cd_fast(b)
                cd, eff_shape = _process_coeffs_dict_new(cd, "symmetric")
                return self.__class__(
                    coeff_dict=cd,
                    data_shape=eff_shape,
                    dgcvType=a.dgcvType,
                    _simplifyKW=a._simplifyKW,
                    variable_spaces=a._merged_variable_spaces(b),
                )

            ab = a._shape_product(b, kind="general")
            ba = b._shape_product(a, kind="general")
            cd = dict(ab.coeff_dict)
            for k, v in ba.coeff_dict.items():
                if not _scalar_is_zero(v):
                    cd[k] = cd.get(k, 0) + v
            cd, eff_shape = _process_coeffs_dict_new(cd, "symmetric")
            return self.__class__(
                coeff_dict=cd,
                data_shape=eff_shape,
                dgcvType=a.dgcvType,
                _simplifyKW=a._simplifyKW,
                variable_spaces=a._merged_variable_spaces(b),
            )

        raise ValueError(f"Unknown product kind '{kind}'.")

    def tp(self, *others):
        return self.tensor_product(*others)

    def tensor_product(self, *others):
        out = self
        for o in others:
            if check_dgcv_category(o):
                coerce = getattr(o, "as_tensor_field", None)
                if callable(coerce):
                    o = coerce()
            if not isinstance(o, self.__class__):
                return NotImplemented
            out = out._shape_product(o, kind="general")
        return out

    def skew_product(self, *others):
        out = self
        df = query_dgcv_categories(out, {"differential_form"})  # bool
        for o in others:
            if check_dgcv_scalar(o):
                out = o * out
                continue
            if not get_dgcv_category(o) == "tensor_field":
                return NotImplemented

            if not (df and query_dgcv_categories(o, {"differential_form"})):
                if check_dgcv_category(out):
                    coerce = getattr(out, "as_tensor_field", None)
                    if callable(coerce):
                        out = coerce()
                if check_dgcv_category(o):
                    coerce = getattr(o, "as_tensor_field", None)
                    if callable(coerce):
                        o = coerce()
            out = out._shape_product(o, kind="skew")
        return out

    def wedge(self, *others):
        # alias
        return self.skew_product(*others)

    def symmetric_product(self, *others):
        out = self
        for o in others:
            if not get_dgcv_category(o) == "tensor_field":
                return NotImplemented
            coerce = getattr(o, "as_tensor_field", None)
            if callable(coerce):
                o = coerce()
            coerce = getattr(self, "as_tensor_field", None)
            s = coerce() if callable(coerce) else self
            out = s._shape_product(o, kind="symmetric")
        return out
