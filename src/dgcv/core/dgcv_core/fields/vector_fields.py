from __future__ import annotations

from math import prod
from numbers import Integral
from typing import Optional

from ...._aux._backends._calculus import diff
from ...._aux._backends._display import latex as _backend_latex
from ...._aux._backends._symbolic_router import _scalar_is_zero, simplify, subs
from ...._aux._backends._types_and_constants import (
    check_dgcv_scalar,
    imag_unit,
    rational,
    verify_conjugate_re_im_free,
)
from ...._aux._utilities._config import dgcv_warning, get_dgcv_settings_registry
from ...._aux._vmf._safeguards import (
    check_dgcv_category,
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_passkey,
)
from ...._aux._vmf.vmf import vmf_lookup
from ...base import dgcv_class
from ...conversions.conversions import allToHol, allToReal, allToSym
from .general import tensor_field_class


class vector_field_class(tensor_field_class):
    _dgcv_categories = {"vector_field"}

    def __init__(
        self,
        varSpace=None,
        coeffs=None,
        *,
        coeff_dict=None,
        valence=None,
        data_shape: str = "all",
        dgcvType: str = "standard",
        _simplifyKW=None,
        variable_spaces=None,
        parameters=set(),
        _inheritance=None,
    ):
        self._vf_cache = None
        if _simplifyKW is None:
            _simplifyKW = {
                "simplify_rule": None,
                "simplify_ignore_list": None,
                "preferred_basis_element": None,
            }
        self.parameters = parameters
        if coeff_dict is not None:
            if varSpace is not None or coeffs is not None:
                raise TypeError(
                    "Provide either `coeff_dict=...` or (`varSpace`, `coeffs`), not both."
                )

            super().__init__(
                coeff_dict=coeff_dict,
                data_shape=data_shape,
                dgcvType=dgcvType,
                _simplifyKW=_simplifyKW,
                variable_spaces=variable_spaces,
                _inheritance=_inheritance,
            )

            if self.valence != (1,) and self.valence != tuple():
                raise ValueError(
                    f"vector_field expects valence=(1,), not {self.valence}"
                )

            self._vf_cache = None
            return

        if varSpace is None or coeffs is None:
            raise TypeError(
                "Provide either `coeff_dict=...` or (`varSpace`, `coeffs`)."
            )

        vs = tuple(varSpace)
        cs = list(coeffs)

        if len(vs) != len(cs):
            raise ValueError("`varSpace` and `coeffs` must have the same length.")
        if len(vs) != len(set(vs)):
            raise TypeError("`varSpace` must not contain repeated variables.")

        syslbl = None
        if len(vs) > 0:
            if isinstance(variable_spaces, dict) and variable_spaces:
                if len(variable_spaces) == 1:
                    syslbl = next(iter(variable_spaces.keys()))
                else:
                    raise ValueError(
                        "vector_field legacy init requires a single system in `variable_spaces`."
                    )
            else:
                info0 = vmf_lookup(vs[0], path=True, relatives=False)
                p0 = info0.get("path")
                if not (isinstance(p0, tuple) and len(p0) >= 2):
                    raise KeyError(
                        "vector_field legacy init requires variables registered in the VMF or `variable_spaces={...}`."
                    )
                syslbl = p0[1]

        if syslbl is None:
            syslbl = "__anon__"

        cd = {(i, 1, syslbl): c for i, c in enumerate(cs) if not _scalar_is_zero(c)}
        if not cd:
            cd = {tuple(): 0}

        if variable_spaces is None:
            variable_spaces = {syslbl: vs}
        else:
            variable_spaces = dict(variable_spaces)
            variable_spaces.setdefault(syslbl, vs)
        super().__init__(
            coeff_dict=cd,
            data_shape=data_shape,
            dgcvType=dgcvType,
            _simplifyKW=_simplifyKW,
            variable_spaces=variable_spaces,
            _inheritance=_inheritance,
        )

    def _vf_view(self):
        cache = getattr(self, "_vf_cache", None)
        if cache is not None:
            return cache
        if self._is_scalar_coeff_dict(self.coeff_dict):
            self._vf_cache = (tuple(), [])
            return self._vf_cache

        syslbl = None
        n = 0
        items = []

        for k, v in self.coeff_dict.items():
            if _scalar_is_zero(v) or not isinstance(k, tuple) or len(k) != 3:
                continue
            i, vb, s = k
            if vb != 1:
                raise ValueError(
                    "vector_field expects contravariant bit 1 in coeff_dict keys."
                )
            if syslbl is None:
                syslbl = s
            elif syslbl != s:
                raise ValueError(
                    "vector_field view requires a single system label in coeff_dict."
                )
            if isinstance(i, Integral):
                ii = int(i)
                if ii >= n:
                    n = ii + 1
                items.append((ii, v))
            else:
                raise TypeError(
                    "vector_field expects integer indices in coeff_dict keys."
                )

        if syslbl is None:
            self._vf_cache = (tuple(), [])
            return self._vf_cache

        vs = self._variable_spaces.get(syslbl)
        if not isinstance(vs, tuple):
            raise KeyError(f"vector_field missing `variable_spaces[{syslbl!r}]`.")

        coeffs = [0] * max(n, len(vs))
        for ii, v in items:
            if ii >= len(coeffs):
                coeffs.extend([0] * (ii + 1 - len(coeffs)))
            coeffs[ii] = coeffs[ii] + v

        self._vf_cache = (vs, coeffs)
        return self._vf_cache

    def as_tensor_field(self, data_shape: Optional[str] = None) -> tensor_field_class:
        vs = getattr(self, "_variable_spaces", None)
        if not isinstance(vs, dict):
            vs = None
        data_shape = data_shape if data_shape else self.data_shape
        return tensor_field_class(
            coeff_dict=self.coeff_dict,
            data_shape=data_shape,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=vs,
            parameters=self.parameters,
        )

    @property
    def coeffs(self):
        if self._coeffs is None:
            self._coeffs = list(self.coeff_dict.values())
        return self._coeffs

    def simplify_format(self, format_type=None, skipVar=None):
        if format_type not in {None, "holomorphic", "real", "symbolic_conjugate"}:
            dgcv_warning(
                "simplify_format() received an unsupported first argument. Try None, 'holomorphic', 'real', or 'symbolic_conjugate'."
            )
        return self.__class__(
            coeff_dict=self.coeff_dict,
            dgcvType=self.dgcvType,
            _simplifyKW={"simplify_rule": format_type, "simplify_ignore_list": skipVar},
            variable_spaces=self._variable_spaces,
        )

    def _eval_simplify(self, **kwargs):
        rule = self._simplifyKW.get("simplify_rule", None)
        ign = self._simplifyKW.get("simplify_ignore_list", None)

        if rule is None:

            def f(c):
                return simplify(c, **kwargs)
        elif rule == "holomorphic":

            def f(c):
                return simplify(allToHol(c, skipVar=ign), **kwargs)
        elif rule == "real":

            def f(c):
                return simplify(allToReal(c, skipVar=ign), **kwargs)
        elif rule == "symbolic_conjugate":

            def f(c):
                return simplify(allToSym(c, skipVar=ign), **kwargs)
        else:

            def f(c):
                return simplify(c, **kwargs)

        cd = {}
        for k, v in self.coeff_dict.items():
            if _scalar_is_zero(v):
                continue
            nv = f(v)
            if not _scalar_is_zero(nv):
                cd[k] = cd.get(k, 0) + nv

        if not cd:
            cd = {tuple(): 0}

        return self.__class__(
            coeff_dict=cd,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces,
            data_shape=self.data_shape,
        )

    def subs(self, subsData):
        cd = {}
        for k, v in self.coeff_dict.items():
            if _scalar_is_zero(v):
                continue
            nv = subs(v, subsData)
            if not _scalar_is_zero(nv):
                cd[k] = cd.get(k, 0) + nv

        if not cd:
            cd = {tuple(): 0}

        return self.__class__(
            coeff_dict=cd,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=self._variable_spaces,
            data_shape=self.data_shape,
        )

    def __neg__(self):
        cd = {k: -v for k, v in self.coeff_dict.items() if not _scalar_is_zero(v)}
        if not cd:
            cd = {tuple(): 0}
        return self.__class__(
            coeff_dict=cd,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=getattr(self, "_variable_spaces", None),
        )

    def _as_tensor_maybe(self, obj):
        if isinstance(obj, tensor_field_class):
            return obj
        if check_dgcv_category(obj):
            coerce = getattr(obj, "as_tensor_field", None)
            if callable(coerce):
                return coerce()
        return None

    def __add__(self, other):
        if _scalar_is_zero(other):
            return self

        if isinstance(other, self.__class__):
            vs = self._merged_variable_spaces(other)
            new_cd = dict(self.coeff_dict)
            for k, v in other.coeff_dict.items():
                if not _scalar_is_zero(v):
                    new_cd[k] = new_cd.get(k, 0) + v
            new_cd = {k: v for k, v in new_cd.items() if not _scalar_is_zero(v)} or {
                tuple(): 0
            }
            return self.__class__(
                coeff_dict=new_cd,
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=vs,
                data_shape=self.data_shape,
            )

        if check_dgcv_scalar(other):
            return self.as_tensor_field().__add__(other)

        tf = self._as_tensor_maybe(other)
        if tf is not None:
            return self.as_tensor_field().__add__(tf)

        return NotImplemented

    def __sub__(self, other):
        if _scalar_is_zero(other):
            return self

        if isinstance(other, self.__class__):
            vs = self._merged_variable_spaces(other)
            new_cd = dict(self.coeff_dict)
            for k, v in other.coeff_dict.items():
                if not _scalar_is_zero(v):
                    new_cd[k] = new_cd.get(k, 0) - v
            new_cd = {k: v for k, v in new_cd.items() if not _scalar_is_zero(v)} or {
                tuple(): 0
            }
            return self.__class__(
                coeff_dict=new_cd,
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=vs,
                data_shape=self.data_shape,
            )

        if check_dgcv_scalar(other):
            return self.as_tensor_field().__sub__(other)

        tf = self._as_tensor_maybe(other)
        if tf is not None:
            return self.as_tensor_field().__sub__(tf)

        return NotImplemented

    def __radd__(self, other):
        if _scalar_is_zero(other):
            return self
        return self.__add__(other)

    def __rsub__(self, other):
        if check_dgcv_scalar(other):
            return self.as_tensor_field().__rsub__(other)

        tf = self._as_tensor_maybe(other)
        if tf is not None:
            return tf.__sub__(self.as_tensor_field())

        return NotImplemented

    def __mul__(self, other):
        if check_dgcv_scalar(other):
            return super().__mul__(other)

        tf = other
        if check_dgcv_category(tf):
            coerce = getattr(tf, "as_tensor_field", None)
            if callable(coerce):
                tf = coerce()

        if not (get_dgcv_category(tf) == "tensor_field"):
            return NotImplemented

        return self.as_tensor_field(data_shape="skew")._shape_product(
            tf,
            kind="skew",
        )

    def __matmul__(self, other):
        tf = other
        if check_dgcv_category(tf):
            coerce = getattr(tf, "as_tensor_field", None)
            if callable(coerce):
                tf = coerce()

        if not (get_dgcv_category(tf) == "tensor_field"):
            return NotImplemented

        return self.as_tensor_field(data_shape="all")._shape_product(
            tf,
            kind="general",
        )

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __pow__(self, other):
        return self._to_diff_op_class().__pow__(other)

    def _to_diff_op_class(self):
        new_dict = dict()
        for k, v in self.coeff_dict.items():
            new_key = []
            degree = len(k) // 3
            for idx in range(degree):
                var_idx, syss = k[idx], k[2 * degree + idx]
                new_key.append(self._variable_spaces[syss][var_idx])
            new_key = tuple(sorted(new_key, key=lambda x: str(x)))
            new_dict[new_key] = new_dict.get(new_key, 0) + v
        return differential_operator(new_dict, validated=True)

    def __call__(self, *args, ignore_complex_handling=None):
        if len(args) != 1:
            raise ValueError("vector_field expects exactly one argument.")
        other = args[0]

        if get_dgcv_category(other) == "array":
            return other.apply(self.__call__)

        if get_dgcv_category(other) == "tensor_field":
            if self._validated_format == "complex":
                if other._validated_format in {"mixed", "real"}:
                    return self.__call__(other._to_complex())
            elif self._validated_format == "real":
                if other._validated_format in {"mixed", "complex"}:
                    return self.__call__(other._to_real())
            elif self._validated_format == "mixed":
                if other._validated_format == "real":
                    return self.__call__(other._to_real())
                if other._validated_format == "complex":
                    return self.__call__(other._to_complex())
                if other._validated_format == "mixed":
                    pref = get_dgcv_settings_registry()["preferred_variable_format"]
                    if pref == "real":
                        return self._to_real().__call__(other._to_real())
                    else:
                        return self._to_complex().__call__(other._to_complex())
            if other._is_scalar():
                key, val = next(iter(other.coeff_dict.items()))
                return self(val)
            return super().__call__(other)

        if (
            get_dgcv_category(other) == "differential_form"
            and getattr(other, "degree", None) == 0
        ):
            c0 = getattr(other, "coeffsInKFormBasis", None)
            if isinstance(c0, (list, tuple)) and c0:
                other = c0[0]

        diff_local = diff
        fmt = self._coordinate_format_info()

        if ignore_complex_handling or fmt.get("dgcv_type") == "standard":
            out = 0
            for k, c in self.coeff_dict.items():
                if _scalar_is_zero(c):
                    continue
                d = len(k) // 3
                if d != 1:
                    continue
                idx = k[0]
                sys = k[2]
                vs = self._variable_spaces.get(sys)
                if not isinstance(vs, tuple | dict):
                    continue
                v = vs[idx]
                out += c * diff_local(other, v)
            return out

        half = rational(1, 2)
        imu = imag_unit()

        has_conj = not verify_conjugate_re_im_free(other)
        a = allToSym(other) if has_conj else other

        out = 0

        for k, c in self.coeff_dict.items():
            if _scalar_is_zero(c):
                continue

            d = len(k) // 3
            if d != 1:
                continue

            idx = k[0]
            sys = k[2]

            vs = self._variable_spaces.get(sys)
            if not isinstance(vs, tuple):
                continue

            v = vs[idx]

            info = vmf_lookup(v, flattened_relatives=True)
            st = info.get("sub_type")
            rel = info.get("flattened_relatives")

            if st is None or rel is None or not isinstance(rel, tuple) or len(rel) != 4:
                out += c * diff_local(a, v)
                continue

            z, zb, x, y = rel

            if has_conj:
                if st == "holo":
                    out += c * diff_local(a, z)
                elif st == "anti":
                    out += c * diff_local(a, zb)
                elif st == "real":
                    out += c * (diff_local(a, z) + diff_local(a, zb))
                elif st == "imag":
                    out += c * imu * (diff_local(a, z) - diff_local(a, zb))
                else:
                    out += c * diff_local(a, v)
                continue

            if st == "holo":
                out += c * (
                    diff_local(a, z)
                    + half * (diff_local(a, x) - imu * diff_local(a, y))
                )
            elif st == "anti":
                out += c * (
                    diff_local(a, zb)
                    + half * (diff_local(a, x) + imu * diff_local(a, y))
                )
            elif st == "real":
                out += c * (diff_local(a, x) + diff_local(a, z) + diff_local(a, zb))
            elif st == "imag":
                out += c * (
                    diff_local(a, y) + imu * (diff_local(a, z) - diff_local(a, zb))
                )
            else:
                out += c * diff_local(a, v)

        return out

    def tensor_product(self, *others, coerce_shapes: bool = False):
        return self.as_tensor_field().tensor_product(*others)


class differential_operator(dgcv_class):
    def __init__(self, operator_dictionary, validated=False):
        """
        PRELIMINARY CLASS UNDER DEVELOPMENT
        operator_dictionary should contain coordinate tuples as keys, mapped to a coefficient value
        """
        if validated is False:
            od = dict()
            for c, coeff in operator_dictionary.items():
                if coeff:
                    key = (
                        tuple(sorted(c, key=lambda x: str(x)))
                        if isinstance(c, (tuple, list))
                        else (c,)
                    )
                    od[key] = od.get(key, 0) + coeff
        else:
            od = operator_dictionary
        self._od = od
        self._vfs = dict()
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "differential_operator"

        super().__init__()

    def _coordinate_vf(self, atom):
        if not isinstance(self._vfs, dict):
            self._vfs = dict()
        if atom not in self._vfs:
            from ...vector_fields_and_differential_forms import (
                coordinate_vector_field,
            )

            vf = coordinate_vector_field(atom)
            if vf:
                self._vfs[atom] = vf
            else:
                raise TypeError(
                    "unsupported coordinate in differntial_operator parametr"
                )
        return self._vfs[atom]

    def _repr_latex_(self, raw=False, **kwargs):

        from ...._aux.printing.printing._spaces import _tf2_latex_coeff

        formatted_terms = []
        for k, v in self._od.items():
            coeff = _tf2_latex_coeff(v)
            order = len(k)
            if order == 0:
                basis_elem = ""
            elif order == 1:
                basis_elem = f"\\frac{{\\partial}}{{\\partial {_backend_latex(k[0])}}}"
            else:
                varprod = prod(k)
                basis_elem = f"\\frac{{\\partial^{{{order}}}}}{{\\partial {_backend_latex(varprod)}}}"

            if coeff == "":
                formatted_terms.append(basis_elem)
            elif coeff == "-":
                formatted_terms.append(rf"- {basis_elem}")
            else:
                formatted_terms.append(rf"{coeff} {basis_elem}")
        latex_str = (
            r"\operatorname{Id}"
            if not formatted_terms
            else " + ".join(formatted_terms).replace("+ -", "- ")
        )
        return latex_str if raw else f"$\\displaystyle {latex_str}$"

    def __mul__(self, other):
        if check_dgcv_scalar(other):
            if other == 0:
                return differential_operator(dict())
            return differential_operator(
                {k: other * v for k, v in self._od.items()},
                validated=True,
            )
        if query_dgcv_categories(other, {"vector_field"}):
            other = other._to_diff_op_class()
        if get_dgcv_category(other) == "differential_operator":
            new_dict = dict()
            for k1, v1 in self._od.items():
                for k2, v2 in other._od.items():
                    new_key = tuple(sorted(k1 + k2, key=lambda x: str(x)))
                    if new_key in new_dict:
                        new_val = new_dict.get(new_key, 0) + v1 * v2
                        if new_val == 0:
                            _ = new_dict.pop(new_val, None)
                        else:
                            new_dict[new_key] = new_val
                    else:
                        new_dict[new_key] = v1 * v2
            return differential_operator(new_dict, validated=True)
        return NotImplemented

    def __rmul__(self, other):
        if check_dgcv_scalar(other):
            if other == 0:
                return differential_operator(dict())
            return differential_operator(
                {k: other * v for k, v in self._od.items()},
                validated=True,
            )
        return NotImplemented

    def __add__(self, other):
        if other == 0:
            return self
        if check_dgcv_scalar(other):
            other = differential_operator({tuple(): other}, validated=True)
        elif query_dgcv_categories(other, {"vector_field"}):
            other = other._to_diff_op_class()
        if get_dgcv_category(other) == "differential_operator":
            new_dict = dict(self._od)
            for k, v in other._od.items():
                new_val = new_dict.get(k, 0) + v
                if not _scalar_is_zero(new_val):
                    new_dict[k] = new_val
            return differential_operator(new_dict, validated=True)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __pow__(self, other):
        out = differential_operator({tuple(): 1}, validated=True)
        for _ in range(other):
            out *= self
        return out

    def __call__(self, *args, **kwds):
        if len(args) == 0:
            return
        if len(args) > 1:
            return [self.__call__(arg) for arg in args]
        f = args[0]
        out = 0
        for c, coeff in self._od.items():
            vfs = [self._coordinate_vf(var) for var in c]
            derivative = f
            for vf in vfs:
                derivative = vf(derivative)
            out += coeff * derivative
        return out
