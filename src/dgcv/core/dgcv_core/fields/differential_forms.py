from __future__ import annotations

from typing import Optional

from ...._aux._backends._symbolic_router import _scalar_is_zero, simplify, subs
from ...._aux._backends._types_and_constants import (
    check_dgcv_scalar,
)
from ...._aux._utilities._config import dgcv_warning, get_dgcv_settings_registry
from ...._aux._vmf._safeguards import (
    check_dgcv_category,
    get_dgcv_category,
    query_dgcv_categories,
)
from ...._aux._vmf.vmf import vmf_lookup
from ...conversions.conversions import (
    allToHol,
    allToReal,
    allToSym,
)
from .general import tensor_field_class
from .general.workers import _process_coeffs_dict_new


def _slot_coordinate(spaces, idx, sys):
    space = spaces.get(sys) if spaces else None
    if space is None:
        return idx if sys is None else None
    try:
        return space[idx]
    except (KeyError, IndexError, TypeError):
        return None


def _same_coordinate(x, y):
    if x is None or y is None:
        return False
    return x is y or str(x) == str(y)


class differential_form_class(tensor_field_class):
    _dgcv_categories = {"differential_form"}

    def __init__(
        self,
        varSpace=None,
        data_dict=None,
        degree=None,
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
        if _simplifyKW is None:
            _simplifyKW = {
                "simplify_rule": None,
                "simplify_ignore_list": None,
                "preferred_basis_element": None,
            }

        self.parameters = parameters
        if coeff_dict is not None:
            if varSpace is not None or data_dict is not None or degree is not None:
                raise TypeError(
                    "Provide either `coeff_dict=...` or (`varSpace`, `data_dict`, `degree`), not both."
                )

            if data_shape == "all":
                max_deg = 0
                all_covariant = True
                for k, c in coeff_dict.items():
                    if _scalar_is_zero(c):
                        continue
                    d = len(k) // 3
                    if d > max_deg:
                        max_deg = d
                    valence_tuple = k[d : 2 * d]
                    for vb in valence_tuple:
                        if vb != 0:
                            all_covariant = False
                            break
                    if not all_covariant:
                        break
                if all_covariant and max_deg > 1:
                    data_shape = "skew"

            super().__init__(
                coeff_dict=coeff_dict,
                data_shape=data_shape,
                dgcvType=dgcvType,
                _simplifyKW=_simplifyKW,
                variable_spaces=variable_spaces,
                _inheritance=_inheritance,
            )
            return

        if varSpace is None or data_dict is None or degree is None:
            raise TypeError(
                "Provide either `coeff_dict=...` or (`varSpace`, `data_dict`, `degree`)."
            )

        vs = tuple(varSpace)
        if len(vs) != len(set(vs)):
            raise ValueError("`varSpace` must not have duplicate entries.")

        syslbl = None
        if len(vs) > 0:
            if isinstance(variable_spaces, dict) and variable_spaces:
                if len(variable_spaces) == 1:
                    syslbl = next(iter(variable_spaces.keys()))
                else:
                    raise ValueError(
                        "differential_form legacy init requires a single system in `variable_spaces`."
                    )
            else:
                info0 = vmf_lookup(vs[0], path=True, relatives=False)
                p0 = info0.get("path")
                if not (isinstance(p0, tuple) and len(p0) >= 2):
                    raise KeyError(
                        "differential_form legacy init requires variables registered in the VMF or `variable_spaces={...}`."
                    )
                syslbl = p0[1]

        if syslbl is None:
            syslbl = "__anon__"

        if variable_spaces is None:
            variable_spaces = {syslbl: vs}
        else:
            variable_spaces = dict(variable_spaces)
            variable_spaces.setdefault(syslbl, vs)

        vtuple = variable_spaces.get(syslbl)
        idx_map = {v: i for i, v in enumerate(vtuple)}

        deg = int(degree)
        if data_shape == "all" and deg > 1:
            data_shape = "skew"

        zeros = (0,) * deg
        syslbls = (syslbl,) * deg

        cd = {}
        for k, c in data_dict.items():
            if _scalar_is_zero(c):
                continue

            if deg == 0:
                if k == tuple() or k == 0:
                    cd[tuple()] = cd.get(tuple(), 0) + c
                    continue
                raise TypeError("degree=0 forms require scalar key tuple().")

            idxs = []
            for jj in k:
                var = vs[int(jj)]
                idxs.append(idx_map[var])

            nk = tuple(idxs) + zeros + syslbls
            cd[nk] = cd.get(nk, 0) + c

        if not cd:
            cd = {tuple(): 0}

        super().__init__(
            coeff_dict=cd,
            data_shape=data_shape,
            dgcvType=dgcvType,
            _simplifyKW=_simplifyKW,
            variable_spaces=variable_spaces,
            _inheritance=_inheritance,
        )

    def simplify_format(self, format_type=None, skipVar=None):
        if format_type not in {None, "holomorphic", "real", "symbolic_conjugate"}:
            dgcv_warning(
                "simplify_format() received an unsupported first argument. Try None, 'holomorphic', 'real', or 'symbolic_conjugate'."
            )
        return self.__class__(
            coeff_dict=self.coeff_dict,
            dgcvType=self.dgcvType,
            _simplifyKW={"simplify_rule": format_type, "simplify_ignore_list": skipVar},
            variable_spaces=getattr(self, "_variable_spaces", None),
        )

    def _eval_simplify(self, **kwargs):
        rule = self._simplifyKW.get("simplify_rule", None)
        ign = self._simplifyKW.get("simplify_ignore_list", None)

        if self._is_scalar():
            return self.__class__(
                coeff_dict={tuple(): simplify(self._scalar_value(), **kwargs)},
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=getattr(self, "_variable_spaces", None),
            )

        if rule is None:
            new_cd = {k: simplify(v, **kwargs) for k, v in self.coeff_dict.items()}
        elif rule == "holomorphic":
            new_cd = {
                k: simplify(allToHol(v, skipVar=ign), **kwargs)
                for k, v in self.coeff_dict.items()
            }
        elif rule == "real":
            new_cd = {
                k: simplify(allToReal(v, skipVar=ign), **kwargs)
                for k, v in self.coeff_dict.items()
            }
        elif rule == "symbolic_conjugate":
            new_cd = {
                k: simplify(allToSym(v, skipVar=ign), **kwargs)
                for k, v in self.coeff_dict.items()
            }
        else:
            dgcv_warning(f"Unsupported simplify_rule: {rule}.")
            new_cd = {k: simplify(v, **kwargs) for k, v in self.coeff_dict.items()}

        new_cd, _ = _process_coeffs_dict_new(new_cd, "all")
        return self.__class__(
            coeff_dict=new_cd,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=getattr(self, "_variable_spaces", None),
        )

    def subs(self, subsData):
        new_cd = {k: subs(v, subsData) for k, v in self.coeff_dict.items()}
        new_cd, _ = _process_coeffs_dict_new(new_cd, "all")
        return self.__class__(
            coeff_dict=new_cd,
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=getattr(self, "_variable_spaces", None),
        )

    def as_tensor_field(self, data_shape: Optional[str] = None) -> tensor_field_class:
        vs = getattr(self, "_variable_spaces", None)
        if not isinstance(vs, dict):
            vs = None
        return tensor_field_class(
            coeff_dict=self.coeff_dict,
            data_shape=(data_shape if data_shape is not None else self.data_shape),
            dgcvType=self.dgcvType,
            _simplifyKW=self._simplifyKW,
            variable_spaces=vs,
            parameters=self.parameters,
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

        if check_dgcv_scalar(other):
            new_cd = dict(self.coeff_dict)
            new_cd[tuple()] = new_cd.get(tuple(), 0) + other
            new_cd = {k: v for k, v in new_cd.items() if not _scalar_is_zero(v)} or {
                tuple(): 0
            }
            return self.__class__(
                coeff_dict=new_cd,
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=getattr(self, "_variable_spaces", None),
                data_shape=self.data_shape,
            )

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

        tf = self._as_tensor_maybe(other)
        if tf is not None:
            return self.as_tensor_field().__add__(tf)

        return NotImplemented

    def __sub__(self, other):
        if _scalar_is_zero(other):
            return self

        if check_dgcv_scalar(other):
            new_cd = dict(self.coeff_dict)
            new_cd[tuple()] = new_cd.get(tuple(), 0) - other
            new_cd = {k: v for k, v in new_cd.items() if not _scalar_is_zero(v)} or {
                tuple(): 0
            }
            return self.__class__(
                coeff_dict=new_cd,
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=getattr(self, "_variable_spaces", None),
                data_shape=self.data_shape,
            )

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
            new_cd = {
                k: (-v) for k, v in self.coeff_dict.items() if not _scalar_is_zero(v)
            }
            new_cd[tuple()] = new_cd.get(tuple(), 0) + other
            new_cd = {k: v for k, v in new_cd.items() if not _scalar_is_zero(v)} or {
                tuple(): 0
            }
            return self.__class__(
                coeff_dict=new_cd,
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=getattr(self, "_variable_spaces", None),
                data_shape=self.data_shape,
            )

        tf = self._as_tensor_maybe(other)
        if tf is not None:
            return tf.__sub__(self.as_tensor_field())

        return NotImplemented

    def __mul__(self, other):
        if check_dgcv_scalar(other):
            cd = {
                k: other * v
                for k, v in self.coeff_dict.items()
                if not _scalar_is_zero(v)
            }
            if not cd:
                cd = {tuple(): 0}
            return self.__class__(
                coeff_dict=cd,
                data_shape=self.data_shape,
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=self._variable_spaces,
                parameters=getattr(self, "parameters", set()),
            )

        tf = self._as_tensor_maybe(other)
        if tf is None:
            return NotImplemented

        if query_dgcv_categories(tf, {"differential_form"}):
            cd = self._tp_concat_cd_fast(tf, shape="skew")
            return self.__class__(
                coeff_dict=cd,
                data_shape="skew",
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=self._merged_variable_spaces(tf),
                parameters=getattr(self, "parameters", set()),
            )

        if get_dgcv_category(tf) == "tensor_field":
            if tf.data_shape in {"skew", "all"}:
                return self.as_tensor_field().wedge(tf)
            return NotImplemented

        return NotImplemented

    def __matmul__(self, other):
        tf = self._as_tensor_maybe(other)
        if tf is None:
            return NotImplemented
        return self.as_tensor_field()._shape_product(
            tf.as_tensor_field() if hasattr(tf, "as_tensor_field") else tf,
            kind="general",
        )

    def tensor_product(self, *others, coerce_shapes: bool = False):
        return self.as_tensor_field().tensor_product(*others)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __call__(self, *args, **kwargs):
        if args and all(query_dgcv_categories(a, {"vector_field"}) for a in args):
            if len(args) > 1:
                contract = self(args[0])
                if query_dgcv_categories(contract, "differential_form"):
                    return contract(*args[1:])
                else:
                    return 0

            other = args[0]
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

            shape_cd = {}

            for k2, v2 in other.coeff_dict.items():
                if _scalar_is_zero(v2):
                    continue

                a, b, c = k2
                other_spaces = getattr(other, "_variable_spaces", None)
                for k1, v1 in self.coeff_dict.items():
                    if _scalar_is_zero(v1):
                        continue
                    sign = 1
                    deg = len(k1) // 3
                    for idx in range(deg):
                        idx2, idx3 = deg + idx, 2 * deg + idx
                        if b + k1[idx2] == 1 and (
                            (a == k1[idx] and c == k1[idx3])
                            or (
                                (c is None or k1[idx3] is None)
                                and _same_coordinate(
                                    _slot_coordinate(self._variable_spaces, k1[idx], k1[idx3]),
                                    _slot_coordinate(other_spaces, a, c),
                                )
                            )
                        ):
                            new_key = tuple(
                                elem
                                for count, elem in enumerate(k1)
                                if count not in (idx, idx2, idx3)
                            )
                            shape_cd[new_key] = (
                                shape_cd.get(new_key, 0) + sign * v1 * v2
                            )
                            break
                        sign *= -1

            if not shape_cd:
                return 0
            if tuple() in shape_cd and len(shape_cd) == 1:
                return shape_cd[tuple()]

            return differential_form_class(
                coeff_dict=shape_cd,
                data_shape=self.data_shape,
                dgcvType=self.dgcvType,
                _simplifyKW=self._simplifyKW,
                variable_spaces=self._merged_variable_spaces(other),
                parameters=self.parameters | getattr(other, "parameters", set()),
            )

        return super().__call__(*args, **kwargs)
