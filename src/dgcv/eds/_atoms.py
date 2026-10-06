"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

module: dgcv.eds._atoms

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers
from fractions import Fraction
from functools import total_ordering

from ..core.base import dgcv_class
from .._aux._deprecated.dgcv_formatter import process_basis_label
from .._aux._utilities._config import dgcv_warning
from .._aux._vmf._safeguards import retrieve_passkey
from .._aux.printing.printing._eds import conjugation_prefix, ordinal_latex
from .._aux.printing.printing._string_processing import _process_var_label
from .._aux._backends._calculus import diff as _routed_diff
from .._aux._backends._cls_coercion import register_legacy_sympy_class
from .._aux._backends import _engine as _engine_mod
from .._aux._backends._engine import engine_kind
from .._aux._backends._symbolic_router import register_zero_test
from .._aux._backends._symbolic_router import conjugate as _routed_conjugate
from .._aux._backends._types_and_constants import expr_numeric_types

_nf_hooks = {}


def _builtin():
    kind = _engine_mod._engine_kind
    if kind is None:
        kind = engine_kind()
    return kind == "builtin"
from ._roles import (
    ASSUMPTION_NAMES,
    CONJUGATE_ROLE,
    FAMILY_ROLES,
    ROLE_MARKERS,
    family_of_label,
    role_of_label,
)


@total_ordering
class SortableObj:
    def __init__(self, pair: tuple):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("Input must be a tuple of (value, sortable_key)")
        self.pair = pair
        self.value = pair[0]
        self.place = pair[1]

    def __lt__(self, other):
        if not isinstance(other, SortableObj):
            return NotImplemented
        return self.place < other.place  # Comparison is based on the sortable key

    def __eq__(self, other):
        if not isinstance(other, SortableObj):
            return NotImplemented
        return self.place == other.place  # Equality is based on the sortable key

    def __repr__(self):
        return f"SortableObj(value={self.value}, place={self.place})"


@total_ordering
class barSortedStr:
    def __init__(self, label=None):
        if label:
            if not isinstance(label, str):
                raise ValueError(f"Input must be a str. Recieved {label}")
            self.str = label
        else:
            self.str = None

    def __lt__(self, other):
        pref = conjugation_prefix()
        prefL = len(pref)
        if self.str is None:
            return True
        if other is None:
            return False
        if isinstance(other, barSortedStr):
            other = other.str
            if other is None:
                return False
        if not isinstance(other, str):
            return NotImplemented
        if self.str[0:prefL] == pref:
            if other[0:prefL] == pref:
                return self.str < other
            else:
                return False
        else:
            if other[0:prefL] == pref:
                return True
            else:
                return self.str < other

    def __eq__(self, other):
        if self.str is None:
            if other is None or (isinstance(other, barSortedStr) and other.str is None):
                return True
            else:
                return False
        if isinstance(other, barSortedStr):
            other = other.str
            if other is None:
                return False
        if not isinstance(other, str):
            return NotImplemented
        return self.str == other  # Equality is based on str label

    def __repr__(self):
        return f"barSortedStr(value={self.str})"


def _custom_conj(expr):
    if isinstance(
        expr,
        (
            zero_form_class,
            zero_form_atom,
            abstract_differential_form_atom,
            abstract_differential_form_monomial,
            abstract_differential_form,
        ),
    ):
        return expr._eval_conjugate()
    else:
        return _routed_conjugate(expr)


_EMPTY_COFRAME = None


def _empty_coframe():
    global _EMPTY_COFRAME
    if _EMPTY_COFRAME is None:
        _EMPTY_COFRAME = abstract_coframe(tuple(), {})
    return _EMPTY_COFRAME


def _imag_unit():
    from .._aux._backends._types_and_constants import imag_unit

    return imag_unit()


def _half():
    from .._aux._backends._types_and_constants import half

    return half()


def _generic_re(atom):
    return zero_form_class(("mul", _half(), ("add", atom, atom._eval_conjugate())))


def _generic_im(atom):
    return zero_form_class(
        ("mul", -1, _imag_unit(), _half(), ("sub", atom, atom._eval_conjugate()))
    )


def _new_bridge():
    from ..core.solvers._bridge import SolveBridge

    return SolveBridge()


def _zero_obstruction(obj):
    bridge = _new_bridge()
    return obj.__dgcv_solve_bridge__(bridge), bridge.symbols


class zero_form_atom(dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "zeroFormAtom"
    _dgcv_categories = {"zeroFormAtom", "eds_scalar"}

    def __init__(
        self,
        label,
        coframe_derivatives=tuple(),
        coframe=None,
        _markers=frozenset(),
        coframe_independants=dict(),
    ):
        """
        Create a new zeroFormAtom instance.

        Parameters:
        - label (str): The base function label.
        - coframe_derivatives (tuple, optional): tuple of tuples whose first entry is a coframe and whose subsequent entries are indices of coframe derivatives.
        - coframe (abstract_coframe, optional): marks the primary abstract_coframe w.r.t. the zero forms printing/display behavior may be adjusted
        """
        if not isinstance(label, str):
            raise TypeError(
                f"label must be type `str`. Instead received `{type(label)}`\n given label: {label}"
            )

        if not isinstance(coframe_derivatives, (tuple, list)):
            raise ValueError(
                f"coframe_derivatives must be a tuple of tuples whose first entry is a coframe and whose subsequent entries are indices of coframe derivatives.\n given `coframe_derivatives` of type: {type(coframe_derivatives)}"
            )
        for elem in coframe_derivatives:
            if not len(elem) == 0 and not isinstance(elem[0], abstract_coframe):
                raise ValueError(
                    f"first elements in the `coframe_derivatives` tuples must be coframe type. Given object {elem} instead with type {type(elem)}"
                )
            if not all(
                isinstance(order, numbers.Integral)
                and order in range(elem[0].dimension)
                for order in elem[1:]
            ):
                raise ValueError(
                    f"tuples in `coframe_derivatives` must begin with an abstCoframe instance followed by non-negative integers in the range of the corresponding coframe dimension. Instead of an integer tuple, recieved: {elem[1:]} \n with associated coframe basis: {elem[0]}"
                )
        coframe_derivatives = tuple(
            [tuple(elem) for elem in coframe_derivatives if len(elem[1:]) > 0]
        )
        if coframe is None:
            if len(coframe_derivatives) > 0:
                coframe = coframe_derivatives[0][0]
            else:
                coframe = _empty_coframe()
        elif len(coframe_derivatives) > 0 and coframe != coframe_derivatives[0][0]:
            coframe_derivatives = ((coframe,),) + coframe_derivatives
        if not isinstance(coframe, abstract_coframe):
            raise TypeError(
                "Expected given `coframe` to be None or have type `abstract_coframe`"
            )

        # Using SymPy's Basic constructor
        self.label = label
        self.coframe_derivatives = coframe_derivatives
        self.coframe = coframe

        self._markers = _markers
        self.coframe_independants = coframe_independants
        self.is_constant = "constant" in _markers
        self.is_one = self.label == "_1" and self.is_constant
        self._is_zero = self.label == "_0" and self.is_constant
        self._rf_cache = None
        self.secondary_coframes = [
            elem[0] for elem in self.coframe_derivatives if elem[0] != self.coframe
        ]
        self.related_coframes = (
            [self.coframe] + self.secondary_coframes
            if self.coframe is not _empty_coframe()
            else self.secondary_coframes
        )

    is_minus_one = False

    def __dgcv_solve_bridge__(self, bridge):
        if engine_kind() == "builtin":
            return [self]
        bridge.register_lifter("eds", _eds_lift)
        return [bridge.standin(self, str(self))]

    def diff(self, *args, **kwargs):
        if engine_kind() != "builtin":
            return self.__dgcv_apply__(lambda e: _routed_diff(e, *args, **kwargs))
        pairs = []
        i = 0
        while i < len(args):
            order = 1
            if i + 1 < len(args) and isinstance(args[i + 1], numbers.Integral):
                order = int(args[i + 1])
                i += 1
            var = args[i]
            if isinstance(var, zero_form_class):
                var = var.base
            pairs.append((var, order))
            i += 1
        total = sum(order for _, order in pairs)
        if total == 0:
            return self
        if total == 1:
            var = next(v for v, order in pairs if order == 1)
            return 1 if var == self else 0
        return 0

    @property
    def __dgcv_zero_obstr__(self):
        return _zero_obstruction(self)

    @property
    def is_zero(self):
        """Property to safely expose the zero check."""
        return self._is_zero

    @property
    def _nf(self):
        rf = getattr(self, "_rf_cache", None)
        if rf is None:
            rf = _nf_hooks["rf_gen"](self)
            self._rf_cache = rf
        return rf

    def _nf_operand(self, other):
        if isinstance(other, (int, Fraction)):
            return not isinstance(other, bool)
        if isinstance(other, zero_form_atom):
            return True
        if isinstance(other, zero_form_class):
            return not _nf_hooks["has_pow_base"](other)
        return _nf_hooks["is_leaf"](other)

    @property
    def differential_order(self):
        if not hasattr(self, "_differential_order"):
            self._differential_order = sum(
                [len(elem[1:]) for elem in self.coframe_derivatives]
            )
        return self._differential_order

    @property
    def role(self):
        for marker in self._markers:
            if marker in ROLE_MARKERS:
                return marker
        return role_of_label(self.label)

    @property
    def family(self):
        return family_of_label(self.label, zero_form_atom)

    @property
    def assumptions(self):
        out = {m for m in self._markers if m in ASSUMPTION_NAMES}
        if self.role in ("real", "imaginary"):
            out.add("real")
        return frozenset(out)

    @property
    def is_real(self):
        a = self.assumptions
        if "real" in a:
            return True
        if self.role in ("holomorphic", "antiholomorphic"):
            return False
        return None

    @property
    def is_positive(self):
        a = self.assumptions
        if "positive" in a:
            return True
        if a & {"negative", "nonpositive"}:
            return False
        return None

    @property
    def is_negative(self):
        a = self.assumptions
        if "negative" in a:
            return True
        if a & {"positive", "nonnegative"}:
            return False
        return None

    @property
    def is_nonzero(self):
        a = self.assumptions
        if a & {"positive", "negative", "nonzero"}:
            return True
        return None

    def _with_label(self, label, markers=None):
        return zero_form_atom(
            label,
            coframe_derivatives=self.coframe_derivatives,
            coframe=self.coframe,
            _markers=self._markers if markers is None else markers,
            coframe_independants=self.coframe_independants,
        )

    def partner(self, role):
        if role not in FAMILY_ROLES:
            raise ValueError(f"unknown complex role {role!r}")
        fam = self.family
        if fam is None:
            return None
        member = fam.partner(role)
        if member.label == self.label:
            return self
        return self._with_label(member.label, member._markers)

    def conjugate_atom(self):
        role = self.role
        target = CONJUGATE_ROLE[role]
        if role in FAMILY_ROLES and target != role:
            partner = self.partner(target)
            if partner is not None:
                return partner
        return None

    def to_real(self):
        role = self.role
        if role not in ("holomorphic", "antiholomorphic"):
            return self
        x = self.partner("real")
        y = self.partner("imaginary")
        if x is None or y is None:
            return self
        sign = 1 if role == "holomorphic" else -1
        return zero_form_class(("add", x, ("mul", sign, _imag_unit(), y)))

    def to_hol(self):
        role = self.role
        if role not in ("real", "imaginary"):
            return self
        z = self.partner("holomorphic")
        zbar = self.partner("antiholomorphic")
        if z is None or zbar is None:
            return self
        if role == "real":
            return zero_form_class(("mul", _half(), ("add", z, zbar)))
        return zero_form_class(("mul", -1, _imag_unit(), _half(), ("sub", z, zbar)))

    def __dgcv_re__(self):
        role = self.role
        if role in ("holomorphic", "antiholomorphic"):
            x = self.partner("real")
            return x if x is not None else _generic_re(self)
        if role in ("real", "imaginary") or "real" in self._markers:
            return self
        return _generic_re(self)

    def __dgcv_im__(self):
        role = self.role
        if role in ("holomorphic", "antiholomorphic"):
            y = self.partner("imaginary")
            if y is None:
                return _generic_im(self)
            return y if role == "holomorphic" else zero_form_class(("mul", -1, y))
        if role in ("real", "imaginary") or "real" in self._markers:
            return 0
        return _generic_im(self)

    def _sage_(self):
        raise AttributeError

    def __eq__(self, other):
        """
        Check equality of two zeroFormAtom instances.
        """
        if not isinstance(other, zero_form_atom):
            return NotImplemented
        return (
            self.label == other.label
            and self.coframe_derivatives == other.coframe_derivatives
        )

    def __hash__(self):
        """
        Hash the zeroFormAtom instance based on its label and coframe_derivatives.
        """
        return hash((self.label, self.coframe_derivatives))

    def __lt__(self, other):
        if not isinstance(other, zero_form_atom):
            return NotImplemented

        self_key = (
            self.label,
            len(self.coframe_derivatives),
            tuple(elem[1:] for elem in self.coframe_derivatives),
        )
        other_key = (
            other.label,
            len(other.coframe_derivatives),
            tuple(elem[1:] for elem in other.coframe_derivatives),
        )
        return self_key < other_key

    def sort_key(self, order=None):
        return (
            3,
            self.label,
            len(self.coframe_derivatives),
            tuple(elem[1:] for elem in self.coframe_derivatives),
        )

    def _eval_conjugate(self):
        """
        Define how `sympy.conjugate()` should behave for zeroFormAtom instances.
        """
        pref = conjugation_prefix()
        prefL = len(pref)
        conjugated_markers = self._markers
        role = self.role
        partner = self.conjugate_atom() if role in FAMILY_ROLES else None
        if partner is not None:
            conjugated_label = partner.label
            conjugated_markers = partner._markers
        elif role in ("real", "imaginary") or "real" in self._markers:
            conjugated_label = self.label
        elif self.label.startswith(pref):
            conjugated_label = self.label[prefL:]  # Remove conjugate prefix
        else:
            conjugated_label = f"{pref}{self.label}"  # Add conjugate prefix

        newCD = []
        for elem in self.coframe_derivatives:
            k = elem[0]
            v = elem[1:]
            newCD += [tuple([k] + [k.conj_rules[index] for index in v])]

        # Return a new zeroFormAtom with the conjugated label
        return zero_form_atom(
            conjugated_label,
            coframe_derivatives=newCD,
            coframe=self.coframe,
            _markers=conjugated_markers,
            coframe_independants=self.coframe_independants,
        )

    def _eval_simplify(self, **kws):
        return self

    def __mul__(self, other):
        """
        Multiplication of zeroFormAtom:
        - With another zeroFormAtom --> Becomes a structured `abstract_ZF` multiplication.
        - With a scalar (int/float/sympy.Expr) --> Wraps in `abstract_ZF`.
        """
        if self.is_one:
            return other
        if self.is_zero:
            return zero_form_class(0)
        if _builtin() and self._nf_operand(other):
            return _nf_hooks["with_nf"](
                ("mul", self, other), lambda: self._nf * _nf_hooks["nf_of"](other)
            )
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            return zero_form_class(("mul", self, other))
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        if _builtin():
            return _nf_hooks["with_nf"](("mul", -1, self), lambda: -self._nf)
        return zero_form_class(("mul", -1, self))

    def __truediv__(self, other):
        """
        Division with zeroFormAtom
        """
        if _builtin() and self._nf_operand(other):
            return _nf_hooks["with_nf"](
                ("div", self, other), lambda: self._nf / _nf_hooks["nf_of"](other)
            )
        if isinstance(other, zero_form_atom) or isinstance(other, expr_numeric_types()):
            return zero_form_class(("div", self, other))
        return NotImplemented

    def __rtruediv__(self, other):
        """
        Division with zeroFormAtom instances
        """
        if _builtin() and self._nf_operand(other):
            return _nf_hooks["with_nf"](
                ("div", other, self), lambda: _nf_hooks["nf_of"](other) / self._nf
            )
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            return zero_form_class(("div", other, self))

        return NotImplemented

    def __pow__(self, exp):
        """
        Exponentiation of zeroFormAtom:
        - Returns a structured `abstract_ZF` exponentiation.
        """
        if (
            _builtin()
            and isinstance(exp, numbers.Integral)
            and not isinstance(exp, bool)
        ):
            return _nf_hooks["with_nf"](
                ("pow", self, exp), lambda: self._nf ** int(exp)
            )
        if isinstance(exp, (zero_form_atom, zero_form_class)) or isinstance(
            exp, expr_numeric_types()
        ):
            return zero_form_class(("pow", self, exp))
        return NotImplemented

    def __rpow__(self, other):
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            return zero_form_class(("pow", other, self))

        return NotImplemented

    def __add__(self, other):
        """
        Addition with zeroFormAtom
        """
        if _builtin() and self._nf_operand(other):
            return _nf_hooks["with_nf"](
                ("add", self, other), lambda: self._nf + _nf_hooks["nf_of"](other)
            )
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            return zero_form_class(("add", self, other))
        return NotImplemented

    def __radd__(self, other):
        """
        Addition with zeroFormAtom
        """
        if _builtin() and self._nf_operand(other):
            return _nf_hooks["with_nf"](
                ("add", other, self), lambda: _nf_hooks["nf_of"](other) + self._nf
            )
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            return zero_form_class(("add", self, other))
        return NotImplemented

    def __sub__(self, other):
        """
        Subtraction of zeroFormAtom:
        """
        if _builtin() and self._nf_operand(other):
            return _nf_hooks["with_nf"](
                ("sub", self, other), lambda: self._nf - _nf_hooks["nf_of"](other)
            )
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            return zero_form_class(("sub", self, other))
        return NotImplemented

    def __rsub__(self, other):
        """
        Subtraction of zeroFormAtom
        """
        if _builtin() and self._nf_operand(other):
            return _nf_hooks["with_nf"](
                ("sub", other, self), lambda: _nf_hooks["nf_of"](other) - self._nf
            )
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            return zero_form_class(("sub", other, self))
        return NotImplemented

    def _canonicalize_step(self):
        for count, elem in enumerate(self.coframe_derivatives):
            zf = self
            partialsCount = len(elem)
            if partialsCount > 2:
                for idx1 in range(1, partialsCount - 1):
                    idx2 = idx1 + 1
                    if elem[idx2] < elem[idx1]:
                        return _swap_CFD_order(
                            zf, count, idx1
                        ), False  # False for stabilized status
        return self, True  # True for stabilized status

    def _eval_canonicalize(self, depth=1000):
        zf = self
        stabilized = False
        count = 0
        while stabilized is False and count < depth:
            if hasattr(zf, "_canonicalize_step"):
                zf, stabilized = zf._canonicalize_step()
            count += 1
        return zf

    @property
    def free_symbols(self):
        return {self}

    def is_primitive(self, other, returnCD=False):
        "compute if other element is a coframe derivative (of some order) of self"
        if isinstance(other, zero_form_class) and isinstance(other.base, zero_form_atom):
            other = other.base
        if isinstance(other, zero_form_atom):
            if self.label == other.label:
                trip = False
                CDlen = len(self.coframe_derivatives)
                if CDlen <= len(other.coframe_derivatives):
                    if CDlen == 0:
                        if returnCD:
                            return True, other.coframe_derivatives
                        return True
                    selfTail = self.coframe_derivatives[-1]
                    tailCDlen = len(selfTail)
                    compareCD = list(other.coframe_derivatives[: CDlen - 1])
                    otherCompareTail = other.coframe_derivatives[CDlen - 1][:tailCDlen]
                    if (
                        self.coframe_derivatives[:-1] == compareCD
                        and selfTail == otherCompareTail
                    ):
                        trip = True
                        trailingCD = [
                            [other.coframe_derivatives[CDlen - 1][0]]
                            + list(other.coframe_derivatives[CDlen - 1][tailCDlen:])
                        ]
                        trailingCD = tuple(
                            trailingCD + list(other.coframe_derivatives[CDlen:])
                        )
                    else:
                        trailingCD = tuple()
                    if returnCD:
                        return trip, trailingCD
                    else:
                        return trip, trailingCD
        if returnCD:
            return False, tuple()
        else:
            return False

    def is_diff_corollary(self, other, returnCD=False):
        if isinstance(other, zero_form_class) and isinstance(other.base, zero_form_atom):
            return other.base.is_primitive(self, returnCD=returnCD)
        if isinstance(other, zero_form_atom):
            return other.is_primitive(self, returnCD=returnCD)
        return False

    def as_coeff_Mul(self, **kwds):
        return 1, self

    def as_ordered_factors(self):
        return (self,)

    def _subs_dgcv(self, data, with_diff_corollaries=False):
        # an alias for regular subs so that other functions can know the with_diff_corollaries keyword is available
        return self.subs(data, with_diff_corollaries=with_diff_corollaries)

    def subs(self, data, with_diff_corollaries=False):
        """
        Symbolic substitution in zeroFormAtom.
        """
        if isinstance(data, (list, tuple)) and all(
            isinstance(j, tuple) and len(j) == 2 for j in data
        ):
            l1 = len(data)
            data = dict(data)
            if len(data) < l1:
                dgcv_warning(
                    "Provided substitution rules had repeat keys, and only one was used."
                )

        if isinstance(data, dict):
            if with_diff_corollaries:
                for key in data.keys():
                    truthVal, coefDer = self.is_diff_corollary(key, returnCD=True)
                    if truthVal:
                        new_value = data[key]
                        if isinstance(
                            new_value, (zero_form_atom, zero_form_class)
                        ) or isinstance(new_value, expr_numeric_types()):
                            for cd in coefDer:
                                new_value = coframe_derivative(new_value, *cd)
                            return new_value
                        else:
                            raise TypeError(
                                f"subs() cannot replace a `zeroFormAtom` with object type {type(new_value)}."
                            )
            if self in data:
                new_value = data[self]
                if isinstance(new_value, (zero_form_atom, zero_form_class)) or isinstance(
                    new_value, expr_numeric_types()
                ):
                    return new_value
                else:
                    raise TypeError(
                        f"subs() cannot replace a `zeroFormAtom` with object type {type(new_value)}."
                    )
            else:
                return self
        else:
            raise TypeError("`zeroFormAtom.subs()` received unsupported subs data.")

    def __dgcv_conjugate__(self, symbolic=False):
        return self._eval_conjugate()

    def __dgcv_apply__(self, func, **kwargs):
        bridge = _new_bridge()
        result = bridge.lift_expr(func(bridge.lower(self)[0], **kwargs))
        if isinstance(result, zero_form_class) and result.base is self:
            return self
        return result

    def simplify(self, **kw):
        return self

    def factor(self, **kw):
        return self

    def expand(self, **kw):
        return self

    def trigsimp(self, **kw):
        return self

    def cancel(self, **kw):
        return self

    def together(self, **kw):
        return self

    def apart(self, **kw):
        return self

    def ratsimp(self, **kw):
        return self

    def powsimp(self, **kw):
        return self

    def logcombine(self, **kw):
        return self

    def expand_log(self, **kw):
        return self

    def expand_trig(self, **kw):
        return self

    def expand_power_exp(self, **kw):
        return self

    def expand_power_base(self, **kw):
        return self

    def numer(self, **kw):
        return self

    def denom(self, **kw):
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """
        Fallback string representation.
        """
        if self.is_one:
            return "1"
        if self.is_zero:
            return "0"

        if len(self.coframe_derivatives) > 0 and (
            len(self.coframe_derivatives[0]) > 1 or len(self.coframe_derivatives) > 1
        ):
            return_str = self.label
            count = 0
            partials_str = "_".join(
                map(str, [j + 1 for j in self.coframe_derivatives[0][1:]])
            )
            return_str = f"D_{partials_str}({return_str})"
            if len(self.coframe_derivatives[0]) > 1:
                count = 1
            for elem in self.coframe_derivatives[1:]:
                v = elem[1:]
                count_str = "" if count == 0 else f"_{count}"
                partials_str = "_".join(map(str, v))
                return_str = f"D_{partials_str}({return_str}){count_str}"
                count += 1
            return return_str

        return self.label

    def _latex(self, printer=None, raw=True, **kwargs):
        """
        LaTeX representation for zeroFormAtom.
        """
        if self.is_one:
            return "1"
        if self.is_zero:
            return "0"

        base_label = self.label
        conjugated = False
        pref = conjugation_prefix()
        prefL = len(pref)
        if base_label.startswith(pref):
            base_label = base_label[prefL:]
            conjugated = True
        if (
            "engine_symbol" in self._markers
            and not self.coframe_derivatives
            and _builtin()
        ):
            return _process_var_label(self.label)

        index_start = None
        if "_low_" in base_label:
            index_start = base_label.index("_low_")
        elif "_hi_" in base_label:
            index_start = base_label.index("_hi_")

        if index_start is not None:
            first_part = base_label[:index_start]
            index_part = base_label[index_start:]
        else:
            first_part = base_label
            index_part = ""

        # Process the base part
        formatted_label = process_basis_label(first_part)

        if len(self.coframe_derivatives) > 0 and (
            len(self.coframe_derivatives[0]) > 1 or len(self.coframe_derivatives) > 1
        ):
            partials = True
        else:
            partials = False
        if "_" in formatted_label and index_part is not None and partials:
            formatted_label = f"\\left({formatted_label}\\right)"

        # Extract lower and upper indices
        lower_list, upper_list = [], []
        if "_low_" in index_part:
            lower_part = index_part.split("_low_")[1]
            if "_hi_" in lower_part:
                lower_part, upper_part = lower_part.split("_hi_")
                upper_list = upper_part.split("_")
            lower_list = lower_part.split("_")
        elif "_hi_" in index_part:
            upper_part = index_part.split("_hi_")[1]
            upper_list = upper_part.split("_")

        # Conjugate index formatter
        def cIdx(idx, cf):
            idx = int(idx)
            if isinstance(cf, abstract_coframe) and idx - 1 in cf.inverted_conj_rules:
                return f"\\overline{{{1 + cf.inverted_conj_rules[idx - 1]}}}"
            else:
                return f"{idx}"

        # Convert string indices to LaTeX-compatible integers
        lower_list = [cIdx(idx, self.coframe) for idx in lower_list if idx]
        upper_list = [cIdx(idx, self.coframe) for idx in upper_list if idx]

        # Extract partial derivative indices
        partials_strs = []
        if partials and len(self.coframe_derivatives[0]) > 1:
            new_indices = [j + 1 for j in self.coframe_derivatives[0][1:]]
            new_indices_str = ",".join([cIdx(j, self.coframe) for j in new_indices])
            partials_strs.extend([new_indices_str])
        elif self.coframe is not None:
            partials_strs = [""]
        for elem in self.coframe_derivatives[1:]:
            new_indices = [j + 1 for j in elem[1:]]
            new_indices_str = ",".join([cIdx(j, elem[0]) for j in new_indices])
            partials_strs.extend([new_indices_str])

        # Combine indices into the LaTeX string
        lower_str = ",".join(lower_list)
        # partials_strs = [",".join(map(cIdx, j)) for j in partials_indices]
        first_partials_str = partials_strs[0] if len(partials_strs) > 0 else ""
        upper_str = ",".join(upper_list)

        indices_str = ""
        indices_str_partials = ""  # only update if conjugated==True
        if upper_str:
            indices_str += f"^{{{upper_str}}}"
            if first_partials_str and conjugated:
                indices_str_partials += f"^{{\\vphantom{{{upper_str}}}}}"
        if lower_str or "verbose" in self._markers:
            if conjugated:
                indices_str += (
                    f"_{{{lower_str}\\vphantom{{;{first_partials_str}}}}}".replace(
                        r"\vphantom{;}}", "}"
                    )
                )
                if first_partials_str:
                    indices_str_partials += (
                        f"_{{\\vphantom{{{lower_str}}};{first_partials_str}}}"
                    )
            else:
                indices_str += f"_{{{lower_str};{first_partials_str}}}".replace(
                    ";}", "}"
                ).replace(r"\vphantom{;}}", "}")
        elif first_partials_str:
            if conjugated:
                if upper_str:
                    indices_str_partials += f"_{{;{first_partials_str}}}"
                else:
                    indices_str_partials += f"_{{{first_partials_str}}}"
            else:
                indices_str += f"_{{{first_partials_str}}}"
        pre_final_str = f"{formatted_label}{indices_str}"
        if indices_str_partials != "":  # implies conjugated
            pre_final_str = f"\\smash{{\\overline{{{pre_final_str}}}}}\\vphantom{{{formatted_label}}}{indices_str_partials}"
        elif conjugated:
            pre_final_str = f"\\overline{{{pre_final_str}}}"

        final_str = pre_final_str

        count = 2
        for new_partials_str in partials_strs[1:]:
            final_str = f"\\left.\\smash{{{final_str}}}\\vphantom{{{pre_final_str}}}\\right|_{{{new_partials_str}}}^{{\\boxed{{\\tiny{ordinal_latex(count)}}}}}"
            count += 1
        return final_str

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex() if raw else f"${self._latex()}$"


register_legacy_sympy_class(zero_form_atom)

register_zero_test(zero_form_atom, zero_form_atom.is_zero.fget)

zeroFormAtom = zero_form_atom
