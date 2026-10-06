"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

module: dgcv.eds._constants

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers

from ._atoms import _builtin, _nf_hooks


def _leaf_operand(other):
    return (
        isinstance(other, (numbers.Number, _BuiltinLeaf))
        and not isinstance(other, bool)
        or getattr(other, "_dgcv_category", None) in ("abstract_ZF", "zeroFormAtom")
    )


class _BuiltinLeaf:
    is_zero = False
    is_one = False
    is_minus_one = False
    is_constant = True

    @property
    def free_symbols(self):
        return set()

    def subs(self, data, **kwargs):
        if isinstance(data, (list, tuple)):
            data = dict(data)
        return data.get(self, self)

    def diff(self, *args, **kwargs):
        return 0

    def __dgcv_conjugate__(self, symbolic=False):
        return self._eval_conjugate()

    def _eval_simplify(self, **kwargs):
        return self

    def __dgcv_simplify__(self, method=None, **kwargs):
        return self

    def __dgcv_apply__(self, func, **kwargs):
        return self

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex() if raw else f"${self._latex()}$"

    def _lowered(self):
        if not getattr(self, "_builtin_constant", False):
            return None
        from .._aux._backends._engine import engine_kind

        if engine_kind() == "builtin":
            return None
        from .._aux._backends._types_and_constants import to_active_engine

        out = to_active_engine(self)
        return None if out is self else out

    def _nf_op(self, tree, compute):
        if _nf_hooks and _builtin():
            return _nf_hooks["with_nf"](tree, compute)
        return zero_form_class(tree)

    def __add__(self, other):
        if not _leaf_operand(other):
            low = self._lowered()
            return NotImplemented if low is None else low + other
        return self._nf_op(
            ("add", self, other),
            lambda: _nf_hooks["nf_of"](self) + _nf_hooks["nf_of"](other),
        )

    __radd__ = __add__

    def __sub__(self, other):
        if not _leaf_operand(other):
            low = self._lowered()
            return NotImplemented if low is None else low - other
        return self._nf_op(
            ("sub", self, other),
            lambda: _nf_hooks["nf_of"](self) - _nf_hooks["nf_of"](other),
        )

    def __rsub__(self, other):
        if not _leaf_operand(other):
            low = self._lowered()
            return NotImplemented if low is None else other - low
        return self._nf_op(
            ("sub", other, self),
            lambda: _nf_hooks["nf_of"](other) - _nf_hooks["nf_of"](self),
        )

    def __mul__(self, other):
        if not _leaf_operand(other):
            low = self._lowered()
            return NotImplemented if low is None else low * other
        return self._nf_op(
            ("mul", self, other),
            lambda: _nf_hooks["nf_of"](self) * _nf_hooks["nf_of"](other),
        )

    __rmul__ = __mul__

    def __neg__(self):
        return self._nf_op(("mul", -1, self), lambda: -_nf_hooks["nf_of"](self))

    def __truediv__(self, other):
        if not _leaf_operand(other):
            low = self._lowered()
            return NotImplemented if low is None else low / other
        return self._nf_op(
            ("div", self, other),
            lambda: _nf_hooks["nf_of"](self) / _nf_hooks["nf_of"](other),
        )

    def __rtruediv__(self, other):
        if not _leaf_operand(other):
            low = self._lowered()
            return NotImplemented if low is None else other / low
        return self._nf_op(
            ("div", other, self),
            lambda: _nf_hooks["nf_of"](other) / _nf_hooks["nf_of"](self),
        )

    def __pow__(self, other):
        if not _leaf_operand(other):
            low = self._lowered()
            return NotImplemented if low is None else low**other
        if isinstance(other, numbers.Integral) and not isinstance(other, bool):
            return self._nf_op(
                ("pow", self, other), lambda: _nf_hooks["nf_of"](self) ** int(other)
            )
        return zero_form_class(("pow", self, other))

    def __rpow__(self, other):
        if not _leaf_operand(other):
            low = self._lowered()
            return NotImplemented if low is None else other**low
        return zero_form_class(("pow", other, self))


class ImaginaryUnit(_BuiltinLeaf):
    _builtin_constant = True
    _dgcv_category = "builtin_constant"
    label = "I"

    def __eq__(self, other):
        return other is self

    def __hash__(self):
        return hash("dgcv.builtin.I")

    def __lt__(self, other):
        return NotImplemented

    def _eval_conjugate(self):
        return zero_form_class(("mul", -1, self))

    def conjugate(self):
        return self._eval_conjugate()

    def __str__(self):
        return "I"

    def __repr__(self):
        return "I"

    def _latex(self, printer=None, raw=True, **kwargs):
        return "i"

    def __reduce__(self):
        return (_imaginary_unit, ())


def _imaginary_unit():
    return I


I = ImaginaryUnit()  # noqa: E741


class NamedConstant(_BuiltinLeaf):
    _builtin_constant = True
    _dgcv_category = "builtin_constant"
    _latex_names = {"e": "e", "pi": "\\pi", "EulerGamma": "\\gamma"}

    def __init__(self, label):
        self.label = str(label)

    def __eq__(self, other):
        return isinstance(other, NamedConstant) and other.label == self.label

    def __hash__(self):
        return hash(("dgcv.builtin.constant", self.label))

    def __lt__(self, other):
        if isinstance(other, NamedConstant):
            return self.label < other.label
        return NotImplemented

    def _eval_conjugate(self):
        return self

    def conjugate(self):
        return self

    def __str__(self):
        return self.label

    def __repr__(self):
        return f"NamedConstant({self.label!r})"

    def _latex(self, printer=None, raw=True, **kwargs):
        return self._latex_names.get(self.label, f"\\mathrm{{{self.label}}}")


_named_constants = {}


def named_constant(label):
    label = str(label)
    if label not in _named_constants:
        _named_constants[label] = NamedConstant(label)
    return _named_constants[label]


E = named_constant("e")
PI = named_constant("pi")
