"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

module: dgcv.eds._zero_forms

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers
from fractions import Fraction
from math import prod

from .._aux._backends import _engine as _engine_mod
from .._aux._backends._calculus import diff as _routed_diff
from .._aux._backends._cls_coercion import register_legacy_sympy_class
from .._aux._backends._display import latex as _routed_latex
from .._aux._backends._engine import engine_kind
from .._aux._backends._symbolic_router import (
    _scalar_is_minus_one,
    _scalar_is_one,
    _scalar_is_zero,
    ratio,
    register_zero_test,
    scalars_equal,
)
from .._aux._backends._symbolic_router import as_numer_denom as _routed_as_numer_denom
from .._aux._backends._symbolic_router import (
    engine_method as _routed_engine_method,
)
from .._aux._backends._symbolic_router import simplify as _routed_simplify
from .._aux._backends._types_and_constants import (
    OP_ADD,
    OP_CONJUGATE,
    OP_CONSTANT,
    OP_EXP,
    OP_FUNCTION,
    OP_MUL,
    OP_NUMBER,
    OP_POW,
    OP_SYMBOL,
    e_constant,
    expr_numeric_types,
    expr_operands,
    expr_types,
    op_expr,
    to_active_engine,
    zero,
)
from .._aux._utilities._config import dgcv_warning
from .._aux._vmf._safeguards import retrieve_passkey
from ..core.base import dgcv_class
from . import _constants
from ._atoms import (
    _custom_conj,
    _new_bridge,
    _nf_hooks,
    _zero_obstruction,
    zero_form_atom,
)
from ._constants import ImaginaryUnit, NamedConstant, _BuiltinLeaf
from ._zf_ops import ARITY, LATEX, LOWER, TEXT, check_arity, op_spec, register_op

_builtin_leaf_types = (zero_form_atom, _BuiltinLeaf)


def _builtin():
    kind = _engine_mod._engine_kind
    if kind is None:
        kind = engine_kind()
    return kind == "builtin"


def _is_engine_leaf(x):
    return isinstance(x, expr_numeric_types()) and not isinstance(x, _builtin_leaf_types)


_SECONDARY_METHODS = frozenset(
    {
        "trigsimp",
        "apart",
        "powsimp",
        "logcombine",
        "expand_log",
        "expand_trig",
        "expand_power_exp",
        "expand_power_base",
    }
)
_BUILTIN_METHODS = {
    "simplify": "simplify",
    "cancel": "simplify",
    "ratsimp": "simplify",
    "together": "simplify",
    "expand": "expand",
    "numer": "numer",
    "denom": "denom",
}


def _builtin_unsupported(name):
    return NotImplementedError(
        f"`{name}` is not available in dgcv's builtin symbolic engine; install sympy or sage and set `default_engine` accordingly"
    )


def _bases_equal(a, b):
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_bases_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, tuple) or isinstance(b, tuple):
        return False
    return scalars_equal(a, b)


def hierarchy_rank(obj):
    th = _type_hierarchy.get(type(obj), -1)
    if th < 1:
        return (th, th)
    if th == 1:
        return (th, th)
    elif th == 2:
        return (th, obj.label)
    else:
        return (th, th)


def is_zero_check(x):
    """Helper function to check if x is zero."""
    return _scalar_is_zero(x)


def is_one_check(x):
    """Helper function to check if x is one."""
    return _scalar_is_one(x)


def _literal_zero(x):
    if isinstance(x, tuple):
        return False if _builtin() else zero_form_class(x).is_zero
    return _scalar_is_zero(x)


def _normalize_sub(op, args, base):
    if all(_is_engine_leaf(j) for j in args):
        base = args[0] - args[1]
    elif scalars_equal(args[0], args[1]):
        base = 0
    elif _literal_zero(args[0]):
        base = ("mul", -1, args[1])
    elif _literal_zero(args[1]):
        base = args[0]
    return base


def _normalize_div(op, args, base):
    if all(_is_engine_leaf(j) for j in args):
        base = ratio(args[0], args[1])
    elif scalars_equal(args[0], args[1]) and (not _literal_zero(args[0])):
        base = 1
    elif _literal_zero(args[0]):
        base = 0
    return base


def _normalize_add_mul(op, args, base):
    # Flatten nested structures (("add", ("add", x, y), z) --> ("add", x, y, z))
    flat_args = []
    for arg in args:
        if (
            isinstance(arg, zero_form_class)
            and isinstance(arg.base, tuple)
            and arg.base[0] == op
        ):
            flat_args.extend(arg.base[1:])  # Expand nested elements
        elif isinstance(arg, tuple) and arg[0] == op:
            flat_args.extend(arg[1:])  # Expand nested elements
        else:
            flat_args.append(arg)

    # Sort operands by hierarchy
    flat_args.sort(key=hierarchy_rank)

    # Combine leading numeric terms (int, float, sp.Expr) into a single term
    numeric_terms = [arg for arg in flat_args if hierarchy_rank(arg)[0] < 2]
    other_terms = [arg for arg in flat_args if hierarchy_rank(arg)[0] >= 2]

    if op == "mul" and (
        any(_scalar_is_zero(j) for j in numeric_terms)
        or any(_scalar_is_zero(j) for j in other_terms if not isinstance(j, tuple))
    ):
        base = 0
    else:
        if op == "mul":
            other_terms = [
                j for j in other_terms if isinstance(j, tuple) or not _scalar_is_one(j)
            ]

        if op == "add":
            new_other_terms = {}

            for term in other_terms:
                if isinstance(term, zero_form_class):
                    term = term.base  # Extract base representation

                # Case 1: Standalone atomic term (e.g., A_low_1_2_hi_1)
                if isinstance(term, _builtin_leaf_types):
                    new_other_terms[term] = new_other_terms.get(term, 0) + 1

                # Case 2: Multiplication structure (e.g., ('mul', 1, A))
                elif (
                    isinstance(term, tuple)
                    and term[0] == "mul"
                    and len(term) == 3
                    and _is_engine_leaf(term[1])
                ):
                    coeff, base_term = term[1], term[2]
                    new_other_terms[base_term] = (
                        new_other_terms.get(base_term, 0) + coeff
                    )

                # Case 3: Any other term, store as-is
                else:
                    new_other_terms[term] = new_other_terms.get(term, 0) + 1

            # Reconstruct terms, applying simplifications
            other_terms = []
            for key, coeff in new_other_terms.items():
                if _scalar_is_zero(coeff):
                    continue

                if _scalar_is_one(coeff):
                    other_terms.append(key)
                elif _scalar_is_minus_one(coeff):
                    other_terms.append(("mul", -1, key))
                else:
                    other_terms.append(("mul", coeff, key))

        # Combine numeric terms into a single sum/prod and insert if nonzero
        numeric_terms = [j for j in numeric_terms if j is not None]
        if numeric_terms:
            if op == "add":
                combined_numeric = sum(numeric_terms, zero())
                if not _scalar_is_zero(combined_numeric):
                    other_terms.insert(0, combined_numeric)
                elif len(other_terms) == 0:
                    other_terms = [combined_numeric]
            elif op == "mul":
                combined_numeric = prod(numeric_terms)
                if _scalar_is_zero(combined_numeric):
                    other_terms = [combined_numeric]
                elif not _scalar_is_one(combined_numeric):
                    other_terms.insert(0, combined_numeric)
                elif len(other_terms) == 0:
                    other_terms = [combined_numeric]

        # Update base
        if len(other_terms) > 1:
            base = (op, *other_terms)
        elif len(other_terms) == 0:
            base = 0
        elif hasattr(other_terms[0], "base"):
            base = other_terms[0].base
        else:
            base = other_terms[0]
    return base


def _normalize_pow(op, args, base):
    left, right = args

    if is_zero_check(right):
        if not is_zero_check(left):
            base = 1
    elif is_one_check(right):
        base = left
    else:
        if is_zero_check(left):
            base = 0
        elif is_one_check(left):
            base = 1
        elif _is_engine_leaf(left) and _is_engine_leaf(right):
            if (
                isinstance(right, numbers.Integral)
                and right < 0
                and isinstance(left, (numbers.Integral, Fraction))
            ):
                base = Fraction(left) ** right
                if base.denominator == 1:
                    base = base.numerator
            elif (
                _builtin()
                and isinstance(right, Fraction)
                and isinstance(left, (numbers.Integral, Fraction))
            ):
                pass
            else:
                base = left**right
        elif (
            isinstance(left, zero_form_class)
            and isinstance(left.base, tuple)
            and left.base[0] == "pow"
        ):
            inner_base, inner_exp = left.base[1], left.base[2]

            base = ("pow", inner_base, ("mul", inner_exp, right))
    return base


for _tag, _normalize in (
    ("add", _normalize_add_mul),
    ("mul", _normalize_add_mul),
    ("pow", _normalize_pow),
    ("sub", _normalize_sub),
    ("div", _normalize_div),
):
    register_op(_tag, ARITY[_tag], _normalize, LOWER[_tag], TEXT[_tag], LATEX[_tag])


def _top_level_ops(expr_str):
    ops = set()
    depth = 0
    prev = ""
    i = 0
    while i < len(expr_str):
        ch = expr_str[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif depth == 0:
            if ch == "+":
                ops.add("+")
            elif ch == "-":
                ops.add("-" if prev and prev not in "*/(" else "neg")
            elif ch == "*":
                if expr_str[i + 1 : i + 2] == "*":
                    ops.add("**")
                    i += 1
                else:
                    ops.add("*")
            elif ch == "/":
                ops.add("/")
        if not ch.isspace():
            prev = ch
        i += 1
    return ops


def _str_needs_parentheses(expr_str, context_op, position):
    ops = _top_level_ops(expr_str)
    if context_op == "add":
        return False
    if context_op == "sub":
        return position > 0 and bool(ops & {"+", "-", "neg"})
    if context_op == "mul":
        if ops & {"+", "-"}:
            return True
        return position > 0 and "neg" in ops and expr_str != "-1"
    if context_op == "div":
        if position == 0:
            return bool(ops & {"+", "-"})
        return bool(ops & {"+", "-", "neg", "*", "/"})
    if context_op == "pow":
        return bool(ops)
    return False


def _base_str(base, display=False):
    if not isinstance(base, tuple):
        return str(base)
    op, *args = base
    formatted_args = []
    for count, arg in enumerate(args):
        if isinstance(arg, tuple):
            arg = zero_form_class(arg)
            arg_str = _base_str(arg._base, True) if display else str(arg)
        else:
            arg_str = str(arg)
        if _str_needs_parentheses(arg_str, op, count):
            arg_str = f"({arg_str})"
        formatted_args.append(arg_str)
    return op_spec(op).text(formatted_args)


class zero_form_class(dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "abstract_ZF"
    _dgcv_categories = {"abstract_ZF", "eds_scalar"}

    def __dgcv_conjugate__(self, symbolic=False):
        return self._eval_conjugate()

    def __dgcv_apply__(self, func, **kwargs):
        if _builtin():
            if getattr(self, "_applying", False):
                raise _builtin_unsupported(getattr(func, "__name__", repr(func)))
            self._applying = True
            try:
                return func(self, **kwargs)
            finally:
                self._applying = False
        return _loop_ZF_format_conversions(
            self, reformatter=lambda expr: func(expr, **kwargs)
        )

    def _engine_method(self, name, **kwargs):
        if _builtin():
            target = _BUILTIN_METHODS.get(name)
            if target == "simplify":
                return self._builtin_simplify()
            if target == "expand":
                return self._builtin_expand()
            if target == "numer":
                return self._builtin_numer_denom()[0]
            if target == "denom":
                return self._builtin_numer_denom()[1]
            if name == "factor":
                return self._builtin_factor(bool(kwargs.get("try_hard", False)))
            if name in _SECONDARY_METHODS:
                return self._secondary_method(name, kwargs)
            raise _builtin_unsupported(name)
        kwargs.pop("try_hard", None)
        return self.__dgcv_apply__(
            lambda expr, **kw: _routed_engine_method(expr, name, **kw), **kwargs
        )

    def simplify(self, **kwargs):
        return self._eval_simplify(**kwargs)

    def factor(self, **kwargs):
        return self._engine_method("factor", **kwargs)

    def expand(self, **kwargs):
        return self._engine_method("expand", **kwargs)

    def trigsimp(self, **kwargs):
        return self._engine_method("trigsimp", **kwargs)

    def cancel(self, **kwargs):
        return self._engine_method("cancel", **kwargs)

    def together(self, **kwargs):
        return self._engine_method("together", **kwargs)

    def apart(self, **kwargs):
        return self._engine_method("apart", **kwargs)

    def ratsimp(self, **kwargs):
        return self._engine_method("ratsimp", **kwargs)

    def powsimp(self, **kwargs):
        return self._engine_method("powsimp", **kwargs)

    def logcombine(self, **kwargs):
        return self._engine_method("logcombine", **kwargs)

    def expand_log(self, **kwargs):
        return self._engine_method("expand_log", **kwargs)

    def expand_trig(self, **kwargs):
        return self._engine_method("expand_trig", **kwargs)

    def expand_power_exp(self, **kwargs):
        return self._engine_method("expand_power_exp", **kwargs)

    def expand_power_base(self, **kwargs):
        return self._engine_method("expand_power_base", **kwargs)

    """
    Symbolic expression class that represents abstract zero forms. Supports representations of combinations of many scalar-like class, such as `float`, `int`, `zeroFormAtom`, and many sympy expressions
    """

    def __init__(self, base):
        """
        Creates a new abstract_ZF instance.
        """
        if base is None or base == list() or base == tuple():
            base = 0
        if isinstance(base, list):
            base = tuple(base)
        if (
            isinstance(base, abstract_differential_form_atom) and base.degree == 0
        ):  ###!!!
            base = base.coeff
        if isinstance(base, zero_form_class):
            base = base.base
        if isinstance(base, tuple):
            op, *args = base
            new_args = []
            for arg in args:
                if (
                    isinstance(arg, abstract_differential_form_atom) and arg.degree == 0
                ):  ###!!!
                    new_args.append(arg.coeff)
                elif isinstance(arg, zero_form_class):
                    new_args.append(arg.base)
                else:
                    new_args.append(to_active_engine(arg))
            base = (op, *new_args)

        if isinstance(base, tuple):
            op, *args = base
            spec = check_arity(op, args)
            base = spec.normalize(op, args, base)
            if isinstance(base, (tuple, list)):
                base = tuple(
                    [j.base if isinstance(j, zero_form_class) else j for j in base]
                )
        elif not (
            isinstance(base, (zero_form_atom, zero_form_class, _BuiltinLeaf))
            or isinstance(base, expr_numeric_types())
        ):
            raise TypeError(
                "Base must be zeroFormAtom, int, float, sympy.Expr, abstract_ZF, or an operation tuple."
            )
        self._base = base
        self._nf_cache = None
        if isinstance(base, (zero_form_atom, zero_form_class, _BuiltinLeaf)):
            self._is_zero = base.is_zero
            self._is_one = base.is_one
        elif isinstance(base, tuple):
            self._is_zero = False
            self._is_one = False
        else:
            self._is_zero = _scalar_is_zero(base)
            self._is_one = _scalar_is_one(base)

    @property
    def base(self):
        b = self._base
        if b is None:
            fac = self._factored
            if fac is not None:
                tmp = zero_form_class(_nf_hooks["base_from_factored"](fac))
            else:
                tmp = zero_form_class(_nf_hooks["base_from_nf"](self._nf_cache))
            b = tmp._base
            self._base = b
            self._is_zero = tmp._is_zero
            self._is_one = tmp._is_one
        return b

    @base.setter
    def base(self, value):
        self._base = value

    @property
    def _nf(self):
        rf = self._nf_cache
        if rf is None:
            rf = _nf_hooks["nf_from_base"](self.base)
            self._nf_cache = rf
        return rf

    @property
    def is_zero(self):
        if self._is_zero:
            return True
        if _builtin():
            rf = self._nf_cache
            if rf is not None:
                return rf.is_zero
            if isinstance(self.base, tuple):
                return self._nf.is_zero
        return self._is_zero

    @property
    def is_one(self):
        if self._is_one:
            return True
        if _builtin():
            rf = self._nf_cache
            if rf is not None:
                return rf.is_one
            if isinstance(self.base, tuple):
                return self._nf.is_one
        return self._is_one

    @property
    def is_minus_one(self):
        if _builtin():
            rf = self._nf_cache
            if rf is not None:
                return rf.is_minus_one
            if isinstance(self.base, tuple):
                return self._nf.is_minus_one
        return _scalar_is_zero(zero_form_class(("add", self, 1)))

    @classmethod
    def _from_nf(cls, rf):
        obj = object.__new__(cls)
        obj._base = None
        obj._nf_cache = rf
        obj._is_zero = rf.is_zero
        obj._is_one = False
        return obj

    _factored = None

    @classmethod
    def _from_factored(cls, rf, factored):
        obj = object.__new__(cls)
        obj._base = None
        obj._nf_cache = rf
        obj._is_zero = rf.is_zero
        obj._is_one = False
        obj._factored = factored
        return obj

    def _builtin_simplify(self):
        return _nf_hooks["simplify"](self)

    def _secondary_method(self, name, kwargs, kind=None):
        return _nf_hooks["secondary_method"](self, name, kwargs, kind)

    def _builtin_factor(self, try_hard=False):
        return _nf_hooks["factor"](self, try_hard)

    def _builtin_expand(self):
        return _nf_hooks["expand"](self)

    def _builtin_numer_denom(self):
        return _nf_hooks["numer_denom"](self)

    def _builtin_collect(self, syms):
        return _nf_hooks["collect"](self, syms)

    def _builtin_diff(self, *args):
        return _nf_hooks["diff"](self, *args)

    def _map_atoms(self, method):
        rules = {}
        for leaf in self.free_symbols:
            if isinstance(leaf, zero_form_atom):
                image = getattr(leaf, method)()
                if image is not leaf:
                    rules[leaf] = image
        if not rules:
            return self
        return self.subs(rules)

    def to_real(self):
        return self._map_atoms("to_real")

    def to_hol(self):
        return self._map_atoms("to_hol")

    def __dgcv_re__(self):
        if not _builtin():
            from .._aux._backends._symbolic_router import re as _routed_re

            return self.__dgcv_apply__(_routed_re)
        return _nf_hooks["re"](self)

    def __dgcv_im__(self):
        if not _builtin():
            from .._aux._backends._symbolic_router import im as _routed_im

            return self.__dgcv_apply__(_routed_im)
        return _nf_hooks["im"](self)

    def diff(self, *args, **kwargs):
        if _builtin():
            return self._builtin_diff(*args)
        return self.__dgcv_apply__(lambda e: _routed_diff(e, *args, **kwargs))

    def collect(self, syms):
        if not isinstance(syms, (list, tuple, set, frozenset)):
            syms = [syms]
        if _builtin():
            return self._builtin_collect(list(syms))
        from .._aux._backends._symbolic_router import collect as _routed_collect

        return self.__dgcv_apply__(lambda e: _routed_collect(e, list(syms)))

    @property
    def tree_leaves(self):
        if not hasattr(self, "_leaves"):
            self._leaves = None
        if self._leaves is None:

            def gather_leaves(base):
                if isinstance(base, zero_form_class):
                    leaves = gather_leaves(base.base)
                elif isinstance(base, tuple):
                    leaves = set()
                    op, *args = base
                    for arg in args:
                        leaves |= gather_leaves(arg)
                else:
                    leaves = {base}
                return leaves

            self._leaves = gather_leaves(self.base)
        return self._leaves

    @property
    def free_symbols(self):
        if not hasattr(self, "_free_symbols"):
            self._free_symbols = None
        if self._free_symbols is None:
            FS = set()
            rf = self._nf_cache
            if rf is not None and self._base is None:
                FS = _nf_hooks["nf_free_symbols"](rf)
            else:
                for leaf in self.tree_leaves:
                    if hasattr(leaf, "free_symbols"):
                        FS |= leaf.free_symbols
                    elif isinstance(leaf, zero_form_atom):
                        FS |= {leaf}
            self._free_symbols = FS
        return self._free_symbols

    def _sage_(self):
        raise AttributeError

    def __hash__(self):
        return hash(self.base)

    def __eq__(self, other):
        if not isinstance(other, zero_form_class):
            if _builtin() and (
                isinstance(other, _builtin_leaf_types)
                or (isinstance(other, numbers.Number) and not isinstance(other, bool))
            ):
                return self._nf.equal(_nf_of(other))
            return NotImplemented
        if _builtin():
            return self._nf.equal(other._nf)
        return _bases_equal(self.base, other.base)

    def sort_key(self, order=None):
        return (4, self.base)  # some tedious details here... perhaps remove

    def _subs_dgcv(self, data, with_diff_corollaries=False):
        return self.subs(data, with_diff_corollaries=with_diff_corollaries)

    def subs(self, data, with_diff_corollaries=False):
        """
        Symbolic substitution in abstract_ZF.
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
        if _builtin() and not with_diff_corollaries and isinstance(data, dict):
            fast = _nf_hooks["subs"](self, data)
            if fast is not None:
                return fast
        if isinstance(self.base, numbers.Number):
            return self
        if isinstance(self.base, zero_form_atom):
            return zero_form_class(
                self.base.subs(data, with_diff_corollaries=with_diff_corollaries)
            )
        if isinstance(self.base, _BuiltinLeaf):
            return zero_form_class(self.base.subs(data))
        if isinstance(self.base, expr_types()):
            new_subs = dict()
            spare_subs = dict()
            for k, v in data.items():
                if isinstance(k, expr_types()):
                    if isinstance(v, expr_numeric_types()):
                        new_subs[k] = v
                    else:
                        spare_subs[k] = v
            new_base = self.base
            if len(new_subs) > 0:
                new_base = new_base.subs(new_subs)
            if len(spare_subs) > 0:
                new_base = _sympy_to_abstract_ZF(new_base, spare_subs)
            return zero_form_class(new_base)
        if isinstance(self.base, tuple):
            op, *args = self.base

            def sub_process(arg, sub_data):
                if isinstance(arg, tuple):
                    arg = zero_form_class(arg)
                if (
                    isinstance(arg, abstract_differential_form_atom) and arg.degree == 0
                ):  ###!!! for review later, may signal a bug...
                    arg = arg.coeff
                if isinstance(arg, _BuiltinLeaf):
                    return arg.subs(data)
                if isinstance(arg, (zero_form_atom, zero_form_class)):
                    newArg = arg.subs(
                        sub_data, with_diff_corollaries=with_diff_corollaries
                    )
                    if isinstance(newArg, abstract_differential_form_atom):
                        if newArg.degree == 0:
                            newArg = newArg.coeff
                        else:
                            raise ValueError(
                                "dgcv subs methods do not support replacing 0-forms with higher degree forms."
                            )
                    return newArg
                if isinstance(arg, expr_types()):
                    new_subs = dict()
                    spare_subs = dict()
                    for k, v in data.items():
                        if isinstance(k, expr_types()):
                            if isinstance(v, expr_numeric_types()):
                                new_subs[k] = v
                            else:
                                spare_subs[k] = v
                    if len(new_subs) > 0:
                        arg = arg.subs(new_subs)
                    if len(spare_subs) > 0:
                        arg = zero_form_class(_sympy_to_abstract_ZF(arg, spare_subs))
                    if isinstance(arg, abstract_differential_form_atom):
                        if arg.degree == 0:
                            arg = arg.coeff
                        else:
                            raise ValueError(
                                "dgcv subs methods do not support replacing 0-forms with higher degree forms."
                            )
                    return arg
                return arg

            new_base = tuple([op] + [sub_process(arg, data) for arg in args])
            return _loop_ZF_format_conversions(zero_form_class(new_base))

    def _eval_conjugate(self):
        if _builtin():
            out = _nf_hooks["conjugate"](self)
            if out is not None:
                return out

        def recursive_conjugate(expr):
            if isinstance(expr, tuple):
                op, *args = expr
                return tuple([op] + [recursive_conjugate(arg) for arg in args])
            out = _custom_conj(expr)
            return out.base if isinstance(out, zero_form_class) else out

        return zero_form_class(recursive_conjugate(self.base))

    @classmethod
    def _dgcv_multiadd_scaled(cls, pairs, start=0):
        if not isinstance(pairs, (list, tuple)):
            pairs = list(pairs)
        if _builtin():
            nf_pairs = []
            for c, t in pairs:
                if type(c) is int:
                    if c == 0:
                        continue
                    cn = _nf_of(c)
                elif type(c) is cls and c._nf_cache is not None:
                    cn = c._nf_cache
                else:
                    nf_pairs = None
                    break
                if type(t) is cls and t._nf_cache is not None:
                    tn = t._nf_cache
                elif type(t) is int:
                    tn = _nf_of(t)
                else:
                    nf_pairs = None
                    break
                nf_pairs.append((cn, tn))
            if nf_pairs is not None:
                if type(start) is cls and start._nf_cache is not None:
                    nf_pairs.append((_nf_of(1), start._nf_cache))
                elif not (type(start) is int and start == 0):
                    nf_pairs = None
            if nf_pairs is not None:
                total = _nf_hooks["msum"](nf_pairs)
                if total is not None:
                    return cls._from_nf(total)
        out = start
        for c, t in pairs:
            out = out + c * t
        return out

    @classmethod
    def _dgcv_multiadd(cls, terms, start=0):
        return cls._dgcv_multiadd_scaled([(1, t) for t in terms], start)

    def __add__(self, other):
        if not (
            isinstance(other, (zero_form_class, zero_form_atom))
            or isinstance(other, expr_numeric_types())
        ):
            return NotImplemented
        if type(other) is zero_form_class and _builtin():
            a = self._nf_cache
            b = other._nf_cache
            if a is not None and b is not None:
                if other._is_zero:
                    return self
                if self._is_zero:
                    return other
                return zero_form_class._from_nf(a + b)
        if _scalar_is_zero(other):
            return self
        if _builtin():
            return _with_nf(("add", self, other), lambda: self._nf + _nf_of(other))
        return zero_form_class(("add", self, other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, zero_form_atom):
            other = zero_form_class(other)
        if not (
            isinstance(other, (zero_form_class))
            or isinstance(other, expr_numeric_types())
        ):
            return NotImplemented
        if type(other) is zero_form_class and _builtin():
            a = self._nf_cache
            b = other._nf_cache
            if a is not None and b is not None:
                if other._is_zero:
                    return self
                return zero_form_class._from_nf(a - b)
        if _scalar_is_zero(other):
            return self
        if _builtin():
            return _with_nf(("sub", self, other), lambda: self._nf - _nf_of(other))
        return zero_form_class(("sub", self, other))

    def __rsub__(self, other):
        return -1 * (self - other)

    def __mul__(self, other):
        if type(other) is int and _builtin() and self._nf_cache is not None:
            if other == 1:
                return self
            if other == -1:
                return zero_form_class._from_nf(-self._nf_cache)
        if type(other) is zero_form_class and _builtin():
            a = self._nf_cache
            b = other._nf_cache
            if (
                a is not None
                and b is not None
                and self._base is None
                and other._base is None
            ):
                try:
                    return zero_form_class._from_nf(a * b)
                except (ZeroDivisionError, NotImplementedError):
                    pass
        if (
            _builtin()
            and not _has_pow_base(self)
            and (
                isinstance(other, _builtin_leaf_types)
                or isinstance(other, expr_numeric_types())
                or (isinstance(other, zero_form_class) and not _has_pow_base(other))
            )
        ):
            return _with_nf(("mul", self, other), lambda: self._nf * _nf_of(other))
        if isinstance(other, zero_form_class):
            # If multiplying same base, add exponents (x^a * x^b --> x^(a + b))
            if (
                isinstance(self.base, tuple)
                and self.base[0] == "pow"
                and isinstance(other.base, tuple)
                and other.base[0] == "pow"
            ):
                base1, exp1 = self.base[1], self.base[2]
                base2, exp2 = other.base[1], other.base[2]
                if base1 == base2:
                    return zero_form_class(
                        ("pow", base1, ("add", exp1, exp2))
                    )  # x^(a+b)

            if _builtin():
                return _with_nf(
                    ("mul", self.base, other.base), lambda: self._nf * other._nf
                )
            return zero_form_class(("mul", self.base, other.base))

        elif isinstance(other, _builtin_leaf_types):
            if _builtin():
                return _with_nf(
                    ("mul", self.base, other), lambda: self._nf * _nf_of(other)
                )
            return zero_form_class(("mul", self.base, other))
        elif isinstance(other, expr_numeric_types()):
            if (
                isinstance(self.base, tuple)
                and self.base[0] == "mul"
                and _is_engine_leaf(self.base[1])
            ):
                factors = tuple(
                    ["mul"]
                    + [
                        other * f if count == 0 else f
                        for count, f in enumerate(self.base[1:])
                    ]
                )
            else:
                factors = ("mul", other, self.base)
            if _builtin():
                return _with_nf(factors, lambda: self._nf * _nf_of(other))
            return zero_form_class(factors)
        elif isinstance(other, zero_form_atom):
            return zero_form_class(("mul", self.base, other))

        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return -1 * self

    def __truediv__(self, other):
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            if _builtin():
                return _with_nf(("div", self, other), lambda: self._nf / _nf_of(other))
            return zero_form_class(("div", self, other))

        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            if _builtin():
                return _with_nf(("div", other, self), lambda: _nf_of(other) / self._nf)
            return zero_form_class(("div", other, self))

        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            if (
                _builtin()
                and isinstance(other, numbers.Integral)
                and not isinstance(other, bool)
            ):
                return _with_nf(("pow", self, other), lambda: self._nf ** int(other))
            return zero_form_class(("pow", self, other))

        return NotImplemented

    def __rpow__(self, other):
        if isinstance(other, (zero_form_atom, zero_form_class)) or isinstance(
            other, expr_numeric_types()
        ):
            return zero_form_class(("pow", other, self))

        return NotImplemented

    def _eval_simplify(self, try_hard=False, **kwargs):
        if _builtin():
            out = self._builtin_simplify()
            if try_hard:
                out = out._secondary_simplify_if_transcendental()
            return out
        return _loop_ZF_format_conversions(self, withSimplify=True)

    def _secondary_simplify_if_transcendental(self):
        return _nf_hooks["secondary_simplify"](self)

    def _apply_with_sympify_loop(self, func_or_method_name, assume_method=False, **kw):
        def formatter(elem):
            """Set `func_or_method_name` to string label if `assume_method`, and function handle otherwise"""
            if assume_method:
                if isinstance(func_or_method_name, str):
                    method_name = func_or_method_name
                    if hasattr(elem, method_name):
                        return getattr(elem, method_name)(**kw)
                    else:
                        return elem
                else:
                    raise TypeError(
                        "If assume_method=True, you must pass a string method name."
                    )
            else:
                return func_or_method_name(elem)

        return _loop_ZF_format_conversions(
            self, withSimplify=False, reformatter=formatter
        )

    def numer(self):
        if _builtin():
            return self._builtin_numer_denom()[0]

        def numer_from_sympy(x):
            if isinstance(x, expr_types()):
                return _routed_as_numer_denom(x)[0]
            return x

        return self._apply_with_sympify_loop(numer_from_sympy)

    def denom(self):
        if _builtin():
            return self._builtin_numer_denom()[1]

        def denom_from_sympy(x):
            if isinstance(x, expr_types()):
                return _routed_as_numer_denom(x)[1]
            return 1

        return self._apply_with_sympify_loop(denom_from_sympy)

    def as_numer_denom(self):
        return self.numer(), self.denom()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """
        Returns a reading-friendly string representation of the expression.
        """
        if _builtin():
            shown = _nf_hooks["display_base"](self)
            if shown is not None:
                return _base_str(zero_form_class(shown)._base, True)
        return _base_str(self.base)

    def _latex(self, printer=None, raw=True, **kwargs):
        """
        Returns a LaTeX representation of the expression.
        """
        if _builtin():
            out = _nf_hooks["latex"](self)
            if out is not None:
                return out
        if isinstance(self.base, tuple):
            op, *args = self.base

            def needs_parentheses(expr, context_op, position):
                """
                Determines whether an expression needs parentheses based on its operator.
                """
                expr_str = str(expr)
                if context_op == "mul" and any(
                    j in expr_str for j in {"+", "-", "add", "sub"}
                ):
                    if position == 0 and (
                        "-" not in expr_str[1:] and "+" not in expr_str[1:]
                    ):
                        return False
                    return True  # Wrap sums inside products
                if (
                    context_op == "pow"
                    and any(
                        j in expr_str
                        for j in {"+", "-", "*", "/", "add", "sub", "mul", "div"}
                    )
                    and position == 0
                ):
                    return True  # base of expontents with sums/products/divs should be wrapped
                if (
                    context_op == "div"
                    and position == 0
                    and (
                        any(j in expr_str for j in {"+", "/", "add", "div"})
                        or "-" in expr_str[1:]
                        or "sub" in expr_str[1:]
                    )
                ):
                    return True  # base of expontents with sums/products/divs should be wrapped
                if context_op == "sub" and (
                    j in expr_str for j in {"+", "-", "add", "sub"}
                ):
                    return True
                return False

            formatted_args = []
            for count, arg in enumerate(args):
                if count == 0 and op == "mul" and arg in {1, 1.0, -1, -1.0}:
                    if arg in {1, 1.0}:
                        if len(args) == 1:
                            formatted_args.append("1")
                    elif arg in {-1, -1.0}:
                        if len(args) == 1:
                            formatted_args.append("-1")
                        else:
                            formatted_args.append("-")
                elif (
                    op == "pow"
                    and isinstance(args[1], numbers.Rational)
                    and all(
                        isinstance(j, numbers.Integral) and j > 0
                        for j in [args[1].numerator, args[1].denominator - 1]
                    )
                ):
                    op = "_handled"
                    if isinstance(arg, tuple):
                        arg_latex = f"{{{_routed_latex(zero_form_class(arg))}}}"
                    else:
                        arg_latex = (
                            f"{{{_routed_latex(arg)}}}"
                            if hasattr(arg, "_latex")
                            else _routed_latex(arg)
                        )
                    if args[1].numerator == 1:
                        if args[1].denominator == 2:
                            formatted_str = f"\\sqrt{{{arg_latex}}}"
                        else:
                            formatted_str = (
                                f"\\sqrt[{args[1].denominator}]{{{arg_latex}}}"
                            )
                    else:
                        if args[1].denominator == 2:
                            formatted_str = f"\\left(\\sqrt{{{arg_latex}}}\\right)^{{{args[1].numerator}}}"
                        else:
                            formatted_str = f"\\left(\\sqrt[{args[1].denominator}]{{{arg_latex}}}\\right)^{{{args[1].numerator}}}"
                else:
                    if isinstance(arg, tuple):
                        arg_latex = f"{{{_routed_latex(zero_form_class(arg))}}}"
                    else:
                        arg_latex = (
                            f"{{{_routed_latex(arg)}}}"
                            if hasattr(arg, "_latex")
                            else _routed_latex(arg)
                        )
                    if needs_parentheses(arg, op, count):
                        arg_latex = f"\\left({arg_latex}\\right)"
                    formatted_args.append(arg_latex)

            if op != "_handled":
                formatted_str = op_spec(op).latex(formatted_args)

            return (
                formatted_str.replace("+ {\\left(-1\\right) ", "- { ")
                .replace("+ -", "-")
                .replace("+ {-", "-{")
            )

        return _routed_latex(self.base)

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex() if raw else f"${self._latex()}$"

    def __dgcv_solve_bridge__(self, bridge):
        if _builtin():
            return [self]
        return [_lower_zf_base(self.base, bridge)]

    @property
    def __dgcv_zero_obstr__(self):
        return _zero_obstruction(self)

    def to_engine(self):
        return _new_bridge().lower(self)[0]

    def to_sympy(self, subs_rules=None):
        if subs_rules:
            dgcv_warning(
                "`abstract_ZF.to_sympy` no longer accepts `subs_rules`; the argument was ignored."
            )
        if _builtin():
            return _nf_hooks["to_sympy"](self.base)
        expr = self.to_engine()
        if engine_kind() == "sage" and hasattr(expr, "_sympy_"):
            return expr._sympy_()
        return expr

    def _canonicalize_step(self):
        if isinstance(self.base, zero_form_class):
            zf = self.base
        else:
            zf = self
        if isinstance(zf.base, zero_form_atom):
            zf, stabilized = zf.base._canonicalize_step()
            if isinstance(zf, zero_form_atom):
                return zero_form_class(zf), stabilized
            else:
                return zf, stabilized
        if isinstance(zf.base, expr_numeric_types()):
            return zf, True  # True for stabilized
        if isinstance(zf.base, tuple):
            stabilized = True  # default, may change
            op, *args = zf.base
            new_base = [op]
            for arg in args:
                if isinstance(arg, tuple):
                    arg = zero_form_class(arg)
                if hasattr(arg, "_canonicalize_step"):
                    new_arg, stab = arg._canonicalize_step()
                    stabilized = stabilized and stab
                else:
                    new_arg = arg
                if isinstance(new_arg, zero_form_class):
                    new_arg = new_arg.base
                new_base.append(new_arg)
            return zero_form_class(tuple(new_base)), stabilized
        return zf, True

    def _eval_canonicalize(self, depth=1000):
        zf = self
        stabilized = False
        count = 0
        while stabilized is False and count < depth:
            if hasattr(zf, "_canonicalize_step"):
                zf, stabilized = zf._canonicalize_step()
            count += 1
        return zf


_type_hierarchy = {
    int: 0,
    float: 0,
    Fraction: 0,
    zero_form_atom: 2,
    ImaginaryUnit: 2,
    NamedConstant: 2,
    zero_form_class: 3,
    tuple: 4,
}


_constants.zero_form_class = zero_form_class


def _has_pow_base(z):
    b = z._base
    return b is not None and isinstance(b, tuple) and b[0] == "pow"


_NUMERIC_NF = {}


def _nf_of(x):
    if isinstance(x, zero_form_class):
        return x._nf
    if isinstance(x, zero_form_atom):
        return x._nf
    if isinstance(x, (int, Fraction)) and not isinstance(x, bool):
        rf = _NUMERIC_NF.get(x)
        if rf is None:
            rf = _nf_hooks["nf_from_base"](x)
            if len(_NUMERIC_NF) < 4096:
                _NUMERIC_NF[x] = rf
        return rf
    return _nf_hooks["nf_from_base"](x)


def _with_nf(base, compute):
    try:
        rf = compute()
    except (ZeroDivisionError, NotImplementedError):
        return zero_form_class(base)
    return zero_form_class._from_nf(rf)


_nf_hooks.update(
    with_nf=_with_nf,
    nf_of=_nf_of,
    has_pow_base=_has_pow_base,
    is_leaf=lambda x: isinstance(x, _BuiltinLeaf),
)
register_zero_test(zero_form_class, zero_form_class.is_zero.fget)


def _base_free_symbols(base):
    if isinstance(base, zero_form_class):
        return set(base.free_symbols)
    if isinstance(base, tuple):
        out = set()
        for a in base[1:]:
            out |= _base_free_symbols(a)
        return out
    if isinstance(base, zero_form_atom):
        return {base}
    fs = getattr(base, "free_symbols", None)
    return set(fs) if fs else set()


def _as_leaf(x):
    if isinstance(x, zero_form_class):
        x = x.base
    if isinstance(x, _builtin_leaf_types):
        return x
    return None


def _zf_coeff_str(coeff, rendered):
    base = getattr(coeff, "base", None)
    if not (isinstance(base, tuple) and len(base) > 0 and base[0] == "add"):
        return rendered
    return f"({rendered})"


def _lower_zf_base(base, bridge):
    if isinstance(base, zero_form_class):
        return _lower_zf_base(base.base, bridge)
    if isinstance(base, expr_numeric_types()) or op_expr(base) in (
        OP_NUMBER,
        OP_CONSTANT,
    ):
        return base
    if isinstance(base, zero_form_atom) or getattr(base, "_compound", False):
        return bridge.lower(base)[0]
    if isinstance(base, tuple):
        op, *args = base
        new_args = []
        for arg in args:
            if isinstance(arg, tuple):
                arg = zero_form_class(arg)
            if isinstance(arg, zero_form_class):
                new_args.append(_lower_zf_base(arg.base, bridge))
            else:
                new_args += bridge.lower(arg)
        return op_spec(op).lower(new_args)
    raise ValueError(
        f"`_lower_zf_base` was given an unsupported expression, of type {type(base)}"
    )


def _eds_lift(bridge, expr):
    if _builtin():
        if isinstance(expr, (zero_form_class, numbers.Number)):
            return expr
        if isinstance(expr, _builtin_leaf_types):
            return zero_form_class(expr)
        return None
    if not isinstance(expr, expr_types()) or isinstance(expr, zero_form_atom):
        try:
            return expr.subs(bridge.reverse)
        except Exception:
            return zero_form_class(_engine_to_abstract_ZF(expr, bridge.reverse))
    return zero_form_class(_engine_to_abstract_ZF(expr, bridge.reverse))


def _seeded_bridge(varDict):
    bridge = _new_bridge()
    for identifier, (original, formatted) in varDict.items():
        bridge.adopt(identifier, original, formatted[0])
    return bridge


def _generate_str_id(base_str, *dicts):
    from ..core.solvers._bridge import _generate_str_id as generate

    return generate(base_str, *dicts)


def _sympify_abst_ZF(zf, varDict):
    bridge = _seeded_bridge(varDict)
    lowered = _lower_zf_base(zf, bridge)
    merged = dict(varDict)
    merged.update(
        {k: (v[0], [v[1]]) for k, v in bridge.standins.items() if k not in varDict}
    )
    return [lowered], merged


def _equation_formatting(eqn, variables_dict):
    bridge = _seeded_bridge(variables_dict)
    formatted = bridge.lower(eqn)
    new = {
        k: (v[0], [v[1]]) for k, v in bridge.standins.items() if k not in variables_dict
    }
    return formatted, new


def _engine_to_abstract_ZF(expr, subs_rules={}):
    if isinstance(expr, zero_form_class):
        return expr.subs(subs_rules).base if subs_rules else expr.base
    if isinstance(expr, _builtin_leaf_types):
        return subs_rules.get(expr, expr) if subs_rules else expr
    expr = to_active_engine(expr)
    """
    Convert a symbolic-engine expression to abstract_ZF format, applying symbol
    substitutions. Dispatches through the symbolic router, so it accepts expressions
    from any engine dgcv supports.

    Parameters:
    - expr: The engine expression to convert.
    - subs_rules (dict): Dictionary mapping engine symbols to zeroFormAtom or abstract_ZF instances.

    Returns:
    - A tuple representing the expression in abstract_ZF format.
    """
    head = op_expr(expr)

    # Base case: Replace symbols if they are in the substitution dictionary
    if head == OP_SYMBOL:
        return subs_rules.get(expr, expr)  # Replace if found, else return as-is

    # If the expr is already a number
    if head == OP_NUMBER or (
        isinstance(expr, numbers.Number) and not isinstance(expr, bool)
    ):
        return expr  # Directly return simple atomic elements

    # Handle operators that map directly to abstract_ZF:
    if head == OP_ADD:
        return (
            "add",
            *[_engine_to_abstract_ZF(arg, subs_rules) for arg in expr_operands(expr)],
        )

    if head == OP_MUL:
        return (
            "mul",
            *[_engine_to_abstract_ZF(arg, subs_rules) for arg in expr_operands(expr)],
        )

    if head == OP_POW:
        args = expr_operands(expr)
        if len(args) != 2:
            raise ValueError("Pow must have exactly 2 arguments.")
        base, exp = args
        return (
            "pow",
            _engine_to_abstract_ZF(base, subs_rules),
            _engine_to_abstract_ZF(exp, subs_rules),
        )

    # Handle conjugation
    if head == OP_CONJUGATE:
        return (
            zero_form_class(_engine_to_abstract_ZF(expr_operands(expr)[0], subs_rules))
            ._eval_conjugate()
            .base
        )

    if head == OP_EXP:
        return (
            "pow",
            e_constant(),
            _engine_to_abstract_ZF(expr_operands(expr)[0], subs_rules),
        )

    # Raise error for unsupported operations
    if head == OP_FUNCTION:
        name = getattr(getattr(expr, "func", None), "__name__", None)
        if name is None:
            name = str(getattr(expr, "operator", lambda: expr)())
        raise ValueError(
            f"Unsupported operation: {name} is not yet supported for the dgcv 0-form classes. Error for type: {type(expr)}"
        )

    if head == OP_CONSTANT:
        return expr  # named mathematical constants

    raise ValueError(
        f"Unsupported operation: {expr} cannot be mapped to abstract_ZF. Error for type: {type(expr)}"
    )


_sympy_to_abstract_ZF = _engine_to_abstract_ZF


def _loop_ZF_format_conversions(expr, withSimplify=False, reformatter=None):
    def format(elem):
        if reformatter is None or not callable(reformatter):
            return elem
        else:
            return reformatter(elem)

    if isinstance(expr, abstract_differential_form_atom) and expr.degree == 0:
        expr = expr.coeff
    if _builtin():
        zf = expr if isinstance(expr, zero_form_class) else zero_form_class(expr)
        if reformatter is not None and callable(reformatter):
            return zf.__dgcv_apply__(reformatter)
        return zf._builtin_simplify()
    bridge = _new_bridge()
    lowered = bridge.lower(expr)[0]
    lowered = _routed_simplify(format(lowered)) if withSimplify else format(lowered)
    return zero_form_class(_engine_to_abstract_ZF(lowered, bridge.reverse))


# supporting scripts for old dgcv versions:
register_legacy_sympy_class(zero_form_class)

abstract_ZF = zero_form_class
