# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

import cmath
import math
import numbers
from fractions import Fraction

from .._aux._backends._types_and_constants import to_active_engine
from .._aux._vmf._safeguards import retrieve_passkey
from .._aux.printing.printing._eds import conjugation_prefix
from .._aux.printing.printing._string_processing import (
    _process_label,
    _verbose_labels,
    latex_superscript,
)
from ..eds import _zero_forms as _zf_mod
from ..eds._atoms import zero_form_atom
from ..eds._constants import PI, E, I, NamedConstant
from ..eds._zero_forms import _nf_of, zero_form_class
from . import _latex as _latex_mod
from . import _poly
from ._factor import _integer_root
from ._latex import nf_latex
from ._normal_form import (
    _HOOKS,
    ELEMENTARY,
    FUNCTION,
    NUMBER_RADICAL,
    RADICAL,
    RF,
    LeafKey,
    _CompoundLeaf,
    _fold_fraction,
    _leaf_base,
    _leaf_keys,
    _leaf_sort_key,
    _term_key,
    base_from_nf,
    conj_rf,
    natural_key,
    nf_from_base,
    rf_gen,
    rf_number,
    rf_one,
    rf_zero,
)

_CIRCULAR = ("sin", "cos", "tan", "cot", "sec", "csc")
_HYPERBOLIC = ("sinh", "cosh", "tanh", "coth", "sech", "csch")
_INVERSE_CIRCULAR = ("asin", "acos", "atan")
_INVERSE_HYPERBOLIC = ("asinh", "acosh", "atanh")
_RECIPROCAL_ARGUMENT = {
    "acot": "atan",
    "asec": "acos",
    "acsc": "asin",
    "acoth": "atanh",
    "asech": "acosh",
    "acsch": "asinh",
}
ELEMENTARY_NAMES = (
    ("exp", "log")
    + _CIRCULAR
    + _HYPERBOLIC
    + _INVERSE_CIRCULAR
    + _INVERSE_HYPERBOLIC
    + tuple(_RECIPROCAL_ARGUMENT)
)
_ODD = frozenset(
    {
        "sin",
        "tan",
        "cot",
        "csc",
        "sinh",
        "tanh",
        "coth",
        "csch",
        "asin",
        "atan",
        "asinh",
        "atanh",
    }
)
_EVEN = frozenset({"cos", "sec", "cosh", "sech"})
_AT_ZERO = {
    "exp": 1,
    "cos": 1,
    "sec": 1,
    "cosh": 1,
    "sech": 1,
    "sin": 0,
    "tan": 0,
    "sinh": 0,
    "tanh": 0,
    "asin": 0,
    "atan": 0,
    "asinh": 0,
    "atanh": 0,
}
_FLOAT_RECIPROCAL = {
    "cot": "tan",
    "sec": "cos",
    "csc": "sin",
    "coth": "tanh",
    "sech": "cosh",
    "csch": "sinh",
}
_CIRCULAR_POLE = {
    "tan": Fraction(1, 2),
    "sec": Fraction(1, 2),
    "cot": Fraction(0),
    "csc": Fraction(0),
}
_CIRCULAR_SHIFT = {
    "sin": (("cos", 1), ("sin", -1), ("cos", -1)),
    "cos": (("sin", -1), ("cos", -1), ("sin", 1)),
    "tan": (("cot", -1), ("tan", 1), ("cot", -1)),
    "cot": (("tan", -1), ("cot", 1), ("tan", -1)),
    "sec": (("csc", -1), ("sec", -1), ("csc", 1)),
    "csc": (("sec", 1), ("csc", -1), ("sec", -1)),
}
_HYPERBOLIC_SHIFT = {
    "sinh": (("cosh", "I"), ("sinh", -1), ("cosh", "-I")),
    "cosh": (("sinh", "I"), ("cosh", -1), ("sinh", "-I")),
    "tanh": (("coth", 1), ("tanh", 1), ("coth", 1)),
    "coth": (("tanh", 1), ("coth", 1), ("tanh", 1)),
    "sech": (("csch", "-I"), ("sech", -1), ("csch", "I")),
    "csch": (("sech", "-I"), ("csch", -1), ("sech", "I")),
}
_IMAGINARY_ARGUMENT = {
    "sin": ("sinh", "I"),
    "cos": ("cosh", 1),
    "tan": ("tanh", "I"),
    "cot": ("coth", "-I"),
    "sec": ("sech", 1),
    "csc": ("csch", "-I"),
    "sinh": ("sin", "I"),
    "cosh": ("cos", 1),
    "tanh": ("tan", "I"),
    "coth": ("cot", "-I"),
    "sech": ("sec", 1),
    "csch": ("csc", "-I"),
}
_CLOSED_ROOTS_OF_UNITY = (1, 2, 3, 4, 6)
_mergeable = [False]
_FAMILY = dict.fromkeys(_CIRCULAR, "circular")
_FAMILY.update(dict.fromkeys(_HYPERBOLIC, "hyperbolic"))
_FAMILY.update({"asin": "arcsine", "acos": "arcsine"})
_HEAD_KIND = {"exp": _poly.K_EXP, "log": _poly.K_LOG}
_HEAD_KIND.update(dict.fromkeys(_CIRCULAR, _poly.K_CIRCULAR))
_HEAD_KIND.update(dict.fromkeys(_HYPERBOLIC, _poly.K_HYPERBOLIC))
_HEAD_KIND.update(dict.fromkeys(_INVERSE_CIRCULAR, _poly.K_INVERSE_CIRCULAR))
_HEAD_KIND.update(dict.fromkeys(_INVERSE_HYPERBOLIC, _poly.K_INVERSE_HYPERBOLIC))
_TEX_MACROS = frozenset(
    {"log", "sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh", "tanh", "coth"}
)
_NOT_REAL_VALUED = frozenset({"asin", "acos", "acosh", "atanh"})
_CONTINUOUS_FROM_BELOW_BEYOND_ONE = frozenset({"asin", "acos", "atanh"})
_SIN_TABLE = {
    Fraction(0): rf_zero,
    Fraction(1, 6): lambda: rf_number(Fraction(1, 2)),
    Fraction(1, 4): lambda: radical(rf_number(2), 2) * rf_number(Fraction(1, 2)),
    Fraction(1, 3): lambda: radical(rf_number(3), 2) * rf_number(Fraction(1, 2)),
    Fraction(1, 2): rf_one,
}
_TAN_TABLE = {
    Fraction(0): rf_zero,
    Fraction(1, 6): lambda: radical(rf_number(3), 2) * rf_number(Fraction(1, 3)),
    Fraction(1, 4): rf_one,
    Fraction(1, 3): lambda: radical(rf_number(3), 2),
    Fraction(2, 3): lambda: -radical(rf_number(3), 2),
    Fraction(3, 4): lambda: -rf_one(),
    Fraction(5, 6): lambda: -(radical(rf_number(3), 2) * rf_number(Fraction(1, 3))),
}


def _arg_base(x):
    x = to_active_engine(x)
    return base_from_nf(_nf_of(x))


def _arg_tex(base):
    return nf_latex(nf_from_base(base))


def _arg_str(base):
    if isinstance(base, tuple):
        return str(zero_form_class(base))
    return str(base)


_declared_markers = {}


def _declaration(markers):
    style = "partial" if "partial_style" in markers else "subscript"
    return f"real={'real' in markers}, deriv_style={style!r}"


class FunctionAtom(_CompoundLeaf):
    __hash__ = _CompoundLeaf.__hash__
    _head_kind = _poly.K_FUNCTION

    def __init__(self, label, args, derivs=None, markers=None):
        self.label_base = str(label)
        self.args = tuple(args)
        self.derivs = tuple(derivs) if derivs else (0,) * len(self.args)
        if markers is None:
            markers = _declared_markers.setdefault(self.label_base, frozenset())
        self._markers = frozenset(markers)
        self._hash = hash(
            ("dgcv.builtin.function", self.label_base, self.args, self.derivs)
        )

    def __eq__(self, other):
        return (
            isinstance(other, FunctionAtom)
            and other.label_base == self.label_base
            and other.derivs == self.derivs
            and other.args == self.args
        )

    def partials(self):
        out = []
        for k, arg in enumerate(self.args):
            bumped = list(self.derivs)
            bumped[k] += 1
            out.append(
                (
                    rf_gen(
                        FunctionAtom(self.label_base, self.args, bumped, self._markers)
                    ),
                    arg,
                )
            )
        return out

    def with_args(self, new_args):
        return rf_gen(
            FunctionAtom(
                self.label_base,
                tuple(_arg_base(a) for a in new_args),
                self.derivs,
                self._markers,
            )
        )

    def conj_rf(self):
        pref = conjugation_prefix()
        if "real" in self._markers:
            label = self.label_base
        elif self.label_base.startswith(pref):
            label = self.label_base[len(pref) :]
        else:
            label = pref + self.label_base
        args = tuple(base_from_nf(conj_rf(nf_from_base(a))) for a in self.args)
        return rf_gen(FunctionAtom(label, args, self.derivs, self._markers))

    def _head(self):
        pref = conjugation_prefix()
        lb = self.label_base
        conj = lb.startswith(pref)
        if conj:
            lb = lb[len(pref) :]
        return lb, conj

    def _index_tex(self, k):
        a = self.args[k]
        if isinstance(a, zero_form_atom):
            return a._latex()
        return str(k + 1)

    def _index_str(self, k):
        a = self.args[k]
        if isinstance(a, zero_form_atom):
            return str(a)
        return str(k + 1)

    def _head_latex(self):
        lb, conj = self._head()
        head = _process_label(lb)
        if conj:
            head = f"\\overline{{{head}}}"

        if any(self.derivs):
            if "partial_style" in self._markers:
                order = sum(self.derivs)
                dens = "".join(
                    f"\\partial {self._index_tex(k)}" * n
                    for k, n in enumerate(self.derivs)
                    if n
                )
                sup = f"^{{{order}}}" if order > 1 else ""
                head = f"\\frac{{\\partial{sup} {head}}}{{{dens}}}"
            else:
                indices = []
                for k, n in enumerate(self.derivs):
                    indices.extend([self._index_tex(k)] * n)

                body = " ".join(indices)
                if "_" in head or "^" in head:
                    head = f"{{{head}}}_{{{body}}}"
                else:
                    head = f"{head}_{{{body}}}"

        return head

    def _arguments_latex(self):
        if _verbose_labels() and self.args:
            return "\\left(" + ", ".join(_arg_tex(a) for a in self.args) + "\\right)"
        return ""

    def _latex(self, printer=None, raw=True, **kwargs):
        return self._head_latex() + self._arguments_latex()

    def _power_latex(self, e):
        return latex_superscript(self._head_latex(), str(e)) + self._arguments_latex()

    def __str__(self):
        name = self.label_base
        if any(self.derivs):
            indices = []
            for k, n in enumerate(self.derivs):
                indices.extend([self._index_str(k)] * n)
            name = f"D_{'_'.join(indices)}({name})"
        return f"{name}({', '.join(_arg_str(a) for a in self.args)})"

    def _sort_key(self):
        lb, conj = self._head()
        return LeafKey(conj, FUNCTION, 0, natural_key(lb), sum(self.derivs), str(self))


class ElementaryFunction(_CompoundLeaf):
    __hash__ = _CompoundLeaf.__hash__

    def __init__(self, name, arg):
        self.name = name
        self.args = (arg,)
        self._hash = hash(("dgcv.builtin.elementary", name, arg))

    def __eq__(self, other):
        return (
            isinstance(other, ElementaryFunction)
            and other.name == self.name
            and other.args == self.args
        )

    def _kind_bits(self):
        return _HEAD_KIND[self.name] | self._arg_rf().kinds()

    def partials(self):
        return [(_derivative(self.name, self, self._arg_rf()), self.args[0])]

    def with_args(self, new_args):
        return elementary(self.name, _nf_of(to_active_engine(new_args[0])))

    def conj_rf(self):
        return elementary(self.name, conj_rf(self._arg_rf()))

    def _tex_head(self):
        if self.name in _TEX_MACROS:
            return f"\\{self.name}"
        return f"\\operatorname{{{self.name}}}"

    def _latex(self, printer=None, raw=True, **kwargs):
        tex = _arg_tex(self.args[0])
        if self.name == "exp":
            return f"e^{{{tex}}}"
        return f"{self._tex_head()}\\left({tex}\\right)"

    def _power_latex(self, e):
        if self.name == "exp":
            return f"e^{{{nf_latex(self._arg_rf() * rf_number(e))}}}"
        tex = _arg_tex(self.args[0])
        return f"{latex_superscript(self._tex_head(), str(e))}\\left({tex}\\right)"

    def __str__(self):
        return f"{self.name}({_arg_str(self.args[0])})"

    def _sort_key(self):
        rank = ELEMENTARY_NAMES.index(self.name)
        return LeafKey(False, ELEMENTARY, rank, (), 0, _arg_str(self.args[0]))


class Radical(_CompoundLeaf):
    __hash__ = _CompoundLeaf.__hash__
    _head_kind = _poly.K_RADICAL

    def __init__(self, arg, q):
        self.args = (arg,)
        self.q = int(q)
        self._hash = hash(("dgcv.builtin.radical", self.q, arg))

    def __eq__(self, other):
        return (
            isinstance(other, Radical) and other.q == self.q and other.args == self.args
        )

    def partials(self):
        return [
            (
                rf_one() / (rf_number(self.q) * rf_gen(self) ** (self.q - 1)),
                self.args[0],
            )
        ]

    def with_args(self, new_args):
        return radical(_nf_of(to_active_engine(new_args[0])), self.q)

    def conj_rf(self):
        return radical(conj_rf(self._arg_rf()), self.q)

    def _latex(self, printer=None, raw=True, **kwargs):
        tex = _arg_tex(self.args[0])
        if self.q == 2:
            return f"\\sqrt{{{tex}}}"
        return f"\\sqrt[{self.q}]{{{tex}}}"

    def _power_latex(self, e):
        arg = self.args[0]
        frac = f"\\frac{{{e}}}{{{self.q}}}"
        if isinstance(arg, zero_form_atom):
            return latex_superscript(arg._latex(), frac)
        return f"\\left({_arg_tex(arg)}\\right)^{{{frac}}}"

    def __str__(self):
        s = _arg_str(self.args[0])
        if self.q == 2:
            return f"sqrt({s})"
        return f"({s})**(1/{self.q})"

    def _sort_key(self):
        arg = self.args[0]

        if isinstance(arg, zero_form_atom):
            return _leaf_sort_key(_poly.gen_id(arg))._replace(order=self.q)

        if isinstance(arg, numbers.Number):
            return LeafKey(False, NUMBER_RADICAL, 0, natural_key(str(arg)), self.q, "")

        return LeafKey(False, RADICAL, 0, (), self.q, _arg_str(arg))


class FunctionHead:
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "function"

    def __init__(self, label, real=False, deriv_style="subscript"):
        if deriv_style not in ("subscript", "partial"):
            raise ValueError(
                f"builtin engine: `deriv_style` is 'subscript' or 'partial', received {deriv_style!r}"
            )

        self.label = str(label)
        markers = set()
        if real:
            markers.add("real")
        if deriv_style == "partial":
            markers.add("partial_style")
        self._markers = frozenset(markers)

        declared = _declared_markers.setdefault(self.label, self._markers)
        if declared != self._markers:
            raise ValueError(
                f"builtin engine: the function `{self.label}` is already declared with {_declaration(declared)} and cannot be declared again with {_declaration(self._markers)}"
            )

        self.__name__ = self.label

    def __call__(self, *args):
        return zero_form_class(
            FunctionAtom(
                self.label, tuple(_arg_base(a) for a in args), None, self._markers
            )
        )

    def __repr__(self):
        return f"function_dgcv({self.label!r})"

    def __str__(self):
        return self.label


def _pi_multiple(rf):
    if not rf.is_polynomial or len(rf.num) != 1:
        return None
    ((m, c),) = rf.num.items()
    if len(m) != 1 or m[0][1] != 1:
        return None
    leaf = _poly.gen_leaf(m[0][0])
    if not (isinstance(leaf, NamedConstant) and leaf.label == "pi"):
        return None
    if isinstance(c, float):
        return None
    return Fraction(c) / Fraction(rf.den[()])


def _sin_of(r):
    r = r % 2
    sign = 1
    if r > 1:
        sign = -1
        r -= 1
    if r > Fraction(1, 2):
        r = 1 - r
    entry = _SIN_TABLE.get(r)
    if entry is None:
        return None
    val = entry()
    return -val if sign < 0 else val


def _trig_special(name, r):
    if name == "sin":
        return _sin_of(r)
    if name == "cos":
        return _sin_of(r + Fraction(1, 2))
    if name == "tan":
        entry = _TAN_TABLE.get(r % 1)
        return None if entry is None else entry()

    s = _sin_of(r)
    c = _sin_of(r + Fraction(1, 2))
    if s is None or c is None:
        return None
    if name == "cot":
        return c / s
    if name == "sec":
        return rf_one() / c
    return rf_one() / s


def _numeric_rf(rf):
    for g in rf.gens():
        leaf = _poly.gen_leaf(g)
        if not (isinstance(leaf, Radical) and isinstance(leaf.args[0], numbers.Number)):
            return False
    return True


def _inverse_special(name, rf):
    if not _numeric_rf(rf):
        return None
    if name == "atan":
        for r in (Fraction(1, 6), Fraction(1, 4), Fraction(1, 3)):
            if rf.equal(_TAN_TABLE[r]()):
                return rf_gen(PI) * rf_number(r)
        return None
    for r, entry in _SIN_TABLE.items():
        if r and rf.equal(entry()):
            if name == "acos":
                r = Fraction(1, 2) - r
            return rf_gen(PI) * rf_number(r)
    return None


def _pole(name, at):
    return ZeroDivisionError(f"builtin engine: {name} has a pole at {at}")


def float_value(name, v):
    try:
        if isinstance(v, complex):
            return _float_value(cmath, name, v)

        try:
            return _float_value(math, name, v)
        except ValueError:
            return _float_value(cmath, name, v)
    except (ValueError, ZeroDivisionError):
        raise _pole(name, v) from None
    except OverflowError:
        raise OverflowError(
            f"builtin engine: {name}({v}) exceeds the float range"
        ) from None


def _float_value(module, name, v):
    target = _RECIPROCAL_ARGUMENT.get(name)
    if target is not None:
        if v == 0 and name == "acot":
            return math.pi / 2
        if v == 0 and name == "acoth":
            return complex(0.0, math.pi / 2)
        return _float_value(module, target, 1.0 / v)
    base = _FLOAT_RECIPROCAL.get(name)
    if base is not None:
        return 1.0 / getattr(module, base)(v)

    if (
        module is cmath
        and name in _CONTINUOUS_FROM_BELOW_BEYOND_ONE
        and not isinstance(v, complex)
        and v > 1
    ):
        v = complex(v, -0.0)
    return getattr(module, name)(v)


def _half_pi():
    return rf_gen(PI) * rf_number(Fraction(1, 2))


def _derivative(name, leaf, a):
    if name == "exp":
        return rf_gen(leaf)
    if name == "log":
        return rf_one() / a
    if name == "sin":
        return elementary("cos", a)
    if name == "cos":
        return -elementary("sin", a)
    if name == "tan":
        return rf_one() + rf_gen(leaf) ** 2
    if name == "cot":
        return -rf_one() - rf_gen(leaf) ** 2
    if name == "sec":
        return elementary("sin", a) / elementary("cos", a) ** 2
    if name == "csc":
        return -elementary("cos", a) / elementary("sin", a) ** 2
    if name == "sinh":
        return elementary("cosh", a)
    if name == "cosh":
        return elementary("sinh", a)
    if name in ("tanh", "coth"):
        return rf_one() - rf_gen(leaf) ** 2
    if name == "sech":
        return -elementary("sinh", a) / elementary("cosh", a) ** 2
    if name == "csch":
        return -elementary("cosh", a) / elementary("sinh", a) ** 2
    if name == "asin":
        return rf_one() / radical(rf_one() - a**2, 2)
    if name == "acos":
        return -rf_one() / radical(rf_one() - a**2, 2)
    if name == "atan":
        return rf_one() / (rf_one() + a**2)
    if name == "asinh":
        return rf_one() / radical(a**2 + rf_one(), 2)
    if name == "acosh":
        return rf_one() / (radical(a - rf_one(), 2) * radical(a + rf_one(), 2))
    if name == "atanh":
        return rf_one() / (rf_one() - a**2)
    raise ValueError(
        f"builtin engine: no derivative rule for the function head `{name}`"
    )


def _compose(name, inner, u):
    if name in _CIRCULAR:
        if inner == "asin":
            s, c = u, radical(rf_one() - u**2, 2)
        elif inner == "acos":
            s, c = radical(rf_one() - u**2, 2), u
        elif inner == "atan":
            c = rf_one() / radical(rf_one() + u**2, 2)
            s = u * c
        else:
            return None

        k = _CIRCULAR.index(name)
    elif name in _HYPERBOLIC:
        if inner == "asinh":
            s, c = u, radical(u**2 + rf_one(), 2)
        elif inner == "acosh":
            s, c = radical(u - rf_one(), 2) * radical(u + rf_one(), 2), u
        elif inner == "atanh":
            c = rf_one() / radical(rf_one() - u**2, 2)
            s = u * c
        else:
            return None

        k = _HYPERBOLIC.index(name)
    else:
        return None

    if k == 0:
        return s
    if k == 1:
        return c
    if k == 2:
        return s / c
    if k == 3:
        return c / s
    if k == 4:
        return rf_one() / c
    return rf_one() / s


_COLLAPSE = {
    "tan": lambda a: elementary("sin", a) / elementary("cos", a),
    "cot": lambda a: elementary("cos", a) / elementary("sin", a),
    "sec": lambda a: rf_one() / elementary("cos", a),
    "csc": lambda a: rf_one() / elementary("sin", a),
    "tanh": lambda a: elementary("sinh", a) / elementary("cosh", a),
    "coth": lambda a: elementary("cosh", a) / elementary("sinh", a),
    "sech": lambda a: rf_one() / elementary("cosh", a),
    "csch": lambda a: rf_one() / elementary("sinh", a),
    "acos": lambda a: _half_pi() - elementary("asin", a),
}


def _single_gen(rf):
    if not rf.is_polynomial or len(rf.num) != 1 or rf.den.get(()) != 1:
        return None
    ((m, c),) = rf.num.items()
    if c != 1 or len(m) != 1 or m[0][1] != 1:
        return None
    return _poly.gen_leaf(m[0][0])


def _scaled(rf, factor):
    if factor == 1:
        return rf
    if factor == -1:
        return -rf
    if factor == "I":
        return rf * rf_gen(I)
    return -(rf * rf_gen(I))


def _imaginary_argument(name, rf):
    if not rf.num or _poly.p_has_float(rf.num) or _poly.p_has_float(rf.den):
        return None
    for m in rf.num:
        if not m or m[0][0] != _poly.I_GEN:
            return None
    target, factor = _IMAGINARY_ARGUMENT[name]
    return _scaled(elementary(target, -(rf * rf_gen(I))), factor)


def _shifted(name, rf):
    if len(rf.num) < 2 or not _poly.p_is_const(rf.den):
        return None
    pi_gid = _poly.gen_id(PI)
    if name in _CIRCULAR:
        key = ((pi_gid, 1),)
        table = _CIRCULAR_SHIFT
    else:
        key = ((_poly.I_GEN, 1), (pi_gid, 1))
        table = _HYPERBOLIC_SHIFT
    c = rf.num.get(key)
    d = rf.den[()]
    if c is None or isinstance(c, float) or isinstance(d, float):
        return None
    quarter_turns = Fraction(2 * c, d)
    if quarter_turns.denominator != 1:
        return None
    rest = dict(rf.num)
    del rest[key]
    rest = RF(rest, rf.den)
    k = quarter_turns.numerator % 4
    if k == 0:
        return elementary(name, rest)
    target, factor = table[name][k - 1]
    return _scaled(elementary(target, rest), factor)


def _exp_special(rf):
    if len(rf.num) != 1 or not _poly.p_is_const(rf.den):
        return None
    ((m, c),) = rf.num.items()
    d = rf.den[()]
    if c != 1 or type(d) is not int:
        return None
    if m == ((_poly.I_GEN, 1), (_poly.gen_id(PI), 1)):
        if d in _CLOSED_ROOTS_OF_UNITY:
            r = Fraction(1, d)
            return _trig_special("cos", r) + rf_gen(I) * _trig_special("sin", r)
        leaf = ElementaryFunction("exp", base_from_nf(rf))
        gid = _poly.gen_id(leaf)
        if gid not in _poly.REDUCTIONS:
            _poly.register_reduction(gid, d, lambda: -rf_one())
        _mergeable[0] = True
        _register_exp(gid, rf)
        return rf_gen(leaf)
    if len(m) == 1 and m[0][1] == 1:
        leaf = _poly.gen_leaf(m[0][0])
        if isinstance(leaf, ElementaryFunction) and leaf.name == "log":
            return radical(leaf._arg_rf(), d)
    return None


def _register_exp(gid, rf):
    if gid in _poly._GRADED or _poly.p_has_float(rf.num) or _poly.p_has_float(rf.den):
        return
    index = _poly.p_int_content(rf.den)
    primitive = rf * rf_number(index)

    def factory(common, primitive=primitive):
        return elementary("exp", primitive * rf_number(Fraction(1, common)))

    key = ("exp", base_from_nf(primitive))
    _poly.register_graded(gid, key, index, factory)
    if primitive.is_one:
        _poly.register_graded(_poly.gen_id(E), key, 1, factory)


def _factor_integer(n):
    if n > _poly.TRIAL_DIVISION_LIMIT:
        return None
    out = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            n //= d
            out[d] = out.get(d, 0) + 1
        d += 1 if d == 2 else 2
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def _log_rational(v):
    if v < 0:
        inner = _log_rational(-v)
        if inner is None:
            return None
        return inner + rf_gen(I) * rf_gen(PI)
    fa = _factor_integer(v.numerator)
    fb = _factor_integer(v.denominator)
    if fa is None or fb is None:
        return None
    out = rf_zero()
    for sign, factors in ((1, fa), (-1, fb)):
        for p, e in factors.items():
            out = out + rf_number(sign * e) * rf_gen(ElementaryFunction("log", p))
    return out


def _positive_exp_argument(leaf):
    if leaf == E:
        return rf_one()
    if isinstance(leaf, ElementaryFunction) and leaf.name == "exp":
        arg = leaf._arg_rf()
        if _all_real(arg):
            return arg
    return None


def _log_of_positive(leaf):
    arg = _positive_exp_argument(leaf)
    if arg is not None:
        return arg
    if isinstance(leaf, Radical):
        v = leaf.args[0]
        if isinstance(v, numbers.Rational) and v > 0:
            inner = _log_rational(Fraction(v))
            if inner is not None:
                return inner * rf_number(Fraction(1, leaf.q))
    return None


def _log_split(rf):
    if _poly.p_has_float(rf.num) or _poly.p_has_float(rf.den):
        return None
    cn = _poly.p_int_content(rf.num)
    cd = _poly.p_int_content(rf.den)
    num, den = rf.num, rf.den
    total = None
    if cn != cd:
        total = _log_rational(Fraction(cn, cd))
        if total is not None:
            num = {m: v // cn for m, v in num.items()}
            den = {m: v // cd for m, v in den.items()}
    if len(num) == 1 and len(den) == 1:
        ((mn, sn),) = num.items()
        ((md, sd),) = den.items()
        kept = []
        for sign, mono in ((1, mn), (-1, md)):
            rest = []
            for g, e in mono:
                val = _log_of_positive(_poly.gen_leaf(g))
                if val is None:
                    rest.append((g, e))
                else:
                    val = val * rf_number(sign * e)
                    total = val if total is None else total + val
            kept.append(tuple(rest))
        if total is None:
            return _log_special(mn, sn, md, sd)
        u = RF({kept[0]: sn}, canonical=True) / RF({kept[1]: sd}, canonical=True)
        return total + elementary("log", u)
    if total is None:
        return None
    return total + elementary("log", RF(num, den))


def _log_special(mn, sn, md, sd):
    if md or sd != 1 or len(mn) != 1:
        return None
    g, e = mn[0]
    if g == _poly.I_GEN and sn in (1, -1):
        return rf_gen(I) * rf_gen(PI) * rf_number(Fraction(sn, 2))
    leaf = _poly.gen_leaf(g)
    if sn == 1 and isinstance(leaf, Radical):
        return elementary("log", leaf._arg_rf()) * rf_number(Fraction(e, leaf.q))
    return None


def elementary(name, rf):
    if name == "exp" and len(rf.num) > 1:
        den = RF(rf.den, canonical=True)
        out = rf_one()
        for m, c in rf.num.items():
            out = out * elementary("exp", RF({m: c}, canonical=True) / den)
        return out

    target = _RECIPROCAL_ARGUMENT.get(name)
    if target is not None:
        if rf.is_zero:
            if name == "acot":
                return _half_pi()
            if name == "acoth":
                return rf_gen(I) * _half_pi()
            raise _pole(name, 0)

        if name == "acoth" and rf.is_constant and rf.constant_value() in (1, -1):
            raise _pole(name, rf.constant_value())
        return elementary(target, rf_one() / rf)

    if rf.is_constant:
        v = rf.constant_value()
        if isinstance(v, float):
            return rf_number(float_value(name, v))
        if v == 0:
            val = _AT_ZERO.get(name)
            if val is not None:
                return rf_number(val)
            if name == "acos":
                return _half_pi()
            if name == "acosh":
                return rf_gen(I) * _half_pi()
            raise _pole(name, 0)

        if name == "atanh" and v in (1, -1):
            raise _pole(name, v)
        if v == 1 and name in ("log", "acosh"):
            return rf_zero()
        if name == "exp" and isinstance(v, int):
            return rf_gen(E) ** v
        if name == "log":
            val = _log_rational(Fraction(v))
            if val is not None:
                return val

    leaf = _single_gen(rf)
    if leaf is not None:
        if name == "log" and leaf == E:
            return rf_one()
        if isinstance(leaf, ElementaryFunction):
            if name == "log" and leaf.name == "exp":
                arg = leaf._arg_rf()
                if _all_real(arg):
                    return arg
            elif name == "exp" and leaf.name == "log":
                return leaf._arg_rf()
            else:
                val = _compose(name, leaf.name, leaf._arg_rf())
                if val is not None:
                    return val

    if name == "log":
        val = _log_split(rf)
        if val is not None:
            return val
    elif name in _IMAGINARY_ARGUMENT:
        val = _imaginary_argument(name, rf)
        if val is None:
            val = _shifted(name, rf)
        if val is not None:
            return val

    if name in _CIRCULAR:
        r = _pi_multiple(rf)
        if r is not None:
            if r % 1 == _CIRCULAR_POLE.get(name):
                raise _pole(name, zero_form_class._from_nf(rf))
            val = _trig_special(name, r)
            if val is not None:
                return val

    if _poly.p_leading_sign(rf.num) < 0:
        if name in _ODD:
            return -elementary(name, -rf)
        if name in _EVEN:
            return elementary(name, -rf)
        if name == "exp":
            return rf_one() / elementary("exp", -rf)
        if name == "acos":
            return rf_gen(PI) - elementary("acos", -rf)

    if name == "exp":
        c = _poly.p_int_content(rf.num)
        if c > 1:
            return elementary("exp", rf * rf_number(Fraction(1, c))) ** c
        val = _exp_special(rf)
        if val is not None:
            return val
        _mergeable[0] = True
    elif name in ("asin", "acos", "atan"):
        val = _inverse_special(name, rf)
        if val is not None:
            return val

    leaf = ElementaryFunction(name, base_from_nf(rf))
    family = _FAMILY.get(name)
    if family is None:
        if name == "exp":
            _register_exp(_poly.gen_id(leaf), rf)
        return rf_gen(leaf)
    gid = _poly.gen_id(leaf)
    if gid not in _poly.GEN_GROUP:
        collapse = _COLLAPSE.get(name)
        _poly.register_family(
            gid,
            family,
            leaf.args[0],
            None if collapse is None else (lambda rf=rf, f=collapse: f(rf)),
        )
        if name == "cos":
            _poly.register_reduction(
                gid, 2, lambda rf=rf: rf_one() - elementary("sin", rf) ** 2
            )
        elif name == "cosh":
            _poly.register_reduction(
                gid, 2, lambda rf=rf: rf_one() + elementary("sinh", rf) ** 2
            )

    return RF(_poly.p_gen(gid), canonical=True)


def _radical_const(r, q):
    if r == 0:
        return rf_zero()
    if isinstance(r, float):
        return rf_number(r ** (1.0 / q))
    r = Fraction(r)
    if r < 0:
        if q == 2:
            return rf_gen(I) * _radical_const(-r, q)
        return _radical_leaf(rf_number(r), q)

    a, b = r.numerator, r.denominator
    fa = _factor_integer(a)
    fb = _factor_integer(b)
    if fa is None or fb is None:
        n = a * b ** (q - 1)
        root = _integer_root(n, q)
        if root**q == n:
            return rf_number(Fraction(root, b))
        return rf_number(Fraction(1, b)) * _radical_leaf(rf_number(n), q)

    for p, e in fb.items():
        fa[p] = fa.get(p, 0) + e * (q - 1)
    out = rf_number(Fraction(1, b))
    for p in sorted(fa):
        whole, e = divmod(fa[p], q)
        if whole:
            out = out * rf_number(p**whole)
        if e:
            g = math.gcd(e, q)
            out = out * _radical_leaf(rf_number(p), q // g) ** (e // g)

    return out


def _radical_core(core, q):
    if len(core.num) == 1 and core.den.get(()) == 1 and len(core.den) == 1:
        ((m, c),) = core.num.items()
        if c == 1:
            if len(m) == 1:
                leaf = _poly.gen_leaf(m[0][0])
                if isinstance(leaf, Radical):
                    return radical(leaf._arg_rf(), leaf.q * q) ** m[0][1]
            out = None
            rest = []
            for g, e in m:
                arg = _positive_exp_argument(_poly.gen_leaf(g))
                if arg is None:
                    rest.append((g, e))
                else:
                    val = elementary("exp", arg * rf_number(Fraction(e, q)))
                    out = val if out is None else out * val
            if out is not None:
                if rest:
                    out = out * _radical_leaf(RF({tuple(rest): 1}, canonical=True), q)
                return out
    return _radical_leaf(core, q)


def _radical_leaf(core, q):
    leaf = Radical(base_from_nf(core), q)
    gid = _poly.gen_id(leaf)
    if gid not in _poly.REDUCTIONS:
        _poly.register_reduction(gid, q, lambda core=core: core)
        _poly.register_graded(
            gid,
            ("radical", leaf.args[0]),
            q,
            lambda index, core=core: _radical_leaf(core, index),
        )
    if core.is_constant:
        _mergeable[0] = True
    return rf_gen(leaf)


def radical(rf, q):
    index = int(q)
    if index != q or index < 1:
        raise ValueError(
            f"builtin engine: a radical takes a positive integer index, received {q!r}"
        )
    q = index
    if q == 1:
        return rf

    if rf.is_zero:
        return rf_zero()
    if rf.is_constant:
        return _radical_const(rf.constant_value(), q)
    if _poly.p_has_float(rf.num) or _poly.p_has_float(rf.den):
        return rf_gen(Radical(base_from_nf(rf), q))
    if q == 2 and len(rf.num) == 1 and _poly.p_is_const(rf.den):
        c = rf.num.get(((_poly.I_GEN, 1),))
        if c is not None:
            half = _radical_const(Fraction(abs(c), 2 * rf.den[()]), 2)
            if c > 0:
                return half * (rf_one() + rf_gen(I))
            return half * (rf_one() - rf_gen(I))
    c = _poly.p_int_content(rf.num)
    if c != 1:
        core = RF({m: v // c for m, v in rf.num.items()}, rf.den, canonical=True)
    else:
        core = rf
    d = rf.den[()] if _poly.p_is_const(rf.den) else 1
    if d != 1:
        core = RF(core.num, canonical=True)
    return _radical_const(Fraction(c, d), q) * _radical_core(core, q)


def _all_real(rf):
    for g in rf.gens():
        if g == _poly.I_GEN:
            return False
        leaf = _poly.gen_leaf(g)
        if isinstance(leaf, zero_form_atom):
            if not (leaf.role in ("real", "imaginary") or "real" in leaf._markers):
                return False
        elif isinstance(leaf, ElementaryFunction):
            if leaf.name in _NOT_REAL_VALUED or not _all_real(leaf._arg_rf()):
                return False
        elif isinstance(leaf, FunctionAtom):
            if "real" not in leaf._markers or not all(
                _all_real(nf_from_base(a)) for a in leaf.args
            ):
                return False
        elif isinstance(leaf, NamedConstant):
            continue
        else:
            return False

    return True


def _is_exp_leaf(leaf):
    return leaf == E or (isinstance(leaf, ElementaryFunction) and leaf.name == "exp")


def _exp_leaf_argument(leaf):
    return rf_one() if leaf == E else leaf._arg_rf()


def _is_number_radical(leaf):
    return isinstance(leaf, Radical) and type(leaf.args[0]) is int and leaf.args[0] > 0


def merge_monomial(m, extra=None):
    if not _mergeable[0]:
        return None
    exp_count = 0 if extra is None else 2
    radicals = {}
    for g, e in m:
        leaf = _poly.gen_leaf(g)
        if _is_exp_leaf(leaf):
            exp_count += 1
        elif _is_number_radical(leaf):
            radicals.setdefault((leaf.q, e), []).append(leaf.args[0])
    merge_exp = exp_count > 1
    if not merge_exp and all(len(v) < 2 for v in radicals.values()):
        return None

    out = []
    total = extra
    slot = None
    for g, e in m:
        leaf = _poly.gen_leaf(g)
        if merge_exp and _is_exp_leaf(leaf):
            term = _exp_leaf_argument(leaf) * rf_number(e)
            total = term if total is None else total + term
            if slot is None:
                slot = len(out)
                out.append(None)
            continue
        if _is_number_radical(leaf):
            members = radicals.pop((leaf.q, e), None)
            if members is None:
                continue
            if len(members) > 1:
                out.append((Radical(math.prod(members), leaf.q), e))
                continue
            radicals[(leaf.q, e)] = None
        out.append((g, e))

    if merge_exp:
        if slot is None:
            slot = len(out)
            out.append(None)
        if total.is_zero:
            del out[slot]
        else:
            out[slot] = (ElementaryFunction("exp", base_from_nf(total)), 1)
    return out


def split_exp_denominator(rf):
    den = rf.den
    if not _mergeable[0] or len(den) != 1 or _poly.p_is_const(den):
        return den, None
    ((m, c),) = den.items()
    total = None
    for g, e in m:
        leaf = _poly.gen_leaf(g)
        if not _is_exp_leaf(leaf):
            return den, None
        term = _exp_leaf_argument(leaf) * rf_number(e)
        total = term if total is None else total + term
    return {(): c}, -total


def _display_poly(p, scale, extra, changed):
    keys = _leaf_keys(_poly.p_gens(p))
    parts = []
    for m, c in sorted(p.items(), key=lambda kv: _term_key(kv[0], keys)):
        if scale != 1:
            c = _fold_fraction(Fraction(c) * scale)
        shown = merge_monomial(m, extra)
        if shown is None:
            shown = m
        else:
            changed[0] = True
        factors = []
        for x, e in shown:
            b = _leaf_base(x) if type(x) is int else x
            factors.append(("pow", b, e) if e != 1 else b)
        if not factors:
            parts.append(c)
        elif c == 1:
            parts.append(("mul", *factors) if len(factors) > 1 else factors[0])
        else:
            parts.append(("mul", c, *factors))
    if len(parts) == 1:
        return parts[0]
    return ("add", *parts)


def display_base(zf):
    if not _mergeable[0] or zf._factored is not None:
        return None
    try:
        rf = zf._nf
    except (ZeroDivisionError, NotImplementedError):
        return None
    if not rf.num or _poly.p_has_float(rf.num) or _poly.p_has_float(rf.den):
        return None
    den, extra = split_exp_denominator(rf)
    changed = [extra is not None]
    if _poly.p_is_const(den):
        d = den[()]
        out = _display_poly(rf.num, 1 if d == 1 else Fraction(1, d), extra, changed)
    else:
        out = (
            "div",
            _display_poly(rf.num, 1, None, changed),
            _display_poly(den, 1, None, changed),
        )
    return out if changed[0] else None


def _leaf_kind(leaf):
    if isinstance(leaf, zero_form_atom):
        role = leaf.role
        if role == "antiholomorphic" or (
            role == "standard" and leaf.label.startswith(conjugation_prefix())
        ):
            return _poly.K_CONJUGATED
        return 0
    bits = getattr(leaf, "_kind_bits", None)
    return bits() if bits is not None else 0


_poly.leaf_kind = _leaf_kind
_latex_mod._merge_monomial = merge_monomial
_latex_mod._split_exp_denominator = split_exp_denominator
_zf_mod._type_hierarchy.update(
    dict.fromkeys((FunctionAtom, ElementaryFunction, Radical), 2)
)
_HOOKS["exp"] = lambda rf: elementary("exp", rf)
_HOOKS["radical"] = radical
