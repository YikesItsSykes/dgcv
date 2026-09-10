"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

module: dgcv.eds.eds


Description: Core EDS definitions and functions

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers
import random
import string
from collections import Counter
from functools import total_ordering
from math import prod

from .._aux._backends._cls_coercion import register_legacy_sympy_class
from .._aux._backends._display import latex as _routed_latex
from .._aux._backends._engine import _get_sympy_module, sympy_module_if_available
from .._aux._backends._symbolic_router import (
    _scalar_is_one,
    _scalar_is_zero,
    get_free_symbols,
    ratio,
)
from .._aux._backends._symbolic_router import (
    as_numer_denom as _routed_as_numer_denom,
)
from .._aux._backends._symbolic_router import (
    conjugate as _routed_conjugate,
)
from .._aux._backends._symbolic_router import (
    log as _routed_log,
)
from .._aux._backends._symbolic_router import (
    simplify as _routed_simplify,
)
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
    as_engine_scalar,
    e_constant,
    expr_numeric_types,
    expr_operands,
    expr_types,
    integer,
    op_expr,
    symbol,
    zero,
)
from .._aux._deprecated.dgcv_formatter import process_basis_label
from .._aux._utilities._config import (
    dgcv_warning,
    get_variable_registry,
    update_working_namespace,
    update_working_namespace_k_v,
    working_namespace,
)
from .._aux._vmf._safeguards import (
    create_key,
    get_dgcv_category,
    retrieve_passkey,
    validate_label,
)
from .._aux._vmf.vmf import clearVar
from .._aux.printing.printing._eds import (
    bar_label_latex,
    conjugation_prefix,
    ext_der_latex,
    ext_der_repr,
    ext_der_str,
    ordinal_latex,
)
from ..core.base import dgcv_class
from ..core.combinatorics.combinatorics import carProd, weightedPermSign

sp = sympy_module_if_available()


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
        expr, (abstract_ZF, zeroFormAtom, abstDFAtom, abstDFMonom, abstract_DF)
    ):
        return expr._eval_conjugate()
    else:
        return _routed_conjugate(expr)


class zeroFormAtom(dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "zeroFormAtom"

    def __new__(
        cls,
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
        - coframe (abst_coframe, optional): marks the primary abst_coframe w.r.t. the zero forms printing/display behavior may be adjusted
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
            if not len(elem) == 0 and not isinstance(elem[0], abst_coframe):
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
                coframe = abst_coframe(tuple(), {})
        elif len(coframe_derivatives) > 0 and coframe != coframe_derivatives[0][0]:
            coframe_derivatives = ((coframe,),) + coframe_derivatives
        if not isinstance(coframe, abst_coframe):
            raise TypeError(
                "Expected given `coframe` to be None or have type `abst_coframe`"
            )

        # Using SymPy's Basic constructor
        obj = object.__new__(cls)
        obj.label = label
        obj.coframe_derivatives = coframe_derivatives
        obj.coframe = coframe
        return obj

    def __init__(
        self,
        label,
        coframe_derivatives=tuple(),
        coframe=None,
        _markers=frozenset(),
        coframe_independants=dict(),
    ):
        """
        Initialize attributes (already set by __new__).
        """
        self._markers = _markers
        self.coframe_independants = coframe_independants
        self.is_constant = "constant" in _markers
        self.is_one = self.label == "_1" and self.is_constant
        self._is_zero = self.label == "_0" and self.is_constant
        self.secondary_coframes = [
            elem[0] for elem in self.coframe_derivatives if elem[0] != self.coframe
        ]
        self.related_coframes = (
            [self.coframe] + self.secondary_coframes
            if self.coframe is not abst_coframe(tuple(), {})
            else self.secondary_coframes
        )

    @property
    def is_zero(self):
        """Property to safely expose the zero check."""
        return self._is_zero

    @property
    def differential_order(self):
        if not hasattr(self, "_differential_order"):
            self._differential_order = sum(
                [len(elem[1:]) for elem in self.coframe_derivatives]
            )
        return self._differential_order

    def _sage_(self):
        raise AttributeError

    def __eq__(self, other):
        """
        Check equality of two zeroFormAtom instances.
        """
        if not isinstance(other, zeroFormAtom):
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
        if not isinstance(other, zeroFormAtom):
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
        if "real" in self._markers:
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
        return zeroFormAtom(
            conjugated_label,
            coframe_derivatives=newCD,
            coframe=self.coframe,
            _markers=self._markers,
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
            return abstract_ZF(0)
        if isinstance(other, (zeroFormAtom, abstract_ZF)) or isinstance(
            other, expr_numeric_types()
        ):
            return abstract_ZF(("mul", self, other))
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return abstract_ZF(("mul", -1, self))

    def __truediv__(self, other):
        """
        Division with zeroFormAtom
        """
        if isinstance(other, zeroFormAtom) or isinstance(other, expr_numeric_types()):
            return abstract_ZF(("div", self, other))
        return NotImplemented

    def __rtruediv__(self, other):
        """
        Division with zeroFormAtom instances
        """
        if isinstance(other, (zeroFormAtom, abstract_ZF)) or isinstance(
            other, expr_numeric_types()
        ):
            return abstract_ZF(("div", other, self))

        return NotImplemented

    def __pow__(self, exp):
        """
        Exponentiation of zeroFormAtom:
        - Returns a structured `abstract_ZF` exponentiation.
        """
        if isinstance(exp, (zeroFormAtom, abstract_ZF)) or isinstance(
            exp, expr_numeric_types()
        ):
            return abstract_ZF(("pow", self, exp))
        return NotImplemented

    def __rpow__(self, other):
        if isinstance(other, (zeroFormAtom, abstract_ZF)) or isinstance(
            other, expr_numeric_types()
        ):
            return abstract_ZF(("pow", other, self))

        return NotImplemented

    def __add__(self, other):
        """
        Addition with zeroFormAtom
        """
        if isinstance(other, (zeroFormAtom, abstract_ZF)) or isinstance(
            other, expr_numeric_types()
        ):
            return abstract_ZF(("add", self, other))
        return NotImplemented

    def __radd__(self, other):
        """
        Addition with zeroFormAtom
        """
        if isinstance(other, (zeroFormAtom, abstract_ZF)) or isinstance(
            other, expr_numeric_types()
        ):
            return abstract_ZF(("add", self, other))
        return NotImplemented

    def __sub__(self, other):
        """
        Subtraction of zeroFormAtom:
        """
        if isinstance(other, (zeroFormAtom, abstract_ZF)) or isinstance(
            other, expr_numeric_types()
        ):
            return abstract_ZF(("sub", self, other))
        return NotImplemented

    def __rsub__(self, other):
        """
        Subtraction of zeroFormAtom
        """
        if isinstance(other, (zeroFormAtom, abstract_ZF)) or isinstance(
            other, expr_numeric_types()
        ):
            return abstract_ZF(("sub", other, self))
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
        if isinstance(other, abstract_ZF) and isinstance(other.base, zeroFormAtom):
            other = other.base
        if isinstance(other, zeroFormAtom):
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
        if isinstance(other, abstract_ZF) and isinstance(other.base, zeroFormAtom):
            return other.base.is_primitive(self, returnCD=returnCD)
        if isinstance(other, zeroFormAtom):
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
                            new_value, (zeroFormAtom, abstract_ZF)
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
                if isinstance(new_value, (zeroFormAtom, abstract_ZF)) or isinstance(
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

    def expand(self, **kw):
        return self

    def factor(self, **kw):
        return self

    def numer(self, **kw):
        return self

    def denom(self, **kw):
        return self

    def __repr__(self):
        return f"zeroFormAtom({self.__str__()!r})"

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
            if isinstance(cf, abst_coframe) and idx - 1 in cf.inverted_conj_rules:
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


def createZeroForm(
    labels,
    index_set={},
    initial_index=1,
    assumeReal=False,
    coframe=None,
    coframe_independants=dict(),
    verbose_labeling=False,
    remove_guardrails=None,
):
    if not isinstance(labels, str):
        raise TypeError(
            "`createZeroForm` requires its first argument to be a string, which will be used in lables for the created zero forms."
        )

    def reformat_string(input_string: str):
        # Replace commas with spaces, then split on spaces
        substrings = input_string.replace(",", " ").split()
        # Return the list of non-empty substrings
        return [s for s in substrings if len(s) > 0]

    if isinstance(index_set, numbers.Integral) and index_set > 0:
        if not isinstance(initial_index, numbers.Integral):
            initial_index = 1
        index_set = {"lower": list(range(initial_index, index_set + initial_index))}
    elif not isinstance(index_set, dict):
        index_set = {}
    if isinstance(labels, str):
        labels = reformat_string(labels)

    for label in labels:
        _zeroFormFactory(
            label,
            index_set=index_set,
            assumeReal=assumeReal,
            coframe=coframe,
            coframe_independants=coframe_independants,
            verbose_labeling=verbose_labeling,
            _tempVar=None,
            _doNotUpdateVar=None,
            remove_guardrails=remove_guardrails,
        )


def _zeroFormFactory(
    label,
    index_set={},
    assumeReal=False,
    coframe=None,
    coframe_independants=dict(),
    verbose_labeling=False,
    _tempVar=None,
    _doNotUpdateVar=None,
    remove_guardrails=None,
):
    """
    Initializes zeroFormAtom systems, registers them in the VMF,
    and updates caller globals().

    Parameters:
    - label (str): Base label for the zeroFormAtom instances.
    - index_set (dict, optional): Determines tuple structure with 'upper' and 'lower' keys.
    - assumeReal (bool, optional): If True, marks instances as real and skips conjugate handling.
    - coframe (abstCoframe, optional): sets the primary coframe associated with the form.
    - _tempVar (None, optional): If set, marks the variable system as temporary.
    - _doNotUpdateVar (None, optional): If set, prevents clearing/replacing existing instances.
    - remove_guardrails (None, optional): If set, bypasses safeguards for label validation.
    """
    variable_registry = get_variable_registry()
    eds_atoms = variable_registry["eds"]["atoms"]
    passkey = retrieve_passkey()

    label = validate_label(label) if not remove_guardrails else label

    if _doNotUpdateVar is None:
        clearVar(label, report=False)

    if _tempVar == passkey:
        variable_registry["temporary_variables"].add(label)
        _tempVar = True
    else:
        _tempVar = None

    family_type = "single"
    if index_set:
        family_type = "tuple"

        def expand_indices(index):
            if index is None:
                return []
            if isinstance(index, numbers.Integral):
                return [[index]]
            if isinstance(index, list):
                return [[i] if isinstance(i, numbers.Integral) else i for i in index]
            raise ValueError("Indices must be an integer or a list of integers/lists.")

        upper_indices = expand_indices(index_set.get("upper", None))
        lower_indices = expand_indices(index_set.get("lower", None))
        lhPairs = index_set.get("low_hi_pairs", None)
        if (
            isinstance(lhPairs, (list, tuple))
            and len(lhPairs) == 1
            and len(lhPairs[0][0]) == 0
            and len(lhPairs[0][1]) == 0
        ):
            lhPairs = None

        if upper_indices and lower_indices:
            index_combinations = list(carProd(lower_indices, upper_indices))
            if lhPairs:
                index_combinations += list(lhPairs)
        elif lower_indices:
            index_combinations = [
                (lo, []) for lo in lower_indices
            ]  # Treat upper as empty
            if lhPairs:
                index_combinations += list(lhPairs)
        elif upper_indices:
            index_combinations = [
                ([], hi) for hi in upper_indices
            ]  # Treat lower as empty
            if lhPairs:
                index_combinations += list(lhPairs)
        elif lhPairs:
            index_combinations = list(lhPairs)
        else:
            index_combinations = [((), ())]  # No indices

        def labeler(index_pair, verbose):
            lower, upper = index_pair
            if verbose is True or len(upper) != 0 or len(lower) > 1:
                return (
                    f"{label}"
                    + (f"_low_{'_'.join(map(str, lower))}" if lower else "")
                    + (f"_hi_{'_'.join(map(str, upper))}" if upper else "")
                )
            elif len(lower) == 1:
                return f"{label}" + str(lower[0])
            else:
                return f"{label}"

        family_names = [
            labeler(index_pair, verbose_labeling) for index_pair in index_combinations
        ]
    else:
        family_names = [label]

    family_values = tuple(
        zeroFormAtom(
            name,
            coframe=coframe,
            _markers={"real"} if assumeReal else frozenset(),
            coframe_independants=coframe_independants,
        )
        for name in family_names
    )

    # handle conjugates
    family_relatives = {}
    conjugates = {}
    for atom in family_values:
        if assumeReal:
            family_relatives[atom.label] = (atom, atom)  # Self-conjugate if real
        else:
            conj_atom = atom._eval_conjugate()
            family_relatives[atom.label] = (atom, conj_atom)
            family_relatives[conj_atom.label] = (atom, conj_atom)
            conjugates[conj_atom.label] = conj_atom

    # store in VMF
    eds_atoms[label] = {
        "family_type": family_type,
        "primary_coframe": coframe,
        "degree": 0,
        "family_values": family_values,
        "family_names": tuple(family_names),
        "tempVar": _tempVar,
        "real": assumeReal if assumeReal else None,
        "conjugates": conjugates,
        "family_relatives": family_relatives,
    }
    variable_registry["_labels"][label] = {
        "path": ("eds", "atoms", label),
        "children": set(family_names + list(conjugates.keys())),
    }

    # update active namespace
    update_working_namespace_k_v(
        label, family_values if family_type == "tuple" else family_values[0]
    )
    update_working_namespace(zip(family_names, family_values))
    if assumeReal is not True:
        pref = conjugation_prefix()
        update_working_namespace_k_v(f"{pref}{label}", tuple(conjugates.values()))
    update_working_namespace(conjugates)


class abstract_ZF(dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "abstract_ZF"

    """
    Symbolic expression class that represents abstract zero forms. Supports representations of combinations of many scalar-like class, such as `float`, `int`, `zeroFormAtom`, and many sympy expressions
    """

    def __new__(cls, base):
        """
        Creates a new abstract_ZF instance.
        """
        if base is None or base == list() or base == tuple():
            base = 0
        if isinstance(base, list):
            base = tuple(base)
        if isinstance(base, abstDFAtom) and base.degree == 0:  ###!!!
            base = base.coeff
        if isinstance(base, abstract_ZF):
            base = base.base
        if isinstance(base, tuple):
            op, *args = base
            new_args = []
            for arg in args:
                if isinstance(arg, abstDFAtom) and arg.degree == 0:  ###!!!
                    new_args.append(arg.coeff)
                elif isinstance(arg, abstract_ZF):
                    new_args.append(arg.base)
                else:
                    new_args.append(arg)
            base = (op, *new_args)

        # Define sorting hierarchy: lower index = lower precedence
        type_hierarchy = {
            int: 0,
            float: 0,
            zeroFormAtom: 2,
            abstract_ZF: 3,
            tuple: 4,
        }

        # Helper function to get the hierarchy value for sorting
        def hierarchy_rank(obj):
            th = type_hierarchy.get(type(obj), -1)
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

        # If `base` is a tuple, process it
        if isinstance(base, tuple):
            op, *args = base  # Extract operator and operands
            if op == "sub":
                if all(isinstance(j, expr_numeric_types()) for j in args):
                    base = args[0] - args[1]
                elif args[0] == args[1]:
                    base = 0
                elif abstract_ZF(args[0]).is_zero:
                    base = ("mul", -1, args[1])
                elif abstract_ZF(args[1]).is_zero:
                    base = args[0]
            elif op == "div":
                if all(isinstance(j, expr_numeric_types()) for j in args):
                    base = ratio(args[0], args[1])
                elif args[0] == args[1] and (not abstract_ZF(args[0]).is_zero):
                    base = 1
                elif abstract_ZF(args[0]).is_zero:
                    base = 0
            elif op in {"add", "mul"}:
                # Flatten nested structures (("add", ("add", x, y), z) --> ("add", x, y, z))
                flat_args = []
                for arg in args:
                    if (
                        isinstance(arg, abstract_ZF)
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
                    any((j == 0 or j == 0.0) for j in numeric_terms)
                    or any(j.is_zero for j in other_terms if not isinstance(j, tuple))
                ):
                    base = 0
                else:
                    if op == "mul":
                        other_terms = [
                            j
                            for j in other_terms
                            if isinstance(j, tuple) or not j.is_one
                        ]

                    if op == "add":
                        new_other_terms = {}

                        for term in other_terms:
                            if isinstance(term, abstract_ZF):
                                term = term.base  # Extract base representation

                            # Case 1: Standalone atomic term (e.g., A_low_1_2_hi_1)
                            if isinstance(term, zeroFormAtom):
                                new_other_terms[term] = new_other_terms.get(term, 0) + 1

                            # Case 2: Multiplication structure (e.g., ('mul', 1, A))
                            elif (
                                isinstance(term, tuple)
                                and term[0] == "mul"
                                and len(term) == 3
                                and isinstance(term[1], expr_numeric_types())
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
                            if coeff == 0:
                                continue  # Skip zero terms

                            if coeff == 1:
                                other_terms.append(key)
                            elif coeff == -1:
                                other_terms.append(("mul", -1, key))
                            else:
                                other_terms.append(("mul", coeff, key))

                    # Combine numeric terms into a single sum/prod and insert if nonzero
                    numeric_terms = [j for j in numeric_terms if j is not None]
                    if numeric_terms:
                        if op == "add":
                            combined_numeric = sum(numeric_terms, zero())
                            if combined_numeric != 0:
                                other_terms.insert(0, combined_numeric)
                            elif (
                                combined_numeric == 0 or combined_numeric == 0.0
                            ) and len(other_terms) == 0:
                                other_terms = [combined_numeric]
                        elif op == "mul":
                            combined_numeric = prod(numeric_terms)
                            if combined_numeric == 0 or combined_numeric == 0.0:
                                other_terms = [combined_numeric]
                            elif combined_numeric != 1 and combined_numeric != 1.0:
                                other_terms.insert(0, combined_numeric)
                            elif (
                                combined_numeric == 1 or combined_numeric == 1.0
                            ) and len(other_terms) == 0:
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

            elif op == "pow":
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
                    elif isinstance(left, expr_numeric_types()) and isinstance(
                        right, expr_numeric_types()
                    ):
                        base = left**right
                    elif (
                        isinstance(left, abstract_ZF)
                        and isinstance(left.base, tuple)
                        and left.base[0] == "pow"
                    ):
                        inner_base, inner_exp = left.base[1], left.base[2]

                        base = ("pow", inner_base, ("mul", inner_exp, right))

            # Assign updated base
            if isinstance(base, (tuple, list)):
                base = tuple(
                    [j.base if isinstance(j, abstract_ZF) else j for j in base]
                )
            obj = object.__new__(cls)
            obj.base = base
            return obj

        elif not (
            isinstance(base, (zeroFormAtom, abstract_ZF))
            or isinstance(base, expr_numeric_types())
        ):
            raise TypeError(
                "Base must be zeroFormAtom, int, float, sympy.Expr, abstract_ZF, or an operation tuple."
            )

        obj = object.__new__(cls)
        obj.base = base
        return obj

    def __init__(self, base):
        self._is_zero = (
            self.base == 0
            or self.base == 0.0
            or (isinstance(self.base, zeroFormAtom) and self.base.is_zero)
            or (isinstance(self.base, abstract_ZF) and self.base.is_zero)
        )

        self.is_one = (
            self.base == 1
            or self.base == 1.0
            or (isinstance(self.base, zeroFormAtom) and self.base.is_one)
            or (isinstance(self.base, abstract_ZF) and self.base.is_one)
        )

    @property
    def is_zero(self):
        """Returns True if the expression simplifies to zero."""
        return self._is_zero

    @property
    def tree_leaves(self):
        if not hasattr(self, "_leaves"):
            self._leaves = None
        if self._leaves is None:

            def gather_leaves(base):
                if isinstance(base, abstract_ZF):
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
            for leaf in self.tree_leaves:
                if hasattr(leaf, "free_symbols"):
                    FS |= leaf.free_symbols
                elif isinstance(leaf, zeroFormAtom):
                    FS |= {leaf}
            self._free_symbols = FS
        return self._free_symbols

    def _sage_(self):
        raise AttributeError

    def __hash__(self):
        """
        Hash the abstract_ZF instance for use in sets and dicts.
        """
        return hash(self.base)

    def __eq__(self, other):
        """
        Check equality of two abstract_ZF instances.
        """
        if not isinstance(other, abstract_ZF):
            return NotImplemented
        return self.base == other.base

    def sort_key(self, order=None):  # for the sympy sorting.py default_sort_key
        return (4, self.base)  # 4 is to group with function-like objects

    def _subs_dgcv(self, data, with_diff_corollaries=False):
        # an alias for regular subs so that other functions can know the with_diff_corollaries keyword is available
        return self.subs(data, with_diff_corollaries=with_diff_corollaries)

    def subs(self, data, with_diff_corollaries=False):
        """
        Symbolic substitution in abstract_ZF.
        """
        if isinstance(self.base, numbers.Number):
            return self
        if isinstance(data, (list, tuple)) and all(
            isinstance(j, tuple) and len(j) == 2 for j in data
        ):
            l1 = len(data)
            data = dict(data)
            if len(data) < l1:
                dgcv_warning(
                    "Provided substitution rules had repeat keys, and only one was used."
                )
        if isinstance(self.base, zeroFormAtom):
            return abstract_ZF(
                self.base.subs(data, with_diff_corollaries=with_diff_corollaries)
            )
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
            return abstract_ZF(new_base)
        if isinstance(self.base, tuple):
            op, *args = self.base

            def sub_process(arg, sub_data):
                if isinstance(arg, tuple):
                    arg = abstract_ZF(arg)
                if (
                    isinstance(arg, abstDFAtom) and arg.degree == 0
                ):  ###!!! for review later, may signal a bug...
                    arg = arg.coeff
                if isinstance(arg, (zeroFormAtom, abstract_ZF)):
                    newArg = arg.subs(
                        sub_data, with_diff_corollaries=with_diff_corollaries
                    )
                    if isinstance(newArg, abstDFAtom):
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
                        arg = abstract_ZF(_sympy_to_abstract_ZF(arg, spare_subs))
                    if isinstance(arg, abstDFAtom):
                        if arg.degree == 0:
                            arg = arg.coeff
                        else:
                            raise ValueError(
                                "dgcv subs methods do not support replacing 0-forms with higher degree forms."
                            )
                    return arg
                return arg

            new_base = tuple([op] + [sub_process(arg, data) for arg in args])
            return _loop_ZF_format_conversions(abstract_ZF(new_base))

    def _eval_conjugate(self):
        def recursive_conjugate(expr):
            if isinstance(expr, tuple):
                op, *args = expr
                return tuple([op] + [recursive_conjugate(arg) for arg in args])
            else:
                return _custom_conj(expr)

        return abstract_ZF(recursive_conjugate(self.base))

    def __add__(self, other):
        """
        Addition of abstract_ZF instances.
        Supports addition with int, float, and sympy.Expr.
        """
        if not (
            isinstance(other, (abstract_ZF, zeroFormAtom))
            or isinstance(other, expr_numeric_types())
        ):
            return NotImplemented
        if other == 0:
            return self
        return abstract_ZF(("add", self, other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtraction of abstract_ZF instances.
        Supports subtraction with int, float, and sympy.Expr.
        """
        if isinstance(other, zeroFormAtom):
            other = abstract_ZF(other)
        if not (
            isinstance(other, (abstract_ZF)) or isinstance(other, expr_numeric_types())
        ):
            return NotImplemented
        if other == 0:
            return self
        return abstract_ZF(("sub", self, other))

    def __rsub__(self, other):
        return -1 * (self - other)

    def __mul__(self, other):
        """
        Multiplication of abstract_ZF instances.
        Supports multiplication with int, float, and sympy.Expr.
        """
        if isinstance(other, abstract_ZF):
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
                    return abstract_ZF(("pow", base1, ("add", exp1, exp2)))  # x^(a+b)

            return abstract_ZF(
                ("mul", self.base, other.base)
            )  # Default multiplication for abstract_ZF instances

        elif isinstance(other, expr_numeric_types()):
            if (
                isinstance(self.base, tuple)
                and self.base[0] == "mul"
                and isinstance(self.base[1], expr_numeric_types())
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
            return abstract_ZF(factors)
        elif isinstance(other, zeroFormAtom):
            return abstract_ZF(("mul", self.base, other))

        return NotImplemented

    def __rmul__(self, other):
        """
        Multiplication of abstract_ZF instances.
        Supports multiplication with int, float, and sympy.Expr.
        """
        return self.__mul__(other)

    def __neg__(self):
        return -1 * self

    def __truediv__(self, other):
        """
        Division of abstract_ZF instances.
        Supports division with int, float, and sympy.Expr.
        """
        if isinstance(other, (zeroFormAtom, abstract_ZF)) or isinstance(
            other, expr_numeric_types()
        ):
            return abstract_ZF(("div", self, other))

        return NotImplemented

    def __rtruediv__(self, other):
        """
        Division of abstract_ZF instances.
        Supports division with int, float, and sympy.Expr.
        """
        if isinstance(other, (zeroFormAtom, abstract_ZF)) or isinstance(
            other, expr_numeric_types()
        ):
            return abstract_ZF(("div", other, self))

        return NotImplemented

    def __pow__(self, other):
        """
        Exponentiation of abstract_ZF instances.
        Supports exponentiation with int, float, and sympy.Expr.
        """
        if isinstance(other, (zeroFormAtom, abstract_ZF)) or isinstance(
            other, expr_numeric_types()
        ):
            return abstract_ZF(("pow", self, other))

        return NotImplemented

    def __rpow__(self, other):
        """
        Exponentiation of abstract_ZF instances.
        """
        if isinstance(other, (zeroFormAtom, abstract_ZF)) or isinstance(
            other, expr_numeric_types()
        ):
            return abstract_ZF(("pow", other, self))

        return NotImplemented

    def _eval_simplify(
        self,
        ratio=None,
        measure=None,
        inverse=True,
        doit=True,
        rational=True,
        expand=False,
        **kwargs,
    ):
        """
        Simplifies the abstract_ZF instance using algebraic rules.
        """
        return _loop_ZF_format_conversions(self, withSimplify=True)
        # if isinstance(self.base, tuple):
        #     op, *args = self.base

        #     # Recursively simplify operands with the same keyword arguments
        #     args = [arg._eval_simplify(ratio=ratio, measure=measure, inverse=inverse,
        #                             doit=doit, rational=rational, expand=expand, **kwargs)
        #             if isinstance(arg, abstract_ZF) else arg for arg in args]

        #     # Apply `expand=True` to distribute multiplication over addition
        #     if expand:
        #         if op == "mul":
        #             expanded_terms = []
        #         for term in args:
        #             if isinstance(term, abstract_ZF) and isinstance(term.base, tuple) and term.base[0] == "add":
        #                 expanded_products = [
        #                     abstract_ZF(("mul", term2, *args[:i], *args[i+1:]))
        #                     for i, term2 in enumerate(term.base[1:])
        #                 ]
        #                 expanded_terms.append(abstract_ZF(("add", *expanded_products)))
        #             elif isinstance(term, tuple) and term[0] == "add":
        #                 expanded_products = [
        #                     abstract_ZF(("mul", *args[:i], term2, *args[i+1:]))
        #                     for i, term2 in enumerate(term[1:])
        #                 ]
        #                 expanded_terms.append(abstract_ZF(("add", *expanded_products)))
        #             else:
        #                 expanded_terms.append(term)
        #             return abstract_ZF(("mul", *expanded_terms))

        #         if op == "pow":
        #             base, exp = args
        #             if isinstance(exp, numbers.Integral) and exp>0:
        #                 if isinstance(base, abstract_ZF) and isinstance(base.base, tuple) and base.base[0] == "add":
        #                     base = base.base
        #                 if isinstance(base, tuple) and base[0] == "add":
        #                     terms = base[1:]
        #                     expanded = [('mul',term) for term in terms]
        #                 for j in range(exp):
        #                     expanded = [prod+(term,) for prod in expanded for term in terms]

        #                 expanded = abstract_ZF(('sum',)+tuple(expanded))
        #                 return expanded._eval_simplify()
        #             else:
        #                 return abstract_ZF(('pow',base,exp))

        #     # Use `ratio` to decide whether to factor (a * b + a * c --> a * (b + c))
        #     if op == "add" and ratio is not None:
        #         common_factors = set()
        #         for term in args:
        #             if isinstance(term, abstract_ZF) and isinstance(term.base, tuple) and term.base[0] == "mul":
        #                 term = term.base
        #             if isinstance(term, tuple) and term[0] == "mul":
        #                 factors = set(term[1:])
        #                 if not common_factors:
        #                     common_factors = factors
        #                 else:
        #                     common_factors &= factors  # Intersect common factors
        #             elif isinstance(term,(int,float,sp.Expr,zeroFormAtom,abstract_ZF)):
        #                 if not common_factors:
        #                     common_factors = set([term])
        #                 else:
        #                     common_factors &= set([term])

        #         if common_factors and ratio > 0.1:  # Custom threshold for factoring
        #             remaining_terms = []
        #             for term in args:
        #                 if isinstance(term, abstract_ZF) and isinstance(term.base, tuple) and term.base[0] == "mul":
        #                     factors = set(term.base[1:])
        #                 elif isinstance(term, tuple) and term[0] == "mul":
        #                     factors = set(term[1:])
        #                 else:
        #                     factors = {term}

        #                 reduced_factors = factors - common_factors
        #                 remaining_terms.append(("mul", *reduced_factors) if reduced_factors else 1)

        #             return abstract_ZF(("mul", *common_factors, ("add", *remaining_terms)))

        #     # Return simplified operation
        #     return abstract_ZF((op, *args))

        # elif isinstance(self.base, sp.Expr):
        #     return abstract_ZF(self.base.simplify(ratio=ratio, measure=measure, rational=rational))

        # return self  # Return unchanged if base is not an operation

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
        def numer_from_sympy(x):
            if isinstance(x, expr_types()):
                return _routed_as_numer_denom(x)[0]
            return x

        return self._apply_with_sympify_loop(numer_from_sympy)

    def denom(self):
        def denom_from_sympy(x):
            if isinstance(x, expr_types()):
                return _routed_as_numer_denom(x)[1]
            return 1

        return self._apply_with_sympify_loop(denom_from_sympy)

    def as_numer_denom(self):
        return self.numer(), self.denom()

    def __repr__(self):
        """
        Returns a detailed string representation showing the AST.
        """
        if isinstance(self.base, tuple):
            return f"abstract_ZF({self.base})"
        return f"abstract_ZF({repr(self.base)})"

    def __str__(self):
        """
        Returns a reading-friendly string representation of the expression.
        """
        if isinstance(self.base, tuple):
            op, *args = self.base

            def needs_parentheses(expr, context_op, position):
                """
                Determines whether an expression needs parentheses based on its operator.
                """
                expr_str = str(expr)
                if context_op == "mul" and ("+" in expr_str or "-" in expr_str):
                    if position == 0 and (
                        "-" not in expr_str[1:] and "+" not in expr_str[1:]
                    ):
                        return False
                    return True  # Wrap sums inside products
                if (
                    context_op == "pow"
                    and any(j in expr_str for j in {"+", "-", "*", "/"})
                    and position == 0
                ):
                    return True  # base of expontents with sums/products/divs should be wrapped
                if (
                    context_op == "div"
                    and position == 0
                    and (any(j in expr_str for j in {"+", "/"}) or "-" in expr_str[1:])
                ):
                    return True  # base of expontents with sums/products/divs should be wrapped
                if context_op == "sub" and ("+" in expr_str or "-" in expr_str):
                    return True
                return False

            formatted_args = []
            for count, arg in enumerate(args):
                arg_str = str(arg)
                if needs_parentheses(arg, op, count):
                    arg_str = f"({arg_str})"
                formatted_args.append(arg_str)

            if op == "add":
                formatted_str = " + ".join(formatted_args)
                formatted_str = formatted_str.replace("+ -", "-")
            elif op == "sub":
                return " - ".join(formatted_args)
            elif op == "mul":
                return " * ".join(formatted_args)
            elif op == "div":
                return f"({formatted_args[0]}) / ({formatted_args[1]})"
            elif op == "pow":
                return f"{formatted_args[0]}**{formatted_args[1]}"

        return str(self.base)

    def _latex(self, printer=None, raw=True, **kwargs):
        """
        Returns a LaTeX representation of the expression.
        """
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
                        arg_latex = f"{{{_routed_latex(abstract_ZF(arg))}}}"
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
                        arg_latex = f"{{{_routed_latex(abstract_ZF(arg))}}}"
                    else:
                        arg_latex = (
                            f"{{{_routed_latex(arg)}}}"
                            if hasattr(arg, "_latex")
                            else _routed_latex(arg)
                        )
                    if needs_parentheses(arg, op, count):
                        arg_latex = f"\\left({arg_latex}\\right)"
                    formatted_args.append(arg_latex)

            if op == "add":
                formatted_str = " + ".join(formatted_args)
                formatted_str = formatted_str.replace("+ -", "-")
            elif op == "sub":
                formatted_str = " - ".join(formatted_args)
            elif op == "mul":
                formatted_str = " ".join(formatted_args)
            elif op == "div":
                formatted_str = f"\\frac{{{formatted_args[0]}}}{{{formatted_args[1]}}}"
            elif op == "pow":
                formatted_str = f"{formatted_args[0]}^{{{formatted_args[1]}}}"

            return (
                formatted_str.replace("+ {\\left(-1\\right) ", "- { ")
                .replace("+ -", "-")
                .replace("+ {-", "-{")
            )

        return _routed_latex(self.base)

    def _repr_latex_(self, raw=False, **kwargs):
        """
        Jupyter Notebook LaTeX representation for abstract_ZF.
        """
        return self._latex() if raw else f"${self._latex()}$"

    def to_sympy(self, subs_rules={}):
        return _sympify_abst_ZF(self, subs_rules)[0][0]

    def _canonicalize_step(self):
        if isinstance(self.base, abstract_ZF):
            zf = self.base
        else:
            zf = self
        if isinstance(zf.base, zeroFormAtom):
            zf, stabilized = zf.base._canonicalize_step()
            if isinstance(zf, zeroFormAtom):
                return abstract_ZF(zf), stabilized
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
                    arg = abstract_ZF(arg)
                if hasattr(arg, "_canonicalize_step"):
                    new_arg, stab = arg._canonicalize_step()
                    stabilized = stabilized and stab
                else:
                    new_arg = arg
                if isinstance(new_arg, abstract_ZF):
                    new_arg = new_arg.base
                new_base.append(new_arg)
            return abstract_ZF(tuple(new_base)), stabilized
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


def _zf_coeff_str(coeff, rendered):
    base = getattr(coeff, "base", None)
    if not (isinstance(base, tuple) and len(base) > 0 and base[0] == "add"):
        return rendered
    return f"({rendered})"


class abstDFAtom(dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "abstDFAtom"

    def __new__(
        cls, coeff, degree, label=None, ext_deriv_order=0, _markers=frozenset()
    ):
        if _scalar_is_zero(coeff):
            coeff = 0
        elif _scalar_is_one(coeff):
            coeff = 1
        if isinstance(coeff, numbers.Integral):
            coeff = integer(coeff)

        obj = object.__new__(cls)
        obj.label = label
        obj.degree = degree
        obj.coeff = coeff
        obj._coeff = coeff
        obj.ext_deriv_order = ext_deriv_order
        obj._markers = _markers
        return obj

    def __init__(
        self, coeff, degree, label=None, ext_deriv_order=0, _markers=frozenset()
    ):
        self.coeffs = [coeff]

    @property
    def is_zero(self):
        return self._coeff == 0

    def _sage_(self):
        raise AttributeError

    def __eq__(self, other):
        """
        Check equality of two abstDFAtom instances.
        """
        if not isinstance(other, abstDFAtom):
            return NotImplemented
        return (
            self.coeff == other.coeff
            and self.degree == other.degree
            and self.label == other.label
            and self.ext_deriv_order == other.ext_deriv_order
        )

    def __hash__(self):
        """
        Hash the abstDFAtom instance based on its attributes.
        """
        return hash((self.coeff, self.degree, self.label, self.ext_deriv_order))

    def _eval_conjugate(self):
        if self.label:
            pref = conjugation_prefix()
            prefL = len(pref)
            if "real" in self._markers:
                label = self.label
            elif self.label[0:prefL] == pref:
                label = self.label[prefL:]
            else:
                label = f"{pref}{self.label}"
        else:
            label = None

        def cMarkers(marker):
            if marker == "holomorphic":
                return "antiholomorphic"
            if marker == "antiholomorphic":
                return "holomorphic"
            return marker

        new_markers = frozenset([cMarkers(j) for j in self._markers])
        coeff = _custom_conj(self.coeff)
        return abstDFAtom(
            coeff,
            self.degree,
            label=label,
            ext_deriv_order=self.ext_deriv_order,
            _markers=new_markers,
        )

    def __repr__(self):
        """String representation for abstDFAtom."""

        if isinstance(self.coeff, (zeroFormAtom, abstract_ZF)):
            coeff_str = _zf_coeff_str(self.coeff, self.coeff.__repr__())
            return ext_der_repr(
                f"{coeff_str}{self.label}" if self.label else coeff_str,
                self.ext_deriv_order,
            )
        coeff_sympy = as_engine_scalar(self.coeff)
        if (
            not get_free_symbols(coeff_sympy)
            and self.ext_deriv_order is not None
            and self.ext_deriv_order > 0
        ):
            return "0"
        if coeff_sympy == 1:
            return str(self.label) if self.label else "1"
        elif coeff_sympy == -1:
            return f"-{self.label}" if self.label else "-1"
        else:
            # Wrap in parentheses if there are multiple terms
            coeff_str = (
                f"({coeff_sympy})"
                if op_expr(coeff_sympy) == OP_ADD
                else str(coeff_sympy)
            )
            return (
                ext_der_repr(f"{coeff_str}{self.label}", self.ext_deriv_order)
                if self.label
                else ext_der_repr(coeff_str, self.ext_deriv_order)
            )

    def _latex(self, printer=None, raw=True, **kwargs):
        """LaTeX representation for abstDFAtom."""

        if isinstance(self.coeff, (zeroFormAtom, abstract_ZF)):
            coeff_latex = _routed_latex(self.coeff)
            return ext_der_latex(
                f"{coeff_latex}{bar_label_latex(self.label)}"
                if self.label
                else coeff_latex,
                self.ext_deriv_order,
            )
        coeff_sympy = as_engine_scalar(self.coeff)
        if (
            not get_free_symbols(coeff_sympy)
            and self.ext_deriv_order is not None
            and self.ext_deriv_order > 0
        ):
            return "0"
        if coeff_sympy == 1:
            return bar_label_latex(self.label) if self.label else "1"
        elif coeff_sympy == -1:
            return f"-{bar_label_latex(self.label)}" if self.label else "-1"
        else:
            # Wrap in parentheses if there are multiple terms
            coeff_latex = (
                f"\\left({_routed_latex(coeff_sympy)}\\right)"
                if op_expr(coeff_sympy) == OP_ADD
                else _routed_latex(coeff_sympy)
            )
            return (
                ext_der_latex(
                    f"{coeff_latex}{bar_label_latex(self.label)}", self.ext_deriv_order
                )
                if self.label
                else coeff_latex
            )

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex() if raw else f"${self._latex()}$"

    def __str__(self):
        if isinstance(self.coeff, (zeroFormAtom, abstract_ZF)):
            coeff_str = _zf_coeff_str(self.coeff, self.coeff.__repr__())
            return ext_der_str(
                f"{coeff_str}{self.label}" if self.label else coeff_str,
                self.ext_deriv_order,
            )
        coeff_sympy = as_engine_scalar(self.coeff)
        if (
            not get_free_symbols(coeff_sympy)
            and self.ext_deriv_order is not None
            and self.ext_deriv_order > 0
        ):
            return "0"
        if coeff_sympy == 1:
            return str(self.label) if self.label else "1"
        elif coeff_sympy == -1:
            return f"-{self.label}" if self.label else "-1"
        else:
            # Wrap in parentheses if there are multiple terms
            coeff_str = (
                f"({coeff_sympy})"
                if op_expr(coeff_sympy) == OP_ADD
                else str(coeff_sympy)
            )
            return (
                ext_der_str(f"{coeff_str}{self.label}", self.ext_deriv_order)
                if self.label
                else ext_der_str(coeff_str, self.ext_deriv_order)
            )

    def to_sympy(self):
        if self.degree == 0:
            if self.label is not None:
                dgcv_warning(
                    "`abstDFAtom.to_sympy()` was called for an instance with nontrivial basis label, and that label is not encoded in the method's output."
                )
            if hasattr(self.coeff, "to_sympy"):
                return self.coeff.to_sympy()
            else:
                return _get_sympy_module().sympify(self.coeff)
        else:
            dgcv_warning(
                "`abstDFAtom.to_sympy()` was called for an instance with positive degree, so `None` was returned."
            )

    def has_common_factor(self, other):
        if not isinstance(other, (abstDFAtom, abstDFMonom)):
            return False

        if isinstance(other, abstDFAtom):
            # Special case: label None and degree 0
            if self.label is None and self.degree == 0:
                return other.label is None and other.degree == 0
            # Otherwise, compare labels
            return self.label == other.label

        elif isinstance(other, abstDFMonom):
            # Match against all factors in the monomial
            return any(
                self.has_common_factor(factor) for factor in other.factors_sorted
            )

        return False

    def _eval_simplify(
        self,
        ratio=None,
        measure=None,
        inverse=True,
        doit=True,
        rational=True,
        expand=False,
        **kwargs,
    ):
        return abstDFAtom(
            _routed_simplify(self.coeff),
            self.degree,
            self.label,
            _markers=self._markers,
        )

    def _eval_canonicalize(self, depth=1000):
        if hasattr(self.coeff, "_eval_canonicalize"):
            new_coeff = self.coeff._eval_canonicalize(depth=depth)
        else:
            new_coeff = self.coeff
        return abstDFAtom(new_coeff, self.degree, self.label, _markers=self._markers)

    def _induce_method_from_descending(self, method_name, **kwargs):
        new_coeff = (
            getattr(self.coeff, method_name)(**kwargs)
            if hasattr(self.coeff, method_name)
            else self.coeff
        )
        return abstDFAtom(
            new_coeff,
            self.degree,
            self.label,
            ext_deriv_order=self.ext_deriv_order,
            _markers=self._markers,
        )

    def _subs_dgcv(self, data, with_diff_corollaries=False):
        # an alias for regular subs so that other functions can know the with_diff_corollaries keyword is available
        return self.subs(data, with_diff_corollaries=with_diff_corollaries)

    def subs(self, subs_data, with_diff_corollaries=False):
        if isinstance(subs_data, (list, tuple)) and all(
            isinstance(j, tuple) and len(j) == 2 for j in subs_data
        ):
            l1 = len(subs_data)
            subs_data = dict(subs_data)
            if len(subs_data) < l1:
                dgcv_warning(
                    "Provided substitution rules had repeat keys, and only one was used."
                )
        if self in subs_data:
            return subs_data[self]
        new_coeff = None
        if isinstance(self.coeff, (zeroFormAtom, abstract_ZF)):
            new_coeff = (self.coeff).subs(
                subs_data, with_diff_corollaries=with_diff_corollaries
            )
        elif isinstance(self.coeff, expr_types()):
            if not all(
                isinstance(k, expr_types()) and isinstance(v, expr_numeric_types())
                for k, v in subs_data.items()
            ):
                new_coeff = abstract_ZF(
                    _sympy_to_abstract_ZF(self.coeff, subs_rules=subs_data)
                )
            else:
                new_coeff = (self.coeff).subs(subs_data)
        for k, v in subs_data.items():
            if isinstance(k, abstDFAtom):
                if (
                    self.degree == k.degree
                    and self.label == k.label
                    and self.ext_deriv_order == k.ext_deriv_order
                    and self._markers == k._markers
                ):
                    if new_coeff is None:
                        return (self.coeff / k.coeff) * v
                    else:
                        return (new_coeff / k.coeff) * v
        if new_coeff is None:
            return self
        else:
            return abstDFAtom(
                new_coeff,
                self.degree,
                self.label,
                self.ext_deriv_order,
                _markers=self._markers,
            )

    @property
    def free_symbols(self):
        if hasattr(self.coeff, "free_symbols"):
            return self.coeff.free_symbols
        return set()

    @property
    def _seperated_form(self):
        if self.coeff == 1:
            newAtom = self
        else:
            newAtom = abstDFAtom(
                1,
                self.degree,
                label=self.label,
                ext_deriv_order=self.ext_deriv_order,
                _markers=self._markers,
            )
        return newAtom, self.coeff

    def __mul__(self, other):
        """Handle left multiplication."""
        if isinstance(other, abstDFAtom):
            # Combine two atoms into a monomial product
            return abstDFMonom([self, other])
        elif isinstance(other, abstDFMonom):
            # Prepend this atom as a factor to the monomial's factors
            return abstDFMonom([self] + other.factors_sorted)
        elif isinstance(other, (zeroFormAtom, abstract_ZF)) or isinstance(
            other, expr_numeric_types()
        ):
            return abstDFAtom(
                self.coeff * other, self.degree, self.label, _markers=self._markers
            )
        else:
            return NotImplemented

    def __rmul__(self, other):
        """Handle right multiplication."""
        if isinstance(other, abstDFMonom):
            # Append this atom as a factor to the monomial
            return abstDFMonom(other.factors_sorted + [self])
        elif isinstance(other, (zeroFormAtom, abstract_ZF)) or isinstance(
            other, expr_numeric_types()
        ):
            return abstDFAtom(
                self.coeff * other, self.degree, self.label, _markers=self._markers
            )
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, abstDFAtom) and other.degree == 0:
            if other.label is None or other.label == "":
                other = other.coeff
            else:
                other = other.coeff * zeroFormAtom(other.label, _markers=other._markers)
        if isinstance(other, (abstract_ZF, zeroFormAtom)) or isinstance(
            other, expr_numeric_types()
        ):
            if other == 0:
                raise ZeroDivisionError("division by zero")
            return (1 / other) * self
        return NotImplemented

    def __neg__(self):
        return -1 * self

    def __add__(self, other):
        """Addition with another atom or monomial returns an abstract_DF."""
        if other is None:
            return self
        if isinstance(other, expr_numeric_types()):
            other = abstract_ZF(other)
        if isinstance(other, (abstDFMonom, abstDFAtom)):
            return abstract_DF([abstDFMonom([self]), other])
        elif isinstance(other, zeroFormAtom):
            return abstract_DF(
                [
                    abstDFMonom([self]),
                    abstDFMonom([abstDFAtom(other, 0, _markers=other._markers)]),
                ]
            )
        elif isinstance(other, abstract_ZF):
            return abstract_DF(
                [abstDFMonom([self]), abstDFMonom([abstDFAtom(other, 0)])]
            )
        elif isinstance(other, abstract_DF):
            return abstract_DF((abstDFMonom([self]),) + tuple(other.terms))
        else:
            raise TypeError(
                f"Unsupported operand type {type(other)} for + with `abstDFAtom`"
            )

    def __sub__(self, other):
        """Subtraction with another atom or monomial returns an abstract_DF."""
        if other is None:
            return self
        if isinstance(other, expr_numeric_types()):
            other = abstract_ZF(other)
        elif isinstance(other, (abstDFMonom, abstDFAtom)):
            return abstract_DF([abstDFMonom([self]), -1 * other])
        elif isinstance(other, abstract_DF):
            negated_terms = tuple([-1 * term for term in other.terms])
            return abstract_DF([abstDFMonom([self])] + list(negated_terms))
        elif isinstance(other, (zeroFormAtom, abstract_ZF)):
            return abstract_DF(
                [abstDFMonom([self]), abstDFMonom([abstDFAtom(-1 * other, 0)])]
            )
        else:
            raise TypeError(
                f"Unsupported operand type for - with `abstDFAtom`: {type(other)}"
            )

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return other + (-self)

    def __lt__(self, other):
        """
        Lexicographic comparison: First by degree, then by label.
        """
        if not isinstance(other, abstDFAtom):
            return NotImplemented

        # Primary comparison: degree
        if self.degree != other.degree:
            return self.degree < other.degree

        # Secondary comparison: label (None precedes any string)
        if self.label is None:
            return True
        if other.label is None:
            return False
        return self.label < other.label


def createDiffForm(
    labels,
    degree,
    number_of_variables=None,
    initialIndex=1,
    assumeReal=False,
    remove_guardrails=None,
    return_obj=False,
):
    """
    Initializes abstDFAtom systems, registers them in the VMF,
    and updates caller globals().

    Parameters:
    - label (str): Base label for the abstDFAtom instances.
    - degree (int): Degree of DF to be created.
    - number_of_variables (int, optional): Determines the number of DF created.
    - initialIndex (int, optional): Determines the starting index for DF enumeration
    - assumeReal (bool, optional): If True, marks instances as real and skips conjugate handling.
    - remove_guardrails (None, optional): If set, bypasses safeguards for label validation.
    """
    if isinstance(labels, (list, tuple)):
        if len(set(labels)) < len(labels):
            raise NameError(
                "`createDiffForm` was given the same label repeatedly. Labels need to be distinct."
            )
        if isinstance(degree, (list, tuple)):
            if len(degree) < len(labels):
                degree = list(degree) + ([degree[-1]] * (len(labels) - len(degree)))
        else:
            degree = [degree] * len(labels)
        if isinstance(number_of_variables, (list, tuple)):
            if len(number_of_variables) < len(labels):
                number_of_variables = list(number_of_variables) + (
                    [number_of_variables[-1]] * (len(labels) - len(number_of_variables))
                )
        else:
            number_of_variables = [number_of_variables] * len(labels)
        if isinstance(initialIndex, (list, tuple)):
            if len(initialIndex) < len(labels):
                initialIndex = list(initialIndex) + (
                    [initialIndex[-1]] * (len(labels) - len(initialIndex))
                )
        else:
            initialIndex = [initialIndex] * len(labels)
        if isinstance(assumeReal, (list, tuple)):
            if len(assumeReal) < len(labels):
                assumeReal = list(assumeReal) + (
                    [assumeReal[-1]] * (len(labels) - len(assumeReal))
                )
        else:
            assumeReal = [assumeReal] * len(labels)
        if isinstance(remove_guardrails, (list, tuple)):
            if len(remove_guardrails) < len(labels):
                remove_guardrails = list(remove_guardrails) + (
                    [remove_guardrails[-1]] * (len(labels) - len(remove_guardrails))
                )
        else:
            remove_guardrails = [remove_guardrails] * len(labels)
        if isinstance(return_obj, (list, tuple)):
            if len(return_obj) < len(labels):
                return_obj = list(return_obj) + (
                    [return_obj[-1]] * (len(labels) - len(return_obj))
                )
        else:
            return_obj = [return_obj] * len(labels)
        for idx in range(len(labels)):
            createDiffForm(
                labels[idx],
                degree[idx],
                number_of_variables[idx],
                initialIndex[idx],
                assumeReal[idx],
                remove_guardrails[idx],
                return_obj[idx],
            )
    else:
        if not isinstance(labels, str):
            raise TypeError(
                "`createDiffForm` requires its first argument to be a string, which will be used in lables for the created DF."
            )

        def reformat_string(input_string: str):
            # Replace commas with spaces, then split on spaces
            substrings = input_string.replace(",", " ").split()
            # Return the list of non-empty substrings
            return [s for s in substrings if len(s) > 0]

        if isinstance(labels, str):
            labels = reformat_string(labels)
        if return_obj is True:
            returnList = []
        for label in labels:
            if return_obj is True:
                returnList.append(
                    _DFFactory(
                        label,
                        degree,
                        number_of_variables=number_of_variables,
                        initialIndex=initialIndex,
                        assumeReal=assumeReal,
                        remove_guardrails=remove_guardrails,
                        return_obj=True,
                    )
                )
            else:
                _DFFactory(
                    label,
                    degree,
                    number_of_variables=number_of_variables,
                    initialIndex=initialIndex,
                    assumeReal=assumeReal,
                    remove_guardrails=remove_guardrails,
                )
        if return_obj is True:
            if len(returnList) == 1:
                return returnList[0]
            return returnList


def _DFFactory(
    label,
    degree,
    number_of_variables=None,
    initialIndex=1,
    assumeReal=False,
    _tempVar=None,
    _doNotUpdateVar=None,
    remove_guardrails=None,
    return_obj=False,
):
    """
    Initializes abstDFAtom systems, registers them in the VMF,
    and updates caller globals().

    Parameters:
    - label (str): Base label for the zeroFormAtom instances.
    - degree (int): Degree of DF to be created.
    - number_of_variables (int, optional): Determines the number of DF created.
    - initialIndex (int, optional): Determines the starting index for DF enumeration
    - assumeReal (bool, optional): If True, marks instances as real and skips conjugate handling.
    - _tempVar (None, optional): If set, marks the variable system as temporary.
    - _doNotUpdateVar (None, optional): If set, prevents clearing/replacing existing instances.
    - remove_guardrails (None, optional): If set, bypasses safeguards for label validation.
    """
    pref = conjugation_prefix()
    variable_registry = get_variable_registry()
    eds_atoms = variable_registry["eds"]["atoms"]
    passkey = retrieve_passkey()

    label = validate_label(label) if not remove_guardrails else label

    if _doNotUpdateVar is None:
        clearVar(label, report=False)

    if _tempVar == passkey:
        variable_registry["temporary_variables"].add(label)
        _tempVar = True
    else:
        _tempVar = None

    family_type = "single"
    if number_of_variables:
        family_type = "tuple"

        family_names = [
            f"{label}{index}"
            for index in range(initialIndex, number_of_variables + initialIndex)
        ]
    else:
        family_names = [label]

    family_values = tuple(
        abstDFAtom(
            1,
            degree,
            label=name,
            ext_deriv_order=0,
            _markers={"real"} if assumeReal else frozenset(),
        )
        for name in family_names
    )

    # handle conjugates
    family_relatives = {}
    conjugates = {}
    for atom in family_values:
        if assumeReal:
            family_relatives[atom.label] = (atom, atom)  # Self-conjugate if real
        else:
            conj_atom = atom._eval_conjugate()
            family_relatives[atom.label] = (atom, conj_atom)
            family_relatives[conj_atom.label] = (atom, conj_atom)
            conjugates[conj_atom.label] = conj_atom

    # store in VMF
    eds_atoms[label] = {
        "family_type": family_type,
        "primary_coframe": None,
        "degree": degree,
        "family_values": family_values,
        "family_names": tuple(family_names),
        "tempVar": _tempVar,
        "real": assumeReal if assumeReal else None,
        "conjugates": conjugates,
        "family_relatives": family_relatives,
    }

    children_set = set(family_names)
    if not assumeReal:
        children_set.update(conjugates.keys())
    variable_registry["_labels"][label] = {
        "path": ("eds", "atoms", label),
        "children": children_set,
    }

    # update namespaces
    label_value = family_values if family_type == "tuple" else family_values[0]
    out = {"original": label_value}
    update_working_namespace_k_v(label, label_value)
    update_working_namespace(zip(family_names, family_values))
    if assumeReal is not True:
        cl_value = tuple(conjugates.values())
        out["conjugated"] = cl_value if family_type == "tuple" else cl_value[0]
        update_working_namespace_k_v(f"{pref}{label}", cl_value)
    update_working_namespace(conjugates)
    if return_obj:
        return out


class abstDFMonom(dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "abstDFMonom"

    def __new__(cls, factors):
        if not isinstance(factors, (list, tuple)):
            raise TypeError("`abstDFMonom` expects `factors` to be a list or tuple")
        if not all(isinstance(elem, abstDFAtom) for elem in factors):
            raise TypeError(
                "`abstDFMonom` expects `factors` to be a list of `abstDFAtom`"
            )

        return object.__new__(cls)

    def __init__(self, factors):
        class DegreeLabelSortable:
            def __init__(self, atom):
                self.degree = atom.degree
                self.label = atom.label

            def __lt__(self, other):
                if self.degree != other.degree:
                    return self.degree < other.degree
                if self.label is None:
                    return True
                if other.label is None:
                    return False
                return self.label < other.label

            def __eq__(self, other):
                return self.degree == other.degree and self.label == other.label

            def __le__(self, other):
                return self < other or self == other

        weighted_objs = [
            SortableObj((j, (j.degree, barSortedStr(j.label)))) for j in factors
        ]
        parity, objs_sorted, _ = weightedPermSign(
            weighted_objs,
            [DegreeLabelSortable(j) for j in factors],
            returnSorted=True,
            use_degree_attribute=True,
        )

        if parity == -1:
            objs_sorted = [SortableObj((abstDFAtom(-1, 0), (0, None)))] + objs_sorted

        coeffFactor = 1
        new_objs = []

        for j in objs_sorted:
            if coeffFactor != 0:
                coeffFactor = coeffFactor * j.value.coeff if j.value.coeff else 0
            if j.place[0] != 0:
                new_objs.append(
                    abstDFAtom(
                        1,
                        j.value.degree,
                        j.value.label,
                        j.value.ext_deriv_order,
                        j.value._markers,
                    )
                )

        consolidated_factor = abstDFAtom(coeffFactor, 0)  ### check !!!
        if coeffFactor == 0:
            self.factors_sorted = [abstDFAtom(0, 0)]
            self._coeff = 0
        else:
            self.factors_sorted = [consolidated_factor] + new_objs
            self._coeff = coeffFactor
        if (
            len(self.factors_sorted) != len(set(self.factors_sorted))
            or len(self.factors_sorted) == 0
        ):
            self.factors_sorted = [abstDFAtom(0, 0)]
        self.factors = factors
        self.str_ids = tuple(
            "<Coeff>"
            if i == 0 and factor.label is None
            else (factor.label if factor.label is not None else "<None>")
            for i, factor in enumerate(self.factors_sorted)
        )
        self.degree = sum(factor.degree for factor in self.factors if factor.coeff != 0)

    def _sage_(self):
        raise AttributeError

    def __eq__(self, other):
        """
        Check equality of two abstDFMonom instances.
        """
        if not isinstance(other, abstDFMonom):
            return NotImplemented
        return (
            self.factors_sorted == other.factors_sorted and self.degree == other.degree
        )

    def __hash__(self):
        """
        Hash the abstDFMonom instance based on sorted factors.
        """
        return hash((tuple(self.factors_sorted), self.degree))

    def sort_key(self, order=None):  # for the sympy sorting.py default_sort_key
        return (
            4,
            self.degree,
            tuple(self.factors_sorted),
        )  # 4 is to group with function-like objects

    @property
    def is_zero(self):
        return self._coeff == 0

    @property
    def coeff(self):
        coeff = 1
        for factor in self.factors_sorted:
            if isinstance(factor, abstDFAtom) and factor.degree == 0:
                factor = factor.coeff
            if factor == 0:
                coeff = 0
                break
            if isinstance(factor, (abstract_ZF, zeroFormAtom)) or isinstance(
                factor, expr_numeric_types()
            ):
                coeff *= factor
        return coeff

    @property
    def coeffs(self):
        return [self.coeff]

    def _eval_conjugate(self):
        return abstDFMonom([j._eval_conjugate() for j in self.factors])

    def _eval_simplify(
        self,
        ratio=None,
        measure=None,
        inverse=True,
        doit=True,
        rational=True,
        expand=False,
        **kwargs,
    ):
        return abstDFMonom([j._eval_simplify() for j in self.factors])

    def _eval_canonicalize(self, depth=1000):
        def _canon(obj):
            if hasattr(obj, "_eval_canonicalize"):
                return obj._eval_canonicalize(depth=depth)
            return obj

        return abstDFMonom([_canon(j) for j in self.factors])

    def _subs_dgcv(self, data, with_diff_corollaries=False):
        # an alias for regular subs so that other functions can know the with_diff_corollaries keyword is available
        return self.subs(data, with_diff_corollaries=with_diff_corollaries)

    def subs(self, subs_data, with_diff_corollaries=False):
        return abstDFMonom(
            [
                j._subs_dgcv(subs_data, with_diff_corollaries=with_diff_corollaries)
                for j in self.factors_sorted
            ]
        )

    def _induce_method_from_descending(self, method_name, **kwargs):
        new_factors = [
            getattr(j, method_name)(**kwargs) if hasattr(j, method_name) else j
            for j in self.factors_sorted
        ]
        return abstDFMonom(new_factors)

    def _latex(self, printer=None, raw=True, **kwargs):
        """
        LaTeX representation
        """
        # Handle the leading degree 0 factor (coefficient)
        coeff0 = self.factors_sorted[0]
        if isinstance(coeff0, abstDFAtom):
            coeff_inner = coeff0.coeff
        else:
            coeff_inner = coeff0
        if isinstance(coeff_inner, zeroFormAtom):
            if coeff_inner.is_one:
                coeff_latex = ""
            elif ((-1) * coeff_inner).is_one:
                coeff_latex = "-"
            elif coeff_inner.is_zero:
                coeff_latex = "0"
            else:
                coeff_latex = _routed_latex(coeff0)
        elif isinstance(coeff_inner, abstract_ZF):
            if coeff_inner.is_one:
                coeff_latex = ""
            elif (-1 * coeff_inner).is_one:
                coeff_latex = "-"
            elif coeff_inner.is_zero:
                coeff_latex = "0"
            else:
                coeff_latex = _routed_latex(coeff0)
                if isinstance(coeff_inner.base, tuple) and coeff_inner.base[0] in {
                    "sub",
                    "add",
                }:
                    coeff_latex = f"\\left({coeff_latex}\\right)"
        else:
            if coeff_inner == 1 or coeff_inner == 1.0:
                coeff_latex = ""
            elif coeff_inner == -1 or coeff_inner == -1.0:
                coeff_latex = "-"
            elif coeff_inner == 0 or coeff_inner == 0.0:
                coeff_latex = "0"
            else:
                coeff_latex = _routed_latex(coeff0)

        # Join the remaining factors using '\wedge'
        if len(self.factors_sorted) > 1:
            # Generate LaTeX for all non-coefficient factors
            factors_latex = " \\wedge ".join(
                _routed_latex(factor) for factor in self.factors_sorted[1:]
            )

            # Combine coefficient and factors, but omit '\cdot' if coeff_latex is empty or "-"
            if coeff_latex in ["", "-"]:
                return f"{coeff_latex}{factors_latex}"
            else:
                return f"{coeff_latex} \\cdot {factors_latex}"
        else:
            # Only the degree 0 factor (no other factors to join)
            if coeff_latex == "":
                return "1"
            elif coeff_latex == "-":
                return "-1"
            return coeff_latex

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex() if raw else f"${self._latex()}$"

    def __repr__(self):
        """
        String representation for abstDFMonom.
        """
        # Handle the coefficient (first factor)
        coeff0 = self.factors_sorted[0]
        if isinstance(coeff0, abstDFAtom):
            coeff_inner = coeff0.coeff
        else:
            coeff_inner = coeff0
        if isinstance(coeff_inner, zeroFormAtom):
            coeff_str = coeff0.__str__()
        elif isinstance(coeff_inner, abstract_ZF):
            coeff_str = coeff0.__str__()
            if isinstance(coeff_inner.base, tuple) and coeff_inner.base[0] in {
                "sub",
                "add",
            }:
                coeff_str = f"({coeff_str})"
        else:
            coeff_str = str(coeff0)
        # Join the other factors using '*'
        if len(self.factors_sorted) > 1:
            factors_str = "*".join(str(factor) for factor in self.factors_sorted[1:])

            # Combine coefficient and factors, but omit '*' if coeff_str is empty or "-"
            if coeff_str in ["", "-"]:
                return f"{coeff_str}{factors_str}"
            else:
                return f"{coeff_str}*{factors_str}"
        else:
            # Only the degree 0 factor (no other factors to join)
            return coeff_str

    def __str__(self):
        """Fallback to the string representation."""
        return self.__repr__()

    def __mul__(self, other):
        """Handle left multiplication."""
        if isinstance(other, abstDFMonom):
            # Combine factors of both monomials
            return abstDFMonom(self.factors_sorted + other.factors_sorted)
        elif isinstance(other, abstDFAtom):
            # Append the atom as a factor to this monomial
            return abstDFMonom(self.factors_sorted + [other])
        elif isinstance(other, (abstract_ZF, zeroFormAtom)) or isinstance(
            other, expr_numeric_types()
        ):
            # Scalar multiplication (prepend as an atom with degree 0)
            other_sympy = as_engine_scalar(other)
            return abstDFMonom([abstDFAtom(other_sympy, 0)] + self.factors_sorted)
        elif isinstance(other, (abstract_ZF, zeroFormAtom)):
            return abstDFMonom([abstDFAtom(other, 0)] + self.factors_sorted)
        else:
            # Allow Python to try __rmul__ of the other operand
            return NotImplemented

    def __rmul__(self, other):
        """Handle right multiplication (symmetrically supports scalar * abstDFMonom)."""
        if isinstance(other, expr_numeric_types()):
            other_sympy = as_engine_scalar(other)
            return abstDFMonom([abstDFAtom(other_sympy, 0)] + self.factors_sorted)
        elif isinstance(other, (abstract_ZF, zeroFormAtom)):
            return abstDFMonom([abstDFAtom(other, 0)] + self.factors_sorted)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for *: '{type(other).__name__}' and 'abstDFMonom'"
            )

    def __truediv__(self, other):
        if isinstance(other, abstDFAtom) and other.degree == 0:
            if other.label is None or other.label == "":
                other = other.coeff
            else:
                other = other.coeff * zeroFormAtom(other.label, _markers=other._markers)
        if isinstance(other, (abstract_ZF, zeroFormAtom)) or isinstance(
            other, expr_numeric_types()
        ):
            if other == 0:
                raise ZeroDivisionError("division by zero")
            return (1 / other) * self
        return NotImplemented

    def __neg__(self):
        return -1 * self

    def __add__(self, other):
        """Addition with another monomial or atom returns an abstract_DF."""
        if other is None:
            return self
        if isinstance(other, expr_numeric_types()):
            other = abstract_ZF(other)
        if isinstance(other, (abstDFMonom, abstDFAtom)):
            return abstract_DF([self, other])
        elif isinstance(other, abstract_DF):
            return abstract_DF((self,) + tuple(other.terms))
        elif isinstance(other, (abstract_ZF, zeroFormAtom)):
            return abstract_DF([self, abstDFAtom(other, 0)])
        elif isinstance(other, expr_numeric_types()):
            other_sympy = as_engine_scalar(other)
            return abstract_DF([self, abstDFAtom(other_sympy, 0)])
        else:
            raise TypeError("Unsupported operand type for + with `abstDFMonom`")

    def __sub__(self, other):
        """Subtraction with another monomial or atom returns an abstract_DF."""
        if other is None:
            return self
        if isinstance(other, expr_numeric_types()):
            other = abstract_ZF(other)
        elif isinstance(other, (abstDFMonom, abstDFAtom)):
            return abstract_DF([self, -1 * other])
        elif isinstance(other, abstract_DF):
            negated_terms = tuple([-1 * term for term in other.terms])
            return abstract_DF([self] + negated_terms)
        elif isinstance(other, (abstract_ZF, zeroFormAtom)):
            return abstract_DF([self, abstDFAtom(-1 * other, 0)])
        elif isinstance(other, expr_numeric_types()):
            other_sympy = -1 * as_engine_scalar(other)
            return abstract_DF([self, abstDFAtom(other_sympy, 0)])
        else:
            raise TypeError("Unsupported operand type for - with `abstDFMonom`")

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return other + (-self)

    @property
    def free_symbols(self):
        var_set = set()
        for factor in self.factors_sorted:
            if hasattr(factor, "free_symbols"):
                var_set |= factor.free_symbols
        return var_set


class abstract_DF(dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "abstract_DF"

    def __new__(cls, terms):
        # Validate terms input
        if not isinstance(terms, (list, tuple)):
            raise TypeError("`abstract_DF` expects `terms` to be a list or tuple")
        if not all(isinstance(elem, (abstDFMonom, abstDFAtom)) for elem in terms):
            raise TypeError(
                "`abstract_DF` expects `terms` to be a list of `abstDFMonom` or `abstDFAtom`"
            )
        return object.__new__(cls)

    def __init__(self, terms):
        """
        Initialize abstract_DF with a simplified list of terms.
        """

        def process_abstDF(elem):
            """
            Check that all elements are abstDFMonom/Atom instances.
            """
            if isinstance(elem, abstDFMonom):
                return elem
            elif isinstance(elem, abstDFAtom):
                return abstDFMonom([elem])
            else:
                raise TypeError(
                    "`abstract_DF` initializer expects `abstDFMonom` or `abstDFAtom` instances"
                )

        # Process terms into abstDFMonom instances
        processed_terms = [
            process_abstDF(term) for term in terms if not process_abstDF(term).is_zero
        ]
        # Handle empty terms: default to trivial zero form
        if not terms:
            terms = [abstDFAtom(0, 0)]

        # Simplify terms by combining like terms and removing zeros
        collected_terms = tuple(self.simplify_terms(processed_terms))
        if len(collected_terms) == 0:
            collected_terms = (abstDFMonom([abstDFAtom(0, 0)]),)
        self.terms = collected_terms
        self.degree = (
            self.terms[0].degree
            if all(j.degree == self.terms[0].degree for j in self.terms)
            else None
        )

    def _sage_(self):
        raise AttributeError

    def simplify_terms(self, terms):
        """
        Simplify a list of terms by combining like terms and removing zero terms.
        Terms with "<None>" in their `str_ids` will not be combined.
        """
        term_dict = {}

        for term in terms:
            # Use str_ids as the key for grouping like terms
            key = tuple(term.str_ids)

            # Skip combining terms that contain "<None>" in their key
            if "<None>" in key:
                # Treat these terms individually
                term_dict[id(term)] = {
                    "coeff": term.factors_sorted[0].coeff,
                    "tail": term.factors_sorted[1:],
                }
                continue

            # Extract the leading coefficient and trailing factors
            coeff = term.factors_sorted[0].coeff  # Leading coefficient
            tail = term.factors_sorted[
                1:
            ]  # Trailing factors (actual abstDFAtom instances)

            # Combine coefficients for like terms
            if key in term_dict:
                term_dict[key]["coeff"] = coeff + term_dict[key]["coeff"]
            else:
                term_dict[key] = {"coeff": coeff, "tail": tail}

        # Rebuild simplified terms list
        simplified_terms = [
            abstDFMonom([abstDFAtom(data["coeff"], 0)] + data["tail"])
            for key, data in term_dict.items()
            if data["coeff"] != 0
        ]

        if not simplified_terms:
            return [abstDFMonom([abstDFAtom(0, 0)])]

        # Return the simplified, sorted terms
        return tuple(sorted(simplified_terms, key=lambda t: t.factors_sorted))

    @property
    def coeffs(self):
        coeff_list = []
        for term in self.terms:
            if hasattr(term, "factors_sorted") and len(term.factors_sorted) > 0:
                coeff_list += [term.factors_sorted[0]]
        return coeff_list

    @property
    def is_zero(self):
        return all(term.is_zero for term in self.terms)

    def __eq__(self, other):
        """
        Check equality of two abstract_DF instances.
        """
        if not isinstance(other, abstract_DF):
            return NotImplemented
        return self.terms == other.terms and self.degree == other.degree

    def __hash__(self):
        """
        Hash the abstract_DF instance based on its terms.
        """
        return hash((self.terms, self.degree))

    def sort_key(self, order=None):  # for the sympy sorting.py default_sort_key
        return (4, self.degree, self.terms)  # 4 is to group with function-like objects

    def _eval_conjugate(self):
        return abstract_DF([j._eval_conjugate() for j in self.terms])

    def _eval_simplify(
        self,
        ratio=None,
        measure=None,
        inverse=True,
        doit=True,
        rational=True,
        expand=False,
        **kwargs,
    ):
        return abstract_DF([j._eval_simplify() for j in self.terms])

    def _eval_canonicalize(self, depth=1000):
        def _canon(obj):
            if hasattr(obj, "_eval_canonicalize"):
                return obj._eval_canonicalize(depth=depth)
            return obj

        return abstract_DF([_canon(j) for j in self.terms])

    def _subs_dgcv(self, data, with_diff_corollaries=False):
        # an alias for regular subs so that other functions can know the with_diff_corollaries keyword is available
        return self.subs(data, with_diff_corollaries=with_diff_corollaries)

    def subs(self, subs_data, with_diff_corollaries=False):
        return abstract_DF(
            [
                j._subs_dgcv(subs_data, with_diff_corollaries=with_diff_corollaries)
                for j in self.terms
            ]
        )

    def _induce_method_from_descending(self, method_name, **kwargs):
        new_terms = [
            getattr(j, method_name)(**kwargs) if hasattr(j, method_name) else j
            for j in self.terms
        ]
        return abstract_DF(new_terms)

    def __add__(self, other):
        if isinstance(other, expr_numeric_types()):
            other = abstract_ZF(other)
        if isinstance(other, (abstract_ZF, zeroFormAtom)):
            other = abstDFAtom(other, 0)
        elif isinstance(other, expr_numeric_types()):
            other = abstDFAtom(as_engine_scalar(other), 0)
        if other is None:
            return self
        elif isinstance(other, abstract_DF):
            return abstract_DF(self.terms + other.terms)
        elif isinstance(other, (abstDFMonom, abstDFAtom)):
            return abstract_DF(self.terms + (other,))
        else:
            raise TypeError("Unsupported operand type for + with `abstract_DF`")

    def __sub__(self, other):
        if isinstance(other, expr_numeric_types()):
            other = abstract_ZF(other)
        if isinstance(other, (abstract_ZF, zeroFormAtom)):
            other = abstDFAtom(other, 0)
        elif isinstance(other, expr_numeric_types()):
            other = abstDFAtom(as_engine_scalar(other), 0)
        if other is None:
            return self
        elif isinstance(other, abstract_DF):
            negated_terms = tuple([-1 * term for term in other.terms])
            return abstract_DF(self.terms + negated_terms)
        elif isinstance(other, (abstDFMonom, abstDFAtom)):
            return self + (-1 * other)
        else:
            raise TypeError("Unsupported operand type for - with `abstract_DF`")

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if isinstance(other, (abstract_ZF, zeroFormAtom)):
            other = abstDFAtom(other, 0)
        elif isinstance(other, expr_numeric_types()):
            other = abstDFAtom(as_engine_scalar(other), 0)
        if isinstance(other, expr_numeric_types()):
            # Scalar multiplication
            return abstract_DF([term * other for term in self.terms])
        if isinstance(other, abstract_DF):
            # Distribute over terms
            new_terms = [t1 * t2 for t1 in self.terms for t2 in other.terms]
            return abstract_DF(new_terms)
        if isinstance(other, abstDFAtom):
            other = abstDFMonom([other])
        if isinstance(other, abstDFMonom):
            # Multiply each term by the monomial
            return abstract_DF([term * other for term in self.terms])
        else:
            NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (abstract_ZF, zeroFormAtom)):
            other = abstDFAtom(other, 0)
        elif isinstance(other, expr_numeric_types()):
            other = abstDFAtom(as_engine_scalar(other), 0)
        if isinstance(other, expr_numeric_types()):
            return self * other
        if isinstance(other, abstDFAtom):
            other = abstDFMonom([other])
        if isinstance(other, abstDFMonom):
            other = abstract_DF([other])
        if isinstance(other, abstract_DF):
            return other.__mul__(self)
        else:
            NotImplemented

    def __neg__(self):
        return -1 * self

    def __repr__(self):
        """String representation for abstract_DF."""
        if len(self.terms) == 1:
            return repr(self.terms[0])

        # Build the string for terms
        terms_repr = [repr(term) for term in self.terms]

        result = terms_repr[0] if len(terms_repr) > 0 else ""
        for term in terms_repr[1:]:
            if term.startswith("-"):
                result += (
                    f" - {term[1:]}"  # Add space before "-" and strip the leading "-"
                )
            else:
                result += f" + {term}"  # Add "+" before positive terms
        return result

    def _latex(self, printer=None, raw=True, **kwargs):
        """LaTeX representation for SymPy's LaTeX printer."""
        if len(self.terms) == 1:
            return _routed_latex(self.terms[0])

        # Build the LaTeX string for terms
        terms_latex = [_routed_latex(term) for term in self.terms]

        result = terms_latex[0] if len(terms_latex) > 0 else ""
        for term in terms_latex[1:]:
            if term.startswith("-"):
                result += (
                    " - " + term[1:]
                )  # Add space before "-" and strip the leading "-"
            else:
                result += " + " + term  # Add "+" before positive terms
        return result

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex() if raw else f"${self._latex()}$"

    def __str__(self):
        return self.__repr__()

    @property
    def free_symbols(self):
        var_set = set()
        for term in self.terms:
            if hasattr(term, "free_symbols"):
                var_set |= term.free_symbols
        return var_set


class abst_coframe(dgcv_class):
    _dgcv_class_check = retrieve_passkey()
    _dgcv_category = "abst_coframe"

    def __new__(cls, coframe_basis, structure_equations, min_conj_rules={}):
        """
        Create a new abst_coframe instance.

        Parameters:
        ==========
            - structure_equations (dict): A dictionary where keys are abstDFAtom instances (representing 1-forms), and values are either abstract_DF instances (representing the differential) or None.
        """

        # Validate basis
        if not isinstance(coframe_basis, (list, tuple)) and len(coframe_basis) > 0:
            raise TypeError(
                f"Given `coframe_basis` must be a non-empty list/tuple. Instead recieved type {type(coframe_basis)}"
            )
        if len(coframe_basis) != len(set(coframe_basis)):
            counter = Counter(coframe_basis)
            repeated_elements = [item for item, count in counter.items() if count > 1]
            raise TypeError(
                f"`coframe_basis` should not have repeated elements\nRepeated elements: {repeated_elements}"
            )
        # Validate basis elements
        for df in coframe_basis:
            if not isinstance(df, abstDFAtom):
                raise TypeError(
                    f"Given `coframe_basis` should contain only `abstDFAtom` instances. Instead recieved {type(df)}"
                )

        # Validate structure_equations input
        if not isinstance(structure_equations, dict):
            raise TypeError("structure_equations must be a dictionary.")

        # Validate keys and values
        for key, value in structure_equations.items():
            if key not in coframe_basis:
                raise TypeError(
                    f"Keys in `structure_equations` dict must be also be in `coframe_basis`.\nErroneus key recieved{key}"
                )
            if value is None:
                structure_equations[key] = abstract_DF([])
            elif not isinstance(value, abstract_DF):
                raise TypeError(
                    "Values in structure_equations dict must be abstract_DF instances or None."
                )

        forms_tuple = tuple(coframe_basis)
        #### the following may be a useful alternative for resolving ordering ambiguity
        # forms_tuple = tuple(sorted(structure_equations.keys(), key=lambda x: (x.degree, x.label)))

        conj_atoms = list(min_conj_rules.keys()) + list(min_conj_rules.values())
        if (
            len(conj_atoms) != len(set(conj_atoms))
            or not all(isinstance(j, numbers.Integral) for j in conj_atoms)
            or not all(j in range(len(forms_tuple)) for j in conj_atoms)
        ):
            raise ValueError(
                f"`conj_rules` should be an invertible dict containing `int` indices in the range 0 to {len(forms_tuple)}"
            )
        # Create the instance
        obj = object.__new__(cls)
        obj.forms = forms_tuple
        obj.structure_equations = structure_equations  # dictionary (mutable)
        obj.min_conj_rules = min_conj_rules
        obj.inverted_conj_rules = {v: k for k, v in min_conj_rules.items()}
        obj.conj_rules = (
            min_conj_rules
            | {v: k for k, v in min_conj_rules.items()}
            | {j: j for j in range(len(forms_tuple)) if j not in conj_atoms}
        )
        obj.hash_key = create_key("hash_key", key_length=16)
        obj.dimension = len(forms_tuple)
        return obj

    def __init__(self, *args, **kwargs):
        """
        Override the default __init__ to do nothing.
        Initialization is fully handled in __new__.
        """
        pass

    def __str__(self):
        return self.__repr__()

    def _sage_(self):
        raise AttributeError

    def structure_coeff(self, lo1, lo2, hi):
        if not (
            0 <= lo1 < self.dimension
            and 0 <= lo2 < self.dimension
            and 0 <= hi < self.dimension
        ):
            raise IndexError(
                f"indices out of bounds: lo1={lo1}, lo2={lo2}, hi={hi}; "
                f"expected in range 0 to {self.dimension - 1}"
            )
        df = self.structure_equations[self.forms[hi]]
        form1 = self.forms[lo1]
        form2 = self.forms[lo2]
        coeff = 0

        if isinstance(df, (abstDFAtom, abstDFMonom)):
            df = abstract_DF(df)

        if isinstance(df, abstract_DF):
            for term in df.terms:
                scale = 1
                termCoeff = term.coeff
                if isinstance(termCoeff, abstDFAtom) and termCoeff.degree == 0:
                    scale = termCoeff
                elif isinstance(termCoeff, (zeroFormAtom, abstract_ZF)) or isinstance(
                    termCoeff, expr_numeric_types()
                ):
                    scale = termCoeff

                for count, factor in enumerate(term.factors_sorted):
                    if factor == form1:
                        if form2 in term.factors_sorted[count + 1 :]:
                            return scale
                    elif factor == form2:
                        if form1 in term.factors_sorted[count + 1 :]:
                            return -scale

        return coeff

    def copy(self):
        """
        Return another abst_coframe instance with the same coframe_basis and current structure equations, but with a new hash key. Useful for modifying structure equations of the copy without changing the original.
        """
        return abst_coframe(self.forms, self.structure_equations, self.min_conj_rules)

    def __eq__(self, other):
        if not isinstance(other, abst_coframe):
            return NotImplemented
        return self.forms == other.forms and self.hash_key == other.hash_key

    def __hash__(self):
        return hash((self.forms, self.hash_key))

    def sort_key(self, order=None):  # for the sympy sorting.py default_sort_key
        return (10, self.forms)  # 10 is to group with misc objects

    def __lt__(self, other):
        if not isinstance(other, abst_coframe):
            return NotImplemented
        return self.hash_key < other.hash_key

    def __repr__(self):
        """
        String representation of the coframe as a list of 1-forms.
        """
        return f"abst_coframe({', '.join(map(str, self.forms))})"

    def _latex(self, printer=None, raw=True, **kwargs):
        """
        LaTeX representation of the coframe as a list of 1-forms.
        """
        return r"\{" + r", ".join(_routed_latex(form) for form in self.forms) + r"\}"

    def _repr_latex_(self, raw=False, **kwargs):
        return self._latex() if raw else f"${self._latex()}$"

    def update_structure_equations(
        self, replace_symbols={}, replace_eqns={}, simplify=True
    ):
        """
        Update the structure equation for a specific form.

        Parameters:
        - form: An abstDFAtom instance representing the 1-form to update.
        - equation: An abstract_DF instance or None representing the new structure equation.
        """
        if not (isinstance(replace_symbols, dict) and isinstance(replace_eqns, dict)):
            raise TypeError(
                "If specified, `replace_symbols` and `replace_eqns` should be `dict` type"
            )
        for key in replace_symbols.keys():
            if key in self.structure_equations.items():
                dgcv_warning(
                    "It appears `replace_symbols` dictionary passed to `update_structure_equations` contains a 1-form in the coframe. Probably this dictionary key-value pair should be assigned to the `replace_eqns` dictionary instead, i.e. use `update_structure_equations(replace_eqns=...)` instead "
                )
        for key, value in self.structure_equations.items():
            if hasattr(value, "subs") and callable(getattr(value, "subs")):
                if simplify:
                    if isinstance(value, abstDFAtom):
                        self.structure_equations[key] = (
                            value.subs(replace_symbols)
                        )._eval_simplify()
                    else:
                        self.structure_equations[key] = _routed_simplify(
                            value.subs(replace_symbols)
                        )
                else:
                    self.structure_equations[key] = value.subs(replace_symbols)
        for key, value in replace_eqns.items():
            if key in self.forms:
                self.structure_equations[key] = value
        if simplify:
            for key, value in self.structure_equations.items():
                if value is None:
                    self.structure_equations[key] = abstract_DF([])
                elif isinstance(value, abstDFAtom):
                    self.structure_equations[key] = key._eval_simplify(value)
                else:
                    self.structure_equations[key] = _routed_simplify(value)


def createCoframe(
    label,
    coframe_labels,
    str_eqns=None,
    str_eqns_labels=None,
    complete_to_complex_cf=None,
    integrable_complex_struct=False,
    markers=dict(),
    remove_guardrails=False,
):
    """
    Create a coframe with specified 1-forms and structure equations.

    Parameters:
    - label (str): The name of the coframe.
    - coframe_labels (list of str): Labels for 1-forms in the coframe.
    - str_eqns (dict, optional): Pre-populated structure equations, keyed by (i, j, k) tuples.
    - str_eqns_labels (str, optional): Prefix for generating labels for missing terms in str_eqns.
    - markers (dict, optional): a dict whose key are strings from `coframe_labels`, and values of sets of properties associated with each coframe element
    - complete_to_complex_cf (any, optional): builds a larger coframe by include complex conjugate duals of coframe elements marked as holomorphic/antiholomorphic
    - remove_guardrails (bool, optional): Pass to validate_label for customization.
    """

    if str_eqns is None:
        str_eqns = {}
    min_conj_rules = {}

    coframe_labels = list(coframe_labels)

    if complete_to_complex_cf is None:
        complete_to_complex_cf = ["standard"] * len(coframe_labels)
    elif complete_to_complex_cf is True:
        complete_to_complex_cf = []
        for coVec in coframe_labels:
            if "holomorphic" in markers.get(coVec, {}):
                complete_to_complex_cf += ["holomorphic"]
                if "antiholomorphic" in markers.get(coVec, {}):
                    dgcv_warning(
                        'A coframe label was given to `createCoframe` with conflicting property markers "holomorphic" and "antiholomorphic". The coframe was created assuming "holomorphic" is correct'
                    )
                if "real" in markers.get(coVec, {}):
                    dgcv_warning(
                        'A coframe label was given to `createCoframe` with conflicting property markers "holomorphic" and "real". The coframe was created assuming "holomorphic" is correct'
                    )
            elif "antiholomorphic" in markers.get(coVec, {}):
                complete_to_complex_cf += ["antiholomorphic"]
                if "real" in markers.get(coVec, {}):
                    dgcv_warning(
                        'A coframe label was given to `createCoframe` with conflicting property markers "antiholomorphic" and "real". The coframe was created assuming "antiholomorphic" is correct'
                    )
            elif "real" in markers.get(coVec, {}):
                complete_to_complex_cf += ["real"]
            else:
                complete_to_complex_cf += ["standard"]
    elif complete_to_complex_cf == "fromHol":
        complete_to_complex_cf = ["holomorphic"] * len(coframe_labels)
    elif complete_to_complex_cf == "fromAntihol":
        complete_to_complex_cf = ["antiholomorphic"] * len(coframe_labels)
    elif not isinstance(complete_to_complex_cf, (list, tuple)) or len(
        complete_to_complex_cf
    ) != len(coframe_labels):
        dgcv_warning(
            "`createCoframe` was given an unexpected value for `complete_to_complex_cf`. Proceeding as if `complete_to_complex_cf=None`."
        )
        complete_to_complex_cf = ["standard"] * len(coframe_labels)

    closed_assumptions = [
        next(iter(markers[j].intersection({"closed"})), None) if j in markers else None
        for j in coframe_labels
    ]

    # Create abstDFAtom instances for coframe labels
    elem_list = []
    conjugates_list = []
    conjugates_labels = []
    augments_counter = 0
    for count, coframe_label in enumerate(coframe_labels):
        if complete_to_complex_cf[count] == "real":
            coframe_label = validate_label(
                coframe_label, remove_guardrails=remove_guardrails
            )
            elem = _DFFactory(coframe_label, 1, assumeReal=True, return_obj=True)[
                "original"
            ]
            elem_list.append(elem)
        elif (
            complete_to_complex_cf[count] == "holomorphic"
            or complete_to_complex_cf[count] == "antiholomorphic"
        ):
            pref = conjugation_prefix()
            prefL = len(pref)
            if coframe_label[0:prefL] == pref:
                paired_label = validate_label(
                    coframe_label[prefL:], remove_guardrails=remove_guardrails
                )
                elems = _DFFactory(paired_label, 1, return_obj=True)
                elem = elems["conjugated"]
                cElem = elems["original"]
            else:
                coframe_label = validate_label(
                    coframe_label, remove_guardrails=remove_guardrails
                )
                elems = _DFFactory(coframe_label, 1, return_obj=True)
                elem = elems["original"]
                cElem = elems["conjugated"]
            elem_list.append(elem)
            conjugates_list.append(cElem)
            conjugates_labels.append(cElem.label)
            min_conj_rules = min_conj_rules | {
                count: len(coframe_labels) + augments_counter
            }
            augments_counter += 1
        else:
            coframe_label = validate_label(
                coframe_label, remove_guardrails=remove_guardrails
            )
            elem = _DFFactory(coframe_label, 1, return_obj=True)["original"]
            elem_list.append(elem)

    init_dimension = len(coframe_labels)
    coframe_labels += conjugates_labels
    elem_list += conjugates_list

    coframe_dict = {elem: abstract_DF([]) for elem in elem_list}

    # register the coframe in VMF
    coframe = abst_coframe(elem_list, coframe_dict, min_conj_rules)
    label = validate_label(label, remove_guardrails=remove_guardrails)
    update_working_namespace_k_v(label, coframe)

    # populate missing terms in str_eqns
    coeff_labels = []
    low_hi_pairs = []
    if str_eqns_labels is not None:
        if integrable_complex_struct:
            first_index_bound = init_dimension
        else:
            first_index_bound = len(coframe_labels)
        for i in range(first_index_bound):
            for j in range(i + 1, len(coframe_labels)):
                for k in range(init_dimension):
                    if (i, j, k) not in str_eqns or str_eqns[(i, j, k)] is None:
                        low_hi_pairs.append([[i + 1, j + 1], [k + 1]])
                        scale = 0 if closed_assumptions[k] == "closed" else 1
                        coeff_label = (
                            f"{str_eqns_labels}_low_{i + 1}_{j + 1}_hi_{k + 1}"
                        )
                        coeff_label = validate_label(
                            coeff_label, remove_guardrails=remove_guardrails
                        )

                        coeff_labels.append(coeff_label)

                        str_eqns[coeff_label] = [(i, j, k), scale]
        _zeroFormFactory(
            str_eqns_labels,
            {"low_hi_pairs": low_hi_pairs},
            coframe=coframe,
            remove_guardrails=remove_guardrails,
        )
        str_eqns = {
            (k if isinstance(k, tuple) else v[0]): (
                v if isinstance(k, tuple) else v[1] * working_namespace()[k]
            )
            for k, v in str_eqns.items()
        }
    else:
        # Fill missing terms with None
        for i in range(len(coframe_labels)):
            for j in range(i + 1, len(coframe_labels)):
                for k in range(len(coframe_labels)):
                    if (i, j, k) not in str_eqns:
                        str_eqns[(i, j, k)] = None

    # update coframe!!!!
    update_dict = {}
    for k in range(init_dimension):
        kth_term_list = []
        for i in range(len(coframe_labels)):
            for j in range(i + 1, len(coframe_labels)):
                if (i, j, k) in str_eqns and str_eqns[(i, j, k)] is not None:
                    kth_term_list.append(
                        str_eqns[(i, j, k)] * elem_list[i] * elem_list[j]
                    )
        if kth_term_list:
            update_dict[elem_list[k]] = abstract_DF(kth_term_list)
    inv_dict = {v: k for k, v in min_conj_rules.items()}
    for v in range(init_dimension, len(coframe_labels)):
        if update_dict[elem_list[inv_dict[v]]]:
            update_dict[elem_list[v]] = update_dict[
                elem_list[inv_dict[v]]
            ]._eval_conjugate()

    coframe.update_structure_equations(replace_eqns=update_dict)

    # update VMF
    vr = get_variable_registry()
    vr["eds"]["coframes"][label] = {
        "dimension": len(coframe_labels),
        "children": coframe_labels,
        "cousins": coeff_labels,
        "cousins_vals": working_namespace()[str_eqns_labels],
        "cousins_parent": str_eqns_labels,
    }
    vr["_labels"][label] = {
        "path": ("eds", "coframes", label),
        "children": set(coframe_labels + coeff_labels),
    }


def coframe_derivative(df, coframe, *cfIndex):
    """
    Compute the coframe derivative of an expression with respect to `coframe.forms[cfIndex]`.

    Parameters:
    - df: A `zeroFormAtom` or `abstract_ZF` instance.
    - coframe: The coframe basis.
    - cfIndex: The index of the coframe element w.r.t. which differentiation is performed.

    Returns:
    - The coframe derivative of `df`.
    """
    if len(cfIndex) == 0:
        return df
    if len(cfIndex) > 1:
        result = df
        for idx in cfIndex:
            if not isinstance(idx, numbers.Integral) or idx < 0:
                raise ValueError(
                    f"`coframe_derivative` indices (i.e., optional arguments) must all be non-negative integers. Received {idx}."
                )
            result = coframe_derivative(result, coframe, idx)
        return result
    cfIndex = cfIndex[0]
    if not isinstance(cfIndex, numbers.Integral) or cfIndex < 0:
        raise ValueError(
            f"optional `cfIndex` arguments must all be non-negative integers. Recieved {cfIndex}."
        )
    if cfIndex >= coframe.dimension:
        raise IndexError(
            f"`cfIndex` {cfIndex} is out of bounds for coframe with {coframe.dimension} forms."
        )

    if isinstance(df, zeroFormAtom):
        return _cofrDer_zeroFormAtom(df, coframe, cfIndex)
    elif isinstance(df, abstract_ZF):
        return _cofrDer_abstract_ZF(df, coframe, cfIndex)
    elif isinstance(df, abstDFAtom) and df.degree == 0:
        coeff = df.coeff
        if isinstance(df.label, str):
            other = zeroFormAtom(df.label, _markers=df._markers)
        else:
            other = 1
        newDF = coeff * other
        return coframe_derivative(newDF, coframe, cfIndex)
    elif isinstance(df, expr_numeric_types()):
        return 0
    else:
        if isinstance(df, abstDFAtom):
            raise TypeError(
                f"`coframe_derivative` does not support type `{type(df).__name__}` with nonzero degree."
            )
        raise TypeError(
            f"`coframe_derivative` does not support type `{type(df).__name__}`."
        )


def extDer(df, coframe=None, order=1, with_canonicalize=False, with_simplify=False):
    """
    Exterior derivative operator `extDer()` for various differential forms.

    Parameters:
    - df: The differential form (`zeroFormAtom`,
          `abstDFAtom`, `abstDFMonom`, or `abstract_DF`).
    - coframe: Optional `abst_coframe` object representing the coframe.
    - order: Optional positive integer, denoting the number of times `extDer` is applied.

    Returns:
    - The exterior derivative of the form.
    """
    if hasattr(df, "_dgcv_eds_applyfunc"):
        return df._dgcv_eds_applyfunc(
            lambda elem: extDer(
                elem,
                coframe=coframe,
                order=order,
                with_canonicalize=with_canonicalize,
                with_simplify=with_simplify,
            )
        )

    if not isinstance(order, numbers.Integral) or order < 1:
        raise ValueError("`order` must be a positive integer.")
    # Recursive case for order > 1
    if order > 1:
        ddf = extDer(extDer(df, coframe=coframe), coframe=coframe, order=order - 1)
    else:
        if coframe is None:
            if isinstance(df, abstDFAtom):
                return abstDFAtom(
                    df.coeff,
                    df.degree,
                    label=df.label,
                    ext_deriv_order=df.ext_deriv_order + order,
                    _markers=df._markers,
                )
            if isinstance(df, (zeroFormAtom, abstract_ZF)):
                markers = df._markers if hasattr(df, "_markers") else frozenset()
                return extDer(
                    abstDFAtom(df, 0, _markers=markers), coframe=None, order=order
                )

        # distribute cases to helper functions based on the type of `df`
        if isinstance(df, zeroFormAtom):
            ddf = _extDer_zeroFormAtom(df, coframe)
        elif isinstance(df, abstract_ZF):
            ddf = _extDer_abstract_ZF(df, coframe)
        elif isinstance(df, abstDFAtom):
            if df.degree == 0:
                ddf = _extDer_abstract_ZF(abstract_ZF(df), coframe)
            else:
                ddf = _extDer_abstDFAtom(df, coframe)
        elif isinstance(df, abstDFMonom):
            ddf = _extDer_abstDFMonom(df, coframe)
        elif isinstance(df, abstract_DF):
            ddf = _extDer_abstract_DF(df, coframe)
        elif isinstance(df, expr_numeric_types()):
            return 0
        else:
            raise TypeError(f"`extDer` does not support type `{type(df).__name__}`.")
    if with_canonicalize and with_simplify:
        if hasattr(ddf, "_eval_canonicalize"):
            ddf = ddf._eval_canonicalize()
        if hasattr(ddf, "_eval_simplify"):
            return ddf._eval_simplify()
        else:
            return ddf
    if with_canonicalize:
        if hasattr(ddf, "_eval_canonicalize"):
            return ddf._eval_canonicalize()
    if with_simplify:
        if hasattr(ddf, "_eval_simplify"):
            return ddf._eval_simplify()
    return ddf


def _extDer_zeroFormAtom(df, coframe):
    """
    Compute the exterior derivative for zeroFormAtom.
    """
    return _extDer_abstract_ZF(abstract_ZF(df), coframe)


def _extDer_abstract_ZF(df, coframe):
    """
    Compute the exterior derivative of an `abstract_ZF` expression.

    Parameters:
    - df: An instance of `abstract_ZF`.
    - coframe: The coframe basis (optional).

    Returns:
    - The exterior derivative as an `abstract_DF` expression.
    """
    if coframe is None:
        return abstDFAtom(
            df,
            1,
            ext_deriv_order=1,
            _markers=frozenset(
                [j for j in df._markers if j not in {"holomorphic", "antiholomorphic"}]
            ),
        )

    # Compute one-form terms using `coframe_derivative`
    oneForms = [
        coframe_derivative(df, coframe, j) * coframe.forms[j]
        for j in range(coframe.dimension)
    ]

    # Sum the terms
    return sum(oneForms[1:], oneForms[0])


def _extDer_abstDFAtom(df: abstDFAtom, coframe: abst_coframe):
    """
    Compute the exterior derivative for abstDFAtom.
    """
    if coframe is None:
        order = df.ext_deriv_order + 1 if df.ext_deriv_order else 1
        return abstDFAtom(
            df.coeff,
            df.degree + 1,
            df.label,
            order,
            _markers=frozenset(
                [
                    j
                    for j in df._markers
                    if (j != "holomorphic" and j != "antiholomorphic")
                ]
            ),
        )
    str_eqns = {
        dfKey._seperated_form[0]: (value, dfKey._seperated_form[1])
        for dfKey, value in coframe.structure_equations.items()
    }
    dfAtom, coeff = df._seperated_form
    if dfAtom in str_eqns:
        dfData = str_eqns[dfAtom]
        if isinstance(coeff, (zeroFormAtom, abstract_ZF)):
            new_markers = frozenset(
                [
                    j
                    for j in df._markers
                    if (j != "holomorphic" and j != "antiholomorphic")
                ]
            )
            return (
                extDer(coeff, coframe=coframe)
                * abstDFAtom(
                    1, df.degree, df.label, df.ext_deriv_order, _markers=new_markers
                )
                + coeff * dfData[1] * dfData[0]
            )
        if coeff == 1 and dfData[1] == 1:
            return dfData[0]
        return coeff * dfData[1] * dfData[0]
    if isinstance(df.coeff, (zeroFormAtom, abstract_ZF)):
        new_markers = frozenset(
            [j for j in df._markers if (j != "holomorphic" and j != "antiholomorphic")]
        )
        return extDer(df.coeff, coframe=coframe) * abstDFAtom(
            1, df.degree, df.label, df.ext_deriv_order, _markers=new_markers
        ) + (df.coeff) * extDer(
            abstDFAtom(
                1, df.degree, df.label, df.ext_deriv_order, _markers=df._markers
            ),
            coframe=coframe,
        )
    if isinstance(df.coeff, expr_types()) and len(get_free_symbols((df.coeff))) > 0:
        new_markers = frozenset(
            [j for j in df._markers if (j != "holomorphic" and j != "antiholomorphic")]
        )
        return abstDFAtom(df.coeff, 1, ext_deriv_order=1) * abstDFAtom(
            1, df.degree, df.label, df.ext_deriv_order, _markers=new_markers
        ) + (df.coeff) * extDer(
            abstDFAtom(
                1, df.degree, df.label, df.ext_deriv_order, _markers=df._markers
            ),
            coframe=coframe,
        )
    # if df.label:
    #     order = df.ext_deriv_order+1 if df.ext_deriv_order else 1
    #     return (df.coeff)*abstDFAtom(1,df.degree,df.label,order,_markers=df._markers)
    return abstDFAtom(0, df.degree + 1)


def _extDer_abstDFMonom(df, coframe):
    """
    Compute the exterior derivative for abstDFMonom.
    """
    result = abstract_DF([])
    fs = df.factors_sorted
    next_degree = 0
    for idx, factor in enumerate(fs):
        sign = 1 if next_degree % 2 == 0 else -1
        first_part = df.factors_sorted[:idx]
        last_part = df.factors_sorted[idx + 1 :]
        if len(df.factors_sorted) == 1:
            term = extDer(factor, coframe=coframe)
        elif idx == 0:
            term = extDer(factor, coframe=coframe) * abstDFMonom(last_part)
        elif idx == len(df.factors_sorted) - 1:
            term = sign * abstDFMonom(first_part) * extDer(factor, coframe=coframe)
        else:
            term = (
                sign
                * abstDFMonom(first_part)
                * extDer(factor, coframe=coframe)
                * abstDFMonom(last_part)
            )
        if isinstance(factor, abstDFAtom):
            next_degree = factor.degree
        else:
            next_degree = 0
        result += term
    return result


def _extDer_abstract_DF(df, coframe):
    """
    Compute the exterior derivative for abstract_DF.
    """
    result = abstract_DF([])
    for term in df.terms:
        result += extDer(term, coframe=coframe)
    return result


def _cofrDer_zeroFormAtom(zf, cf, cfIndex):
    """
    Compute the coframe derivative of a `zeroFormAtom` with respect to `cf.forms[cfIndex]`.

    Parameters:
    - zf: The `zeroFormAtom` instance to differentiate.
    - cf: The `abst_coframe` representing the coframe basis.
    - cfIndex: The index of the coframe element w.r.t. which differentiation is performed.

    Returns:
    - A new `zeroFormAtom` representing the coframe derivative.
    """
    if not isinstance(zf, zeroFormAtom):
        raise TypeError("`zf` must be an instance of `zeroFormAtom`.")
    if not isinstance(cf, abst_coframe):
        raise TypeError("`cf` must be an instance of `abst_coframe`.")
    if not isinstance(cfIndex, numbers.Integral) or cfIndex < 0:
        raise ValueError("`cfIndex` must be a non-negative integer.")

    if cfIndex >= cf.dimension:
        raise IndexError(
            f"`cfIndex` {cfIndex} is out of bounds for coframe with {cf.dimension} forms."
        )

    if cf in zf.coframe_independants and cfIndex in zf.coframe_independants[cf]:
        return 0 * zf

    # Helper function to increment the partial derivative orders
    def raise_indices(int_list, int_index):
        new_list = list(int_list)
        new_list[int_index] += 1
        return tuple(new_list)

    if len(zf.coframe_derivatives) > 0 and zf.coframe_derivatives[-1][0] == cf:
        new_cd_elem = tuple(zf.coframe_derivatives[-1]) + (cfIndex,)
        new_CD = tuple(zf.coframe_derivatives[:-1]) + (new_cd_elem,)
    else:
        new_CD = zf.coframe_derivatives + ((cf, cfIndex),)

    # Compute the new derivative
    return zeroFormAtom(
        zf.label,
        coframe_derivatives=new_CD,
        coframe=zf.coframe,
        _markers=zf._markers,
        coframe_independants=zf.coframe_independants,
    )


def _cofrDer_abstract_ZF(df, cf, cfIndex):
    """
    Compute the coframe derivative of an `abstract_ZF` expression or elements in its AST.

    MUST OPERATE ON ANY ELEMENT IN THE AST

    Parameters:
    - df: An instance of `abstract_ZF` or kind of element in AST.
    - cf: The coframe basis.
    - cfIndex: The index of the coframe element w.r.t. which differentiation is performed.

    Returns:
    - The coframe derivative as an `abstract_ZF` expression.
    """
    if isinstance(df, expr_numeric_types()):  # Scalars differentiate to 0
        return 0
    if hasattr(df, "base"):
        df = df.base
    if isinstance(df, zeroFormAtom):
        return _cofrDer_zeroFormAtom(df, cf, cfIndex)
    if isinstance(df, tuple):
        op, *args = df

        if op == "add":  # d(f + g) = df + dg
            return abstract_ZF(
                ("add", *[_cofrDer_abstract_ZF(arg, cf, cfIndex) for arg in args])
            )

        if op == "sub":  # d(f - g) = df - dg
            return abstract_ZF(
                ("sub", *[_cofrDer_abstract_ZF(arg, cf, cfIndex) for arg in args])
            )

        elif op == "mul":  # Product Rule: d(fg) = df g + f dg
            terms = []
            for i, term in enumerate(args):
                other_factors = args[:i] + args[i + 1 :]
                terms.append(
                    abstract_ZF(
                        ("mul", _cofrDer_abstract_ZF(term, cf, cfIndex), *other_factors)
                    )
                )
            return abstract_ZF(("add", *terms))

        elif op == "div":  # Quotient Rule: d(f/g) = (df g - f dg) / g²
            num, denom = args
            dnum = _cofrDer_abstract_ZF(num, cf, cfIndex)
            ddenom = _cofrDer_abstract_ZF(denom, cf, cfIndex)
            return abstract_ZF(
                (
                    "div",
                    (
                        "sub",
                        abstract_ZF(("mul", dnum, denom)),
                        abstract_ZF(("mul", num, ddenom)),
                    ),
                    ("pow", denom, 2),
                )
            )

        elif op == "pow":  # Power Rule: d(f^g)
            base, exponent = args
            if isinstance(exponent, expr_numeric_types()):
                dbase = _cofrDer_abstract_ZF(base, cf, cfIndex)
                new_exp = exponent - 1
                return abstract_ZF(("mul", exponent, ("pow", base, new_exp), dbase))
            elif isinstance(base, expr_numeric_types()):
                return abstract_ZF(
                    (
                        "mul",
                        _routed_log(base),
                        ("pow", base, exponent),
                        _cofrDer_abstract_ZF(exponent, cf, cfIndex),
                    )
                )
            else:
                raise NotImplementedError(
                    f"coframe derivatives are not implemented for type {type(base)} raised to type {type(exponent)}"
                )

    return 0  # If df is constant, return 0


def simplify_with_PDEs(expr, PDEs: dict, tryLess=False, iterations=1):
    """
    Simplifies expressions under quasilinear PDE constraints. Given `PDEs` should be a dictionary whose key is either a sympy.Symbol or a dgcv.zeroFormAtom. For zeroFormAtom keys if their differential order is nonzero, then their corresponding key value must represent an expression whose differential order is not higher. The algorithm is optimized for the case where such key values has strictly lower order, and edge case optimizations for the genearl case will be implemented later.
    """
    if iterations > 1:
        return simplify_with_PDEs(
            simplify_with_PDEs(expr, PDEs, tryLess=tryLess, iterations=iterations - 1),
            PDEs,
            tryLess=tryLess,
            iterations=1,
        )

    def expr_order(expression):
        ex_order = 0
        if hasattr(expression, "free_symbols"):
            vars = expression.free_symbols
            for var in vars:
                if hasattr(var, "differential_order"):
                    ex_order = max(ex_order, var.differential_order)
        return ex_order

    ex_order = expr_order(expr)
    regular_handling = True
    subs_order = 0
    trip = False
    if ex_order > 0:
        regular_handling = False
        trip = True
        for k, v in PDEs.items():
            kOrder = expr_order(k)
            vOrder = expr_order(v)
            subs_order = max(subs_order, kOrder)
            if kOrder < vOrder:  # implies v has free_symbols attribute
                for elem in v.free_symbols:
                    if (
                        isinstance(elem, zeroFormAtom)
                        and isinstance(k, zeroFormAtom)
                        and k.is_primitive(elem)
                    ):
                        raise ValueError(
                            "`simplify_with_PDEs` recieved `PDEs` dictionary in an unsupported format. All PDEs should be solved for a variable whose higher order partials do not appear elsewhere in the expression."
                        )
            if kOrder == vOrder and kOrder > 0:
                trip = False
    standardEQs = {
        v: k
        for v, k in PDEs.items()
        if isinstance(v, expr_types()) and not hasattr(v, "_subs_dgcv")
    }

    def _custom_subs(elem):
        if hasattr(elem, "_subs_dgcv"):
            return elem._subs_dgcv(PDEs, with_diff_corollaries=True)
        elif hasattr(elem, "subs"):
            return elem.subs(standardEQs)
        else:
            return elem

    new_expr = _custom_subs(expr)
    if trip or not tryLess or not regular_handling:
        for _ in range(ex_order - 1):
            new_expr = _custom_subs(new_expr)
    if hasattr(new_expr, "_subs_dgcv"):
        if hasattr(new_expr, "_eval_simplify"):
            return new_expr._eval_simplify()
        else:
            return new_expr
    else:
        return _routed_simplify(new_expr)


def _sympify_abst_ZF(zf: abstract_ZF, varDict):
    if isinstance(zf.base, abstract_ZF):
        return _sympify_abst_ZF(zf.base, varDict)
    if isinstance(zf.base, expr_numeric_types()) or op_expr(zf.base) in (
        OP_NUMBER,
        OP_CONSTANT,
    ):
        return [zf.base], varDict
    if isinstance(zf.base, zeroFormAtom):
        return _equation_formatting(zf.base, varDict)
    if isinstance(zf.base, tuple):
        op, *args = zf.base
        new_args = []
        constructedVarDict = varDict
        for arg in args:
            if isinstance(arg, tuple):
                arg = abstract_ZF(arg)
            if isinstance(arg, abstract_ZF):
                new_arg, new_dict = _sympify_abst_ZF(arg, constructedVarDict)
            else:
                new_data = _equation_formatting(arg, constructedVarDict)
                if new_data:
                    new_arg, new_dict = new_data
                else:
                    new_arg = []
                    new_dict = dict()
            new_args += new_arg
            constructedVarDict |= new_dict
        if op == "mul":
            zf_formatted = [prod(new_args)]
        if op == "add":
            zf_formatted = [sum(new_args)]
        if op == "pow":
            zf_formatted = [new_args[0] ** new_args[1]]
        if op == "sub":
            zf_formatted = [new_args[0] - new_args[1]]
        if op == "div":
            if all(isinstance(arg, numbers.Number) for arg in new_args):
                zf_formatted = [sp.Rational(new_args[0], new_args[1])]
            else:
                zf_formatted = [new_args[0] / new_args[1]]
        return zf_formatted, constructedVarDict
    raise ValueError(
        f"`_sympify_abst_ZF` was given an unsupport expression, of type {type(zf.base)}"
    )


def _engine_to_abstract_ZF(expr, subs_rules={}):
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
            abstract_ZF(_engine_to_abstract_ZF(expr_operands(expr)[0], subs_rules))
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

    if isinstance(expr, abstDFAtom) and expr.degree == 0:
        expr = (
            expr.coeff
        )  # gaurantess expr is scalar (i.e., float/int/abstract_ZF etc.)
    expr, varD = _sympify_abst_ZF(expr, {})
    expr = _routed_simplify(format(expr[0])) if withSimplify else format(expr[0])
    varD = {symbol(k): v[0] for k, v in varD.items()}
    return abstract_ZF(_sympy_to_abstract_ZF(expr, varD))


def _generate_str_id(base_str: str, *dicts: dict) -> str:
    """
    Generates a unique identifier based on base_str.
    Filters against the provided dictionaries to make sure the generated str is not in them.
    """
    candidate = base_str
    while any(candidate in d for d in dicts):
        random_suffix = "".join(
            random.choices(string.ascii_letters + string.digits, k=6)
        )
        candidate = f"{base_str}_{random_suffix}"

    return candidate


def _equation_formatting(eqn, variables_dict):
    var_dict = dict()
    if (isinstance(eqn, (tuple, list)) and len(eqn) == 0) or eqn is None:
        return [], var_dict
    if get_dgcv_category(eqn) == "expression":
        eqn = eqn.polyExpr
    if isinstance(eqn, expr_numeric_types()) and not isinstance(eqn, zeroFormAtom):
        return [as_engine_scalar(eqn)], var_dict
    if get_dgcv_category(eqn) in [
        "algebra_element",
        "subalgebra_element",
        "vector_space_element",
    ]:
        return [as_engine_scalar(term) for term in eqn.coeffs] or [], var_dict
    if get_dgcv_category(eqn) in ["tensorProduct", "fastTensorProduct"]:
        return [
            as_engine_scalar(term) for term in eqn.coeff_dict.values()
        ] or [], var_dict
    if isinstance(eqn, zeroFormAtom):
        not_found_filter = True
        for k, v in variables_dict.items():
            if eqn == v[0]:
                identifier = k
                eqn_formatted = v[1]
                not_found_filter = False
                break
        if not_found_filter:
            candidate_str = eqn.__str__()
            if candidate_str in variables_dict:
                identifier = candidate_str
                eqn_formatted = variables_dict[candidate_str][1]
                # nothing new to add to var_dict here.
            else:
                identifier = _generate_str_id(
                    candidate_str, variables_dict, working_namespace()
                )
                eqn_formatted = [
                    symbol(identifier)
                ]  # The single variable is the equation
                var_dict[identifier] = (
                    eqn,
                    eqn_formatted,
                )  # string label --> (original, formatted)
        return eqn_formatted, var_dict
    if isinstance(eqn, abstract_ZF):
        eqn_formatted, var_dict = _sympify_abst_ZF(eqn, variables_dict)
        return eqn_formatted, var_dict
    elif isinstance(eqn, abstDFAtom):
        eqn_formatted, var_dict = _equation_formatting(eqn._coeff, variables_dict)
        return eqn_formatted, var_dict
    elif isinstance(eqn, abstDFMonom):
        eqn_formatted, var_dict = _equation_formatting(eqn._coeff, variables_dict)
        return eqn_formatted, var_dict
    elif isinstance(eqn, abstract_DF):
        terms = []
        var_dict = dict()
        for term in eqn.terms:
            new_term, new_var_dict = _equation_formatting(
                term, variables_dict | var_dict
            )
            var_dict = var_dict | new_var_dict
            terms += new_term
        return terms, var_dict


def _swap_CFD_order(zf: zeroFormAtom, co1, co2):
    cf_derivatives = list(zf.coframe_derivatives)
    target_elem = cf_derivatives[co1]
    cf = target_elem[0]
    swap_form_idx1 = target_elem[co2]
    swap_form_idx2 = target_elem[co2 + 1]

    def permIdx(idx):
        if idx == co2:
            return co2 + 1
        if idx == co2 + 1:
            return co2
        return idx

    permuted_elem = tuple(
        [target_elem[permIdx(idx)] for idx in range(len(target_elem))]
    )
    lower_order_starts = [
        [k if idx == co2 else target_elem[idx] for idx in range(co2 + 1)]
        for k in range(cf.dimension)
    ]
    lower_order_tail = list(target_elem[co2 + 2 :])
    if len(lower_order_tail) > 0:
        injection_CFD = tuple(
            [tuple([target_elem[0]] + lower_order_tail)] + cf_derivatives[co1 + 1 :]
        )
    else:
        injection_CFD = tuple(cf_derivatives[co1 + 1 :])
    lower_order_CFD = tuple(
        [cf_derivatives[:co1] + [tuple(j)] for j in lower_order_starts]
    )
    lower_order_atoms = [
        zeroFormAtom(
            zf.label,
            coframe_derivatives=CFD,
            coframe=zf.coframe,
            _markers=zf._markers,
            coframe_independants=zf.coframe_independants,
        )
        for CFD in lower_order_CFD
    ]
    lower_order_terms = []
    for idx, atom in enumerate(lower_order_atoms):
        coeffZF = cf.structure_coeff(swap_form_idx1, swap_form_idx2, idx)
        lower_ord_zf = coeffZF * atom
        if coeffZF != 0:
            for elem in injection_CFD:
                lower_ord_zf = coframe_derivative(lower_ord_zf, *elem)
            lower_order_terms.append(lower_ord_zf)

    swapped_CFD = tuple(
        cf_derivatives[:co1] + [permuted_elem] + cf_derivatives[co1 + 1 :]
    )
    swapped_atom = zeroFormAtom(
        zf.label,
        coframe_derivatives=swapped_CFD,
        coframe=zf.coframe,
        _markers=zf._markers,
        coframe_independants=zf.coframe_independants,
    )
    for term in lower_order_terms:
        swapped_atom += term
    return swapped_atom


def _add_passthrough_methods(cls, method_names):
    for name in method_names:
        if not hasattr(cls, name):
            setattr(cls, name, lambda self, **kw: self)
    return cls


for cls in [zeroFormAtom]:
    _add_passthrough_methods(
        cls,
        [
            "factor",
            "expand",
            "simplify",
            "trigsimp",
            "cancel",
            "together",
            "apart",
            "ratsimp",
            "powsimp",
            "logcombine",
            "expand_log",
            "expand_trig",
            "expand_power_exp",
            "expand_power_base",
            "numer",
            "denom",
        ],
    )


def _add_induced_methods(cls, method_names):
    def make_method(method_name):
        def method(self, **kwargs):
            return self._induce_method_from_descending(method_name, **kwargs)

        return method

    for name in method_names:
        if not hasattr(cls, name):
            setattr(cls, name, make_method(name))
    return cls


for cls in [abstDFAtom, abstDFMonom, abstract_DF]:
    _add_induced_methods(
        cls,
        [
            "factor",
            "expand",
            "simplify",
            "trigsimp",
            "cancel",
            "together",
            "apart",
            "ratsimp",
            "powsimp",
            "logcombine",
            "expand_log",
            "expand_trig",
            "expand_power_exp",
            "expand_power_base",
            "numer",
            "denom",
        ],
    )


def _add_sympify_loop_methods(cls, method_names):
    for name in method_names:
        if not hasattr(cls, name):

            def method(self, _name=name, **kw):
                func = getattr(sp, _name)
                return self._apply_with_sympify_loop(func, assume_method=False, **kw)

            setattr(cls, name, method)


_add_sympify_loop_methods(
    abstract_ZF,
    [
        "factor",
        "expand",
        "simplify",
        "trigsimp",
        "cancel",
        "together",
        "apart",
        "ratsimp",
        "powsimp",
        "logcombine",
        "expand_log",
        "expand_trig",
        "expand_power_exp",
        "expand_power_base",
    ],
)


def _add_dgcv_conjugate(cls):
    if not hasattr(cls, "__dgcv_conjugate__"):

        def __dgcv_conjugate__(self, symbolic=False):
            return self._eval_conjugate()

        cls.__dgcv_conjugate__ = __dgcv_conjugate__


for _cls in (
    zeroFormAtom,
    abstract_ZF,
    abstDFAtom,
    abstDFMonom,
    abstract_DF,
):
    _add_dgcv_conjugate(_cls)


for _cls in (
    zeroFormAtom,
    abstract_ZF,
    abstDFAtom,
    abstDFMonom,
    abstract_DF,
    abst_coframe,
):
    register_legacy_sympy_class(_cls)
