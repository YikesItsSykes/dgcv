"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

module: dgcv.eds._calculus

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers

from .._aux._backends._symbolic_router import _scalar_is_zero, get_free_symbols
from .._aux._backends._symbolic_router import log as _routed_log
from .._aux._backends._symbolic_router import simplify as _routed_simplify
from .._aux._backends._types_and_constants import expr_numeric_types, expr_types
from ._zero_forms import _is_engine_leaf
from ._atoms import zero_form_atom
from ._coframes import abstract_coframe
from ._forms import abstract_differential_form, abstract_differential_form_atom, abstract_differential_form_monomial
from ._zero_forms import zero_form_class
from ._zf_ops import op_spec


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

    if isinstance(df, zero_form_atom):
        return _cofrDer_zeroFormAtom(df, coframe, cfIndex)
    elif isinstance(df, zero_form_class):
        return _cofrDer_abstract_ZF(df, coframe, cfIndex)
    elif isinstance(df, abstract_differential_form_atom) and df.degree == 0:
        coeff = df.coeff
        if isinstance(df.label, str):
            other = zero_form_atom(df.label, _markers=df._markers)
        else:
            other = 1
        newDF = coeff * other
        return coframe_derivative(newDF, coframe, cfIndex)
    elif isinstance(df, expr_numeric_types()):
        return 0
    else:
        if isinstance(df, abstract_differential_form_atom):
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
          `abstract_DF_atom`, `abstract_DF_monomial`, or `abstract_DF`).
    - coframe: Optional `abstract_coframe` object representing the coframe.
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
            if isinstance(df, abstract_differential_form_atom):
                return abstract_differential_form_atom(
                    df.coeff,
                    df.degree,
                    label=df.label,
                    ext_deriv_order=df.ext_deriv_order + order,
                    _markers=df._markers,
                )
            if isinstance(df, (zero_form_atom, zero_form_class)):
                markers = df._markers if hasattr(df, "_markers") else frozenset()
                return extDer(
                    abstract_differential_form_atom(df, 0, _markers=markers), coframe=None, order=order
                )

        # distribute cases to helper functions based on the type of `df`
        if isinstance(df, zero_form_atom):
            ddf = _extDer_zeroFormAtom(df, coframe)
        elif isinstance(df, zero_form_class):
            ddf = _extDer_abstract_ZF(df, coframe)
        elif isinstance(df, abstract_differential_form_atom):
            if df.degree == 0:
                ddf = _extDer_abstract_ZF(zero_form_class(df), coframe)
            else:
                ddf = _extDer_abstDFAtom(df, coframe)
        elif isinstance(df, abstract_differential_form_monomial):
            ddf = _extDer_abstDFMonom(df, coframe)
        elif isinstance(df, abstract_differential_form):
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
    return _extDer_abstract_ZF(zero_form_class(df), coframe)


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
        return abstract_differential_form_atom(
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


def _extDer_abstDFAtom(df: abstract_differential_form_atom, coframe: abstract_coframe):
    """
    Compute the exterior derivative for abstract_DF_atom.
    """
    if coframe is None:
        order = df.ext_deriv_order + 1 if df.ext_deriv_order else 1
        return abstract_differential_form_atom(
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
        if isinstance(coeff, (zero_form_atom, zero_form_class)):
            new_markers = frozenset(
                [
                    j
                    for j in df._markers
                    if (j != "holomorphic" and j != "antiholomorphic")
                ]
            )
            return (
                extDer(coeff, coframe=coframe)
                * abstract_differential_form_atom(
                    1, df.degree, df.label, df.ext_deriv_order, _markers=new_markers
                )
                + coeff * dfData[1] * dfData[0]
            )
        if coeff == 1 and dfData[1] == 1:
            return dfData[0]
        return coeff * dfData[1] * dfData[0]
    if isinstance(df.coeff, (zero_form_atom, zero_form_class)):
        new_markers = frozenset(
            [j for j in df._markers if (j != "holomorphic" and j != "antiholomorphic")]
        )
        return extDer(df.coeff, coframe=coframe) * abstract_differential_form_atom(
            1, df.degree, df.label, df.ext_deriv_order, _markers=new_markers
        ) + (df.coeff) * extDer(
            abstract_differential_form_atom(
                1, df.degree, df.label, df.ext_deriv_order, _markers=df._markers
            ),
            coframe=coframe,
        )
    if isinstance(df.coeff, expr_types()) and len(get_free_symbols((df.coeff))) > 0:
        new_markers = frozenset(
            [j for j in df._markers if (j != "holomorphic" and j != "antiholomorphic")]
        )
        return abstract_differential_form_atom(df.coeff, 1, ext_deriv_order=1) * abstract_differential_form_atom(
            1, df.degree, df.label, df.ext_deriv_order, _markers=new_markers
        ) + (df.coeff) * extDer(
            abstract_differential_form_atom(
                1, df.degree, df.label, df.ext_deriv_order, _markers=df._markers
            ),
            coframe=coframe,
        )
    # if df.label:
    #     order = df.ext_deriv_order+1 if df.ext_deriv_order else 1
    #     return (df.coeff)*abstract_DF_atom(1,df.degree,df.label,order,_markers=df._markers)
    return abstract_differential_form_atom(0, df.degree + 1)


def _extDer_abstDFMonom(df, coframe):
    """
    Compute the exterior derivative for abstract_DF_monomial.
    """
    result = abstract_differential_form([])
    fs = df.factors_sorted
    next_degree = 0
    for idx, factor in enumerate(fs):
        sign = 1 if next_degree % 2 == 0 else -1
        first_part = df.factors_sorted[:idx]
        last_part = df.factors_sorted[idx + 1 :]
        if len(df.factors_sorted) == 1:
            term = extDer(factor, coframe=coframe)
        elif idx == 0:
            term = extDer(factor, coframe=coframe) * abstract_differential_form_monomial(last_part)
        elif idx == len(df.factors_sorted) - 1:
            term = (
                sign
                * abstract_differential_form_monomial(first_part)
                * extDer(factor, coframe=coframe)
            )
        else:
            term = (
                sign
                * abstract_differential_form_monomial(first_part)
                * extDer(factor, coframe=coframe)
                * abstract_differential_form_monomial(last_part)
            )
        if isinstance(factor, abstract_differential_form_atom):
            next_degree = factor.degree
        else:
            next_degree = 0
        result += term
    return result


def _extDer_abstract_DF(df, coframe):
    """
    Compute the exterior derivative for abstract_DF.
    """
    result = abstract_differential_form([])
    for term in df.terms:
        result += extDer(term, coframe=coframe)
    return result


def _cofrDer_zeroFormAtom(zf, cf, cfIndex):
    """
    Compute the coframe derivative of a `zeroFormAtom` with respect to `cf.forms[cfIndex]`.

    Parameters:
    - zf: The `zeroFormAtom` instance to differentiate.
    - cf: The `abstract_coframe` representing the coframe basis.
    - cfIndex: The index of the coframe element w.r.t. which differentiation is performed.

    Returns:
    - A new `zeroFormAtom` representing the coframe derivative.
    """
    if not isinstance(zf, zero_form_atom):
        raise TypeError("`zf` must be an instance of `zeroFormAtom`.")
    if not isinstance(cf, abstract_coframe):
        raise TypeError("`cf` must be an instance of `abstract_coframe`.")
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
    return zero_form_atom(
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
    if hasattr(df, "base"):
        df = df.base
    if isinstance(df, zero_form_atom):
        return _cofrDer_zeroFormAtom(df, cf, cfIndex)
    if getattr(df, "_compound", False):
        terms = [
            ("mul", zero_form_class._from_nf(pk), _cofrDer_abstract_ZF(arg, cf, cfIndex))
            for pk, arg in df.partials()
        ]
        return zero_form_class(("add", *terms)) if terms else 0
    if isinstance(df, tuple):
        op, *args = df
        return op_spec(op).cfd(args, cf, cfIndex)
    if _is_engine_leaf(df):
        return 0

    return 0  # If df is constant, return 0


def _cfd_add(args, cf, cfIndex):
    return zero_form_class(
        ("add", *[_cofrDer_abstract_ZF(arg, cf, cfIndex) for arg in args])
    )


def _cfd_sub(args, cf, cfIndex):
    return zero_form_class(
        ("sub", *[_cofrDer_abstract_ZF(arg, cf, cfIndex) for arg in args])
    )


def _cfd_mul(args, cf, cfIndex):
    terms = []
    for i, term in enumerate(args):
        other_factors = args[:i] + args[i + 1 :]
        terms.append(
            zero_form_class(
                ("mul", _cofrDer_abstract_ZF(term, cf, cfIndex), *other_factors)
            )
        )
    return zero_form_class(("add", *terms))


def _cfd_div(args, cf, cfIndex):
    num, denom = args
    dnum = _cofrDer_abstract_ZF(num, cf, cfIndex)
    ddenom = _cofrDer_abstract_ZF(denom, cf, cfIndex)
    return zero_form_class(
        (
            "div",
            (
                "sub",
                zero_form_class(("mul", dnum, denom)),
                zero_form_class(("mul", num, ddenom)),
            ),
            ("pow", denom, 2),
        )
    )


def _cfd_pow(args, cf, cfIndex):
    base, exponent = args
    if _is_engine_leaf(exponent):
        dbase = _cofrDer_abstract_ZF(base, cf, cfIndex)
        new_exp = exponent - 1
        return zero_form_class(("mul", exponent, ("pow", base, new_exp), dbase))
    if _is_engine_leaf(base):
        return zero_form_class(
            (
                "mul",
                _routed_log(base),
                ("pow", base, exponent),
                _cofrDer_abstract_ZF(exponent, cf, cfIndex),
            )
        )
    raise NotImplementedError(
        f"coframe derivatives are not implemented for type {type(base)} raised to type {type(exponent)}"
    )


for _tag, _cfd in (
    ("add", _cfd_add),
    ("sub", _cfd_sub),
    ("mul", _cfd_mul),
    ("div", _cfd_div),
    ("pow", _cfd_pow),
):
    op_spec(_tag).cfd = _cfd


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
                        isinstance(elem, zero_form_atom)
                        and isinstance(k, zero_form_atom)
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


def _swap_CFD_order(zf: zero_form_atom, co1, co2):
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
        zero_form_atom(
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
        if not _scalar_is_zero(coeffZF):
            for elem in injection_CFD:
                lower_ord_zf = coframe_derivative(lower_ord_zf, *elem)
            lower_order_terms.append(lower_ord_zf)

    swapped_CFD = tuple(
        cf_derivatives[:co1] + [permuted_elem] + cf_derivatives[co1 + 1 :]
    )
    swapped_atom = zero_form_atom(
        zf.label,
        coframe_derivatives=swapped_CFD,
        coframe=zf.coframe,
        _markers=zf._markers,
        coframe_independants=zf.coframe_independants,
    )
    for term in lower_order_terms:
        swapped_atom += term
    return swapped_atom
