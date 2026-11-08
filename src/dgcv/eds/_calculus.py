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

from .._aux._backends._symbolic_router import (
    _scalar_is_one,
    _scalar_is_zero,
    get_free_symbols,
)
from .._aux._backends._symbolic_router import conjugate as _routed_conjugate
from .._aux._backends._symbolic_router import log as _routed_log
from .._aux._backends._symbolic_router import simplify as _routed_simplify
from .._aux._backends._symbolic_router import subs as _routed_subs
from .._aux._backends._types_and_constants import expr_numeric_types, expr_types
from .._aux._utilities._config import dgcv_warning, get_variable_registry
from .._aux._vmf.vmf import vmf_lookup
from ._atoms import _family_entry, _independent_of_form, zero_form_atom
from ._coframes import abstract_coframe, structure_equation_system
from ._forms import (
    abstract_differential_form,
    abstract_differential_form_atom,
    abstract_differential_form_monomial,
)
from ._zero_forms import _builtin, _is_engine_leaf, zero_form_class
from ._zf_ops import op_spec


def _coframe_name(cf):
    for label, entry in get_variable_registry()["eds"]["coframes"].items():
        if entry["coframe"] is cf:
            return label
    return repr(cf)


def _primary_of(expr):
    if isinstance(expr, zero_form_atom):
        return expr.primary_coframe
    if isinstance(expr, zero_form_class):
        found = {
            atom.primary_coframe
            for atom in expr.free_symbols
            if isinstance(atom, zero_form_atom)
        } - {None}
        if len(found) > 1:
            names = ", ".join(sorted(_coframe_name(cf) for cf in found))
            raise ValueError(
                f"{expr} involves zero forms of different coframes ({names}); pass the coframe to differentiate against"
            )
        return found.pop() if found else None
    if isinstance(expr, abstract_differential_form_atom):
        if expr.ext_deriv_order:
            return _primary_of(expr.source_form)
        if expr.label is None:
            return _primary_of(expr.coeff) if expr.degree == 0 else None
        return expr._seperated_form[0].primary_coframe
    return None


def _form_primary(df):
    if df.label is None:
        return _primary_of(df.coeff) if (df.degree == 0 or df.ext_deriv_order) else None
    unit = df.source_form if df.ext_deriv_order else df._seperated_form[0]
    return unit.primary_coframe


def _coordinate_scalar(expr):
    try:
        symbols = get_free_symbols(expr)
    except Exception:
        return False
    if not symbols:
        return False
    return all(vmf_lookup(symbol)["type"] == "coordinate" for symbol in symbols)


def _whole_scalar(df):
    if isinstance(df, zero_form_class):
        if _builtin():
            return df
        base = df.base
        return base if _is_engine_leaf(base) else None
    return df


def _frame_apply(cf, cfIndex, expr):
    return cf.dual_frame_in_coordinates[cfIndex](expr)


def _frame_coefficient_coframe(coframe, coeff):
    if (
        coframe is not None
        and coframe.dual_frame_in_coordinates is not None
        and not isinstance(coeff, numbers.Number)
        and _primary_of(coeff) is None
        and _coordinate_scalar(coeff)
    ):
        return coframe
    return None


def _monomial_frame_coframe(factors, coeff):
    if isinstance(coeff, numbers.Number):
        return None
    owners = {factor.primary_coframe for factor in factors if factor.degree > 0} - {None}
    if len(owners) != 1:
        return None
    return _frame_coefficient_coframe(owners.pop(), coeff)


def _index_in(cf, item):
    if isinstance(item, abstract_differential_form_atom):
        if item in cf.forms:
            return cf.forms.index(item)
        raise ValueError(
            f"{item} is not a basis 1-form of the coframe {_coframe_name(cf)}."
        )
    if not isinstance(item, numbers.Integral) or item < 0:
        raise ValueError(
            f"`coframe_derivative` indices (i.e., optional arguments) must all be non-negative integers. Received {item}."
        )
    if item >= cf.dimension:
        raise IndexError(
            f"`cfIndex` {item} is out of bounds for coframe with {cf.dimension} forms."
        )
    return item


def _direction_of(df, cf, item):
    if isinstance(item, abstract_differential_form_atom):
        owner = item.primary_coframe
        if owner is None:
            raise ValueError(
                f"{item} is not a basis 1-form of a registered coframe; pass the coframe: coframe_derivative(df, coframe, ...)."
            )
        return owner, cf.forms.index(item) if cf is not None and item in cf.forms else owner.forms.index(item)
    if cf is None:
        raise ValueError(
            f"{df} has no coframe to differentiate against; pass one: coframe_derivative(df, coframe, {item}, ...)."
        )
    return cf, _index_in(cf, item)


def coframe_derivative(df, coframe=None, *cfIndex):
    """
    Compute the coframe derivative of an expression with respect to `coframe.forms[cfIndex]`.

    Parameters:
    - df: A `zeroFormAtom` or `abstract_ZF` instance.
    - coframe: The coframe basis, or the first direction (an index or a basis 1-form) when the
      expression's own coframe is meant.
    - cfIndex: The indices (or basis 1-forms) w.r.t. which differentiation is performed.

    Returns:
    - The coframe derivative of `df`.
    """
    directions = tuple(cfIndex) if coframe is None else (coframe,) + tuple(cfIndex)
    if len(directions) == 0:
        return df
    if isinstance(directions[0], abstract_coframe):
        cf = directions[0]
        steps = [(cf, _index_in(cf, item)) for item in directions[1:]]
    else:
        cf = _primary_of(df)
        steps = [_direction_of(df, cf, item) for item in directions]
    result = df
    for step_cf, idx in steps:
        result = _coframe_derivative_step(result, step_cf, idx)
    return result


def _coframe_derivative_step(df, coframe, cfIndex):
    if isinstance(df, zero_form_atom):
        return _cofrDer_zeroFormAtom(df, coframe, cfIndex)
    elif isinstance(df, zero_form_class):
        if coframe.dual_frame_in_coordinates is not None:
            whole = _whole_scalar(df)
            if whole is not None and _coordinate_scalar(whole):
                return _frame_apply(coframe, cfIndex, whole)
        return _cofrDer_abstract_ZF(df, coframe, cfIndex)
    elif isinstance(df, abstract_differential_form_atom) and df.degree == 0:
        coeff = df.coeff
        if isinstance(df.label, str):
            other = zero_form_atom(df.label, _markers=df._markers)
        else:
            other = 1
        newDF = coeff * other
        return _coframe_derivative_step(newDF, coframe, cfIndex)
    elif isinstance(df, expr_numeric_types()):
        if coframe.dual_frame_in_coordinates is not None and not isinstance(df, numbers.Number):
            return _frame_apply(coframe, cfIndex, df)
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
        if coframe is None and isinstance(df, (zero_form_atom, zero_form_class)):
            coframe = _primary_of(df)
            if coframe is None:
                markers = df._markers if hasattr(df, "_markers") else frozenset()
                df = abstract_differential_form_atom(df, 0, _markers=markers)
        if (
            coframe is None
            and isinstance(df, abstract_differential_form_atom)
            and df.degree == 0
            and not df.ext_deriv_order
        ):
            coframe = _form_primary(df)

        # distribute cases to helper functions based on the type of `df`
        if isinstance(df, zero_form_atom):
            ddf = _extDer_zeroFormAtom(df, coframe)
        elif isinstance(df, zero_form_class):
            ddf = _extDer_abstract_ZF(df, coframe)
        elif isinstance(df, abstract_differential_form_atom):
            if df.degree == 0 and coframe is not None:
                ddf = _extDer_abstract_ZF(zero_form_class(df), coframe)
            else:
                ddf = _extDer_abstDFAtom(df, coframe)
        elif isinstance(df, abstract_differential_form_monomial):
            ddf = _extDer_abstDFMonom(df, coframe)
        elif isinstance(df, abstract_differential_form):
            ddf = _extDer_abstract_DF(df, coframe)
        elif isinstance(df, expr_numeric_types()):
            if (
                coframe is not None
                and coframe.dual_frame_in_coordinates is not None
                and not isinstance(df, numbers.Number)
            ):
                ddf = _extDer_abstract_ZF(zero_form_class(df), coframe)
            else:
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
        coframe = _primary_of(df)
    if coframe is None:
        markers = getattr(df, "_markers", None)
        if markers is None:
            markers = getattr(getattr(df, "base", None), "_markers", frozenset())
        return abstract_differential_form_atom(
            df,
            1,
            ext_deriv_order=1,
            _markers=frozenset(
                [j for j in markers if j not in {"holomorphic", "antiholomorphic"}]
            ),
        )

    # Compute one-form terms using `coframe_derivative`
    oneForms = [
        coframe_derivative(df, coframe, j) * coframe.forms[j]
        for j in range(coframe.dimension)
    ]

    # Sum the terms
    return sum(oneForms[1:], oneForms[0])


def _owning_coframe(form):
    primary = form.primary_coframe
    if primary is not None:
        return primary
    owners = {
        entry["coframe"]
        for entry in get_variable_registry()["eds"]["coframes"].values()
        if form in entry["coframe"].forms
    }
    return next(iter(owners)) if len(owners) == 1 else None


def _extDer_abstDFAtom(df: abstract_differential_form_atom, coframe: abstract_coframe):
    """
    Compute the exterior derivative for abstract_DF_atom.
    """
    given = coframe is not None
    if not given:
        coframe = _form_primary(df)
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
    coeff_cf = coframe if given else _frame_coefficient_coframe(coframe, df.coeff)
    if df.label is None and df.ext_deriv_order:
        return extDer(df.coeff, coframe=coframe, order=df.ext_deriv_order + 1)
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
                extDer(coeff, coframe=coeff_cf)
                * abstract_differential_form_atom(
                    1, df.degree, df.label, df.ext_deriv_order, _markers=new_markers
                )
                + coeff * dfData[1] * dfData[0]
            )
        if coeff_cf is not None and coeff_cf.dual_frame_in_coordinates is not None and _coordinate_scalar(coeff):
            new_markers = frozenset(
                [
                    j
                    for j in df._markers
                    if (j != "holomorphic" and j != "antiholomorphic")
                ]
            )
            return (
                extDer(coeff, coframe=coeff_cf)
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
        return extDer(df.coeff, coframe=coeff_cf) * abstract_differential_form_atom(
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
        if coeff_cf is not None and coeff_cf.dual_frame_in_coordinates is not None and _coordinate_scalar(df.coeff):
            d_coeff = extDer(df.coeff, coframe=coeff_cf)
        else:
            d_coeff = abstract_differential_form_atom(df.coeff, 1, ext_deriv_order=1)
        return d_coeff * abstract_differential_form_atom(
            1, df.degree, df.label, df.ext_deriv_order, _markers=new_markers
        ) + (df.coeff) * extDer(
            abstract_differential_form_atom(
                1, df.degree, df.label, df.ext_deriv_order, _markers=df._markers
            ),
            coframe=coframe,
        )
    if df.label is None:
        return abstract_differential_form_atom(0, df.degree + 1)
    if not df.ext_deriv_order and dfAtom in coframe.forms:
        return abstract_differential_form_atom(0, df.degree + 1)
    source = df.source_form
    owner = coframe if source in coframe.forms else _owning_coframe(source)
    if owner is None:
        d_unit = _extDer_abstDFAtom(dfAtom, None)
    else:
        d_unit = extDer(source, coframe=owner, order=df.ext_deriv_order + 1)
    if _scalar_is_one(coeff):
        return d_unit
    return coeff * d_unit


def _extDer_abstDFMonom(df, coframe):
    """
    Compute the exterior derivative for abstract_DF_monomial.
    """
    result = abstract_differential_form([])
    fs = df.factors_sorted
    degree_before = 0
    for idx, factor in enumerate(fs):
        sign = 1 if degree_before % 2 == 0 else -1
        degree_before += factor.degree
        factor_cf = coframe
        if coframe is None and factor.degree == 0:
            factor_cf = _monomial_frame_coframe(fs, factor.coeff)
        dfactor = extDer(factor, coframe=factor_cf)
        first_part = fs[:idx]
        last_part = fs[idx + 1 :]
        if len(fs) == 1:
            term = dfactor
        elif idx == 0:
            term = dfactor * abstract_differential_form_monomial(last_part)
        elif idx == len(fs) - 1:
            term = sign * abstract_differential_form_monomial(first_part) * dfactor
        else:
            term = (
                sign
                * abstract_differential_form_monomial(first_part)
                * dfactor
                * abstract_differential_form_monomial(last_part)
            )
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

    if (
        cf.dual_frame_in_coordinates is not None
        and not zf.coframe_derivatives
        and vmf_lookup(zf)["type"] == "coordinate"
    ):
        return _frame_apply(cf, cfIndex, zf)
    if not zf.coframe_derivatives and cf._potential_index:
        row = cf._potential_index.get(zf)
        if row is not None:
            return row.get(cfIndex, 0)

    family = _family_entry(zf.label)
    if family is None:
        constant = "constant" in zf._markers
        rules = zf._coframe_independants
    else:
        constant = "constant" in zf._markers or "constant" in family.get(
            "declared", {}
        ).get(zf.label, ())
        rules = family.get("independent_of", {}).get(zf.label)
    if constant:
        return 0 * zf
    if rules:
        if cf in rules and cfIndex in rules[cf]:
            return 0 * zf
        if _independent_of_form(rules, cf.forms[cfIndex]):
            return 0 * zf
    if not zf.coframe_derivatives and zf in cf.parameters:
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
    home = zf.primary_coframe
    return zero_form_atom(
        zf.label,
        coframe_derivatives=new_CD,
        coframe=zf.coframe if home is None else home,
        _markers=zf._markers,
        coframe_independants=zf._coframe_independants,
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
    if isinstance(df, zero_form_class):
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
        if cf.dual_frame_in_coordinates is not None and not isinstance(df, numbers.Number):
            return _frame_apply(cf, cfIndex, df)
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
    dexp = (
        0
        if _is_engine_leaf(exponent)
        else _cofrDer_abstract_ZF(exponent, cf, cfIndex)
    )
    dbase = _cofrDer_abstract_ZF(base, cf, cfIndex)
    if _scalar_is_zero(dexp):
        new_exp = exponent - 1
        return zero_form_class(("mul", exponent, ("pow", base, new_exp), dbase))
    try:
        log_base = _routed_log(base)
    except Exception:
        raise NotImplementedError(
            "the coframe derivative of a power whose exponent varies needs the logarithm of its base, which only the builtin engine represents for abstract zero forms"
        ) from None
    if _scalar_is_zero(dbase):
        return zero_form_class(("mul", log_base, ("pow", base, exponent), dexp))
    return zero_form_class(
        (
            "mul",
            ("pow", base, exponent),
            (
                "add",
                ("mul", dexp, log_base),
                ("div", ("mul", exponent, dbase), base),
            ),
        )
    )


for _tag, _cfd in (
    ("add", _cfd_add),
    ("sub", _cfd_sub),
    ("mul", _cfd_mul),
    ("div", _cfd_div),
    ("pow", _cfd_pow),
):
    op_spec(_tag).cfd = _cfd


def _differential_order(expression):
    ex_order = 0
    if hasattr(expression, "free_symbols"):
        vars = expression.free_symbols
        for var in vars:
            if hasattr(var, "differential_order"):
                ex_order = max(ex_order, var.differential_order)
    return ex_order


def simplify_with_PDEs(expr, PDEs: dict, tryLess=False, iterations=1, with_conjugates=True):
    """
    Simplifies expressions under quasilinear PDE constraints. Given `PDEs` should be a dictionary whose key is either a sympy.Symbol or a dgcv.zeroFormAtom. For zeroFormAtom keys if their differential order is nonzero, then their corresponding key value must represent an expression whose differential order is not higher. The algorithm is optimized for the case where such key values has strictly lower order, and edge case optimizations for the genearl case will be implemented later.
    """
    if with_conjugates:
        PDEs = _conjugate_closure(PDEs, "subs")
    if iterations > 1:
        return simplify_with_PDEs(
            simplify_with_PDEs(
                expr, PDEs, tryLess=tryLess, iterations=iterations - 1, with_conjugates=False
            ),
            PDEs,
            tryLess=tryLess,
            iterations=1,
            with_conjugates=False,
        )

    ex_order = _differential_order(expr)
    regular_handling = True
    subs_order = 0
    trip = False
    if ex_order > 0:
        regular_handling = False
        trip = True
        for k, v in PDEs.items():
            kOrder = _differential_order(k)
            vOrder = _differential_order(v)
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
            return elem._subs_dgcv(PDEs, with_diff_corollaries=True, with_conjugates=False)
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


_CONSTRAINTS_MESSAGE = (
    "`constraints` takes a dict of rules or a list of (key, value) pairs, "
    "like subs(); a coframe or its structure equations is not a rule set."
)


def _constraint_rules(rules):
    if isinstance(rules, (structure_equation_system, abstract_coframe)):
        raise TypeError(_CONSTRAINTS_MESSAGE)
    if isinstance(rules, dict):
        return dict(rules)
    if isinstance(rules, (list, tuple)) and all(
        isinstance(j, tuple) and len(j) == 2 for j in rules
    ):
        out = dict(rules)
        if len(out) < len(rules):
            dgcv_warning(
                "Provided substitution rules had repeat keys, and only one was used."
            )
        return out
    raise TypeError(_CONSTRAINTS_MESSAGE)


def _constraint_matches(expr, rules):
    try:
        symbols = get_free_symbols(expr)
    except Exception:
        return True, True
    atom_keys = [k for k in rules if isinstance(k, zero_form_atom)]
    exact = False
    for symbol in symbols:
        if symbol in rules:
            exact = True
        elif isinstance(symbol, zero_form_atom) and any(
            k.is_primitive(symbol) for k in atom_keys
        ):
            return True, True
    return exact, False


def _apply_constraints(expr, rules, with_conjugates=True):
    rules = _constraint_rules(rules)
    if with_conjugates:
        rules = _conjugate_closure(rules, "subs")
    for count in range(max(_differential_order(expr), 1)):
        matches, corollaries = _constraint_matches(expr, rules)
        if count and not matches:
            break
        if hasattr(expr, "_subs_dgcv"):
            expr = expr._subs_dgcv(
                rules, with_diff_corollaries=corollaries, with_conjugates=False
            )
        else:
            expr = _routed_subs(expr, rules)
    if hasattr(expr, "_eval_canonicalize"):
        expr = expr._eval_canonicalize()
    return expr


def _simplify_under_constraints(expr, rules, with_conjugates=True, **kwargs):
    out = _apply_constraints(expr, rules, with_conjugates=with_conjugates)
    if hasattr(out, "_eval_simplify"):
        return out._eval_simplify(**kwargs)
    return _routed_simplify(out, **kwargs)


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
            coframe_independants=zf._coframe_independants,
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
        coframe_independants=zf._coframe_independants,
    )
    for term in lower_order_terms:
        swapped_atom += term
    return swapped_atom


def _conjugate_key(key):
    try:
        conjugated = _routed_conjugate(key)
        hash(conjugated)
    except Exception:
        return None
    return conjugated


_CLOSURE_ESCAPES = {
    "update": " To substitute the rules exactly as written, call update_structure_equations(..., with_conjugates=False).",
    "subs": " To substitute the rules exactly as written, pass with_conjugates=False.",
}


def _conjugate_closure(rules, escape=None):
    if isinstance(rules, dict):
        out = dict(rules)
        for key, value in rules.items():
            conjugated = _conjugate_key(key)
            if conjugated is None or conjugated == key or conjugated in out:
                continue
            if conjugated in get_free_symbols(value):
                raise ValueError(
                    f"The rule {key} -> {value} relates {key} to its own conjugate; adding its conjugate "
                    "rule would make the pair circular (each rewrites the other), so nothing would be "
                    "eliminated. Introduce a real zero form instead: createZeroForm('u', assumeReal=True, "
                    f"coframe=...) and {key}: I*u (or u + I*v for a general relation)."
                    + _CLOSURE_ESCAPES.get(escape, "")
                )
            out[conjugated] = _routed_conjugate(value)
        return out
    if isinstance(rules, (list, tuple, set, frozenset)):
        items = sorted(rules, key=str) if isinstance(rules, (set, frozenset)) else list(rules)
        out = list(items)
        for key in items:
            conjugated = _conjugate_key(key)
            if conjugated is not None and conjugated != key and conjugated not in out:
                out.append(conjugated)
        return out
    raise TypeError("conjugate_closure expects a dict of rules or a list of atoms.")


def conjugate_closure(rules):
    return _conjugate_closure(rules)


def _atoms_in(exprs, found):
    if isinstance(exprs, dict):
        for key, value in exprs.items():
            _atoms_in(key, found)
            _atoms_in(value, found)
    elif isinstance(exprs, (list, tuple, set, frozenset)):
        for item in exprs:
            _atoms_in(item, found)
    else:
        try:
            symbols = get_free_symbols(exprs)
        except Exception:
            symbols = set()
        found.update(s for s in symbols if isinstance(s, zero_form_atom))


def derivative_atoms(exprs, of=None, order=None):
    found = set()
    _atoms_in(exprs, found)
    if of is None:
        sources = None
    elif isinstance(of, (list, tuple, set, frozenset)):
        sources = list(of)
    else:
        sources = [of]
    if order is None:
        orders = None
    elif isinstance(order, numbers.Integral):
        orders = {int(order)}
    else:
        orders = {int(o) for o in order}

    def wanted(atom):
        k = atom.differential_order
        if orders is not None and k not in orders:
            return False
        if sources is None:
            return orders is not None or k > 0
        for source in sources:
            if isinstance(source, zero_form_class) and isinstance(source.base, zero_form_atom):
                source = source.base
            if not isinstance(source, zero_form_atom) or not source.is_primitive(atom):
                continue
            if orders is not None or k > source.differential_order:
                return True
        return False

    return sorted(
        (atom for atom in found if wanted(atom)),
        key=lambda atom: (atom.differential_order, atom.sort_key()),
    )
