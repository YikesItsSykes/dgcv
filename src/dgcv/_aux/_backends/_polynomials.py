"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv._aux._backends

module: dgcv._aux._backends._polynomials


Description: manages dgcv's interfacing with polynomial classes in available CAS libraries.

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ._engine import (
    _get_sage_module,
    _get_sympy_module,
    engine_capability,
    engine_kind,
    engine_module,
)
from ._symbolic_router import (
    _scalar_is_zero,
    as_numer_denom,
    cancel,
    expand,
    get_free_symbols,
)
from ._symbolic_router import factor as factor_routed
from ._types_and_constants import symbol, zero

__all__ = [
    "discriminant",
    "make_poly",
    "poly_gens",
    "poly_monoms",
    "poly_coeffs",
    "poly_total_degree",
    "is_polynomial",
    "extract_polynomial_coeffs",
    "poly_terms",
    "poly_linear_roots_from_factorization",
]


# -----------------------------------------------------------------------------
# uilities
# -----------------------------------------------------------------------------
class PolyBackendError(RuntimeError):
    pass


def _unwrap_poly_expr(x: Any) -> Any:
    u = getattr(x, "poly_obj_unformatted", None)
    return u if u is not None else x


def _normalize_poly_expr(expr: Any, *, assume_polynomial: bool = False) -> Any:
    raw = _unwrap_poly_expr(expr)
    if not assume_polynomial:
        raw = cancel(raw)
        raw = as_numer_denom(raw)[0]
    return expand(raw)


@dataclass(frozen=True)
class _DGCVPolyTerms:
    expr: Any
    gens: Tuple[Any, ...]
    monoms: Tuple[Tuple[int, ...], ...]
    coeffs: Tuple[Any, ...]


def poly_terms(
    expr: Any,
    gens: Sequence[Any],
    *,
    assume_polynomial: bool = False,
    parameters: Optional[Iterable[Any]] = None,
) -> Tuple[Tuple[Any, ...], List[Tuple[int, ...]], List[Any]]:
    raw = _normalize_poly_expr(expr, assume_polynomial=assume_polynomial)
    kind = engine_kind()

    if kind == "sympy":
        gens_t = _stable_stamp(gens)
        sp = _get_sympy_module()
        try:
            P = sp.Poly(raw, *gens_t)
        except Exception as e:
            raise PolyBackendError(
                f"dgcv: SymPy Poly construction failed for gens={gens_t!r} and expr type {type(raw).__name__}"
            ) from e
        monoms = [tuple(m) for m in P.monoms()]
        coeffs = list(P.coeffs())
        return gens_t, monoms, coeffs

    if kind == "sage":
        gens_t = _stable_stamp_by_str(gens)
        monoms, coeffs = _sage_terms_via_coefficient(raw, gens_t)
        return gens_t, monoms, coeffs

    if kind == "builtin":
        return engine_capability("poly_terms")(raw, gens)

    raise PolyBackendError(
        "dgcv: no supported symbolic engine is available for polynomials"
    )


def _sage_terms_via_coefficient(
    expr_sr: Any, gens: Tuple[Any, ...]
) -> Tuple[List[Tuple[int, ...]], List[Any]]:
    sage = _get_sage_module()
    SR = getattr(sage, "SR", None)
    if SR is None:
        raise PolyBackendError("dgcv: Sage backend active but sage.SR is unavailable")

    try:
        f = SR(expr_sr)
    except Exception:
        f = expr_sr

    def _coeff_dict_in_var(f0: Any, v: Any) -> Dict[int, Any]:
        deg = getattr(f0, "degree", None)
        if callable(deg):
            try:
                dmax = int(deg(v))
            except Exception:
                dmax = None
        else:
            dmax = None

        out: Dict[int, Any] = {}

        if dmax is not None and dmax >= 0:
            for k in range(dmax + 1):
                try:
                    ck = f0.coefficient(v, k)
                except Exception:
                    ck = None
                if ck is None or _scalar_is_zero(ck):
                    continue
                out[int(k)] = ck
            return out if out else {0: 0}

        try:
            _ = f0.coefficient(v, 0)
            _ = f0.coefficient(v, 1)
        except Exception:
            return {0: f0}

        cap = 256
        for k in range(cap + 1):
            try:
                ck = f0.coefficient(v, k)
            except Exception:
                break
            if ck is None:
                break
            if not _scalar_is_zero(ck):
                out[int(k)] = ck

        if not out:
            out[0] = f0
        return out

    def _rec(i: int, f0: Any) -> Dict[Tuple[int, ...], Any]:
        if i == len(gens):
            return {(): f0}
        v = gens[i]
        d = _coeff_dict_in_var(f0, v)
        out: Dict[Tuple[int, ...], Any] = {}
        for k, ck in d.items():
            tail = _rec(i + 1, ck)
            for e_tail, c_tail in tail.items():
                out[(int(k),) + e_tail] = c_tail
        return out

    term_dict = _rec(0, f)
    items = [(m, c) for (m, c) in term_dict.items() if not _scalar_is_zero(c)]
    items.sort(key=lambda mc: (sum(mc[0]), mc[0]))
    monoms = [tuple(int(e) for e in m) for m, _ in items]
    coeffs = [c for _, c in items]
    return monoms, coeffs


def extract_polynomial_coeffs(
    expr: Union[Any, Sequence[Any]],
    gens: Sequence[Any],
    *,
    assume_polynomial: bool = False,
    parameters: Optional[Iterable[Any]] = None,
    flatten: bool = True,
) -> List[Any]:
    exprs = list(expr) if isinstance(expr, (list, tuple)) else [expr]

    if flatten:
        out: List[Any] = []
        for e in exprs:
            _, _, coeffs = poly_terms(
                e, gens, assume_polynomial=assume_polynomial, parameters=parameters
            )
            out.extend(list(coeffs))
        return out

    grouped: List[List[Any]] = []
    for e in exprs:
        _, _, coeffs = poly_terms(
            e, gens, assume_polynomial=assume_polynomial, parameters=parameters
        )
        grouped.append(list(coeffs))
    return grouped  # type: ignore[return-value]


def discriminant(expr, var):
    kind = engine_kind()

    if kind == "builtin":
        raise NotImplementedError(
            "`discriminant` is not available in dgcv's builtin symbolic engine; install sympy or sage and set `default_engine` accordingly"
        )

    if kind == "sympy":
        sp = _get_sympy_module()
        try:
            return sp.discriminant(_unwrap_poly_expr(expr), var)
        except Exception as e:
            raise PolyBackendError(
                f"dgcv: discriminant failed for var={var!r} and expr type {type(expr).__name__}"
            ) from e

    if kind == "sage":
        sage = _get_sage_module()
        SR = sage.SR
        try:
            f = SR(_unwrap_poly_expr(expr))
            den = f.denominator()
            if SR(den).has(var):
                raise ValueError(f"{var!r} occurs in the denominator")
            g = SR(f.numerator())
            n = int(g.degree(var))
            if n < 1:
                return zero()
            r = g.resultant(g.derivative(var), var)
            dg = SR((-1) ** (n * (n - 1) // 2) * r / g.coefficient(var, n))
            return cancel(SR(dg / SR(den) ** (2 * n - 2)))
        except Exception as e:
            raise PolyBackendError(
                f"dgcv: discriminant failed for var={var!r} and expr type {type(expr).__name__}"
            ) from e

    raise PolyBackendError(
        "dgcv: no supported symbolic engine is available for polynomials"
    )


def make_poly(
    expr: Any, gens: Sequence[Any], *, parameters: Optional[Iterable[Any]] = None
) -> Any:
    kind = engine_kind()

    if kind == "sympy":
        gens_t = _stable_stamp(gens)
        if len(gens_t) == 0 and len(get_free_symbols(expr)) == 0:
            gens_t = (symbol("_dgcv_atom"),)
        sp = _get_sympy_module()
        try:
            return sp.Poly(expr, *gens_t)
        except Exception as e:
            raise PolyBackendError(
                f"dgcv: SymPy Poly construction failed for gens={gens_t!r} and expr type {type(expr).__name__}"
            ) from e

    if kind in ("sage", "builtin"):
        gens_t = _stable_stamp_by_str(gens) if kind == "sage" else tuple(gens)
        raw = _normalize_poly_expr(expr, assume_polynomial=False)
        _gens_out, monoms, coeffs = poly_terms(
            raw, gens_t, assume_polynomial=True, parameters=parameters
        )
        return _DGCVPolyTerms(
            expr=raw,
            gens=tuple(_gens_out),
            monoms=tuple(tuple(int(e) for e in m) for m in monoms),
            coeffs=tuple(coeffs),
        )

    raise PolyBackendError(
        "dgcv: no supported symbolic engine is available for polynomials"
    )


def poly_gens(P: Any) -> Tuple[Any, ...]:
    if isinstance(P, _DGCVPolyTerms):
        return tuple(P.gens)
    kind = engine_kind()

    if kind == "sympy":
        try:
            return tuple(P.gens)
        except Exception as e:
            raise PolyBackendError("dgcv: failed to read SymPy Poly generators") from e

    if kind == "sage":
        if isinstance(P, _DGCVPolyTerms):
            return tuple(P.gens)
        try:
            return tuple(P.parent().gens())
        except Exception as e:
            raise PolyBackendError(
                "dgcv: failed to read Sage polynomial generators"
            ) from e

    raise PolyBackendError(
        "dgcv: no supported symbolic engine is available for polynomials"
    )


def poly_monoms(P: Any) -> List[Tuple[int, ...]]:
    if isinstance(P, _DGCVPolyTerms):
        return [tuple(int(e) for e in m) for m in P.monoms]
    kind = engine_kind()

    if kind == "sympy":
        try:
            return [tuple(m) for m in P.monoms()]
        except Exception as e:
            raise PolyBackendError(
                "dgcv: failed to extract SymPy polynomial monomials"
            ) from e

    if kind == "sage":
        if isinstance(P, _DGCVPolyTerms):
            return [tuple(int(e) for e in m) for m in P.monoms]
        try:
            d = P.dict()
            items = list(d.items())
            items.sort(key=lambda kv: (sum(kv[0]), kv[0]))
            return [tuple(int(i) for i in k) for k, _ in items]
        except Exception as e:
            raise PolyBackendError(
                "dgcv: failed to extract Sage polynomial monomials"
            ) from e

    raise PolyBackendError(
        "dgcv: no supported symbolic engine is available for polynomials"
    )


def poly_coeffs(P: Any) -> List[Any]:
    if isinstance(P, _DGCVPolyTerms):
        return list(P.coeffs)
    kind = engine_kind()

    if kind == "sympy":
        try:
            return list(P.coeffs())
        except Exception as e:
            raise PolyBackendError(
                "dgcv: failed to extract SymPy polynomial coefficients"
            ) from e

    if kind == "sage":
        if isinstance(P, _DGCVPolyTerms):
            return list(P.coeffs)
        try:
            d = P.dict()
            items = list(d.items())
            items.sort(key=lambda kv: (sum(kv[0]), kv[0]))
            return [v for _, v in items]
        except Exception as e:
            raise PolyBackendError(
                "dgcv: failed to extract Sage polynomial coefficients"
            ) from e

    raise PolyBackendError(
        "dgcv: no supported symbolic engine is available for polynomials"
    )


def poly_total_degree(
    expr: Any, gens: Sequence[Any], *, parameters: Optional[Iterable[Any]] = None
) -> Optional[int]:
    kind = engine_kind()

    if kind == "sympy":
        sp = _get_sympy_module()
        gens_t = _stable_stamp(gens)
        try:
            return int(sp.total_degree(expr, *gens_t))
        except Exception:
            try:
                return int(sp.Poly(expr, *gens_t).total_degree())
            except Exception:
                return None

    if kind in ("sage", "builtin"):
        try:
            P = make_poly(expr, gens, parameters=parameters)
            monoms = poly_monoms(P)
            if not monoms:
                return 0
            return max(sum(int(e) for e in m) for m in monoms)
        except Exception:
            return None

    return None


def is_polynomial(
    expr: Any, gens: Sequence[Any], *, parameters: Optional[Iterable[Any]] = None
) -> bool:
    try:
        make_poly(expr, gens, parameters=parameters)
        return True
    except Exception:
        return False


def _ordered_factors(expr: Any) -> List[Any]:
    fl = getattr(expr, "factor_list", None)
    if callable(fl):
        try:
            out = [base for base, mult in fl() if mult > 0]
            if out:
                return out
        except Exception:
            pass

    aof = getattr(expr, "as_ordered_factors", None)
    if callable(aof):
        try:
            out = list(aof())
            return out if out else [expr]
        except Exception:
            pass

    try:
        if getattr(expr, "is_Mul", False):
            return list(getattr(expr, "args", ())) or [expr]
    except Exception:
        pass

    return [expr]


def _unwrap_pow(f: Any) -> Any:
    try:
        if getattr(f, "is_Pow", False):
            base = getattr(f, "base", None)
            exp = getattr(f, "exp", None)
            if base is not None and exp is not None:
                return base
    except Exception:
        pass

    try:
        if hasattr(f, "args") and len(f.args) == 2:
            base, _exp = f.args
            if str(type(f)).lower().find("pow") != -1:
                return base
    except Exception:
        pass

    return f


def poly_linear_roots_from_factorization(factored_poly: Any, var: Any) -> List[Any]:
    roots: List[Any] = []

    def _try_linear_root_from_poly(f: Any) -> Any:
        as_poly = getattr(f, "as_poly", None)
        if callable(as_poly):
            try:
                p = as_poly(var)
                if p is None:
                    return None
                deg = getattr(p, "degree", None)
                d = deg() if callable(deg) else None
                if d is None:
                    try:
                        d = p.degree()
                    except Exception:
                        d = None
                if d != 1:
                    return None
                all_coeffs = getattr(p, "all_coeffs", None)
                if callable(all_coeffs):
                    a, b = all_coeffs()
                else:
                    a, b = p.all_coeffs()
                if not _scalar_is_zero(a):
                    return -b / a
            except Exception:
                return None
        return None

    def _is_add_like(f: Any) -> bool:
        try:
            return bool(getattr(f, "is_Add", False))
        except Exception:
            return False

    def _try_linear_root_heuristic(f: Any) -> Any:
        try:
            args = getattr(f, "args", None)
            if args is not None and len(args) == 2:
                a0, a1 = args
                if a0 == var:
                    return -a1
                if a1 == var:
                    return -a0
        except Exception:
            pass

        if _is_add_like(f) or hasattr(f, "args"):
            try:
                terms = list(getattr(f, "args", ()))
                if not terms:
                    return None
            except Exception:
                return None

            var_term = None
            const_terms = []

            for t in terms:
                if t == var:
                    if var_term is None:
                        var_term = 1
                    else:
                        return None
                    continue

                try:
                    if getattr(t, "is_Mul", False):
                        mul_args = list(getattr(t, "args", ()))
                        if var in mul_args and mul_args.count(var) == 1:
                            coeff = 1
                            for u in mul_args:
                                if u == var:
                                    continue
                                coeff = coeff * u
                            if var_term is None:
                                var_term = coeff
                            else:
                                return None
                            continue
                except Exception:
                    pass

                const_terms.append(t)

            if var_term is None:
                return None

            b = 0 if not const_terms else const_terms[0]
            for u in const_terms[1:]:
                b = b + u

            try:
                if not _scalar_is_zero(var_term):
                    return -b / var_term
            except Exception:
                return None

        return None

    for f0 in _ordered_factors(factored_poly):
        f = _unwrap_pow(f0)

        r = _try_linear_root_from_poly(f)
        if r is None:
            r = _try_linear_root_heuristic(f)

        if r is not None:
            roots.append(r)

    out: List[Any] = []
    seen: set = set()
    for r in roots:
        k = str(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)

    return out


def _stable_stamp(seq: Iterable[Any]) -> Tuple[Any, ...]:
    out: List[Any] = []
    seen: set = set()
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return tuple(out)


def _stable_stamp_by_str(seq: Iterable[Any]) -> Tuple[Any, ...]:
    out: List[Any] = []
    seen: set = set()
    for x in seq:
        k = str(x)
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return tuple(out)


def groebner(exprs, syms, *, process_rationals=False, **kwargs):
    """
    Compute a Groebner basis for an ideal generated by symbolic expressions.

    Parameters
    ----------
    exprs : list
        Symbolic expressions generating the ideal, understood as equations set
        equal to zero.
    syms : sequence
        Symbols with respect to which the Groebner basis is computed.
    process_rationals : bool, optional
        If True, extract numerators via ``as_numer_denom`` before computing the
        basis. Useful when expressions are rational functions. Default is False.
    fail_quietly : bool, optional
        If True, return (a deduplicated copy of) the input expressions unchanged
        rather than raising on failure. Passed via ``**kwargs``. Default is False.
    **kwargs
        Additional keyword arguments forwarded to the underlying engine's
        Groebner basis routine.

    Returns
    -------
    list
        Basis elements of the Groebner basis, ordered according to the monomial
        order used by the active engine. Returns a deduplicated copy of ``exprs``
        if ``fail_quietly=True`` and computation fails.

    Raises
    ------
    PolyBackendError
        If the computation fails and ``fail_quietly=False``, or if no supported
        engine is available.
    """
    if kwargs.pop("bypass", False):
        return list(set(exprs))
    fail_quietly = kwargs.pop("fail_quietly", False)
    try:
        kind = engine_kind()

        if process_rationals:
            exprs = [as_numer_denom(e)[0] for e in exprs]
        from .._vmf.vmf import order_coordinates

        syms = order_coordinates(syms)
        if kind == "builtin":
            raise NotImplementedError(
                "`groebner` is not available in dgcv's builtin symbolic engine; install sympy or sage and set `default_engine` accordingly"
            )
        if kind == "sympy":
            sp = engine_module()
            try:
                result = sp.groebner(exprs, *syms, **kwargs)
                return list(result)
            except Exception as e:
                raise PolyBackendError(
                    "dgcv: SymPy Groebner basis computation failed"
                ) from e

        if kind == "sage":
            for expr in exprs:
                if not is_polynomial(expr, syms):
                    raise PolyBackendError(
                        f"dgcv: Sage Groebner basis requires polynomial expressions; "
                        f"got {type(expr).__name__}: {expr}"
                    )
            sage = engine_module()
            try:
                polys = [sage.SR(e) for e in exprs]
                ideal = sage.ideal(polys)
                return list(ideal.groebner_basis(**kwargs))
            except Exception as e:
                raise PolyBackendError(
                    "dgcv: Sage Groebner basis computation failed"
                ) from e
    except NotImplementedError:
        if fail_quietly:
            return list(set(exprs))
        raise
    except Exception:
        if fail_quietly:
            return list(set(exprs))
    raise PolyBackendError(
        "dgcv: no supported symbolic engine is available for Groebner basis computation"
    )


def expr_union_primitives(exprs, syms=None, *, process_rationals=False, **kwargs):
    """
    Reduce expressions defining a union of zero loci to irreducible primitives.

    Parameters
    ----------
    exprs : list
        Symbolic expressions defining the union of their zero loci.
    syms : sequence, optional
        Variables of interest. Factors not involving these are dropped.
        If None, any factor with free symbols is kept.
    process_rationals : bool, optional
        If True, extract numerators before factoring. Default is False.
    bypass : bool, optional
        If True, return input unchanged. Passed via ``**kwargs``.
    **kwargs
        Forwarded to engine routines.

    Returns
    -------
    list
        Irreducible factors with identical union of zero loci, with
        unprocessable expressions appended unchanged.
    """
    if kwargs.pop("bypass", False):
        return list(exprs)

    syms_set = set(syms) if syms is not None else None
    unprocessable = []
    processable = []

    for expr in exprs:
        if process_rationals:
            try:
                expr = as_numer_denom(expr)[0]
            except Exception:
                unprocessable.append(expr)
                continue
        try:
            primitive_factors = engine_capability("primitive_factors")
            if primitive_factors is not None:
                builtin_factors = primitive_factors(expr)
                if builtin_factors is not None:
                    factors = []
                    for f in builtin_factors:
                        free = get_free_symbols(f)
                        if free and (syms_set is None or free & syms_set):
                            factors.append(f)
                    if factors:
                        processable.extend(factors)
                    continue
            factored = factor_routed(expr)
            factors = []
            for f0 in _ordered_factors(factored):
                f = _unwrap_pow(f0)
                free = get_free_symbols(f)
                if free and (syms_set is None or free & syms_set):
                    factors.append(f)
            if factors:
                processable.extend(factors)
        except Exception:
            unprocessable.append(expr)

    seen = set()
    out = []
    for f in processable + unprocessable:
        k = _factor_key(f)
        if k not in seen:
            seen.add(k)
            out.append(f)

    return out


def _factor_key(f: Any) -> str:
    try:
        atoms = sorted(get_free_symbols(f), key=str)
        if not atoms:
            return str(f)
        _gens, monoms, coeffs = poly_terms(f, atoms)
        if not monoms or not coeffs:
            return str(f)
        terms = sorted(zip(monoms, coeffs), key=lambda item: item[0], reverse=True)
        lead = terms[0][1]
        if not lead:
            return str(f)
        normalized = []
        for monom, coeff in terms:
            try:
                ratio = coeff / lead
            except Exception:
                return str(f)
            normalized.append((tuple(monom), str(ratio)))
        return repr((tuple(str(a) for a in atoms), tuple(normalized)))
    except Exception:
        pass
    try:
        return min(str(f), str(-f))
    except Exception:
        return str(f)
