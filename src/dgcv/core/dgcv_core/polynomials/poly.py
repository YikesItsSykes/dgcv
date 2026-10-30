from __future__ import annotations

from math import gcd
from numbers import Number
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from ...._aux._backends._calculus import diff, integrate
from ...._aux._backends._display import latex as _backend_latex
from ...._aux._backends._engine import engine_kind
from ...._aux._backends._polynomials import (
    _stable_stamp,
    make_poly,
    poly_coeffs,
    poly_gens,
    poly_monoms,
    poly_terms,
    poly_total_degree,
)
from ...._aux._backends._symbolic_router import (
    _scalar_is_zero,
    as_numer_denom,
    cancel,
    conjugate,
    expand,
    get_free_symbols,
    ilcm,
    im,
    ratio,
    re,
    simplify,
    subs,
)
from ...._aux._backends._symbolic_router import factor as factor_dgcv
from ...._aux._backends._types_and_constants import (
    check_dgcv_scalar,
    integer,
    is_atomic,
    one,
    verify_conjugate_re_im_free,
    zero,
)
from ...._aux._utilities._config import dgcv_warning
from ...._aux._vmf._safeguards import retrieve_passkey
from ...._aux._vmf.vmf import order_coordinates, vmf_lookup
from ...._aux.printing.printing import _unwrap_math_delims
from ...base import dgcv_class
from ...conversions.conversions import (
    allToHol,
    allToReal,
    allToSym,
    holToReal,
    realToHol,
    realToSym,
    symToHol,
    symToReal,
)
from ..coordinate_formats import conj_with_hol_coor
from .vmf_interface import _term_from_monom


class polynomial_dgcv(dgcv_class):
    """
    A class representing polynomial expressions in the dgcv package, providing a light,
    wrapper around the polynomial functionality of the active symbolic backend.

    Interprets a symbolic expression as a polynomial in a specified set of
    generators, with all other atomic symbols treated as parameters (unknown scalar
    constants).

    Parameters
    ----------
    polyExpr : symbolic expression
        The symbolic expression to be interpreted as a polynomial.

    varSpace : list or tuple, optional
        Variables to be treated as polynomial generators. If not provided, the generators
        are inferred from the free symbols of `polyExpr`.

    parameters : list or tuple, optional
        Variables to be treated as parameters (coefficients) rather than polynomial
        generators. If not provided and `varSpace` is specified, parameters are inferred.

    degreeUpperBound : int, optional
        An optional upper bound on the total degree of the polynomial, used in applications
        where only terms up to a given degree are relevant.

    Methods
    -------
    get_monomials(min_degree=0, max_degree=None, formatting='unformatted', return_coeffs=False)
        Returns the monomials (or coefficients) of the polynomial within the specified
        degree range and representation.

    holomorphic_part
        Returns the holomorphic part of the polynomial with respect to dgcv's complex
        variable systems.

    antiholomorphic_part
        Returns the antiholomorphic part of the polynomial.

    pluriharmonic_part
        Returns the pluriharmonic part of the polynomial, defined as the sum of the
        holomorphic and antiholomorphic parts with the constant term counted once.

    mixed_terms
        Returns the mixed terms of the polynomial, involving nonzero powers of both
        holomorphic and antiholomorphic variables.

    simplify_poly(method=None, **kwargs)
        Simplifies the underlying polynomial expression using the active symbolic backend.

    subs(substitutions)
        Substitutes variables or expressions into the polynomial and returns a new
        `polynomial_dgcv` instance.

    diff(*symbols)
        Differentiates the polynomial with respect to the given symbols and returns a new
        `polynomial_dgcv` instance.
    """

    def __init__(
        self,
        polyExpr: Any,
        varSpace: Optional[Sequence[Any]] = None,
        *,
        parameters: Optional[Iterable[Any]] = None,
        degreeUpperBound: Optional[int] = None,
        conjugate_free: Optional[bool] = None,
    ):
        src = polyExpr if isinstance(polyExpr, polynomial_dgcv) else None

        if src is not None:
            if varSpace is None:
                varSpace = src.varSpace
            if degreeUpperBound is None:
                degreeUpperBound = src.degreeUpperBound
            if parameters is None:
                parameters = src._parameters
            polyExpr = src.polyExpr

        hard_filter = tuple(parameters) if parameters is not None else tuple()
        self.polyExpr = (
            polyExpr
            if conjugate_free
            else (
                polyExpr
                if verify_conjugate_re_im_free(polyExpr)
                else allToSym(polyExpr, skipVar=hard_filter)
            )
        )

        if varSpace is None:
            self._parameters = hard_filter
            if isinstance(polyExpr, Number) and not isinstance(polyExpr, bool):
                self.coordinates = ()
            else:
                self.coordinates = order_coordinates(tuple(get_free_symbols(polyExpr)))
        else:
            self.coordinates = order_coordinates(
                polynomial_dgcv._normalize_polynomial_varspace_via_vmf(tuple(varSpace))
            )
            self._parameters = hard_filter if parameters is not None else None
        self.varSpace = self.coordinates

        self.degreeUpperBound = degreeUpperBound

        self._degree = None
        self._poly_obj_unformatted = None
        self._poly_obj_complex = None
        self._poly_obj_real = None
        self._valence_sorting = None
        self._terms = None
        self._holomorphic_part = None
        self._antiholomorphic_part = None
        self._pluriharmonic_part = None
        self._mixed_terms_part = None
        self._holomorphic_dominated_part = None
        self._antiholo_dominated_part = None
        self._balanced_terms = None
        self._is_zero = None
        self._is_one = None
        self._is_minus_one = None
        self._is_constant = None
        self._constant_term = None
        self._is_monomial = None
        self._complex_terms_cache = None
        self._complex_holo_anti_idx_cache = None

        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "expression"
        self._dgcv_categories = {"polynomial"}

        if src is not None:
            same_varspace = self.coordinates == src.coordinates
            same_deg_ub = self.degreeUpperBound == src.degreeUpperBound
            same_params_semantics = self._parameters == src._parameters

            if same_varspace and same_deg_ub and same_params_semantics:
                self._degree = src._degree
                self._poly_obj_unformatted = src._poly_obj_unformatted
                self._poly_obj_complex = src._poly_obj_complex
                self._poly_obj_real = src._poly_obj_real
                self._holomorphic_part = src._holomorphic_part
                self._antiholomorphic_part = src._antiholomorphic_part
                self._pluriharmonic_part = src._pluriharmonic_part
                self._mixed_terms_part = src._mixed_terms_part
                self._is_zero = src._is_zero
                self._is_one = src._is_one
                self._is_minus_one = src._is_minus_one
                self._is_constant = src._is_constant
                self._constant_term = src._constant_term
                self._is_monomial = src._is_monomial

    @property
    def parameters(self) -> Tuple[Any, ...]:
        if self._parameters is None:
            fs = tuple(get_free_symbols(self.polyExpr))
            vs = set(self.coordinates)
            self._parameters = tuple(x for x in _stable_stamp(fs) if x not in vs)
        return self._parameters

    @property
    def free_symbols(self) -> set:
        return get_free_symbols(self.polyExpr)

    @property
    def degree(self) -> Optional[int]:
        if self._degree is None:
            try:
                if engine_kind() == "sympy":
                    self._degree = poly_total_degree(
                        self.polyExpr, self.coordinates, parameters=self.parameters
                    )
                else:
                    monoms = poly_monoms(self.poly_obj_unformatted)
                    self._degree = max(
                        (sum(int(e) for e in m) for m in monoms), default=0
                    )
            except Exception:
                self._degree = None
        return self._degree

    @property
    def to_sym_engine_expr(self):
        return self.polyExpr

    @staticmethod
    def _vmf_coordinate_info(x: Any):
        try:
            info = vmf_lookup(x, relatives=True, system_index=True)
        except Exception:
            return None
        if not isinstance(info, dict):
            return None
        t = info.get("type", None)
        if t == "unregistered":
            return None
        if t != "coordinate":
            return None
        rel = info.get("relatives", None)
        if not isinstance(rel, dict):
            return None
        sys_label = rel.get("system_label", None)
        if not isinstance(sys_label, str) or not sys_label:
            return None
        si = info.get("system_index", None)
        try:
            si_int = int(si)
        except Exception:
            return None
        return sys_label, si_int, rel

    @staticmethod
    def _normalize_polynomial_varspace_via_vmf(
        varSpace_in: Sequence[Any],
    ) -> Tuple[Any, ...]:
        blocks: List[tuple] = []
        block_index_for_slot: Dict[tuple, int] = {}
        seen_standard: set = set()

        def _append_standard(a: Any) -> None:
            if a in seen_standard:
                return
            seen_standard.add(a)
            if blocks and blocks[0][0] == "standard":
                blocks[0][1].append(a)
            else:
                blocks.insert(0, ("standard", [a]))

        for v in varSpace_in:
            if not is_atomic(v):
                raise TypeError(
                    "polynomial_dgcv(varSpace=...): all entries must be atomic symbols; "
                    f"got {v!r} of type {type(v).__name__}"
                )

            info = polynomial_dgcv._vmf_coordinate_info(v)
            if info is None:
                _append_standard(v)
                continue

            sys_label, sys_i, rel = info

            holo = rel.get("holo", None)
            anti = rel.get("anti", None)
            real = rel.get("real", None)
            imag = rel.get("imag", None)

            has_symb_pair = (holo is not None) and (anti is not None)
            has_real_pair = (real is not None) and (imag is not None)
            if not (has_symb_pair or has_real_pair):
                _append_standard(v)
                continue

            if has_symb_pair and (v == holo or v == anti):
                fmt_seen = "symb"
                pair = (holo, anti)
            elif has_real_pair and (v == real or v == imag):
                fmt_seen = "real"
                pair = (real, imag)
            else:
                _append_standard(v)
                continue

            slot_key = (sys_label, sys_i)

            if slot_key not in block_index_for_slot:
                blocks.append(("complex", slot_key, fmt_seen, list(pair)))
                block_index_for_slot[slot_key] = len(blocks) - 1
                continue

            idx = block_index_for_slot[slot_key]
            _tag, _slot, fmt_current, _pair_current = blocks[idx]

            if fmt_current == fmt_seen:
                continue

            blocks[idx] = ("complex", slot_key, fmt_seen, list(pair))

        out: List[Any] = []
        seen_out: set = set()

        for blk in blocks:
            if blk[0] == "standard":
                for a in blk[1]:
                    if a in seen_out:
                        continue
                    seen_out.add(a)
                    out.append(a)
            else:
                _tag, _slot, _fmt, pair_list = blk
                for a in pair_list:
                    if a in seen_out:
                        continue
                    seen_out.add(a)
                    out.append(a)

        return tuple(out)

    def __dgcv_converter__(
        self,
        conv,
        *,
        skipVar=None,
        convert_everything=True,
        conversion_dict=None,
    ):
        conv_map = {
            "holToReal": holToReal,
            "realToSym": realToSym,
            "symToHol": symToHol,
            "realToHol": realToHol,
            "symToReal": symToReal,
            "allToReal": allToReal,
            "allToHol": allToHol,
            "allToSym": allToSym,
        }

        fn_expr = conv_map.get(conv)
        if fn_expr is None:
            return None

        if conv == "realToHol":
            fn_basis = realToSym
        elif conv == "allToHol":
            fn_basis = allToSym
        elif conv == "symToHol":
            fn_basis = None
        else:
            fn_basis = fn_expr

        kw_expr = {"skipVar": skipVar, "convert_everything": convert_everything}
        if conversion_dict is not None and conv in {
            "holToReal",
            "realToSym",
            "symToHol",
            "realToHol",
            "symToReal",
        }:
            kw_expr["_conversion_dict"] = conversion_dict

        new_expr = fn_expr(self.polyExpr, **kw_expr)

        if fn_basis is None:
            new_varSpace = self.coordinates
            new_params = None if self._parameters is None else self._parameters
        else:
            kw_basis = {"skipVar": skipVar, "convert_everything": convert_everything}

            vs_atoms = []
            for v in self.coordinates:
                vv = fn_basis(v, **kw_basis)
                vs_atoms.extend(get_free_symbols(vv))
            new_varSpace = tuple(_stable_stamp(vs_atoms))

            if self._parameters is None:
                new_params = None
            else:
                p_atoms = []
                for p in self._parameters:
                    pp = fn_basis(p, **kw_basis)
                    p_atoms.extend(get_free_symbols(pp))
                new_params = tuple(_stable_stamp(p_atoms))

        return polynomial_dgcv(
            new_expr,
            varSpace=new_varSpace,
            parameters=new_params,
            degreeUpperBound=self.degreeUpperBound,
        )

    def _complex_view(self, *, skipVar=None, convert_everything=True):
        expr_c = allToSym(
            self.polyExpr, skipVar=skipVar, convert_everything=convert_everything
        )

        gens = []
        for v in self.coordinates:
            vv = allToSym(v, skipVar=skipVar, convert_everything=convert_everything)
            gens.extend(get_free_symbols(vv))
        gens_c = _stable_stamp(gens)

        params_c = self.parameters
        return expr_c, gens_c, params_c

    def _real_view(self, *, skipVar=None, convert_everything=True):
        expr_r = allToReal(
            self.polyExpr, skipVar=skipVar, convert_everything=convert_everything
        )

        gens = []
        for v in self.coordinates:
            vv = allToReal(v, skipVar=skipVar, convert_everything=convert_everything)
            gens.extend(get_free_symbols(vv))
        gens_r = _stable_stamp(gens)

        params_r = self.parameters
        return expr_r, gens_r, params_r

    def __dgcv_simplify__(self, *, method=None, **kwargs):
        new_expr = simplify(self.polyExpr, method=method, **kwargs)
        return polynomial_dgcv(
            new_expr,
            varSpace=self.coordinates,
            parameters=self._parameters if self._parameters is not None else None,
            degreeUpperBound=self.degreeUpperBound,
        )

    def __dgcv_conjugate__(self, symbolic=False):
        conj = conjugate if symbolic is False else conj_with_hol_coor
        params = None if self._parameters is None else self._parameters
        return polynomial_dgcv(
            conj(self.polyExpr),
            varSpace=self.coordinates,
            parameters=params,
            degreeUpperBound=self.degreeUpperBound,
        )

    @property
    def poly_obj_unformatted(self):
        if self._poly_obj_unformatted is None:
            self._poly_obj_unformatted = make_poly(
                self.polyExpr, self.coordinates, parameters=self.parameters
            )
        return self._poly_obj_unformatted

    @property
    def poly_obj_complex(self):
        if self._poly_obj_complex is None:
            expr_c, gens_c, params_c = self._complex_view()
            self._poly_obj_complex = make_poly(expr_c, gens_c, parameters=params_c)
        return self._poly_obj_complex

    @property
    def poly_obj_real(self):
        if self._poly_obj_real is None:
            expr_r, gens_r, params_r = self._real_view()
            self._poly_obj_real = make_poly(expr_r, gens_r, parameters=params_r)
        return self._poly_obj_real

    @property
    def is_zero(self) -> bool:
        v = self._is_zero
        if v is not None:
            return v

        expr = self.polyExpr

        if isinstance(expr, Number) and not isinstance(expr, bool):
            self._is_zero = expr == 0
            return self._is_zero

        iz = getattr(expr, "is_zero", None)
        if iz is True:
            self._is_zero = True
            return True
        if callable(iz):
            try:
                self._is_zero = bool(iz())
                return self._is_zero
            except Exception:
                pass

        try:
            P = self.poly_obj_unformatted
            self._is_zero = all(c == 0 for c in poly_coeffs(P))
        except Exception:
            try:
                self._is_zero = expr == 0
            except Exception:
                self._is_zero = False

        return self._is_zero

    @property
    def is_constant(self) -> bool:
        v = self._is_constant
        if v is not None:
            return v

        try:
            P = self.poly_obj_unformatted
            self._is_constant = all(all(int(e) == 0 for e in m) for m in poly_monoms(P))
        except Exception:
            try:
                self._is_constant = len(self.coordinates) == 0
            except Exception:
                self._is_constant = False

        return self._is_constant

    @property
    def constant_term(self):
        v = self._constant_term
        if v is not None:
            return v

        expr = self.polyExpr

        try:
            P = self.poly_obj_unformatted
            for m, c in zip(poly_monoms(P), poly_coeffs(P)):
                if all(int(e) == 0 for e in m):
                    self._constant_term = c
                    return c
            self._constant_term = zero()
        except Exception:
            self._constant_term = expr

        return self._constant_term

    @property
    def is_one(self) -> bool:
        v = self._is_one
        if v is not None:
            return v

        expr = self.polyExpr

        if isinstance(expr, Number) and not isinstance(expr, bool):
            self._is_one = expr == 1
            return self._is_one

        try:
            self._is_one = bool(self.is_constant and (self.constant_term == 1))
        except Exception:
            try:
                self._is_one = expr == 1
            except Exception:
                self._is_one = False

        return self._is_one

    @property
    def is_minus_one(self) -> bool:
        v = self._is_minus_one
        if v is not None:
            return v

        expr = self.polyExpr

        if isinstance(expr, Number) and not isinstance(expr, bool):
            self._is_minus_one = expr == -1
            return self._is_minus_one

        try:
            self._is_minus_one = bool(self.is_constant and (self.constant_term == -1))
        except Exception:
            try:
                self._is_minus_one = expr == -1
            except Exception:
                self._is_minus_one = False

        return self._is_minus_one

    @property
    def is_monomial(self) -> bool:
        v = self._is_monomial
        if v is not None:
            return v

        try:
            coeffs = self.get_monomials(formatting="unformatted", return_coeffs=True)
            nz = 0
            for c in coeffs:
                if not _scalar_is_zero(c):
                    nz += 1
                    if nz > 1:
                        self._is_monomial = False
                        return False
            self._is_monomial = True
            return True
        except Exception:
            expr = self.polyExpr
            if isinstance(expr, Number) and not isinstance(expr, bool):
                self._is_monomial = True
                return True
            self._is_monomial = bool(getattr(expr, "is_Atom", False))
            return self._is_monomial

    def _complex_terms(self):
        """
        Return (gens, monoms, coeffs) for the complex view.
        """
        if self._complex_terms_cache is None:
            expr_c, gens_c, params_c = self._complex_view()

            gens = tuple(gens_c)

            _gens_out, monoms, coeffs = poly_terms(
                expr_c,
                gens,
                assume_polynomial=True,
                parameters=params_c,
            )

            self._complex_terms_cache = (tuple(_gens_out), monoms, coeffs)
        return self._complex_terms_cache

    def _complex_holo_anti_indices(self):
        if self._complex_holo_anti_idx_cache is None:
            gens, _monoms, _coeffs = self._complex_terms()

            holo_idx = []
            anti_idx = []
            standard_idx = []
            for i, g in enumerate(gens):
                g_info = vmf_lookup(g).get("sub_type")
                if g_info == "anti":
                    anti_idx.append(i)
                elif g_info == "holo":
                    holo_idx.append(i)
                else:
                    standard_idx.append(i)

            self._complex_holo_anti_idx_cache = (
                tuple(holo_idx),
                tuple(anti_idx),
                tuple(standard_idx),
            )
        return self._complex_holo_anti_idx_cache

    def get_monomials(
        self,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        *,
        formatting: Literal["unformatted", "complex", "real"] = None,
        return_coeffs: bool = False,
        separate_coeffs: bool = False,
        as_dict: bool = False,
    ):
        if formatting is None:
            formatting = "unformatted"
        elif formatting not in ("unformatted", "complex", "real"):
            dgcv_warning(
                "The `formatting` parameter was set to an unsupported value in `get_monomials`"
            )
            formatting = "unformatted"

        if as_dict:
            return_coeffs = False
            separate_coeffs = True
        elif separate_coeffs:
            return_coeffs = False

        if formatting == "unformatted":
            P = self.poly_obj_unformatted
        elif formatting == "complex":
            P = self.poly_obj_complex
        else:
            P = self.poly_obj_real

        gens_t = poly_gens(P)
        monoms = poly_monoms(P)
        coeffs = poly_coeffs(P)

        if max_degree is None:
            max_degree = self.degree
        if max_degree is None:
            max_degree = max(sum(int(e) for e in m) for m in monoms) if monoms else 0

        if return_coeffs:
            return [
                c
                for m, c in zip(monoms, coeffs)
                if min_degree <= sum(int(e) for e in m) <= max_degree
            ]

        if separate_coeffs:
            out_m = []
            out_c = []
            for m, c in zip(monoms, coeffs):
                d = sum(int(e) for e in m)
                if d < min_degree or d > max_degree:
                    continue
                mon = 1
                for g, e in zip(gens_t, m):
                    ee = int(e)
                    if ee:
                        mon *= g**ee
                out_m.append(mon)
                out_c.append(c)

            if as_dict:
                return dict(zip(out_m, out_c))

            return out_m, out_c

        out = []
        for m, c in zip(monoms, coeffs):
            d = sum(int(e) for e in m)
            if d < min_degree or d > max_degree:
                continue
            term = c
            for g, e in zip(gens_t, m):
                ee = int(e)
                if ee:
                    term *= g**ee
            out.append(term)

        return out

    def get_coeffs(
        self,
        min_degree: int = 0,
        max_degree: Optional[int] = None,
        *,
        formatting: str = "unformatted",
    ):
        return self.get_monomials(
            min_degree=min_degree,
            max_degree=max_degree,
            formatting=formatting,
            return_coeffs=True,
        )

    def _constant_term_in_complex_view(self):
        _gens, monoms, coeffs = self._complex_terms()
        for m, c in zip(monoms, coeffs):
            if all(int(e) == 0 for e in m):
                return c
        return zero()

    @property
    def _valence_sorted_parts(self):
        if self._valence_sorting is None:
            gens, monoms, coeffs = self._complex_terms()
            _holo_idx, anti_idx, standard_idx = self._complex_holo_anti_indices()
            terms = dict()
            holo_part = 0
            anti_part = 0
            mixed_terms = 0
            for m, c in zip(monoms, coeffs):
                h_degree = sum(m[i] for i in _holo_idx) if _holo_idx else 0
                a_degree = sum(m[i] for i in anti_idx) if anti_idx else 0
                s_degree = sum(m[i] for i in anti_idx) if standard_idx else 0
                key, new_term = (
                    (h_degree, a_degree, s_degree),
                    _term_from_monom(gens, m, c),
                )
                terms[key] = terms.get(key, 0) + new_term
                if h_degree == 0:
                    if s_degree == 0:
                        anti_part += new_term
                elif a_degree != 0 or s_degree != 0:
                    mixed_terms += new_term
                if a_degree == 0:
                    if s_degree == 0:
                        holo_part += new_term
                    else:
                        mixed_terms += new_term
            self._antiholomorphic_part = anti_part
            self._holomorphic_part = holo_part
            self._mixed_terms_part = mixed_terms
            self._valence_sorting = terms
        return self._valence_sorting

    @property
    def holomorphic_part(self):
        if self._holomorphic_part is None:
            _ = self._valence_sorted_parts
            # gens, monoms, coeffs = self._complex_terms()
            # _holo_idx, anti_idx = self._complex_holo_anti_indices()

            # terms = []
            # for m, c in zip(monoms, coeffs):
            #     if anti_idx and any(int(m[i]) != 0 for i in anti_idx):
            #         continue
            #     terms.append(_term_from_monom(gens, m, c))

            # self._holomorphic_part = sum(terms, zero()) if terms else zero()
        return self._holomorphic_part

    @property
    def antiholomorphic_part(self):
        if self._antiholomorphic_part is None:
            _ = self._valence_sorted_parts
            # gens, monoms, coeffs = self._complex_terms()
            # holo_idx, _anti_idx = self._complex_holo_anti_indices()

            # terms = []
            # for m, c in zip(monoms, coeffs):
            #     if holo_idx and any(int(m[i]) != 0 for i in holo_idx):
            #         continue
            #     terms.append(_term_from_monom(gens, m, c))

            # self._antiholomorphic_part = sum(terms, zero()) if terms else zero()
        return self._antiholomorphic_part

    @property
    def mixed_terms(self):
        if self._mixed_terms_part is None:
            _ = self._valence_sorted_parts
        #     gens, monoms, coeffs = self._complex_terms()
        #     holo_idx, anti_idx = self._complex_holo_anti_indices()

        #     terms = []
        #     for m, c in zip(monoms, coeffs):
        #         has_holo = holo_idx and any(int(m[i]) != 0 for i in holo_idx)
        #         has_anti = anti_idx and any(int(m[i]) != 0 for i in anti_idx)
        #         if has_holo and has_anti:
        #             terms.append(_term_from_monom(gens, m, c))

        #     self._mixed_terms_part = sum(terms, zero()) if terms else zero()
        return self._mixed_terms_part

    @property
    def pluriharmonic_part(self):
        if self._pluriharmonic_part is None:
            c0 = self._constant_term_in_complex_view()
            self._pluriharmonic_part = (
                self.holomorphic_part + self.antiholomorphic_part - c0
            )
        return self._pluriharmonic_part

    @property
    def holomorphic_dominated_part(self):
        if self._holomorphic_dominated_part is None:
            terms_dict = self._valence_sorted_parts
            h_part, a_part, b_part = 0, 0, 0
            for k, v in terms_dict.items():
                if k[0] == k[1]:
                    h_part += v
                    a_part += v
                    b_part += v
                elif k[0] > k[1]:
                    h_part += v
                else:
                    a_part += v
            self._holomorphic_dominated_part = h_part
            self._antiholo_dominated_part = a_part
            self._balanced_terms = b_part
        return self._holomorphic_dominated_part

    @property
    def antiholo_dominated_part(self):
        if self._antiholo_dominated_part is None:
            _ = self.holomorphic_dominated_part
        return self._antiholo_dominated_part

    @property
    def balanced_terms(self):
        if self._balanced_terms is None:
            _ = self.holomorphic_dominated_part
        return self._balanced_terms

    def homogeneous_terms(
        self,
        degree: int = None,
        holomorphic_degree: int = None,
        anti_holomorphic_degree: int = None,
    ):
        build_needed = False
        if holomorphic_degree is not None or anti_holomorphic_degree is not None:
            term_dict = self._valence_sorted_parts
        else:
            build_needed = True
            if degree is None:
                degree = self.get_degree()
            if self._terms is None:
                _, monom_idxs, coeffs = poly_terms(self.polyExpr, self.coordinates)
                self._terms = dict(zip(monom_idxs, coeffs))
            term_dict = self._terms

        def term_filter(idx):
            if degree is not None and sum(idx) != degree:
                return False
            if holomorphic_degree is not None and idx[0] != holomorphic_degree:
                return False
            if (
                anti_holomorphic_degree is not None
                and idx[1] != anti_holomorphic_degree
            ):
                return False
            return True

        if build_needed:
            out = 0
            for idx, coeff in term_dict.items():
                if term_filter(idx):
                    out += _term_from_monom(self.coordinates, idx, coeff)
            return out
        term_dict = {
            idxs: term for idxs, term in term_dict.items() if term_filter(idxs)
        }
        return sum(term_dict.values())

    def simplify_poly(self, method: Optional[str] = None, **kwargs):
        return polynomial_dgcv(
            simplify(self.polyExpr, method=method, **kwargs),
            varSpace=self.coordinates,
            parameters=self.parameters,
            degreeUpperBound=self.degreeUpperBound,
        )

    def evaluate(self, substitutions: dict):
        return subs(self.polyExpr, substitutions)

    def subs(self, substitutions, **kwargs):
        new_expr = subs(self.polyExpr, substitutions, **kwargs)
        return polynomial_dgcv(
            new_expr,
            varSpace=self.coordinates,
            parameters=self._parameters if self._parameters is not None else None,
            degreeUpperBound=self.degreeUpperBound,
        )

    def diff(self, *args, **kwargs):
        new_expr = diff(self.polyExpr, *args, **kwargs)
        return polynomial_dgcv(
            new_expr,
            varSpace=self.coordinates,
            parameters=self._parameters if self._parameters is not None else None,
            degreeUpperBound=self.degreeUpperBound,
        )

    def scale_to_have_int_coeffs(
        self,
        return_scale_only: bool = False,
        *,
        reduce_by_gcd: bool = True,
        balance_sign: bool = True,
    ):
        coeffs = list(self.get_coeffs())

        def _is_zero(x):
            z = getattr(x, "is_zero", None)
            if z is True:
                return True
            if callable(z):
                try:
                    return bool(z())
                except Exception:
                    pass
            return x == 0

        nz_coeffs = [c for c in coeffs if not _is_zero(c)]
        if not nz_coeffs:
            scale = one()
            return scale if return_scale_only else scale * self

        def _sign_real(x):
            if _is_zero(x):
                return 0
            neg = getattr(x, "is_negative", None)
            pos = getattr(x, "is_positive", None)
            if neg is True:
                return -1
            if pos is True:
                return 1
            try:
                v = float(x)
                return -1 if v < 0 else (1 if v > 0 else 0)
            except Exception:
                return 0

        def _sign_quadrant(c):
            if _is_zero(c):
                return 0

            rr = re(c)
            ii = im(c)

            rr0 = _is_zero(rr)
            ii0 = _is_zero(ii)

            if ii0:
                return _sign_real(rr)

            if rr0:
                return _sign_real(ii)

            rr_nonpos = getattr(rr, "is_nonpositive", None)
            ii_nonpos = getattr(ii, "is_nonpositive", None)
            rr_nonneg = getattr(rr, "is_nonnegative", None)
            ii_nonneg = getattr(ii, "is_nonnegative", None)

            rr_neg = getattr(rr, "is_negative", None)
            ii_neg = getattr(ii, "is_negative", None)
            rr_pos = getattr(rr, "is_positive", None)
            ii_pos = getattr(ii, "is_positive", None)

            neg_corner = (
                rr_nonpos is True
                and ii_nonpos is True
                and (rr_neg is True or ii_neg is True)
            )
            pos_corner = (
                rr_nonneg is True
                and ii_nonneg is True
                and (rr_pos is True or ii_pos is True)
            )

            if neg_corner:
                return -1
            if pos_corner:
                return 1

            try:
                zc = complex(float(rr), float(ii))
                if zc.real <= 0 and zc.imag <= 0 and (zc.real < 0 or zc.imag < 0):
                    return -1
                if zc.real >= 0 and zc.imag >= 0 and (zc.real > 0 or zc.imag > 0):
                    return 1
            except Exception:
                pass

            return 0

        denoms = []
        for c in nz_coeffs:
            _, d = as_numer_denom(c)
            try:
                di = int(d)
            except Exception:
                continue
            if not _scalar_is_zero(di):
                denoms.append(abs(di))

        base = ilcm(*denoms) if denoms else one()
        base = integer(base) if base != one() else base

        if not reduce_by_gcd and not balance_sign:
            scale = base
            return scale if return_scale_only else scale * self

        scaled_coeffs = [base * c for c in nz_coeffs]

        def _as_int(x):
            if _is_zero(x):
                return 0
            try:
                xi = int(x)
            except Exception:
                return None
            try:
                if x == xi:
                    return xi
            except Exception:
                pass
            return None

        ints = []
        for c in scaled_coeffs:
            ci = _as_int(c)
            if ci is None:
                ints = []
                break
            if not _scalar_is_zero(ci):
                ints.append(abs(ci))

        g = 1
        if reduce_by_gcd and ints:
            g = ints[0]
            for u in ints[1:]:
                g = gcd(g, u)
                if g == 1:
                    break

        content = integer(g) if g != 1 else one()
        scale0 = base / content if content != one() else base

        if not balance_sign:
            scale = scale0
            return scale if return_scale_only else scale * self

        def _sign_count_after_scale(s):
            pos = 0
            neg = 0
            for c in nz_coeffs:
                q = _sign_quadrant(s * c)
                if q == 1:
                    pos += 1
                elif q == -1:
                    neg += 1
            return pos, neg

        pos0, neg0 = _sign_count_after_scale(scale0)
        pos1, neg1 = _sign_count_after_scale(-scale0)

        if pos1 > pos0 or (pos1 == pos0 and neg1 < neg0):
            scale = -scale0
        else:
            scale = scale0

        return scale if return_scale_only else scale * self

    def __add__(self, other):
        if _scalar_is_zero(other):
            return self
        if isinstance(other, polynomial_dgcv):
            new_vs = _stable_stamp(self.coordinates + other.coordinates)
            return polynomial_dgcv(
                self.polyExpr + other.polyExpr,
                varSpace=new_vs,
                parameters=None,
                degreeUpperBound=self.degreeUpperBound,
            )
        if check_dgcv_scalar(other):
            inferred_param = self._parameters if self._parameters is not None else None
            OIP = get_free_symbols(other) - set(self.coordinates)
            if OIP:
                inferred_param = (
                    tuple(set(inferred_param) | OIP) if inferred_param else tuple(OIP)
                )
            return polynomial_dgcv(
                self.polyExpr + other,
                varSpace=self.coordinates,
                parameters=inferred_param,
                degreeUpperBound=self.degreeUpperBound,
            )
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, polynomial_dgcv):
            return self.polyExpr / other.polyExpr
        if check_dgcv_scalar(other):
            return ratio(1, other) * self

    def __radd__(self, other):
        if _scalar_is_zero(other):
            return self
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, polynomial_dgcv):
            new_vs = _stable_stamp(self.coordinates + other.coordinates)
            return polynomial_dgcv(
                self.polyExpr - other.polyExpr,
                varSpace=new_vs,
                parameters=None,
                degreeUpperBound=self.degreeUpperBound,
            )
        if check_dgcv_scalar(other):
            inferred_param = self._parameters if self._parameters is not None else None
            OIP = get_free_symbols(other) - set(self.coordinates)
            if OIP:
                inferred_param = (
                    tuple(set(inferred_param) | OIP) if inferred_param else tuple(OIP)
                )
            return polynomial_dgcv(
                self.polyExpr - other,
                varSpace=self.coordinates,
                parameters=inferred_param,
                degreeUpperBound=self.degreeUpperBound,
            )
        return NotImplemented

    def __rsub__(self, other):
        if check_dgcv_scalar(other):
            inferred_param = self._parameters if self._parameters is not None else None
            OIP = get_free_symbols(other) - set(self.coordinates)
            if OIP:
                inferred_param = (
                    tuple(set(inferred_param) | OIP) if inferred_param else tuple(OIP)
                )
            return polynomial_dgcv(
                other - self.polyExpr,
                varSpace=self.coordinates,
                parameters=inferred_param,
                degreeUpperBound=self.degreeUpperBound,
            )
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, polynomial_dgcv):
            new_vs = _stable_stamp(self.coordinates + other.coordinates)

            if self.degreeUpperBound is None and other.degreeUpperBound is None:
                new_bound = None
            else:
                vals = [
                    v
                    for v in (self.degreeUpperBound, other.degreeUpperBound)
                    if v is not None
                ]
                new_bound = min(vals) if vals else None

            return polynomial_dgcv(
                self.polyExpr * other.polyExpr,
                varSpace=new_vs,
                parameters=None,
                degreeUpperBound=new_bound,
            )

        if check_dgcv_scalar(other):
            inferred_param = self._parameters if self._parameters is not None else None
            OIP = get_free_symbols(other) - set(self.coordinates)
            if OIP:
                inferred_param = (
                    tuple(set(inferred_param) | OIP) if inferred_param else tuple(OIP)
                )
            return polynomial_dgcv(
                self.polyExpr * other,
                varSpace=self.coordinates,
                parameters=inferred_param,
                degreeUpperBound=self.degreeUpperBound,
            )
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __dgcv_apply__(self, fun, **kwargs):
        new_expr = fun(self.polyExpr, **kwargs)
        if engine_kind() == "sympy":
            return polynomial_dgcv(
                make_poly(
                    new_expr, self.coordinates, parameters=self.parameters
                ).as_expr(),
                varSpace=self.coordinates,
                parameters=self._parameters,
                degreeUpperBound=self.degreeUpperBound,
            )
        return polynomial_dgcv(
            new_expr,
            varSpace=self.coordinates,
            parameters=self._parameters if self._parameters is not None else None,
            degreeUpperBound=self.degreeUpperBound,
        )

    def __dgcv_expand__(self, **kwargs):
        return self.expand(**kwargs)

    def expand(self, **kwargs):
        if engine_kind() == "sympy":
            return polynomial_dgcv(
                make_poly(
                    self.polyExpr, self.coordinates, parameters=self.parameters
                ).as_expr(),
                varSpace=self.coordinates,
                parameters=self._parameters,
                degreeUpperBound=self.degreeUpperBound,
            )
        return polynomial_dgcv(
            expand(self.polyExpr, **kwargs),
            varSpace=self.coordinates,
            parameters=self._parameters if self._parameters is not None else None,
            degreeUpperBound=self.degreeUpperBound,
        )

    def factor(self, **kwargs):
        new_expr = factor_dgcv(self.polyExpr, **kwargs)
        return polynomial_dgcv(
            new_expr,
            varSpace=self.coordinates,
            parameters=self._parameters if self._parameters is not None else None,
            degreeUpperBound=self.degreeUpperBound,
        )

    def cancel(self, **kwargs):
        new_expr = cancel(self.polyExpr, **kwargs)
        return polynomial_dgcv(
            new_expr,
            varSpace=self.coordinates,
            parameters=self._parameters if self._parameters is not None else None,
            degreeUpperBound=self.degreeUpperBound,
        )

    def integrate(self, *args, **kwargs):
        new_expr = integrate(self.polyExpr, *args, **kwargs)
        return polynomial_dgcv(
            new_expr,
            varSpace=self.coordinates,
            parameters=self._parameters if self._parameters is not None else None,
            degreeUpperBound=self.degreeUpperBound,
        )

    def get_degree(self):
        return self.degree

    def is_homogeneous(self, *, formatting: str = "unformatted") -> bool:
        P = (
            self.poly_obj_unformatted
            if formatting == "unformatted"
            else (
                self.poly_obj_complex if formatting == "complex" else self.poly_obj_real
            )
        )
        monoms = poly_monoms(P)
        if not monoms:
            return True
        degs = {sum(int(e) for e in m) for m in monoms}
        return len(degs) <= 1

    def leading_term(self, *, formatting: str = "unformatted"):
        terms = self.get_monomials(formatting=formatting, return_coeffs=False)
        if not terms:
            return zero()

        def _deg_of_term(t):
            try:
                p = polynomial_dgcv(
                    t, varSpace=self.coordinates, parameters=self.parameters
                )
                d = p.degree
                return -1 if d is None else int(d)
            except Exception:
                return -1

        return max(terms, key=_deg_of_term)

    def latex_representation(self, removeBARs: bool = False) -> str:
        x = getattr(self, "polyExpr", None)
        if x is None:
            return ""

        if not removeBARs:
            try:
                x = symToHol(x, convert_everything=False)
            except Exception:
                pass

        s = _backend_latex(x)
        if isinstance(s, str) and s.strip():
            return _unwrap_math_delims(s)

        return str(x)

    def _repr_latex_(self, raw: bool = False, removeBARs: bool = False, **kwargs):
        s = self.latex_representation(removeBARs=removeBARs)
        s = _unwrap_math_delims(s)

        if raw:
            return s
        return f"$\\displaystyle {s}$"

    def _latex(
        self, printer=None, raw: bool = True, removeBARs: bool = False, **kwargs
    ):
        return self._repr_latex_(raw=raw, removeBARs=removeBARs, **kwargs)

    def __str__(self):
        try:
            return str(self.polyExpr)
        except Exception:
            return self.__repr__()

    def pretty_print(self):
        f = getattr(self.polyExpr, "pretty", None)
        if callable(f):
            try:
                return f()
            except Exception:
                pass
        return str(self.polyExpr)
