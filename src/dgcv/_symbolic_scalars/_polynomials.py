# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

import numbers

from .._aux._backends._polynomials import PolyBackendError
from ..eds._zero_forms import _as_leaf, zero_form_class
from . import _factor, _poly
from ._normal_form import RF, _nf, nf_from_base
from ._normal_form import poly_terms as _rf_terms
from ._poly import _prim


def poly_terms(raw, gens):
    gens_t = tuple(gens)
    leaves = []
    for g in gens_t:
        leaf = _as_leaf(g)
        if leaf is None:
            raise PolyBackendError(
                f"dgcv: builtin polynomial generators must be zero-form atoms, received {type(g).__name__}"
            )

        leaves.append(leaf)

    rf = nf_from_base(raw)
    terms = _rf_terms(rf, [_poly.gen_id(leaf) for leaf in leaves])
    if terms is None:
        raise PolyBackendError(
            "dgcv: the builtin engine received a non-polynomial expression for polynomial term extraction"
        )

    monoms = [tuple(int(e) for e in exps) for exps, _ in terms]
    coeffs = []
    for _, c in terms:
        zf = zero_form_class._from_nf(c)
        coeffs.append(zf.base if not isinstance(zf.base, tuple) else zf)

    return gens_t, monoms, coeffs


_FACTOR_TERM_LIMIT = 200


def primitive_factors(expr):
    if isinstance(expr, numbers.Number):
        return []
    rf = _nf(expr)
    if rf is None:
        return None
    p = rf.num
    if not p or _poly.p_is_const(p) or _poly.p_has_float(p):
        return []
    mono = _poly.p_monomial_content(p)
    factors = [_poly.p_gen(g) for g, _e in mono if g != _poly.I_GEN]
    q, _ = _prim(_poly.p_divide_monomial(p, mono) if mono else p)
    gens = sorted(_poly.p_gens(q) - {_poly.I_GEN})
    if len(q) > _FACTOR_TERM_LIMIT:
        factors.append(q)
    elif gens:
        if _poly._holds_i(q):
            q, _ = _prim(_poly._gaussian_primitive(q)[0])

        factors.extend(f for f, _ in _factor._split_factors(q, gens))

    return [zero_form_class._from_nf(RF(f)) for f in factors]
