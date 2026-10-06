# SPDX-License-Identifier: Apache-2.0
# SPDX-AI-Disclosure: ai-generated
# The above markers pertain to all code within the dgcv._symbolic_scalars subpackage

import random
from fractions import Fraction

from .._aux._backends._symbolic_router import _scalar_is_zero
from .._aux._backends._types_and_constants import exact_fraction
from ..algebras.threads.util import _exact_builtin_paths
from . import _poly
from ._normal_form import _nf as _builtin_nf

_MODULAR_SEED = 24301
_MODULAR_ATTEMPTS = 4
_modular_rng = random.Random(_MODULAR_SEED)


def _modular_pick(prepared):
    if not prepared:
        return None
    alg = getattr(prepared[0][1], "algebra", None)
    dim = getattr(alg, "dimension", None)
    if alg is None or not isinstance(dim, int):
        return None
    vectors = []
    symbolic = False
    gens = set()

    for _, elem in prepared:
        cd = getattr(elem, "coeff_dict", None)
        if not isinstance(cd, dict) or getattr(elem, "algebra", None) is not alg:
            return None
        vec = {}
        for row, value in cd.items():
            if not isinstance(row, int) or row < 0 or row >= dim:
                return None
            if _scalar_is_zero(value):
                continue
            fr = exact_fraction(value)
            if fr is not None:
                vec[row] = fr
                continue

            rf = _builtin_nf(value)
            if rf is None:
                return None
            if _poly.p_has_float(rf.num) or _poly.p_has_float(rf.den):
                return None
            symbolic = True
            gens |= rf.gens()
            vec[row] = rf

        vectors.append(vec)

    if not symbolic:
        return None
    if _poly._RELATED and _poly.has_relation(gens):
        return None
    prime = _poly._COPRIME_PRIME
    for _attempt in range(_MODULAR_ATTEMPTS):
        point = {g: _modular_rng.randrange(2, prime - 1) for g in gens}
        point[_poly.I_GEN] = _poly._COPRIME_I
        images = []
        ok = True

        for vec in vectors:
            img = {}
            for row, value in vec.items():
                if isinstance(value, Fraction):
                    d = value.denominator % prime
                    if d == 0:
                        ok = False
                        break

                    v = value.numerator % prime * pow(d, prime - 2, prime) % prime
                else:
                    n = _poly.p_eval_mod(value.num, point, prime)
                    d = _poly.p_eval_mod(value.den, point, prime)
                    if n is None or d is None:
                        return None
                    if d == 0:
                        ok = False
                        break

                    v = n * pow(d, prime - 2, prime) % prime

                if v:
                    img[row] = v

            if not ok:
                break
            images.append(img)

        if ok:
            break
    else:
        return None

    rows = []
    pivots = []
    kept = []

    for pos, img in enumerate(images):
        b = dict(img)
        for a, piv in enumerate(pivots):
            c = b.get(piv)
            if c is None:
                continue
            for j, v in rows[a].items():
                nv = (b.get(j, 0) - c * v) % prime
                if nv:
                    b[j] = nv
                else:
                    b.pop(j, None)

        if not b:
            continue
        piv = next(iter(b))
        inv = pow(b[piv], prime - 2, prime)
        b = {j: v * inv % prime for j, v in b.items()}
        rows.append(b)
        pivots.append(piv)
        kept.append(pos)

    return kept


def basis_pick(prepared):
    if _exact_builtin_paths():
        return None
    return _modular_pick(prepared)
