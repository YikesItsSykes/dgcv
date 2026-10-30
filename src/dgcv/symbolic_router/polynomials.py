"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.symbolic_router

module: dgcv.symbolic_router.polynomials


---
Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from .._aux._backends._groebner import groebner
from .._aux._backends._polynomials import (
    discriminant,
    extract_polynomial_coeffs,
    is_polynomial,
    make_poly,
    poly_coeffs,
    poly_gens,
    poly_linear_roots_from_factorization,
    poly_monoms,
    poly_terms,
    poly_total_degree,
)

__all__ = [
    "discriminant",
    "extract_polynomial_coeffs",
    "groebner",
    "is_polynomial",
    "make_poly",
    "poly_coeffs",
    "poly_gens",
    "poly_linear_roots_from_factorization",
    "poly_monoms",
    "poly_terms",
    "poly_total_degree",
]
