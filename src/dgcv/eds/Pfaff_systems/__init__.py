"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds.Pfaff_systems


Description:
------------
Pfaff systems on coframes and jet spaces: polar systems, characters and genre, a
Cartan involutivity test, and the Pfaff systems of systems of PDE systems.
The underlying mathematics follows Chapter 5 of

    Alekseevskij, D. V., Vinogradov, A. M. and Lychagin, V. V.,
    Geometry I: Basic Ideas and Concepts of Differential Geometry,
    Encyclopaedia of Mathematical Sciences, vol. 28, Springer-Verlag, 1991,

for the Cartan involutivity test in particular, and the count of parameterizing data of
Pfaff systems' integral submanifolds comes from

    Bryant, R. L., Chern, S. S., Gardner, R. B., Goldschmidt, H. L. and Griffiths, P. A.,
    Exterior Differential Systems,
    Mathematical Sciences Research Institute Publications, vol. 18, Springer-Verlag, 1991.

---
Author (of this sub-package): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from . import _pfaff
from ._jets import createJetSpace, jet_restriction_data, jetSpace
from ._pfaff import PfaffSystem

_pfaff.jetSpace = jetSpace

__all__ = [
    "PfaffSystem",
    "createJetSpace",
    "jetSpace",
    "jet_restriction_data",
]
