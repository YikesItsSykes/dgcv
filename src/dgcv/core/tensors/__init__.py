"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.core.tensors

module: dgcv.core.tensors


---
Author (of this sub-package): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from . import (
    tp_arithmetic,
    tp_brackets,
    tp_contraction,
    tp_contraction_product,
    tp_core,
    tp_evaluation,
    tp_products,
    tp_weights,
)
from .tp_arithmetic import _tp_arithmetic
from .tp_brackets import _tp_brackets
from .tp_contraction import _tp_contraction
from .tp_contraction_product import _tp_contraction_product
from .tp_core import _tp_core
from .tp_evaluation import _tp_evaluation
from .tp_printing import _tp_printing
from .tp_products import _tp_products
from .tp_weights import _tp_weights
from .utilities import multi_tensor_product

__all__ = ["multi_tensor_product", "tensorProduct"]


class tensorProduct(
    _tp_core,
    _tp_weights,
    _tp_printing,
    _tp_arithmetic,
    _tp_products,
    _tp_contraction,
    _tp_contraction_product,
    _tp_evaluation,
    _tp_brackets,
):
    pass


from .lazy import _lazy_class

lazy_tensorProduct = _lazy_class(tensorProduct)

tp_core.tensorProduct = tensorProduct
tp_weights.tensorProduct = tensorProduct
tp_arithmetic.tensorProduct = tensorProduct
tp_products.tensorProduct = tensorProduct
tp_contraction.tensorProduct = tensorProduct
tp_contraction_product.tensorProduct = tensorProduct
tp_evaluation.tensorProduct = tensorProduct
tp_brackets.tensorProduct = tensorProduct
