import numbers

from ..._aux._backends._symbolic_router import simplify, subs
from ..._aux._backends._types_and_constants import rational
from ._protocol import _is_scalar, _is_vs_element
from .promotion import _merge_normalized
from .shapes import _infer_shape


class _tp_arithmetic:
    def subs(self, subs_data):
        new_dict = {j: subs(k, subs_data) for j, k in self.coeff_dict.items()}
        return tensorProduct(new_dict, shape=self.shape)

    def simplify(self, **kwargs):
        new_dict = {j: simplify(k, **kwargs) for j, k in self.coeff_dict.items()}
        return tensorProduct(new_dict, shape=self.shape)

    def __dgcv_simplify__(self, *args, **kwargs):
        new_dict = {key: simplify(value, **kwargs) for key, value in self.coeff_dict.items()}
        return tensorProduct(new_dict, shape=self.shape)

    def _eval_simplify(self, *args, **kwargs):
        new_dict = {key: simplify(value, **kwargs) for key, value in self.coeff_dict.items()}
        return tensorProduct(new_dict, shape=self.shape)

    def _combine(self, other, sign, op):
        if _is_vs_element(other):
            other = other._convert_to_tp()
        elif _is_scalar(other):
            other = tensorProduct({tuple(): other})
        if not isinstance(other, tensorProduct):
            raise TypeError(
                f"`{op}` can only combine `tensorProduct` elements with elements from variants of dgcv vector space type classes."
            )
        new_shape = _infer_shape(self, other)
        state = _merge_normalized(self, other, sign, new_shape, True)
        return tensorProduct(None, shape=new_shape, _prebuilt=state)

    def __add__(self, other):
        return self._combine(other, 1, "+")

    def __radd__(self, other):
        if _is_scalar(other):
            return tensorProduct({tuple(): other}) + self

    def __sub__(self, other):
        return self._combine(other, -1, "-")

    def __rsub__(self, other):
        if _is_scalar(other):
            return tensorProduct({tuple(): other}) - self

    def __truediv__(self, other):
        if isinstance(other, numbers.Integral):
            return rational(1, other) * self
        if _is_scalar(other):
            return (1 / other) * self

    def __neg__(self):
        return -1 * self
