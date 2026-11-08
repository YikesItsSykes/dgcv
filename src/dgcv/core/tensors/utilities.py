from ..._aux._vmf._safeguards import get_dgcv_category
from ._protocol import _is_vs_element


def multi_tensor_product(*tp):
    product = 1
    for elem in tp:
        if get_dgcv_category(elem) != "tensorProduct" and not _is_vs_element(elem):
            raise TypeError(
                f"multi_tensor_product only excepts arguments that `dgcv` can process as factors in a tensor product. Recieved type {type(elem)}"
            )
        product = product @ elem
    return product
