from ..._aux._backends._types_and_constants import expr_numeric_types
from ..._aux._vmf._safeguards import query_dgcv_categories


def _is_vs_element(obj):
    return bool(query_dgcv_categories(obj, "vs_element"))


def _is_scalar(obj):
    return isinstance(obj, expr_numeric_types()) or bool(
        query_dgcv_categories(obj, "eds_scalar")
    )


def _as_tp(obj):
    if _is_vs_element(obj) or query_dgcv_categories(obj, "eds_form"):
        return obj._convert_to_tp()
    return obj


def _vs_bracket(a, b):
    hook = getattr(a, "bracket", None)
    if callable(hook):
        return hook(b)
    return a * b
