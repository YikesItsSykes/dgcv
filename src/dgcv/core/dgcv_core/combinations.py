from ..._aux._backends._symbolic_router import _scalar_is_zero, simplify
from ..._aux._backends._types_and_constants import check_dgcv_scalar
from ..._aux._vmf._safeguards import get_dgcv_category, query_dgcv_categories
from .fields import vector_field_class


def VF_bracket(X, Y, *, doNotSimplify: bool = False, **_ignored):
    if not query_dgcv_categories(X, {"vector_field"}):
        raise TypeError(
            f"VF_bracket expects X to be a vector field, got {type(X).__name__}."
        )
    if not query_dgcv_categories(Y, {"vector_field"}):
        raise TypeError(
            f"VF_bracket expects Y to be a vector field, got {type(Y).__name__}."
        )

    cd_X = getattr(X, "coeff_dict", None)
    cd_Y = getattr(Y, "coeff_dict", None)
    if not isinstance(cd_X, dict) or not isinstance(cd_Y, dict):
        raise TypeError("VF_bracket expects vector fields with a coeff_dict attribute.")

    merged_vs = None
    mv = getattr(X, "_merged_variable_spaces", None)
    if callable(mv):
        merged_vs = mv(Y)
    else:
        merged_vs = getattr(X, "_variable_spaces", None)

    out = {}

    for k, v in cd_Y.items():
        if not _scalar_is_zero(v):
            w = X(v)
            if not _scalar_is_zero(w):
                out[k] = out.get(k, 0) + w

    for k, v in cd_X.items():
        if not _scalar_is_zero(v):
            w = Y(v)
            if not _scalar_is_zero(w):
                out[k] = out.get(k, 0) - w

    if not out:
        out = {tuple(): 0}
    else:
        if not doNotSimplify:
            for k, v in list(out.items()):
                if not _scalar_is_zero(v):
                    out[k] = simplify(v)
        out = {k: v for k, v in out.items() if not _scalar_is_zero(v)}
        if not out:
            out = {tuple(): 0}

    out_cls = X.__class__ if (X.__class__ is Y.__class__) else vector_field_class

    return out_cls(
        coeff_dict=out,
        data_shape="all",
        dgcvType=getattr(X, "dgcvType", "standard"),
        _simplifyKW=getattr(X, "_simplifyKW", None),
        variable_spaces=merged_vs,
    )


def tensor_product(*args, doNotSimplify=False):
    """
    Computes the tensor product of tensor_field_class instances by dispatching the tensor_field_class product method.
    """
    if not all(
        check_dgcv_scalar(arg) or get_dgcv_category(arg) == "tensor_field"
        for arg in args
    ):
        bad_types = []
        for arg in args:
            if not (check_dgcv_scalar(arg) or get_dgcv_category(arg) == "tensor_field"):
                bad_types += [type(arg)]
        bad_types = list(set(bad_types))
        bt_str = ", ".join(bad_types)
        raise Exception(
            f"Expected all arguments to be instances of tensorField or scalar-like objects, not type: {bt_str}"
        )
    if len(args) == 0:
        return
    if len(args) == 1:
        return args[0]
    if check_dgcv_scalar(args[0]):
        return args[0] * tensor_product(*args[1:])
    if len(args) == 2:
        return args[0].tensor_product(args[1])
    return tensor_product(args[0].tensor_product(args[1]), *args[2:])


def exteriorProduct(*args):
    if len(args) == 0:
        raise TypeError("exteriorProduct expects at least one argument.")

    for a in args:
        if not query_dgcv_categories(a, {"differential_form"}):
            raise TypeError(
                "exteriorProduct expects differential_form objects. "
                "Use dgcv creator functions to build differential forms."
            )

    out = args[0]
    for a in args[1:]:
        out = out._shape_product(a, kind="skew")
    return out


def wedge(*tfs):
    types = {get_dgcv_category(tf) for tf in tfs if not check_dgcv_scalar(tf)}
    acceptables = {"algebra_element", "subalgebra_element", "tensorProduct"}
    if types != {"tensor_field"} and not all(
        dgcv_type in acceptables for dgcv_type in types
    ):
        raise TypeError(
            "`wedge` only operates on scalars and dgcv tensor field, tensor algebra, and vector space classes. The field and vector space classes moreover cannot mix."
        )
    if len(tfs) == 0:
        return
    if len(tfs) == 1:
        return tfs[0]
    if len(tfs) > 2:
        return wedge(wedge(tfs[0], tfs[1]), *tfs[2:])
    tf1, tf2 = tfs
    if check_dgcv_scalar(tf1) or check_dgcv_scalar(tf2):
        return tf1 * tf2
    elif get_dgcv_category(tf1) == "tensor_field":
        return tf1.skew_product(tf2)
    elif get_dgcv_category(tf1) in acceptables:
        tf1 = tf1._convert_to_tp()
        return tf1.shape_inferred_tensor_product(tf2, impose_shape="skew")
    else:
        raise TypeError(
            "`wedge` only operates on dgcv tensor field, tensor algebra, and vector space classes."
        )


def symmetric_product(*tfs):
    types = {get_dgcv_category(tf) for tf in tfs if not check_dgcv_scalar(tf)}
    acceptables = {
        "algebra_element",
        "subalgebra_element",
        "tensorProduct",
    }
    if types != {"tensor_field"} and not all(
        dgcv_type in acceptables for dgcv_type in types
    ):
        raise TypeError(
            "`symmetric_product` only operates on scalars and dgcv tensor field, tensor algebra, and vector space classes. The field and vector space classes moreover cannot mix."
        )
    if len(tfs) == 0:
        return
    if len(tfs) == 1:
        return tfs[0]
    if len(tfs) > 2:
        return symmetric_product(symmetric_product(tfs[0], tfs[1]), *tfs[2:])
    tf1, tf2 = tfs
    if check_dgcv_scalar(tf1) or check_dgcv_scalar(tf2):
        return tf1 * tf2
    elif get_dgcv_category(tf1) == "tensor_field":
        return tf1.symmetric_product(tf2)
    elif get_dgcv_category(tf1) in acceptables:
        tf1 = tf1._convert_to_tp()
        return tf1.shape_inferred_tensor_product(tf2, impose_shape="symmetric")
    else:
        raise TypeError(
            "`symmetric_product` only operates on dgcv tensor field, tensor algebra, and vector space classes."
        )


def sum_dgcv(terms, start=0):
    """
    Sum dgcv objects, using a class-provided accumulator when one is available.

    Parameters
    ----------
    terms : iterable
        Objects to sum.
    start : optional
        Initial value included in the sum. Default is 0.

    Returns
    -------
    object
        The sum of `start` and the elements of `terms`.
    """
    if not isinstance(terms, (list, tuple)):
        terms = list(terms)
    if not terms:
        return start
    hook = getattr(type(terms[0]), "_dgcv_multiadd", None)
    if hook is None:
        return sum(terms, start)
    return hook(terms, start)
