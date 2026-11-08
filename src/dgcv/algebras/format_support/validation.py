import numbers

from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    get_free_symbols,
    simplify,
)
from ..._aux._backends._types_and_constants import is_atomic
from ..._aux._utilities._config import dgcv_warning
from ..._aux._vmf._safeguards import get_dgcv_category, query_dgcv_categories
from ...core.arrays import matrix_dgcv
from ..linear_algebra import _structure_array
from .dense_lists import aDataFromNestedLists
from .matrices import algebraDataFromMatRep
from .sparse_dicts import (
    _complete_skew,
    _normalize_sparse_structure,
    _structure_data_from_array,
)
from .tensors import algebraDataFromTensorRep
from .vector_fields import aDataFromVFWithAnsatz


def _validate_structure_data(
    data,
    process_matrix_rep=False,
    assume_skew=False,
    basis_order_for_supplied_str_eqns=None,
    process_tensor_rep=False,
    determinacy_order_ansatz=None,
    dimension=None,
    process_with_decompose=False,
):
    if process_tensor_rep:
        # try:
        return algebraDataFromTensorRep(data), "tensor"
        # except Exception as e:
        #     raise dgcv_exception_note(f"{e}") from None
    if process_matrix_rep:
        mats, warningmessage = [], False
        for obj in data:
            if isinstance(obj, matrix_dgcv):
                m = obj
            try:
                m = matrix_dgcv(obj)
            except Exception:
                m = None
            if m is None or m.nrows != m.ncols:
                mats = None
                if warningmessage is False:
                    warningmessage = "Some basis objects could not be interpreted as square matrices for a matrix algebra interpretation, and were discarded."
                break
            mats.append(m)
        if warningmessage is not False:
            dgcv_warning(warningmessage)
        if mats is not None:
            # try:
            return algebraDataFromMatRep(data), "matrix"
            # except Exception as e:
            #     raise dgcv_exception_note(f"{e}") from None
        elif all(get_dgcv_category(elem) == "tensorProduct" for elem in data):
            dgcv_warning(
                "`_validate_structure_data` was given a list of tensorProduct instance, but `process_matrix_rep` was also marked True. The latter was ignored."
            )
            return _validate_structure_data(
                data,
                process_matrix_rep=False,
                assume_skew=assume_skew,
                basis_order_for_supplied_str_eqns=basis_order_for_supplied_str_eqns,
                process_tensor_rep=True,
            )
        else:
            raise ValueError(
                "matrix representation processing requires a list of square matrices."
            )

    if isinstance(data, (list, tuple)):
        if len(data) > 0:
            if all(query_dgcv_categories(obj, {"vector_field"}) for obj in data):
                return aDataFromVFWithAnsatz(
                    data,
                    determinacy_order_ansatz=determinacy_order_ansatz,
                    process_with_decompose=process_with_decompose,
                )
            elif all(query_dgcv_categories(obj, {"matrix"}) for obj in data):
                return algebraDataFromMatRep(data), "matrix"
            else:
                try:
                    return aDataFromNestedLists(data, assume_skew=assume_skew)
                except ValueError:
                    raise
                except Exception:
                    raise TypeError(
                        "The algebra_class initializer recieved data in an unsupported format."
                    )
        else:
            return _structure_array(dict(), 0 if dimension is None else dimension), set()

    if get_dgcv_category(data) == "array":
        return _structure_data_from_array(data, assume_skew=assume_skew)

    try:
        if isinstance(data, dict):
            if not data:
                return _structure_array(dict(), 0 if dimension is None else dimension), set()
            if all(
                isinstance(key, numbers.Integral)
                and all(query_dgcv_categories(obj, {"vector_field"}) for obj in val)
                for key, val in data.items()
            ):
                try:
                    return aDataFromVFWithAnsatz(
                        data,
                        determinacy_order_ansatz=determinacy_order_ansatz,
                        process_with_decompose=process_with_decompose,
                    )
                except Exception:
                    raise TypeError(
                        "`createAlgebra` could not extract a Lie algebra structure from the given vector fields with the indicated grading. If indicating a grading was unintended, then provide the fields in a list instead; if that also fails, then they may not span a Lie algebra."
                    )
            if all(
                isinstance(key, tuple)
                and len(key) == 2
                and all(is_atomic(idx) for idx in key)
                for key in data
            ):
                tuple_vars = set()
                for key in data:
                    tuple_vars.add(key[0])
                    tuple_vars.add(key[1])
                if basis_order_for_supplied_str_eqns is None:
                    build_basis_order = True
                    basis_order_for_supplied_str_eqns = []
                else:
                    build_basis_order = False
                    if len(tuple_vars) < len(basis_order_for_supplied_str_eqns):
                        tuple_vars |= set(basis_order_for_supplied_str_eqns)
                if not isinstance(
                    basis_order_for_supplied_str_eqns, (list, tuple)
                ) or not all(
                    is_atomic(var) for var in basis_order_for_supplied_str_eqns
                ):
                    raise ValueError(
                        "If initializing an algebra from structure equations and supplying the `basis_order_for_supplied_str_eqns` parameter, this parameter should be a list of the atomic variables appearing in the supplied structure equations."
                    )
                bo_names = {str(var) for var in basis_order_for_supplied_str_eqns}
                for var in tuple_vars:
                    if str(var) not in bo_names:
                        if build_basis_order:
                            basis_order_for_supplied_str_eqns.append(var)
                            bo_names.add(str(var))
                        else:
                            raise ValueError(
                                "If initializing an algebra from structure equations and supplying the `basis_order_for_supplied_str_eqns` parameter, this parameter should be a list containing all atomic variables appearing in the supplied structure equations."
                            )
                ordered_BV = basis_order_for_supplied_str_eqns
                zeroing = {var: 0 for var in ordered_BV}
                dim = len(ordered_BV)
                cells = dict()
                params = set()
                ordered_BV_index = {str(v): i for i, v in enumerate(ordered_BV)}
                for idx_pair, val in data.items():
                    if not _scalar_is_zero(val):
                        params |= get_free_symbols(val)
                        v1, v2 = idx_pair
                        idx1 = ordered_BV_index[str(v1)]
                        idx2 = ordered_BV_index[str(v2)]

                        if hasattr(val, "subs") and _scalar_is_zero(val.subs(zeroing)):
                            coeffs = dict()
                            for idx, var in enumerate(ordered_BV):
                                subdict = zeroing | {var: 1}
                                c = simplify(val.subs(subdict))
                                if not _scalar_is_zero(c):
                                    coeffs[idx] = c
                            if coeffs:
                                cells[(idx1, idx2)] = matrix_dgcv(
                                    coeffs, shape=(dim, 1)
                                )
                        else:
                            raise ValueError(
                                "If initializing an algebra from structure equations, supplied structure equations should be a dictionary whose keys are tuples of atomic variables and whose values are linear combinations of variables representing the product of the elements in the key tuple. If that is the case then you are likely getting this error because you did not supply the algebra creator with a valid value for the `basis_order_for_supplied_str_eqns` parameter. If that paremeter were omitted, it is not always possible to unambiguously infer its proper value from general structure equations data, which can lead to this error."
                            )
                if assume_skew:
                    _complete_skew(cells, dim)
                return _structure_array(cells, dim), {
                    x for x in params if x not in tuple_vars
                }
            return _normalize_sparse_structure(
                data.items(), dim=dimension, assume_skew=assume_skew
            )
        params = set()
        for j in data._data.values():
            params |= get_free_symbols(j)

        return data, params  # structure data array, parameters

    except (ValueError, TypeError):
        raise
    except Exception as e:
        raise ValueError(f"Invalid structure data format: {type(data)} - {e}")
