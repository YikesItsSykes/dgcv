from __future__ import annotations

from typing import Any, List, Sequence

from ..._aux._backends._symbolic_router import get_free_symbols, subs
from ..._aux._vmf._safeguards import get_dgcv_category
from ..solvers import solve_dgcv
from ._solver_support import _local_unknowns, _make_parameters, _require_subcategory
from ._vmf_frames import _coordinate_basis_from_vmf, _infer_coordinate_space_from_objs


def _validated_coordinate_space(objs, coordinate_space, who):
    if coordinate_space is None:
        return _infer_coordinate_space_from_objs(objs)
    if not isinstance(coordinate_space, (list, tuple, set, frozenset)):
        raise TypeError(
            f"`{who}` coordinate_space must be a list/tuple/set if provided."
        )
    return list(coordinate_space)


def _as_scalar_eqns(x):
    if get_dgcv_category(x) == "tensor_field":
        return list(x.coeff_dict.values())
    return [x]


def _pairing_row(pairs):
    row = []
    for val in pairs:
        scalars = _as_scalar_eqns(val)
        if len(scalars) != 1:
            row.append(sum(scalars))
        else:
            row.append(scalars[0])
    return row


def _dual_combinations(
    eval_entries,
    basis,
    who,
    *,
    return_parameters,
    register_parameters,
    parameters_label,
):
    m = len(eval_entries)
    n = len(basis)
    unknowns = _local_unknowns(m * n)

    eqns: List[Any] = []
    for j in range(m):
        row_vars = unknowns[j * n : (j + 1) * n]
        for k in range(m):
            target = 1 if j == k else 0
            s = 0
            Ek = eval_entries[k]
            for u, e in zip(row_vars, Ek):
                s = s + u * e
            eqns.append(s - target)

    sols = solve_dgcv(eqns, unknowns, method="linsolve", simplify_result=False)
    if not sols:
        raise RuntimeError(
            f"`{who}` could not solve for a dual basis (system unsatisfiable)."
        )

    sol = sols[0]

    unknowns_set = set(unknowns)
    free_vars_set = set()
    for u in unknowns:
        free_vars_set |= set(get_free_symbols(sol.get(u, u))) & unknowns_set
    free = [u for u in unknowns if u in free_vars_set]

    if return_parameters and free:
        params = _make_parameters(
            len(free),
            register_parameters=register_parameters,
            parameters_label=parameters_label,
        )
        sub = dict(zip(free, params))
        sol_use = dict(sol)
        for v in free:
            sol_use[v] = v

        def _coeff(u):
            c = sol_use.get(u, u)
            return subs(c, sub)
    else:
        sol_use = dict(sol)
        zero_sub = {v: 0 for v in free}
        for v in free:
            sol_use[v] = 0

        def _coeff(u):
            c = sol_use.get(u, u)
            return subs(c, zero_sub) if zero_sub else c

    out: List[Any] = []
    for j in range(m):
        row_vars = unknowns[j * n : (j + 1) * n]
        obj_j = 0
        for u, b in zip(row_vars, basis):
            obj_j = obj_j + _coeff(u) * b
        out.append(obj_j)

    return out


def get_coframe(
    VFList: Sequence[Any],
    *,
    coordinate_space: Sequence[Any] | None = None,
    return_parameters: bool = False,
    register_parameters: bool = False,
    parameters_label: str | None = None,
):
    VFList = list(VFList)
    if not VFList:
        return []

    _require_subcategory(VFList, {"vector_field"}, "get_coframe")
    coordinate_space = _validated_coordinate_space(VFList, coordinate_space, "get_coframe")

    dfBasis = _coordinate_basis_from_vmf(coordinate_space, coorType="df")
    if len(dfBasis) == 0:
        raise TypeError("`get_coframe` could not obtain a DF basis from VMF.")

    eval_entries = [_pairing_row(df(vf) for df in dfBasis) for vf in VFList]
    return _dual_combinations(
        eval_entries,
        dfBasis,
        "get_coframe",
        return_parameters=return_parameters,
        register_parameters=register_parameters,
        parameters_label=parameters_label,
    )


def get_frame(
    DFList: Sequence[Any],
    *,
    coordinate_space: Sequence[Any] | None = None,
    return_parameters: bool = False,
    register_parameters: bool = False,
    parameters_label: str | None = None,
):
    DFList = list(DFList)
    if not DFList:
        return []

    _require_subcategory(DFList, {"differential_form"}, "get_frame")
    if any(getattr(df, "degree", None) != 1 for df in DFList):
        raise TypeError("`get_frame` expects a list of 1-forms.")
    coordinate_space = _validated_coordinate_space(DFList, coordinate_space, "get_frame")

    vfBasis = _coordinate_basis_from_vmf(coordinate_space, coorType="vf")
    if len(vfBasis) == 0:
        raise TypeError("`get_frame` could not obtain a VF basis from VMF.")

    eval_entries = [_pairing_row(df(vf) for vf in vfBasis) for df in DFList]
    return _dual_combinations(
        eval_entries,
        vfBasis,
        "get_frame",
        return_parameters=return_parameters,
        register_parameters=register_parameters,
        parameters_label=parameters_label,
    )
