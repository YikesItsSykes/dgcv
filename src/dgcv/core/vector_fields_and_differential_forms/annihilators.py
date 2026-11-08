from __future__ import annotations

from typing import Any, Sequence

from ..._aux._backends._symbolic_router import _scalar_is_zero, subs
from ..._aux._utilities._misc import linear_combination
from ..._aux._vmf._safeguards import (
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_passkey,
)
from ..._aux._vmf.vmf import vmf_lookup
from ..combinatorics.combinatorics import chooseOp
from ..conversions.conversions import allToReal
from ..solvers import solve_dgcv
from .decomposition import _extract_basis_over_number_field


def annihilator(
    objList: Sequence[Any],
    coordinate_space: Sequence[Any] | None = None,
    control_distribution: Sequence[Any] | None = None,
    polynomial_bases: bool = False,
    _pass_error_report=None,
    filter_to_basis_over_number_field: bool = False,
    *,
    coherent_coordinates_checked: bool = False,
    frame=None,
    **kwargs,
):
    """
    Finds annihilators for a given list of vector fields or differential forms.

    This function computes objects that "annihilate" the provided list of vector fields or differential forms.
    An annihilator is either the span of differential forms that evaluate to zero on each vector field in the list,
    or vector fields whose interior product annihilates each differential form in the list. `annihilator` dynamically
    handles both real and holomorphic coordinate systems and can convert between them as needed. Additionally,
    solutions can be constrained to a given control distribution by using the `control_distribution` keyword.

    Parameters
    ----------
    objList : list of VFClass or DFClass
        A list of vector fields or differential forms for which the annihilator will be computed. All objects
        in the list must be of the same class (either all vector fields or all differential forms) and have
        consistent `dgcvType` attributes (i.e., 'standard' or 'complex').
    coordinate_Space : list, tuple, or set
        A collection of variables that define the coordinate system in which the annihilator is to be computed.
    polynomial_bases: bool (optional)
        Attempt to scale computed basis elements so that they have polynomial coefficients before returning the list. Can still produce non-nonpolynomial expressions when factoring is not possible.
    filter_to_basis_over_number_field: bool, optional
        If True, the returned list will be linearly independant over C.
    _pass_error_report : optional
        Internal use parameter for handling error reports in certain edge cases (default is None).

    Returns
    -------
    list
        A list of differential forms (if vector fields were provided) or vector fields (if differential forms were
        provided) that annihilate the objects in `objList`.

    Raises
    ------
    TypeError
        If the objects in `objList` are not all of the same type (i.e., all vector fields or all differential forms),
        or if the `coordinate_Space` is not a valid list, tuple, or set.

    Warnings
    --------
    - If the objects in `objList` are defined with inconsistent coordinate system types (real vs. holomorphic), the
    function converts them to a consistent coordinate system and issues a warning.

    Example
    -------
    >>> from dgcv import createVariables, annihilator, exteriorDerivative, complex_struct_op, Del, DelBar, allToReal
    >>> createVariables('z', 'x', 'y', 4, initialIndex=0)
    >>> rho = (x1*x2 + x1**2*x3 - x0)  # A defining equation for a real hypersurface M in C^4
    >>> d_rho = exteriorDerivative(rho)  # Its differential will annihilate TM
    >>> print(d_rho)

    (2*x1*x3 + x2)*d_x1 + x1*d_x2 + x1**2*d_x3 - 1*d_x0

    >>> dfList = [d_rho]
    >>> TMbasis = annihilator(dfList, x+y)  # Use annihilator to compute the tangent bundle TM
    >>> TMbasis

    [(16*x1*x3 + 8*x2)*D_x0 + 8*D_x1, 8*x1*D_x0 + 8*D_x2, 8*x1**2*D_x0 + 8*D_x3, 8*D_y0, 8*D_y1, 8*D_y2, 8*D_y3]

    >>> J_of_TMbasis = [complex_struct_op(vf) for vf in TMbasis]  # Get the image of TM under the complex structure operator.
    >>> J_of_TMbasis

    [(16*x1*x3 + 8*x2)*D_y0 + 8*D_y1, 8*x1*D_y0 + 8*D_y2, 8*x1**2*D_y0 + 8*D_y3, -8*D_x0, -8*D_x1, -8*D_x2, -8*D_x3]

    >>> CR_distribution = annihilator(annihilator(J_of_TMbasis, x+y) + annihilator(TMbasis, x+y), x+y)
    >>> CR_distribution  # Use annihilator to get the CR distribution, which is the intersection of TM with CTM

    [(16*x1*x3 + 8*x2)*D_x0 + 8*D_x1, 8*x1*D_x0 + 8*D_x2, 8*x1**2*D_x0 + 8*D_x3,
    (16*x1*x3 + 8*x2)*D_y0 + 8*D_y1, 8*x1*D_y0 + 8*D_y2, 8*x1**2*D_y0 + 8*D_y3]

    >>> LeviForm = allToReal(Del(DelBar(rho)))  # Apply Dolbeault operators to represent the Levi form
    >>> print(LeviForm._repr_latex_())

    # Output: <LeviForm in LaTeX formatted plain text>

    >>> Levi_kernel = annihilator([LeviForm], x+y, control_distribution=CR_distribution)
    >>> Levi_kernel  # annihilator reveals that the Levi form has a real 2-d. kernel

    [-64*x1**2*D_x0 - 128*x1*D_x2 + 64*D_x3, -64*x1**2*D_y0 - 128*x1*D_y2 + 64*D_y3]

    >>> not_the_Levi_kernel = annihilator([LeviForm], x+y)
    >>> not_the_Levi_kernel  # Without constraining annihilator to the CR distribution, it finds a kernel that is too large.

    [8*D_x0, -16*x1*D_x2 + 8*D_x3, 8*D_y0, -16*x1*D_y2 + 8*D_y3]
    """
    if frame is not None or query_dgcv_categories(objList, {"eds_form"}) or (
        not get_dgcv_category(objList)
        and all(query_dgcv_categories(o, {"eds_form"}) for o in objList)
        and len(list(objList)) > 0
    ):
        from ...eds._vectors import annihilator as _eds_annihilator

        return _eds_annihilator(objList, frame=frame)
    if get_dgcv_category(objList) in {
        "tensor_field",
        "algebra_element",
        "subalgebra_element",
    }:
        objList = [objList]
    objList = list(objList)
    if not objList:
        return []

    if not all(get_dgcv_category(o) == "tensor_field" for o in objList):
        raise TypeError("`annihilator` expects tensor_field objects.")

    is_vf_case = all(query_dgcv_categories(o, {"vector_field"}) for o in objList)
    control_needed, expensive_path_needed = control_distribution is not None, False
    if is_vf_case is True:
        is_df_case = False
    else:
        is_df_case = True
        for o in objList:
            if not query_dgcv_categories(o, {"differential_form"}):
                is_df_case = False
                break
            md, mmd = o.min_degree, o.max_degree
            if control_needed and mmd > 1:
                expensive_path_needed = True
                if md != mmd:
                    raise TypeError(
                        "`annihilator` does not support restriction to control distributions when the objects to annihilate are mixed deree differential forms."
                    )
    if not is_vf_case and not is_df_case:
        raise TypeError(
            "`annihilator` expects a list of all vector fields or all differential forms."
        )
    if control_needed:
        if is_vf_case:
            if get_dgcv_category(control_distribution) == "distribution":
                control_distribution = control_distribution.df_basis
            else:
                control_distribution = list(control_distribution)
                if not all(
                    query_dgcv_categories(o, {"differential_form"})
                    for o in control_distribution
                ):
                    raise TypeError(
                        "`annihilator` control_distribution must be differential forms if given objects to annihilate are VF."
                    )
            basis_elems = control_distribution
        else:
            if get_dgcv_category(control_distribution) == "distribution":
                control_distribution = control_distribution.vf_basis
            else:
                control_distribution = list(control_distribution)
                if not all(
                    query_dgcv_categories(o, {"vector_field"})
                    for o in control_distribution
                ):
                    raise TypeError(
                        "`annihilator` control_distribution must be vector fields if the given objects to annihilater are DF."
                    )
            basis_elems = control_distribution

    elif coordinate_space is not None:
        if not isinstance(coordinate_space, (list, tuple, set)):
            raise TypeError(
                "`annihilator` coordinate_space must be a list/tuple/set if provided."
            )

        coords = list(coordinate_space)
        for a in coords:
            ds = vmf_lookup(a, differential_system=True).get("differential_system")
            if ds is None:
                raise TypeError(
                    "`annihilator` requires manually provided coordinates to be registered in the dgcv VMF with differential objects. "
                    "Suggestion: initialize coordinates with `createVariables(..., withVF=True)` (or `createVariables(..., complex=True)` as appropriate)."
                )

        basis_elems = []
        for a in coords:
            ds = vmf_lookup(a, differential_system=True).get("differential_system")
            elem = ds.get("df") if is_vf_case else ds.get("vf")
            if elem is None:
                raise TypeError(
                    "`annihilator` requires coordinates to be registered in the dgcv VMF with differential objects. "
                    "Suggestion: initialize coordinates with `createVariables(..., withVF=True)` (or `createVariables(..., complex=True)` as appropriate)."
                )
            basis_elems.append(elem)

    else:
        atoms = set()
        for o in objList:
            vs = getattr(o, "_variable_spaces", None)
            if isinstance(vs, dict):
                for tup in vs.values():
                    atoms.update(tup)

        basis_elems = []
        for a in sorted(atoms, key=lambda x: str(x)):
            ds = vmf_lookup(a, differential_system=True).get("differential_system")
            if not ds:
                continue
            elem = ds.get("df") if is_vf_case else ds.get("vf")
            if elem is None:
                continue
            basis_elems.append(elem)

        if not basis_elems:
            raise TypeError(
                "`annihilator` could not obtain a coordinate basis from VMF with differential objects. "
                "Suggestion: initialize coordinates with `createVariables(..., withVF=True)` (or `createVariables(..., complex=True)` as appropriate)."
            )

    if not coherent_coordinates_checked:
        objList = [allToReal(o) for o in objList]
        basis_elems = [allToReal(e) for e in basis_elems]

    n = len(basis_elems)

    general, vars_list = linear_combination(basis_elems)

    if is_vf_case:
        eqns = [general(vf) for vf in objList]
    else:
        if expensive_path_needed:
            eqns = []
            combos = dict()
            for df in objList:
                deg = df.max_degree
                if deg == 0:
                    eqns.append(df)
                else:
                    if deg not in combos:
                        combos[deg] = chooseOp(
                            range(n),
                            deg - 1,
                            withOrder=True,
                            withoutReplacement=True,
                        )
                    for tail in combos[deg]:
                        trailing = [basis_elems[idx] for idx in tail]
                        eqns += [df(general, *trailing) for df in objList]
        else:
            eqns = [df(general) for df in objList]

    scalar_eqns = []
    for e in eqns:
        if get_dgcv_category(e) == "tensor_field":
            scalar_eqns.extend(e.coeff_dict.values())
        else:
            scalar_eqns.append(e)

    sols = solve_dgcv(scalar_eqns, vars_list, method="linsolve", simplify_result=False)
    if not sols:
        if _pass_error_report == retrieve_passkey():
            return "`annihilator` found no solutions."
        return []

    sol = sols[0]

    def _apply_sol(expr, sol_dict, scale=False):
        sol = subs(expr, sol_dict) if isinstance(sol_dict, dict) else expr
        if scale:
            f = getattr(sol, "scale_to_polynomial_attempt", None)
            sol = f() if callable(f) else sol
        return sol

    general_solution = _apply_sol(general, sol)

    free_vars = [u for u in vars_list if str(sol.get(u, u)) == str(u)]

    if not free_vars:
        if polynomial_bases:
            f = getattr(general_solution, "scale_to_polynomial_attempt", None)
            general_solution = f() if callable(f) else general_solution
        return [general_solution]

    out = []
    for v in free_vars:
        assign = {u: 0 for u in free_vars}
        assign[v] = 1
        out.append(_apply_sol(general_solution, assign, scale=polynomial_bases))
    if filter_to_basis_over_number_field:
        out = _extract_basis_over_number_field(out)
    else:
        out = [elem for elem in out if not _scalar_is_zero(elem)]
    return out
