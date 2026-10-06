from __future__ import annotations

from ..._aux._backends._calculus import diff
from ..._aux._backends._symbolic_router import _scalar_is_zero, get_free_symbols, subs
from ..._aux._backends._types_and_constants import is_atomic, symbol
from ..._aux._utilities._config import dgcv_warning
from ..._aux._utilities._misc import linear_combination
from ..._aux._vmf._safeguards import create_key, retrieve_passkey
from ..dgcv_core import variableProcedure
from ..solvers import solve_dgcv
from .independence import _extract_basis_by_wedge_vectorized


def _extract_basis_over_function_ring(objs, dimension_hint=None):
    try:
        if len(objs) == 0:
            return objs
        basis = [objs[0]]
        for obj in objs[1:]:
            if dimension_hint is not None and len(basis) == dimension_hint:
                break
            deco = decompose(obj, basis)[0]
            if len(deco) == len(basis):
                continue
            basis.append(obj)
        return basis

    except Exception:
        raise ValueError(
            "Could not extract basis. Likely the given objects do not adhere vector space axioms in supported formats."
        )


_prepared_spans = {}
_prepared_limit = 64
_format_classes = {"complex": "complex", "real": "real"}


def _coeff_vector(obj):
    cd = getattr(obj, "coeff_dict", None)
    if not isinstance(cd, dict):
        return None
    return cd


def _span_signature(obj):
    return (
        getattr(obj, "_span_class", type(obj)),
        getattr(obj, "data_shape", None),
        id(getattr(obj, "algebra", None)),
    )


def _format_class(obj):
    return _format_classes.get(getattr(obj, "_validated_format", None))


def _prepared_entry(objs, to_vector):
    from ..._aux._backends._engine import engine_kind
    from ...algebras.threads.util import _span_solver

    objs = list(objs)
    if not objs:
        return None
    key = (engine_kind(), tuple(id(o) for o in objs))
    hit = _prepared_spans.get(key)
    if hit is not None:
        return hit[1]
    if len(_prepared_spans) >= _prepared_limit:
        _prepared_spans.clear()
    entry = None
    sig = _span_signature(objs[0])
    fmt = None
    vectors = []
    for o in objs:
        if _span_signature(o) != sig:
            break
        f = _format_class(o)
        if f is not None:
            if fmt is None:
                fmt = f
            elif f != fmt:
                break
        vec = to_vector(o)
        if vec is None:
            break
        vectors.append(vec)
    else:
        solver = _span_solver.build_from_vectors(vectors)
        if solver is not None:
            entry = (solver, sig, fmt)
    _prepared_spans[key] = (objs, entry)
    return entry


def _prepared_span(objs, to_vector=_coeff_vector):
    entry = _prepared_entry(objs, to_vector)
    return None if entry is None else entry[0]


def _prepared_prefix_spans(objs, cuts, to_vector=_coeff_vector):
    from ..._aux._backends._engine import engine_kind
    from ...algebras.threads.util import _span_solver

    objs = list(objs)
    cuts = sorted({c for c in cuts if 0 < c <= len(objs)})
    if not objs or not cuts:
        return None
    kind = engine_kind()
    keys = [(kind, tuple(id(o) for o in objs[:cut])) for cut in cuts]
    if all(_prepared_spans.get(key) is not None for key in keys):
        return [_prepared_spans[key][1] for key in keys]
    sig = _span_signature(objs[0])
    fmt = None
    vectors = []
    for o in objs[: cuts[-1]]:
        if _span_signature(o) != sig:
            return None
        f = _format_class(o)
        if f is not None:
            if fmt is None:
                fmt = f
            elif f != fmt:
                return None
        vec = to_vector(o)
        if vec is None:
            return None
        vectors.append(vec)
    solvers = _span_solver.build_prefixes(vectors, cuts)
    if solvers is None:
        return None
    if len(_prepared_spans) + len(cuts) > _prepared_limit:
        _prepared_spans.clear()
    entries = []
    for key, cut, solver in zip(keys, cuts, solvers):
        entry = (solver, sig, fmt)
        _prepared_spans[key] = (objs[:cut], entry)
        entries.append(entry)
    return entries


def _decompose_prepared(obj, basis, wanted=None):
    entry = _prepared_entry(basis, _coeff_vector)
    if entry is None:
        return None
    solver, sig, fmt = entry
    if _span_signature(obj) != sig:
        return None
    f = _format_class(obj)
    if f is not None and fmt is not None and f != fmt:
        return None
    vec = _coeff_vector(obj)
    if vec is None:
        return None
    return solver.reduce_vector(vec, wanted)


def _decompose_constrained_prepared(obj, basis, constrained):
    from ..._aux._backends._engine import engine_capability

    if engine_capability("span_field") is None:
        return None
    entry = _prepared_entry(basis, _coeff_vector)
    if entry is None:
        return None
    solver, sig, fmt = entry
    if _span_signature(obj) != sig:
        return None
    f = _format_class(obj)
    if f is not None and fmt is not None and f != fmt:
        return None
    vec = _coeff_vector(obj)
    if vec is None:
        return None
    reduced = solver.reduce_vector_residual(vec)
    if reduced is None:
        return None
    coords, residual = reduced
    eqns = [e for e in residual if not _scalar_is_zero(e)]
    if eqns:
        solutions = solve_dgcv(
            eqns, list(constrained), method="linsolve", simplify_result=False
        )
        if not solutions:
            return None
        sol0 = solutions[0]
    else:
        sol0 = {}
    if sol0:
        coords = [subs(c, sol0) for c in coords]
    return (coords, list(basis), {v: sol0.get(v, v) for v in constrained})


def decompose(
    obj,
    basis,
    return_parameters: bool = False,
    new_parameters_label: str | None = None,
    only_check_decomposability: bool = False,
    variables_to_constrain: list = None,
    assume_VTC_linear: bool = False,
    _pass_error_report=None,
    _hand_off=None,
    *,
    assume_basis: bool = False,
    register_parameters: bool = False,
    _no_disposable_solve_vars: bool = False,
    _generic_combination=None,
    _coordinate_indices=None,
):
    """
    Decomposes a vector field or differential form as a linear combination of a given `basis` list.

    This function attempts to express the input `obj` (a VFClass or DFClass object) as a linear combination
    of the elements in the provided `basis` list. The `basis` list does not need to be linearly independent,
    and if the decomposition is not unique, the function will parameterize the solution space. Any parameters
    needed are automatically initialized and registered in dgcv's variable management framework (VMF).

    The function carefully handles variable types based on the `dgcvType` attribute of the objects. For objects
    with `dgcvType='complex'`, it dynamically selects whether to perform real coordinate computations or complex
    coordinate computations, depending on the input data. If a canonical variable formatting decision cannot be
    made naturally from the input, the function will return warnings with explanations.

    Parameters
    ----------
    obj : VFClass or DFClass
        The vector field or differential form to decompose.
    basis : list of VFClass or DFClass
        A list of objects (vector fields or differential forms) to decompose `obj` with respect to.
        The class of objects in the `basis` list must match the class of `obj`.
    return_parameters : bool, optional
        If True, the function will return the parameterized solution when the decomposition is not unique
        (default is False). Parameters are initialized with labels registered within the VMF.
    new_parameters_label : str or None, optional
        If `return_parameters` is True and the decomposition is not unique, this label will be used
        to name the new parameter variables. If None, obscure labels will be generated automatically (default is None).
    _pass_error_report : optional
        Internal use parameter for handling error reports in certain edge cases (default is None).

    Returns
    -------
    list
        The coefficients of the linear combination that expresses `obj` in terms of the `basis` list.
        If the decomposition is parameterized, the returned list contains the parameterized solution.

    Raises
    ------
    TypeError
        If the class of `obj` does not match the class of elements in the `basis` list (i.e., both must
        be either VFClass or DFClass), or if objects in the `basis` list have inconsistent `dgcvType` attributes.

    Warnings
    --------
    - A warning is issued if `obj` is not in the span of the provided `basis` list.
    - If the `basis` list is not linearly independent, the decomposition is not unique, and a parameterized
    solution will be returned. The initialized parameters are registered as 'standard variables' in the VMF.

    Remarks
    -------
    - The function dynamically handles objects based on their `dgcvType` attribute. For `dgcvType='complex'`,
    it distinguishes between real and complex coordinate computations, converting the input as needed
    to ensure consistency in formatting. If this decision cannot be determined from the input data,
    the function issues warnings explaining the necessary canonical formatting.

    Example
    -------
    # Decompose a vector field 'vf' as a linear combination of two basis vector fields 'vf1' and 'vf2'
    coeffs, basis_used = decompose(vf, [vf1, vf2])

    # Decompose a differential form 'df' with a parameterized solution due to non-uniqueness
    coeffs, basis_used = decompose(df, [df1, df2, df3], return_parameters=True, new_parameters_label='p')
    """
    basis = list(basis)
    original_length = len(basis)

    if not return_parameters and not assume_basis:
        basis = _extract_basis_by_wedge_vectorized(basis)
    if only_check_decomposability and variables_to_constrain:
        dgcv_warning(
            "only_check_decomposability=True disables variables_to_constrain, as variables_to_constrain undermines the intended computation savings of only_check_decomposability."
        )
    if not (
        isinstance(variables_to_constrain, (list, tuple))
        and all(is_atomic(x) for x in variables_to_constrain)
    ):
        variables_to_constrain = None

    n = len(basis)
    if n == 0:
        if only_check_decomposability is True or variables_to_constrain:
            ob = getattr(obj, "__dgcv_zero_obstr__", None)
            if ob is None:
                raise TypeError(
                    f"decompose does not operate on objects of type {type(obj).__name__}"
                )
            eqns, _ = ob
            eqns = list(eqns or [])
            if variables_to_constrain:
                sol = solve_dgcv(
                    eqns,
                    variables_to_constrain,
                    method="linsolve",
                    simplify_result=False,
                )
                if len(sol) > 0:
                    return [], basis, sol
            return bool(eqns) and all(_scalar_is_zero(e) for e in eqns)
        return ([], basis)
    if (
        assume_basis
        and not return_parameters
        and not variables_to_constrain
        and not _no_disposable_solve_vars
    ):
        prepared = _decompose_prepared(obj, basis, _coordinate_indices)
        if prepared is not None:
            if only_check_decomposability is True:
                return True
            return (prepared, basis)
    if (
        assume_basis
        and assume_VTC_linear
        and variables_to_constrain
        and not return_parameters
        and not only_check_decomposability
        and not _no_disposable_solve_vars
    ):
        prepared = _decompose_constrained_prepared(obj, basis, variables_to_constrain)
        if prepared is not None:
            return prepared
    use_disposable_solve_vars = (
        assume_basis
        and not return_parameters
        and not variables_to_constrain
        and not _no_disposable_solve_vars
    )
    if use_disposable_solve_vars and _generic_combination is not None:
        gen_combo, variables = _generic_combination
    else:
        gen_combo, variables = linear_combination(
            basis, _disposable=use_disposable_solve_vars
        )
    if variables_to_constrain:
        variables = list(variables) + list(variables_to_constrain)
    system = obj - gen_combo

    ob = getattr(system, "__dgcv_zero_obstr__", None)
    if ob is None:
        raise TypeError(f"{type(system).__name__} lacks __dgcv_zero_obstr__")
    eqns, _ = ob
    eqns = list(eqns or [])

    if variables_to_constrain and not assume_VTC_linear:
        solutions = solve_dgcv(eqns, variables, method="solve", simplify_result=False)
    else:
        solutions = solve_dgcv(
            eqns, variables, method="linsolve", simplify_result=False
        )
    if not solutions:
        if only_check_decomposability is True:
            return False
        if _pass_error_report == retrieve_passkey():
            return (
                f"`decompose` found no solution for equations {eqns} in variables {variables} "
                f"against a spanning family of length {original_length}."
            )
        return ([], basis, solutions) if variables_to_constrain else ([], basis)

    if only_check_decomposability is True:
        return True

    sol0 = solutions[0]
    decomp_coeffs = [sol0.get(v, v) for v in variables]
    if variables_to_constrain:
        vtc_solutions = {v: sol0.get(v, v) for v in variables_to_constrain}

    free = set()
    for expr in decomp_coeffs:
        free |= set(get_free_symbols(expr) or ())
    if variables_to_constrain:
        for expr in vtc_solutions.values():
            free |= set(get_free_symbols(expr) or ())
    varnames = {str(x) for x in variables}
    if len(free) > len(variables):
        if variables_to_constrain:
            vtcnames = {str(x) for x in variables_to_constrain}
            compnames = {str(x) for x in free if str(x) not in vtcnames}
        else:
            compnames = {str(x) for x in free}
        free = {x for x in variables if str(x) in compnames}
    else:
        if variables_to_constrain:
            vtcnames = {str(x) for x in variables_to_constrain}
            compset = {x for x in variables if str(x) not in vtcnames}
        else:
            compset = variables
        free = {x for x in free if str(x) in varnames}
    if use_disposable_solve_vars and free:
        return decompose(
            obj,
            basis,
            return_parameters=return_parameters,
            new_parameters_label=new_parameters_label,
            only_check_decomposability=only_check_decomposability,
            variables_to_constrain=variables_to_constrain,
            assume_VTC_linear=assume_VTC_linear,
            _pass_error_report=_pass_error_report,
            _hand_off=_hand_off,
            assume_basis=assume_basis,
            register_parameters=register_parameters,
            _no_disposable_solve_vars=True,
        )

    if return_parameters and free:
        free = tuple(free)
        if register_parameters:
            params = variableProcedure(
                new_parameters_label
                if isinstance(new_parameters_label, str)
                else "param",
                len(free),
                return_created_object=True,
            )[0]
            params = tuple(params)
        else:
            plabel = (
                new_parameters_label
                if isinstance(new_parameters_label, str) and new_parameters_label
                else create_key(prefix="param")
            )
            params = tuple(symbol(f"{plabel}{i}") for i in range(len(free)))

        subs_dict = {v: p for v, p in zip(free, params)}
        decomp_coeffs = [subs(c, subs_dict) for c in decomp_coeffs]
        if variables_to_constrain:
            vtc_solutions = {k: subs(v, subs_dict) for k, v in vtc_solutions.items()}

    return (
        (decomp_coeffs, basis, vtc_solutions)
        if variables_to_constrain
        else (decomp_coeffs, basis)
    )


def _decompose_over_number_field(
    obj,
    basis: list,
    determinacy_order_ansatz=None,
    return_basis=False,
    only_check_independence=False,
):
    free_symbols = set()
    dim = len(basis)
    if determinacy_order_ansatz is None:
        order_bound = 2
    else:
        order_bound = determinacy_order_ansatz
    for elem in basis:
        free_symbols |= get_free_symbols(elem)
    vlabel = create_key("var")
    variables = [symbol(f"{vlabel}_{idx}") for idx in range(dim)]
    general_combination = sum(var * elem for var, elem in zip(variables, basis))
    eqns = obj - general_combination

    if hasattr(eqns, "__dgcv_zero_obstr__"):
        eqns = [
            eqn for eqn in eqns.__dgcv_zero_obstr__[0] if not _scalar_is_zero(eqn)
        ]
    exhaustion_tree = {0: {tuple(free_symbols): list(eqns)}}
    for order in range(order_bound):
        if order not in exhaustion_tree:
            break
        previous = dict(exhaustion_tree[order])
        new_order = order + 1
        for v_tuple, eqn_branch in previous.items():
            for eqn in list(eqn_branch):
                branch_root = []
                new_branch = []
                for var in v_tuple:
                    neqn = diff(eqn, var)
                    if not _scalar_is_zero(neqn):
                        branch_root.append(var)
                        new_branch.append(neqn)
                if len(branch_root) == 0:
                    continue
                branch_root = tuple(branch_root)
                eqns += new_branch
                if new_order not in exhaustion_tree:
                    exhaustion_tree[new_order] = {branch_root: new_branch}
                else:
                    exhaustion_tree[new_order][branch_root] = (
                        exhaustion_tree[new_order].get(branch_root, []) + new_branch
                    )
    sol = solve_dgcv(eqns, variables, method="linsolve", simplify_result=False)
    if len(sol) == 0:
        if only_check_independence:
            return True
        out = []
    else:
        if only_check_independence:
            return False
        sol = sol[0]
        decomp_coeffs = []
        fv = set()
        for v in variables:
            coeff = sol.get(v)
            fv |= get_free_symbols(coeff)
            decomp_coeffs.append(coeff)
        fv = {v for v in fv if v not in free_symbols}
        zeroing = {v: 0 for v in fv}
        out = [[subs(coeff, zeroing | {v: 1}) for coeff in decomp_coeffs] for v in fv]
    if return_basis:
        return out, basis
    return out


def _extract_basis_over_number_field(
    spanners: list | tuple, determinacy_order_ansatz=None
):
    basis = []
    for elem in spanners:
        if _scalar_is_zero(elem):
            continue
        if len(basis) == 0:
            basis.append(elem)
            continue
        indep = _decompose_over_number_field(
            elem,
            basis,
            determinacy_order_ansatz=determinacy_order_ansatz,
            only_check_independence=True,
        )
        if indep:
            basis.append(elem)
            continue
    return basis
