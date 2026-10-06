from ..._aux._backends._calculus import diff
from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    get_free_symbols,
    simplify,
)
from ..._aux._backends._types_and_constants import symbol
from ..._aux._vmf._safeguards import create_key
from ...core.arrays import array_dgcv, freeze_matrix, matrix_dgcv
from ...core.dgcv_core import VF_bracket
from ...core.solvers import solve_dgcv
from ...core.vector_fields_and_differential_forms import decompose


def aDataFromVFWithAnsatz(
    graded_components, determinacy_order_ansatz=None, process_with_decompose=False
):
    if not isinstance(graded_components, dict):
        grading = None
        basis = graded_components  # assumed to be iterable
        graded_components = {0: graded_components}
    else:
        basis = []
        grading = []
        for weight, component in graded_components.items():
            grading += [weight] * len(component)
            basis += list(component)
    free_symbols = set()
    coordinates = set()
    for vf in basis:
        coordinates |= vf.minimal_coordinate_space
        free_symbols |= get_free_symbols(vf)
    free_symbols = {v for v in free_symbols if v in coordinates}
    vlabel = create_key("var")
    var_dict = {
        k: [symbol(f"{vlabel}{k}_{idx}") for idx in range(len(v))]
        for k, v in graded_components.items()
    }
    gen_elements = {
        k: sum(var * elem for var, elem in zip(var_dict[k], v))
        for k, v in graded_components.items()
    }
    dim = len(basis)

    structure_data = array_dgcv(
        dict(), shape=(dim, dim), null_return=freeze_matrix(matrix_dgcv.zeros(dim, 1))
    )
    params = set()

    if determinacy_order_ansatz is None:
        order_bound = len(basis)
    else:
        order_bound = determinacy_order_ansatz
    for c1, vf1 in enumerate(basis):
        for c, vf2 in enumerate(basis[c1 + 1 :]):
            c2 = c1 + 1 + c
            new_weight = 0 if grading is None else grading[c1] + grading[c2]
            if new_weight in gen_elements:
                genelement = gen_elements[new_weight]
                variables = var_dict[new_weight]
            else:
                genelement = 0
                variables = [symbol("_dgcv_var_")]
            liebracket = VF_bracket(vf1, vf2)
            if process_with_decompose:
                if genelement == 0:
                    sol = {}
                else:
                    coeff_vals = decompose(
                        liebracket, graded_components[new_weight], assume_basis=True
                    )[0]
                    sol = {vari: val for vari, val in zip(variables, coeff_vals)}
            else:
                eqns = [
                    eqn
                    for eqn in (liebracket - (genelement)).coeff_dict.values()
                    if not _scalar_is_zero(eqn)
                ]
                prev_eqns = eqns
                for _ in range(order_bound):
                    p_e = prev_eqns
                    prev_eqns = []
                    for j in p_e:
                        for var in free_symbols:
                            neqn = diff(
                                j, var
                            )  ###!!! may be good to prune free_symbols here
                            if not _scalar_is_zero(neqn):
                                prev_eqns.append(neqn)
                    if len(prev_eqns) == 0:
                        break
                    eqns += prev_eqns
                sol = solve_dgcv(
                    eqns,
                    variables,
                    method="linsolve",
                    pass_to_symbolic_engine=False,
                    simplify_result=False,
                )
                if not sol:
                    raise RuntimeError(
                        f"Given vector field list is not closed under Lie brackets at indices ({c1}, {c2})."
                    )

                sol = sol[0]
            counter, result = 0, (matrix_dgcv.zeros(dim, 1))
            for idx in range(dim):
                weight = 0 if grading is None else grading[idx]
                if weight == new_weight:
                    newcoeff = sol.get(variables[counter])
                    if (
                        not _scalar_is_zero(newcoeff)
                        and get_free_symbols(newcoeff) & coordinates
                    ):
                        newcoeff = simplify(newcoeff)
                    if not _scalar_is_zero(newcoeff):
                        params |= get_free_symbols(newcoeff)
                        result[idx] = newcoeff
                    counter += 1
            structure_data[c1, c2] = result
            structure_data[c2, c1] = -result
    return structure_data, params, grading
