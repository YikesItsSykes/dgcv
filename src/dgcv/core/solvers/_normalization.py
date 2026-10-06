from __future__ import annotations

from ..._aux._backends._symbolic_router import get_free_symbols
from ..._aux._backends._types_and_constants import is_atomic
from ._bridge import SolveBridge


def normalize_equations_and_vars(eqns, vars_to_solve):
    if not isinstance(eqns, (list, tuple)):
        eqns = [eqns]
    formatted = []
    for eqn in eqns:
        if hasattr(eqn, "__dgcv_solve_bridge__"):
            formatted.append(eqn)
            continue
        neqn = getattr(eqn, "__dgcv_zero_obstr__", None)
        if neqn is None:
            formatted.append(eqn)
        else:
            try:
                formatted += list(neqn[0])
            except Exception as solve_exception:
                raise RuntimeError(
                    "Equations data provided to `solve_dgcv` were in an unsupported format."
                ) from solve_exception
    eqns = formatted

    if vars_to_solve is None:
        vars_to_solve = set()
        for eqn in eqns:
            try:
                vars_to_solve |= set(get_free_symbols(eqn))
            except Exception:
                pass

    if isinstance(vars_to_solve, set):
        vars_to_solve = list(vars_to_solve)
    if not isinstance(vars_to_solve, (list, tuple)):
        vars_to_solve = [vars_to_solve]
    return eqns, vars_to_solve


def _equations_preprocessing(eqns: tuple | list, vars: tuple | list, bridge=None):
    if bridge is None:
        bridge = SolveBridge()
    processed_eqns = []
    for eqn in eqns:
        processed_eqns += bridge.lower(eqn)

    system_vars = []
    extra_vars = []
    for var in vars:
        var = bridge.lookup(var)
        if isinstance(var, (list, tuple)) and len(var) == 1:
            var = var[0]
        if is_atomic(var):
            system_vars += [var]
        else:
            extra_vars += [var]
    return processed_eqns, system_vars, extra_vars, bridge
