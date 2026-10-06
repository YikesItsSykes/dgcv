from ..._aux._backends._engine import engine_capability, engine_kind, engine_module
from ..._aux._backends._polynomials import expr_union_primitives
from ..._aux._backends._symbolic_router import get_free_symbols, simplify
from ..._aux._backends._types_and_constants import expr_numeric_types
from ..._aux._utilities._config import get_dgcv_settings_registry
from ..._aux._vmf.vmf import order_coordinates
from ._linsolve import _dgcv_linsolve, _sage_engine_linsolve
from ._normalization import _equations_preprocessing, normalize_equations_and_vars
from ._solution_shapes import (
    _engine_solve_to_dicts,
    _linsolve_to_dicts,
    _sage_solve_to_dicts,
)

_LINEAR_METHODS = ("linear", "linear_parametric")
_METHOD_ALIASES = {"linsolve": "linear"}


def solve_dgcv(
    eqns,
    vars_to_solve=None,
    verbose=False,
    method="auto",
    simplify_result=True,
    print_solve_stats=False,
    return_divisors=False,
    pass_to_symbolic_engine=None,
    sample_if_overdetermined=False,
    allow_underdetermined_solution=True,
    simplify_pivots=False,
    surface_singularities=False,
):
    """
    Solve an equation system for the given variables.

    Parameters
    ----------
    eqns : object
        Equation, list of equations, or dgcv object carrying a zero obstruction.
    vars_to_solve : object, optional
        Variable or list of variables. Defaults to all free symbols in `eqns`.
    method : {'auto', 'linear', 'linear_parametric', 'solve'}, default 'auto'
        Solver dispatch. 'auto' resolves against the `_solve_default` setting,
        then dispatches to the symbolic engine's general solver, falling back
        to dgcv's linear solver if the engine fails.

        The two linear methods both promise a linear system; they differ in how
        heavily the coefficients depend on symbolic parameters, which decides
        the fastest backend strategy:

        - 'linear': little or no parameter dependence (the common case).
          sympy dispatches to `linsolve`; sage to a fraction-field solve over
          the coefficients' own base field. `pass_to_symbolic_engine` False
          pins dgcv's matrix solver instead, in either engine.
        - 'linear_parametric': coefficients are large symbolic expressions.
          sage dispatches to its general solver, which handles these far better
          than elementwise matrix pivoting. That route is taken unless
          `pass_to_symbolic_engine` is False or `return_divisors` is set,
          either of which pins dgcv's own solver. On sympy this method behaves
          exactly like 'linear'.

        `return_divisors` pins dgcv's solver for both linear methods, since no
        engine route reports divisors.

        'linsolve' is accepted as a deprecated alias for 'linear'.
    surface_singularities : bool, default False
        Assume a linear system and use dgcv's own solver, forcing
        `simplify_pivots` and `return_divisors` to True and normalizing the
        returned divisors.

    Returns
    -------
    list of dict
        Solutions, empty when none exists. Paired with a divisor list when
        `return_divisors` is True, and with `system_vars` and `extra_vars`
        when `verbose` is True.
    """
    if method == "auto":
        method = get_dgcv_settings_registry().get("_solve_default", "auto")
    method = _METHOD_ALIASES.get(method, method)
    if surface_singularities is True:
        simplify_pivots = True
        return_divisors = True
        pass_to_symbolic_engine = False
    if pass_to_symbolic_engine is None:
        pass_to_symbolic_engine = (
            get_dgcv_settings_registry().get(
                "pass_solve_requests_to_symbolic_engine", False
            )
            is True
        )

    import time

    t0 = time.perf_counter()

    def _log(*a):
        if print_solve_stats:
            print("[solve_dgcv]", *a)

    if isinstance(eqns, (list, tuple)) and len(eqns) == 0:
        if isinstance(vars_to_solve, (list, tuple)):
            out = [{var: var for var in vars_to_solve}]
        elif isinstance(vars_to_solve, expr_numeric_types()):
            out = [{vars_to_solve: vars_to_solve}]
        else:
            out = [dict()]
        return (out, []) if return_divisors else out

    eqns, vars_to_solve = normalize_equations_and_vars(eqns, vars_to_solve)
    processed_eqns, system_vars, extra_vars, bridge = _equations_preprocessing(
        eqns, vars_to_solve
    )

    def _simplify(x):
        if not simplify_result:
            return x
        try:
            return simplify(x)
        except Exception:
            return x

    def _expr_reformatting(expr):
        return bridge.lift_expr(expr)

    def _extract_reformatting(var):
        return bridge.lift_var(var)

    mod = engine_module()

    def _engine_linsolve(eqns_, vars_):
        if mod is None:
            return None
        if engine_kind() == "sage":
            return _sage_engine_linsolve(eqns_, tuple(vars_))
        fn = getattr(mod, "linsolve", None)
        if not callable(fn):
            return None
        try:
            sols = fn(eqns_, tuple(vars_))
        except Exception:
            return None
        return _linsolve_to_dicts(sols, tuple(vars_))

    def _engine_solve(eqns_, vars_):
        if mod is None:
            return None
        fn = getattr(mod, "solve", None)
        if not callable(fn):
            return None
        sage_engine = engine_kind() == "sage"
        if (
            sage_engine
            and isinstance(eqns_, (list, tuple))
            and len(eqns_) == 1
            and len(vars_) > 1
        ):
            eqns_ = list(eqns_) * 2
        if sage_engine:
            try:
                sols = fn(eqns_, vars_)
            except Exception:
                return None
        else:
            try:
                sols = fn(eqns_, vars_, dict=True)
            except TypeError:
                try:
                    sols = fn(eqns_, vars_)
                except Exception:
                    return None
            except Exception:
                return None
        if sage_engine:
            input_symbols = set()
            for eqn in eqns_ if isinstance(eqns_, (list, tuple)) else [eqns_]:
                try:
                    input_symbols |= set(get_free_symbols(eqn))
                except Exception:
                    pass
            new_sols = _sage_solve_to_dicts(sols, vars_, input_symbols)
        else:
            new_sols = _engine_solve_to_dicts(sols, vars_)
        out = []
        for sol in new_sols:
            out.append({var: sol.get(var, var) for var in vars_})
        return out

    _log(
        f"engine={(engine_kind() or 'unknown')} method={method} #eqns={len(processed_eqns)} #vars={len(system_vars)}"
    )

    preformatted_solutions = []
    divisors = []
    custom_succeeded = False
    custom_ran = False
    engine_linsolve_ran = False
    engine_solve_failed = False

    def _run_custom_linsolve():
        nonlocal preformatted_solutions, divisors, custom_succeeded, custom_ran
        custom_ran = True
        try:
            _run_custom_linsolve_body()
        except NotImplementedError:
            raise
        except Exception:
            preformatted_solutions = []
            custom_succeeded = False

    def _run_custom_linsolve_body():
        nonlocal preformatted_solutions, divisors, custom_succeeded
        if True:
            if return_divisors:
                preformatted_solutions, d = _dgcv_linsolve(
                    processed_eqns,
                    system_vars,
                    return_divisors=True,
                    sample_if_overdetermined=sample_if_overdetermined,
                    allow_underdetermined_solution=allow_underdetermined_solution,
                    simplify_pivots=simplify_pivots,
                )
                divisors = d or []
            else:
                preformatted_solutions = _dgcv_linsolve(
                    processed_eqns,
                    system_vars,
                    return_divisors=False,
                    sample_if_overdetermined=sample_if_overdetermined,
                    allow_underdetermined_solution=allow_underdetermined_solution,
                    simplify_pivots=simplify_pivots,
                )
            custom_succeeded = True

    def _run_engine_linsolve():
        nonlocal preformatted_solutions, engine_linsolve_ran
        try:
            res = _engine_linsolve(processed_eqns, system_vars)
        except Exception:
            res = None
        engine_linsolve_ran = res is not None
        preformatted_solutions = res or []

    def _run_secondary_solve():
        nonlocal preformatted_solutions
        res = engine_capability("solve_secondary")(
            list(processed_eqns), list(system_vars)
        )
        preformatted_solutions = [
            {var: sol.get(var, var) for var in system_vars}
            for sol in (res or [])
            if isinstance(sol, dict)
        ]

    def _run_engine_solve():
        nonlocal preformatted_solutions, engine_solve_failed
        try:
            if engine_kind() == "sage" and not bridge.standins:
                res = _engine_solve(eqns, vars_to_solve)
            else:
                res = _engine_solve(processed_eqns, system_vars)
        except Exception:
            res = None
        engine_solve_failed = res is None
        preformatted_solutions = res or []

    if method not in ("auto", "solve") + _LINEAR_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Use 'auto', 'linear', "
            "'linear_parametric', or 'solve'."
        )

    if engine_kind() == "builtin":
        secondary_kind = engine_capability("secondary_kind")
        if method == "solve" and secondary_kind() is not None:
            _run_secondary_solve()
        else:
            try:
                _run_custom_linsolve_body()
            except NotImplementedError:
                if secondary_kind() is None:
                    raise
                _run_secondary_solve()
    elif method in _LINEAR_METHODS:
        prefer_engine_first = (
            method == "linear_parametric"
            and engine_kind() == "sage"
            and pass_to_symbolic_engine
            and not return_divisors
        )
        if prefer_engine_first:
            _run_engine_solve()
        if not preformatted_solutions and (
            return_divisors or not pass_to_symbolic_engine
        ):
            _run_custom_linsolve()
        if not custom_succeeded and not preformatted_solutions:
            _run_engine_linsolve()
        if (
            not custom_succeeded
            and not engine_linsolve_ran
            and not preformatted_solutions
        ):
            if not prefer_engine_first:
                _run_engine_solve()
            if engine_solve_failed and not custom_ran:
                _run_custom_linsolve()
    else:
        _run_engine_solve()
        if engine_solve_failed and not custom_ran:
            _run_custom_linsolve()

    solutions_formatted = [
        {
            _extract_reformatting(var): _expr_reformatting(_simplify(expr))
            for var, expr in (solution or {}).items()
        }
        for solution in (preformatted_solutions or [])
        if isinstance(solution, dict)
    ]

    _log(
        f"solutions={len(solutions_formatted)} elapsed_s={time.perf_counter() - t0:.6f}"
    )

    if return_divisors:
        present_atoms = set()
        returnables = []
        for v in divisors:
            atoms = get_free_symbols(v)
            if atoms:
                returnables.append(v)
                present_atoms |= atoms
        if surface_singularities:
            divisors = expr_union_primitives(
                returnables,
                order_coordinates(present_atoms),
                process_rationals=True,
                fail_quietly=True,
            )
        if verbose:
            return solutions_formatted, system_vars, extra_vars, divisors
        return solutions_formatted, divisors

    return (
        (solutions_formatted, system_vars, extra_vars)
        if verbose
        else solutions_formatted
    )


def _retry_normalized(eqns):
    try:
        if isinstance(eqns, (list, tuple)):
            return type(eqns)(simplify(eqn) for eqn in eqns)
        return simplify(eqns)
    except Exception:
        return None


def _no_solutions(result):
    payload = result[0] if isinstance(result, tuple) else result
    return not payload


def solve_knowing_solution_exists(eqns, vars_to_solve=None, *, try_hard=True, **kwargs):
    result = solve_dgcv(eqns, vars_to_solve, **kwargs)
    if not try_hard or not _no_solutions(result):
        return result
    normalized = _retry_normalized(eqns)
    if normalized is None:
        return result
    retried = solve_dgcv(normalized, vars_to_solve, **kwargs)
    return result if _no_solutions(retried) else retried
