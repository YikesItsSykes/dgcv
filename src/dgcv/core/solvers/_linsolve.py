import random

from ..._aux._backends._calculus import diff
from ..._aux._backends._engine import _get_sage_module, engine_capability, engine_kind
from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    get_free_symbols,
    subs,
)
from ..._aux._backends._types_and_constants import rational
from ..arrays import matrix_dgcv
from ._predicates import _as_zero_expr, _is_zero, _is_zero_after_simplify

_fraction_field_solver_cache = {}
_fraction_field_cache_limit = 8


def _canonical_row_order(coeff_entries, rhs, n_rows):
    rows = [[] for _ in range(n_rows)]
    for (row_idx, col_idx), value in coeff_entries.items():
        rows[row_idx].append((col_idx, value))
    keyed = []
    for row_idx in range(n_rows):
        entries = sorted(rows[row_idx], key=lambda item: item[0])
        keyed.append((tuple((col, str(val)) for col, val in entries), row_idx, entries))
    keyed.sort(key=lambda item: item[0])

    ordered_entries = {}
    for new_row, (_signature, _old_row, entries) in enumerate(keyed):
        for col_idx, value in entries:
            ordered_entries[(new_row, col_idx)] = value
    signature = tuple(item[0] for item in keyed)
    ordered_rhs = [rhs[item[1]] for item in keyed]
    return signature, ordered_entries, ordered_rhs


def _coerce_system(sage, F, coeff_entries, rhs, n_rows, n_cols):
    try:
        entries = {key: F(value) for key, value in coeff_entries.items()}
        A = sage.matrix(F, n_rows, n_cols, entries, sparse=True)
        b = sage.vector(F, [F(entry) for entry in rhs])
    except (TypeError, ValueError, NotImplementedError, ArithmeticError):
        return None
    return A, b


def _solve_over_field_with(sage, A, b, vars_, system_vars):
    n_cols = len(vars_)
    reduced = A.augment(b.column()).rref()

    pivot_cols = list(reduced.pivots())
    if n_cols in pivot_cols:
        return []
    free_cols = [col for col in range(n_cols) if col not in set(pivot_cols)]

    SR = sage.SR
    x_general = [SR(0)] * n_cols
    for col in free_cols:
        x_general[col] = SR(vars_[col])

    for row_idx, pivot_col in enumerate(pivot_cols):
        value = SR(reduced[row_idx, n_cols])
        for col in free_cols:
            coeff = reduced[row_idx, col]
            if not coeff.is_zero():
                value = value - SR(coeff) * SR(vars_[col])
        x_general[pivot_col] = value

    return [dict(zip(system_vars, x_general))]


_specialization_primes = (2147483647, 2147483629, 2147483587, 2147483579)
_specialization_attempts = 4


def _specialization_targets(sage, base):
    if base is sage.QQ:
        for prime in _specialization_primes:
            Fp = sage.GF(prime)
            yield Fp, Fp
        return
    defining = getattr(base, "defining_polynomial", None)
    if defining is None:
        return
    try:
        poly = defining()
    except Exception:
        return
    for prime in _specialization_primes:
        Fp = sage.GF(prime)
        try:
            roots = poly.change_ring(Fp).roots()
        except Exception:
            continue
        if not roots:
            continue
        root = roots[0][0]

        def convert(value, Fp=Fp, root=root):
            total = Fp(0)
            power = Fp(1)
            for coordinate in value.list():
                total += Fp(coordinate) * power
                power *= root
            return total

        yield Fp, convert


def _evaluate_at_point(poly, point, Fp, convert):
    total = Fp(0)
    for coeff, mono in zip(poly.coefficients(), poly.monomials()):
        exponents = mono.exponents()[0]
        if not hasattr(exponents, "__len__"):
            exponents = (exponents,)
        term = convert(coeff)
        for value, exponent in zip(point, exponents):
            if exponent:
                term *= value**exponent
        total += term
    return total


def _specialized_pivot_rows(F, A, n_rows, n_cols):
    try:
        sage = _get_sage_module()
        R = F.ring()
        gens = R.gens()
        base = R.base_ring()
    except Exception:
        return None
    if not gens:
        return None
    try:
        entries = [
            ((i, j), A[i, j])
            for i in range(n_rows)
            for j in range(n_cols)
            if not A[i, j].is_zero()
        ]
    except Exception:
        return None
    for Fp, convert in _specialization_targets(sage, base):
        prime = int(Fp.characteristic())
        for _ in range(_specialization_attempts):
            point = [Fp(random.randrange(2, prime - 1)) for _ in gens]
            data = {}
            failed = False
            try:
                for key, value in entries:
                    den = _evaluate_at_point(R(value.denominator()), point, Fp, convert)
                    if den == 0:
                        failed = True
                        break
                    num = _evaluate_at_point(R(value.numerator()), point, Fp, convert)
                    if num != 0:
                        data[key] = num / den
                if failed:
                    continue
                M = sage.matrix(Fp, n_rows, n_cols, data)
                return int(M.rank()), list(M.pivot_rows())
            except Exception:
                return None
    return None


def _prepare_frac_field_solver(F, A, n_rows, n_cols):
    prepared = {"F": F, "A": A, "pivots": None, "inverse": None, "extra_rows": ()}
    specialized = _specialized_pivot_rows(F, A, n_rows, n_cols)
    if specialized is not None:
        rank, pivots = specialized
        if rank != n_cols:
            return prepared
    else:
        try:
            pivots = list(A.pivot_rows())
        except Exception:
            return prepared
        if len(pivots) != n_cols:
            return prepared
    try:
        inverse = A.matrix_from_rows(pivots).inverse()
    except Exception:
        return prepared
    pivot_rows = set(pivots)
    prepared["pivots"] = pivots
    prepared["inverse"] = inverse
    prepared["extra_rows"] = tuple(
        (row_idx, [A[row_idx, col_idx] for col_idx in range(n_cols)])
        for row_idx in range(n_rows)
        if row_idx not in pivot_rows
    )
    return prepared


def _apply_frac_field_solver(sage, prepared, rhs, vars_, system_vars):
    F = prepared["F"]
    try:
        b = [F(entry) for entry in rhs]
    except (TypeError, ValueError, NotImplementedError, ArithmeticError):
        return None

    inverse = prepared["inverse"]
    if inverse is None:
        return _solve_over_field_with(
            sage, prepared["A"], sage.vector(F, b), vars_, system_vars
        )

    pivots = prepared["pivots"]
    x = inverse * sage.vector(F, [b[row_idx] for row_idx in pivots])
    for row_idx, row in prepared["extra_rows"]:
        total = F(0)
        for col_idx, coeff in enumerate(row):
            if not coeff.is_zero():
                total += coeff * x[col_idx]
        if not (total - b[row_idx]).is_zero():
            return []

    SR = sage.SR
    return [dict(zip(system_vars, [SR(entry) for entry in x]))]


def _store_frac_field_solver(cache_key, prepared):
    if len(_fraction_field_solver_cache) >= _fraction_field_cache_limit:
        _fraction_field_solver_cache.clear()
    _fraction_field_solver_cache[cache_key] = prepared


def _sage_solve_over_field(sage, F, coeff_entries, rhs, n_rows, vars_, system_vars):
    coerced = _coerce_system(sage, F, coeff_entries, rhs, n_rows, len(vars_))
    if coerced is None:
        return None
    A, b = coerced
    return _solve_over_field_with(sage, A, b, vars_, system_vars)


def _sage_frac_field_solve(sage, coeff_entries, rhs, n_rows, vars_, system_vars):
    signature, coeff_entries, rhs = _canonical_row_order(coeff_entries, rhs, n_rows)
    n_cols = len(vars_)
    SR = sage.SR
    values = list(coeff_entries.values()) + [
        entry for entry in rhs if not _scalar_is_zero(entry)
    ]
    atoms = sorted(
        {str(atom) for value in values for atom in SR(value).variables()}
        - {str(v) for v in vars_}
    )

    cache_key = (signature, n_rows, n_cols, tuple(atoms))
    prepared = _fraction_field_solver_cache.get(cache_key)
    if prepared is not None:
        solution = _apply_frac_field_solver(sage, prepared, rhs, vars_, system_vars)
        if solution is not None:
            return solution

    def field_over(base):
        return sage.PolynomialRing(base, atoms).fraction_field() if atoms else base

    for base in (sage.QQ, sage.QQ[sage.I]):
        F = field_over(base)
        coerced = _coerce_system(sage, F, coeff_entries, rhs, n_rows, n_cols)
        if coerced is None:
            continue
        A, b = coerced
        prepared = _prepare_frac_field_solver(F, A, n_rows, n_cols)
        _store_frac_field_solver(cache_key, prepared)
        solution = _apply_frac_field_solver(sage, prepared, rhs, vars_, system_vars)
        if solution is not None:
            return solution
        return _solve_over_field_with(sage, A, b, vars_, system_vars)

    F_algebraic = field_over(sage.QQbar)
    try:
        algebraic_values = [F_algebraic(value) for value in values]
    except (TypeError, ValueError, NotImplementedError, ArithmeticError):
        return None

    constants = set()
    for value in algebraic_values:
        if atoms:
            constants.update(value.numerator().coefficients())
            constants.update(value.denominator().coefficients())
        else:
            constants.add(value)

    K, _images, _embedding = sage.number_field_elements_from_algebraics(
        sorted(constants, key=str), embedded=True
    )
    return _sage_solve_over_field(
        sage, field_over(K), coeff_entries, rhs, n_rows, vars_, system_vars
    )


def _sage_engine_linsolve(processed_eqns, system_vars):
    if engine_kind() != "sage" or not processed_eqns or not system_vars:
        return None

    sage = _get_sage_module()
    SR = sage.SR
    vars_ = tuple(system_vars)
    zero = rational(0, 1)
    var_columns = {}
    for col_idx, var in enumerate(vars_):
        var_columns[str(var)] = col_idx

    coeff_entries = {}
    rhs = []
    for row_idx, eq in enumerate(processed_eqns):
        expr = _as_zero_expr(eq)
        try:
            present = []
            for atom in SR(expr).variables():
                col_idx = var_columns.get(str(atom))
                if col_idx is not None:
                    present.append((col_idx, atom))
            if not present:
                rhs.append(-expr)
                continue
            for col_idx, atom in present:
                coeff = diff(expr, atom)
                if not _scalar_is_zero(coeff):
                    coeff_entries[(row_idx, col_idx)] = coeff
            rhs.append(-subs(expr, {atom: zero for _, atom in present}))
        except Exception:
            return None

    return _sage_frac_field_solve(
        sage, coeff_entries, rhs, len(processed_eqns), vars_, system_vars
    )


def _dgcv_linsolve(
    processed_eqns,
    system_vars,
    *,
    return_divisors=False,
    simplify_pivots=False,
    sample_if_overdetermined=False,
    allow_underdetermined_solution=True,
):
    if not processed_eqns:
        out = [{v: 0 for v in system_vars}]
        return (out, []) if return_divisors else out

    vars_ = tuple(system_vars)
    n = len(vars_)

    if n == 0:
        ok = True
        for e in processed_eqns:
            try:
                ok = _is_zero(_as_zero_expr(e))
            except Exception:
                ok = False
            if not ok:
                break
        out = [dict()] if ok else []
        return (
            (out, [e for e in processed_eqns if get_free_symbols(e)])
            if return_divisors
            else out
        )

    zero = rational(0, 1)
    var_columns = {str(v): idx for idx, v in enumerate(vars_)}

    def _build_and_solve(eqns):
        builtin_linsolve = engine_capability("linsolve")
        if builtin_linsolve is not None:
            builtin = builtin_linsolve(eqns, vars_, return_divisors)
            if builtin is not None:
                return builtin
        coeff_entries = {}
        rhs = []
        row_count = 0
        for eq in eqns:
            expr = _as_zero_expr(eq)
            present = []
            for atom in get_free_symbols(expr):
                col_idx = var_columns.get(str(atom))
                if col_idx is not None:
                    present.append((col_idx, atom))

            if present:
                c0 = subs(expr, {atom: zero for _, atom in present})
            else:
                c0 = expr

            row_has_entry = False
            for col_idx, atom in present:
                coeff = diff(expr, atom)
                if _is_zero(coeff):
                    continue
                coeff_entries[row_count * n + col_idx] = coeff
                row_has_entry = True

            if not row_has_entry and not _is_zero_after_simplify(c0):
                return ([], [c0]) if get_free_symbols(c0) else ([], [])

            rhs.append(-c0)
            row_count += 1

        A = matrix_dgcv(coeff_entries, shape=(row_count, n))
        b = matrix_dgcv(
            {idx: value for idx, value in enumerate(rhs) if not _is_zero(value)},
            shape=(row_count, 1),
        )
        sol, divs = A.solve_right(
            b,
            return_divisors=True,
            allow_formal_inverse=True,
            parametric_vars=vars_,
            simplify_steps=simplify_pivots,
        )

        if sol is None:
            return [], divs

        return (dict(zip(system_vars, sol)), divs)

    # determine which equations to use
    m = len(processed_eqns)
    if sample_if_overdetermined is True:
        sample_if_overdetermined = 4
    use_sampling = (
        isinstance(sample_if_overdetermined, (int, float))
        and sample_if_overdetermined > 0
        and m > sample_if_overdetermined * n
    )

    if use_sampling:
        sample_size = max(n, int(sample_if_overdetermined * n))
        sampled_eqns = random.sample(processed_eqns, min(sample_size, m))
        sol_dict, divs = _build_and_solve(sampled_eqns)
        if (
            not allow_underdetermined_solution
            and isinstance(sol_dict, dict)
            and any(set(vars_) & get_free_symbols(v) for v in sol_dict.values())
        ):
            sol_dict, divs = _build_and_solve(processed_eqns)
    else:
        sol_dict, divs = _build_and_solve(processed_eqns)

    if not isinstance(sol_dict, dict):
        return ([], divs) if return_divisors else []

    out = [sol_dict]
    return (out, divs) if return_divisors else out
