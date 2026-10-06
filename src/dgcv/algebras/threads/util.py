from __future__ import annotations

import random
import uuid
from fractions import Fraction

from ..._aux._backends._engine import engine_capability
from ..._aux._backends._polynomials import (
    _ordered_factors,
    _unwrap_pow,
    as_numer_denom,
    make_poly,
    poly_coeffs,
    poly_monoms,
    poly_total_degree,
)
from ..._aux._backends._symbolic_router import (
    IndeterminateSignError,
    _scalar_is_zero,
    _scalar_sign,
    factor,
    get_free_symbols,
    ratio,
    simplify,
    subs,
)
from ..._aux._backends._types_and_constants import (
    _disposable_symbols,
    exact_fraction as _exact_fraction,
    rational,
    symbol,
)
from ..._aux._utilities._config import dgcv_warning, get_dgcv_settings_registry
from ..._aux._utilities._misc import zip_sum
from ..._aux._vmf._safeguards import create_key, get_dgcv_category
from ...core.arrays import _as_matrix_dgcv, matrix_dgcv
from ...core.arrays._indexing import _spool
from ...core.solvers import solve_dgcv
from .algebra_classifications import RealFormReport, complex_type_from_root_lengths

# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------


def fast_rank(mat, surface_singularities=False, simplify_singularities=None) -> int:
    M = _as_matrix_dgcv(mat)
    if M is None:
        M = matrix_dgcv(mat)
    return M.rank(
        allow_formal_inverse=surface_singularities,
        simplify_steps=False
        if not surface_singularities
        else simplify_singularities
        if simplify_singularities is not None
        else True,
        record_divisors=surface_singularities,
    )


def _projector_eigenspaces(specialized, expected, rng):
    projector = engine_capability("projector_eigenspaces")
    if projector is None:
        return None
    found = projector(specialized, expected, rng)
    if found is None:
        return None
    if _exact_builtin_paths():
        vectors = [
            {i: v for i, v in enumerate(col) if not _scalar_is_zero(v)}
            for block in found[0]
            for col in block
        ]
        if _span_solver.build_from_vectors(vectors) is None:
            return None
    return found


def _commutant_eigenspace_vectors(mat, free_vars, max_attempts=6):
    ordered_vars = sorted(free_vars, key=str)
    expected = len(ordered_vars)
    dim = mat.shape[0]
    engine_route = engine_capability("projector_eigenspaces") is not None
    for attempt in range(max_attempts):
        rng = random.Random(9000 + attempt)
        weights = rng.sample(range(1, 16 * (attempt + 2)), expected)
        specialized = mat.subs(dict(zip(ordered_vars, weights)))
        vectors = None
        if not (engine_route and get_free_symbols(specialized)):
            try:
                eigen_data = specialized._eigenvects_by_engine()
                if len(eigen_data) == expected:
                    vectors = [list(edata[-1]) for edata in eigen_data]
                    if sum(len(block) for block in vectors) != dim:
                        vectors = None
            except Exception:
                vectors = None
        if vectors is not None:
            return vectors, []
        found = _projector_eigenspaces(specialized, expected, rng)
        if found is not None:
            return found
    return None


def _centroid_ideal_components(solMat, free_vars, attempts=8):
    order = sorted(free_vars, key=str)
    var = symbol(create_key("_cpoly"))
    for attempt in range(attempts):
        bound = 8 * (attempt + 1)
        z = solMat.subs({v: random.randint(-bound, bound) for v in order})
        minimal_poly = _matrix_minimal_polynomial(z, len(order), var)
        if minimal_poly is None:
            continue
        degree, poly_expr = minimal_poly
        if degree != len(order):
            continue
        return _split_along_minimal_polynomial(z, poly_expr, var)
    raise RuntimeError(
        "decompose_semisimple_algebra could not certify a generator of the centroid "
        f"after {attempts} random specializations, so the decomposition into simple "
        "ideals cannot be trusted. This is expected when the given algebra is not "
        "semisimple."
    )


def _matrix_minimal_polynomial(mat, degree_bound, var):
    n = mat.nrows
    columns = []
    power = matrix_dgcv.identity(n)
    for step in range(degree_bound + 1):
        columns.append([power[i] for i in range(n * n)])
        if step < degree_bound:
            power = power @ mat
    relations = matrix_dgcv.from_cols(columns).nullspace()
    if not relations:
        return None
    relation = relations[0]
    degree = None
    for k in range(degree_bound, -1, -1):
        if not _scalar_is_zero(relation[k]):
            degree = k
            break
    if degree is None:
        return None
    lead = relation[degree]
    poly_expr = 0
    for k in range(degree + 1):
        c = relation[k]
        if _scalar_is_zero(c):
            continue
        poly_expr = poly_expr + ratio(c, lead) * var**k
    return degree, poly_expr


def _split_along_minimal_polynomial(mat, poly_expr, var):
    seen = set()
    irreducibles = []
    for f in _ordered_factors(factor(poly_expr)):
        base = _unwrap_pow(f)
        degree = poly_total_degree(base, [var])
        if not degree:
            continue
        stamp = str(base)
        if stamp in seen:
            continue
        seen.add(stamp)
        irreducibles.append((base, degree))

    components = []
    for base, degree in irreducibles:
        coeffs = _univariate_coefficients(base, var, degree)
        centroid_type = None
        if degree == 1:
            centroid_type = "real"
        elif degree == 2:
            try:
                if _scalar_sign(coeffs[1] ** 2 - 4 * coeffs[2] * coeffs[0]) < 0:
                    centroid_type = "complex"
            except IndeterminateSignError:
                pass
        vectors = _evaluate_polynomial_at_matrix(coeffs, mat).nullspace()
        if vectors:
            components.append((vectors, centroid_type))
    components.sort(key=_component_order_key)
    return components


def _univariate_coefficients(expr, var, degree):
    poly = make_poly(expr, [var])
    coeffs = [0] * (degree + 1)
    for monom, c in zip(poly_monoms(poly), poly_coeffs(poly)):
        coeffs[monom[0]] = c
    return coeffs


def _evaluate_polynomial_at_matrix(coeffs, mat):
    n = mat.nrows
    out = matrix_dgcv.zeros(n, n)
    power = matrix_dgcv.identity(n)
    last = len(coeffs) - 1
    for k, c in enumerate(coeffs):
        if not _scalar_is_zero(c):
            out = out + c * power
        if k < last:
            power = power @ mat
    return out


def _component_order_key(component):
    leads = []
    for v in component[0]:
        positions = [i for i in range(len(v)) if not _scalar_is_zero(v[i])]
        leads.append(positions[0] if positions else -1)
    return (len(component[0]), tuple(sorted(leads)))


def adjointRepresentation(alg, list_format=False, assume_Lie_algebra=False):
    if get_dgcv_category(alg) in {"algebra", "subalgebra"}:
        if assume_Lie_algebra is False and not alg.is_Lie_algebra():
            dgcv_warning(
                "The algebra passed to `adjointRepresentation` is not a Lie algebra; there is likely a mistake if applying  `adjointRepresentation`."
            )
        get_slice = alg._structure_data_slice
        shp = (alg.dimension, alg.dimension)
        return [
            matrix_dgcv(get_slice(idx), shape=shp).transpose()
            for idx in range(alg.dimension)
        ]
    else:
        raise Exception(
            "adjointRepresentation expected to receive an algebra instance."
        ) from None


def decompose_semisimple_algebra(
    alg,
    assume_semisimple=False,
    format_as_lists_of_elements=False,
    surface_singularities=False,
    simplify_singularities=None,
    return_centroid_types=False,
):
    """
    Decompose a semisimple Lie algebra into simple ideals.

    Parameters
    ----------
    alg : algebra_class or subalgebra_class
        Algebra to decompose.
    assume_semisimple : bool, default False
        Skip the semisimplicity check.
    format_as_lists_of_elements : bool, default False
        Return each ideal as a list of basis elements rather than a subalgebra.
    surface_singularities : bool, default False
        Also return parameter-space singularities raised by the linear solver.
    simplify_singularities : bool, optional
        Forwarded to the linear solver when surfacing singularities.
    return_centroid_types : bool, default False
        Also return one centroid type per ideal: `"real"` for an absolutely
        simple ideal, `"complex"` for a realification of a complex simple
        algebra, or `None` when undetermined.

    Returns
    -------
    list
        The simple ideals, followed by the centroid types and then the
        singularities when either is requested.
    """
    assert get_dgcv_category(alg) in {"algebra", "subalgebra"}

    sing = []

    def _package(components, centroid_types):
        if return_centroid_types is True:
            if surface_singularities is True:
                return components, tuple(centroid_types), sing
            return components, tuple(centroid_types)
        if surface_singularities is True:
            return components, sing
        return components

    def _whole_algebra(centroid_type):
        components = [list(alg.basis)] if format_as_lists_of_elements else [alg]
        return _package(components, [centroid_type])

    if alg.dimension == 0:
        return _package([alg], [None])
    if assume_semisimple is False and not alg.is_semisimple():
        raise TypeError(
            "decompose_semisimple_algebra was given a non-semisimple algebra to decompose."
        )

    n = alg.dimension
    get_slice = alg._structure_data_slice
    slice_shape = (n, n)
    mbasis = [
        matrix_dgcv(
            {(k, j): v for (j, k), v in get_slice(idx).items()}, shape=slice_shape
        )
        for idx in range(n)
    ]

    pref = create_key("_var")
    variables = _disposable_symbols(pref, n * n)
    vMat = matrix_dgcv(dict(enumerate(variables)), shape=(n, n))

    mats = []
    for mat in mbasis:
        comm = (vMat @ mat) - (mat @ vMat)
        mats += list(comm._data.values())
    if surface_singularities is True:
        sol, sing = solve_dgcv(
            mats,
            variables,
            method="linsolve",
            return_divisors=True,
            pass_to_symbolic_engine=False,
            simplify_pivots=simplify_singularities
            if simplify_singularities is not None
            else True,
            simplify_result=False,
        )
    else:
        sol = solve_dgcv(mats, variables, method="linsolve", simplify_result=False)
    if not sol:
        raise RuntimeError("solve_dgcv failed in decompose_semisimple_algebra.")

    solMat = vMat.subs(sol[0])

    free_vars = set()
    for v in solMat._data.values():
        if v is None:
            continue
        free_vars |= get_free_symbols(v)
    params = getattr(alg, "_parameters", set())
    if params:
        free_vars -= params
    if len(free_vars) < 2:
        return _whole_algebra("real")

    if params or getattr(alg, "base_field", "complex") == "complex":
        found = _commutant_eigenspace_vectors(solMat, free_vars)
        if found is None:
            raise RuntimeError(
                "decompose_semisimple_algebra failed, likely due to unsupported "
                "complexity in the algebra's parameter dependence. Adjusting the "
                "dgcv settings default engine may help."
            )
        blocks, divisors = found
        if surface_singularities is True:
            sing = list(sing) + [d for d in divisors if d not in sing]
        raw = [(vecs, None) for vecs in blocks]
    else:
        raw = _centroid_ideal_components(solMat, free_vars)
        if len(raw) == 1:
            return _whole_algebra(raw[0][1])

    simples = []
    centroid_types = []
    for vectors, centroid_type in raw:
        new_basis = [zip_sum(v, alg.basis) for v in vectors]
        if not new_basis:
            continue
        if format_as_lists_of_elements is True:
            simples.append(new_basis)
        else:
            ideal = alg.subalgebra(new_basis, simplify_basis=True)
            if centroid_type is not None:
                ideal._verified_ideal = True
                ideal._centroid_type = centroid_type
            simples.append(ideal)
        centroid_types.append(centroid_type)

    if not simples:
        return _whole_algebra(None)
    return _package(simples, centroid_types)


def killingForm(alg, assume_Lie_algebra=False):
    if get_dgcv_category(alg) not in {"algebra", "subalgebra"}:
        raise Exception(
            "killingForm expected to receive an algebra instance."
        ) from None
    if alg._killing_form is None:
        if assume_Lie_algebra is False and not alg.is_Lie_algebra():
            raise Exception(
                "killingForm expects argument to be a Lie algebra instance of the algebra"
            ) from None
        aRepLoc = adjointRepresentation(alg, assume_Lie_algebra=assume_Lie_algebra)
        dim = alg.dimension
        supports = [
            frozenset(key for key, _ in mat.iter_nonzero_items()) for mat in aRepLoc
        ]
        transposed = [frozenset((j, i) for i, j in keys) for keys in supports]
        entries = [[0] * dim for _ in range(dim)]
        for j in range(dim):
            for k in range(j, dim):
                if supports[j].isdisjoint(transposed[k]):
                    continue
                value = _trace_of_product(aRepLoc[j], aRepLoc[k])
                entries[j][k] = value
                entries[k][j] = value
        alg._killing_form = matrix_dgcv(entries)

    return alg._killing_form


def _combine_matrices(mats, coeffs):
    out = None
    for coeff, mat in zip(coeffs, mats):
        if _scalar_is_zero(coeff):
            continue
        term = coeff * mat
        out = term if out is None else out + term
    if out is None:
        return matrix_dgcv.zeros(mats[0].nrows, mats[0].ncols)
    return out


def _trace_of_product(left, right):
    total = 0
    data = right._data
    shape = right.shape
    for (i, j), value in left.iter_nonzero_items():
        entry = data.get(_spool((j, i), shape))
        if entry is None or _scalar_is_zero(entry):
            continue
        total = total + value * entry
    return total


def _cartan_casimir(ads, coeffs, dimension):
    cartan = _combine_matrices(ads, coeffs).nullspace()
    rank = len(cartan)
    if rank == 0 or rank >= dimension:
        return None
    adh = [_combine_matrices(ads, [vec[i] for i in range(dimension)]) for vec in cartan]
    for a in range(rank):
        for b in range(a + 1, rank):
            if any(True for _ in (adh[a] @ cartan[b]).iter_nonzero_items()):
                return None
    gram = matrix_dgcv(
        [[_trace_of_product(adh[a], adh[b]) for b in range(rank)] for a in range(rank)]
    )
    if _scalar_is_zero(gram.det()):
        return None
    inverse_gram = gram.inverse()
    casimir = None
    for a in range(rank):
        dual = _combine_matrices(adh, [inverse_gram[a, b] for b in range(rank)])
        term = adh[a] @ dual
        casimir = term if casimir is None else casimir + term
    return rank, casimir


def _as_fraction(value):
    try:
        numerator, denominator = as_numer_denom(value)
        return Fraction(int(numerator), int(denominator))
    except (TypeError, ValueError, AttributeError):
        return None


def _casimir_spectrum(casimir, rank, dimension):
    var = symbol(create_key("_rootLength"))
    minimal = _matrix_minimal_polynomial(casimir, 4, var)
    if minimal is None:
        return None
    degree, poly_expr = minimal
    if degree not in (2, 3):
        return None
    values = []
    for f in _ordered_factors(factor(poly_expr)):
        base = _unwrap_pow(f)
        base_degree = poly_total_degree(base, [var])
        if base_degree == 0:
            continue
        if base_degree != 1:
            return None
        coeffs = _univariate_coefficients(base, var, 1)
        value = _as_fraction(ratio(-coeffs[0], coeffs[1]))
        if value is None or value in values:
            return None
        values.append(value)
    if len(values) != degree or values.count(Fraction(0)) != 1:
        return None
    nonzero = sorted(value for value in values if value)
    root_count = dimension - rank
    if len(nonzero) == 1:
        multiplicities = [root_count]
    else:
        low, high = nonzero
        first = (rank - root_count * high) / (low - high)
        multiplicities = [first, root_count - first]
    lengths = []
    for value, count in zip(nonzero, multiplicities):
        if count <= 0 or count.denominator != 1:
            return None
        lengths.append((value, int(count)))
    if sum(value * count for value, count in lengths) != rank:
        return None
    if _as_fraction(_trace_of_product(casimir, casimir)) != sum(
        value * value * count for value, count in lengths
    ):
        return None
    return tuple(lengths)


def compute_root_length_profile(alg, attempts=4, assume_Lie_algebra=True):
    """
    Measure the root lengths of a complexified simple Lie algebra.

    Parameters
    ----------
    alg : algebra_class or subalgebra_class
        Assumed simple; the caller is responsible for that check.
    attempts : int, default 4
        Number of random elements tried before giving up.
    assume_Lie_algebra : bool, default True
        Skip the Lie algebra check when building the adjoint representation.

    Returns
    -------
    RealFormReport or None
        `None` when no attempt produced a Cartan subalgebra with a rational
        length spectrum, which includes the parametric case.

    Notes
    -----
    The centralizer of a generic element is a Cartan subalgebra `h` exactly
    when it is abelian and the Killing form restricted to it is
    nondegenerate, both of which are checked, so the rank reported here is
    exact rather than an approximation. The operator
    `sum_a ad(h_a) ad(k_a)`, with `k_a` the Killing dual basis of `h`, acts
    on each root space by the squared length of that root and vanishes on
    `h`, so its spectrum is the wanted data and stays rational over any
    real form.
    """
    if get_dgcv_category(alg) not in {"algebra", "subalgebra"}:
        raise Exception(
            "compute_root_length_profile expected to receive an algebra instance."
        ) from None
    dimension = alg.dimension
    if dimension < 3:
        return None
    ads = adjointRepresentation(alg, assume_Lie_algebra=assume_Lie_algebra)
    rng = random.Random(4177)
    for attempt in range(attempts):
        bound = 8 * (attempt + 1)
        coeffs = [rng.randint(-bound, bound) for _ in range(dimension)]
        data = _cartan_casimir(ads, coeffs, dimension)
        if data is None:
            continue
        rank, casimir = data
        lengths = _casimir_spectrum(casimir, rank, dimension)
        if lengths is None:
            continue
        candidates, certain = complex_type_from_root_lengths(dimension, rank, lengths)
        longest = max(value for value, _ in lengths)
        coxeter = 1 / longest
        return RealFormReport(
            lengths=lengths,
            dimension=dimension,
            rank=rank,
            dual_coxeter_number=int(coxeter) if coxeter.denominator == 1 else None,
            candidates=candidates,
            complex_type=candidates[0] if certain else None,
            certain=certain,
        )
    return None


def _ordered_union(first, second):
    out = list(first)
    seen = set(out)
    for item in second:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _fresh_solve_variables(count):
    pref = "v" + uuid.uuid4().hex[:8]
    return _disposable_symbols(pref, count)


def _solve_weight_kwargs(
    heavy, surface_singularities, simplify_singularities, method="linsolve"
):
    kwargs = {"method": method, "simplify_result": False}
    if surface_singularities:
        kwargs["return_divisors"] = True
        kwargs["pass_to_symbolic_engine"] = False
        if heavy:
            kwargs["simplify_pivots"] = True
        else:
            kwargs["simplify_pivots"] = (
                simplify_singularities if simplify_singularities is not None else True
            )
    elif heavy:
        kwargs["simplify_pivots"] = True
    return kwargs


def _indep_check(
    elems,
    newE,
    return_decomp_coeffs=False,
    print_solve_stats=False,
    method="linsolve",
    _solve_variables=None,
    surface_singularities=False,
    simplify_singularities=None,
    _force_eqn_simiplify=False,
    force_heavy_solve=False,
):
    if not isinstance(elems, (list, tuple)) or len(elems) == 0:
        if return_decomp_coeffs:
            return (True, {}, []) if surface_singularities else (True, {})
        return (True, []) if surface_singularities else True
    if _scalar_is_zero(newE):
        if return_decomp_coeffs:
            return (False, [{}], []) if surface_singularities else (False, [{}])
        return (False, []) if surface_singularities else False
    count = len(elems)
    if _solve_variables is None or len(_solve_variables) < count:
        variables = _fresh_solve_variables(count)
    else:
        variables = _solve_variables[:count]
    eqn = zip_sum(variables, elems) - newE
    if _force_eqn_simiplify or force_heavy_solve:
        eqn = simplify(eqn)

    solve_kwargs = _solve_weight_kwargs(
        force_heavy_solve,
        surface_singularities,
        simplify_singularities,
        method=method,
    )
    if surface_singularities:
        sol, sing = solve_dgcv(
            eqn,
            variables,
            print_solve_stats=print_solve_stats,
            **solve_kwargs,
        )
    else:
        sol = solve_dgcv(
            eqn,
            variables,
            print_solve_stats=print_solve_stats,
            **solve_kwargs,
        )
    if len(sol) == 0:
        if return_decomp_coeffs:
            return (True, [], sing) if surface_singularities else (True, [])
        return (True, sing) if surface_singularities else True
    if surface_singularities:
        sing = [subs(v, sol[0]) for v in sing]
    if return_decomp_coeffs:
        s = sol[0]
        coeffs = {idx: s.get(var, 0) for idx, var in enumerate(variables)}
        var_set = set(variables)
        free_vars = set()
        for c in coeffs.values():
            free_vars |= get_free_symbols(c)
        free_vars &= var_set
        if len(free_vars) == 0:
            coeffs = [coeffs]
        else:
            zeroing = {u: 0 for u in free_vars}
            expanded = []
            for v in sorted(free_vars, key=str):
                rule = {**zeroing, v: 1}
                expanded.append({idx: c.subs(rule) for idx, c in coeffs.items()})
            coeffs = expanded
        return (False, coeffs, sing) if surface_singularities else (False, coeffs)
    return (False, sing) if surface_singularities else False


def _elem_scale(elem, surface_singularities=False):
    coeffs = getattr(elem, "coeffs", None)
    if isinstance(coeffs, (list, tuple)):
        for c in coeffs:
            if not _scalar_is_zero(c):
                if get_free_symbols(c):
                    break
                try:
                    out = elem / c
                except Exception:
                    break
                return (out, []) if surface_singularities else out
    return (elem, []) if surface_singularities else elem


def _basis_builder(
    elems,
    newE,
    ALBS=False,
    print_solve_stats=False,
    method="linsolve",
    _solve_variables=None,
    surface_singularities=False,
    simplify_singularities=None,
    force_heavy_solve=False,
):
    if _scalar_is_zero(newE):
        return (list(elems), []) if surface_singularities else list(elems)
    if ALBS is True:
        newE = _elem_scale(newE, surface_singularities=surface_singularities)
        if surface_singularities:
            newE, sing = newE
    elif surface_singularities:
        sing = []
    if not isinstance(elems, (list, tuple)):
        raise TypeError(
            f"_basis_builder expects `elems` to be a list, recieved {elems} of type {type(elems)}"
        )
    if len(elems) == 0:
        out = ([newE], sing) if surface_singularities else [newE]
        return out
    check = _indep_check(
        elems,
        newE,
        print_solve_stats=print_solve_stats,
        method=method,
        return_decomp_coeffs=False,
        _solve_variables=_solve_variables,
        surface_singularities=surface_singularities,
        simplify_singularities=simplify_singularities,
        force_heavy_solve=force_heavy_solve,
    )
    if surface_singularities:
        check, sing2 = check
    if check is True:
        return (
            (list(elems) + [newE], _ordered_union(sing, sing2))
            if surface_singularities
            else list(elems) + [newE]
        )
    else:
        return (
            (list(elems), _ordered_union(sing, sing2))
            if surface_singularities
            else list(elems)
        )


_rank_extraction_stats = {"fast_path": 0, "fallback": {}}


def _exact_builtin_paths():
    return bool(
        get_dgcv_settings_registry().get("forgo_builtin_probabilistic_shortcuts", False)
    )


def rank_extraction_stats():
    return {
        "fast_path": _rank_extraction_stats["fast_path"],
        "fallback": dict(_rank_extraction_stats["fallback"]),
    }


def _decline(reason):
    counts = _rank_extraction_stats["fallback"]
    counts[reason] = counts.get(reason, 0) + 1
    return None


def _span_scalar(value):
    if value.denominator == 1:
        return rational(value.numerator)
    return rational(value.numerator, value.denominator)


def _elem_view(elem, alg):
    cd = getattr(elem, "coeff_dict", None)
    if not isinstance(cd, dict):
        return None
    ealg = getattr(elem, "algebra", None)
    if ealg is alg:
        return cd
    if (
        get_dgcv_category(elem) == "subalgebra_element"
        and getattr(ealg, "ambient", None) is alg
    ):
        rep = elem.ambient_rep
        cd = getattr(rep, "coeff_dict", None)
        if isinstance(cd, dict) and getattr(rep, "algebra", None) is alg:
            return cd
    return None


class _span_solver:
    __slots__ = (
        "alg",
        "dim",
        "count",
        "rows",
        "trans",
        "pivots",
        "numeric",
        "divisors",
        "_reported",
        "keys",
        "_field",
        "_pivot_index",
        "columns",
        "membership_only",
    )

    def __init__(self, alg, dim, numeric):
        self.alg = alg
        self.dim = dim
        self.count = 0
        self.rows = []
        self.trans = []
        self.pivots = []
        self.numeric = numeric
        self.divisors = []
        self._reported = 0
        self.keys = None
        self._field = None
        self._pivot_index = None
        self.columns = None
        self.membership_only = False

    @classmethod
    def build_from_vectors(cls, vectors, force_symbolic=False, membership_only=False):
        if not isinstance(vectors, (list, tuple)) or not vectors:
            return None
        if (
            get_dgcv_settings_registry().get("use_rank_basis_extraction", True)
            is not True
        ):
            return None
        index = {}
        numeric = not force_symbolic
        raw = []
        for vec in vectors:
            if not isinstance(vec, dict):
                return None
            out = {}
            for key, value in vec.items():
                if _scalar_is_zero(value):
                    continue
                row = index.get(key)
                if row is None:
                    row = len(index)
                    index[key] = row
                if numeric and _exact_fraction(value) is None:
                    numeric = False
                out[row] = value
            raw.append(out)
        field = engine_capability("span_field")
        if field is None and not numeric:
            return None
        if field is not None and not numeric:
            field = field.fresh(raw)
        solver = cls(None, len(index), numeric)
        solver._field = field
        solver.keys = index
        solver.membership_only = membership_only
        converted = []
        for vec in raw:
            b = solver._convert(vec)
            if b is None:
                return None
            converted.append(b)
        if membership_only:
            for b in converted:
                if not solver._insert_echelon(b):
                    return None
            return solver
        order = sorted(
            range(len(converted)),
            key=lambda i: (len(converted[i]), solver._weight(converted[i])),
        )
        for i in order:
            if not solver._insert(converted[i], i):
                return None
        solver.count = len(converted)
        return solver

    @classmethod
    def build_prefixes(cls, vectors, cuts):
        if not isinstance(vectors, (list, tuple)) or not vectors:
            return None
        if (
            get_dgcv_settings_registry().get("use_rank_basis_extraction", True)
            is not True
        ):
            return None
        cuts = sorted({c for c in cuts if 0 < c <= len(vectors)})
        if not cuts:
            return None
        index = {}
        numeric = True
        raw = []
        for vec in vectors[: cuts[-1]]:
            if not isinstance(vec, dict):
                return None
            out = {}
            for key, value in vec.items():
                if _scalar_is_zero(value):
                    continue
                row = index.get(key)
                if row is None:
                    row = len(index)
                    index[key] = row
                if numeric and _exact_fraction(value) is None:
                    numeric = False
                out[row] = value
            raw.append(out)
        field = engine_capability("span_field")
        if field is None and not numeric:
            return None
        if field is not None and not numeric:
            field = field.fresh(raw)
        solver = cls(None, len(index), numeric)
        solver._field = field
        solver.keys = index
        converted = []
        for vec in raw:
            b = solver._convert(vec)
            if b is None:
                return None
            converted.append(b)
        snapshots = []
        start = 0
        for cut in cuts:
            order = sorted(
                range(start, cut),
                key=lambda i: (len(converted[i]), solver._weight(converted[i])),
            )
            for i in order:
                if not solver._insert(converted[i], i):
                    return None
            snap = cls(None, solver.dim, numeric)
            snap._field = field
            snap.keys = index
            snap.count = cut
            snap.rows = [dict(r) for r in solver.rows]
            snap.trans = [dict(t) for t in solver.trans]
            snap.pivots = list(solver.pivots)
            snap.divisors = list(solver.divisors)
            snap.columns = (start, cut)
            snapshots.append(snap)
            start = cut
            solver.trans = [
                {i: v for i, v in t.items() if i >= cut} for t in solver.trans
            ]
        return snapshots

    def _covers(self, wanted):
        columns = self.columns
        if columns is None:
            return True
        if wanted is None:
            return False
        lo, hi = columns
        if isinstance(wanted, range):
            return not wanted or (wanted.start >= lo and wanted[-1] < hi)
        return all(lo <= i < hi for i in wanted)

    def reduce_vector(self, vec, wanted=None):
        index = self.keys
        if index is None or not isinstance(vec, dict):
            return None
        if not self._covers(wanted):
            return None
        b = {}
        for key, value in vec.items():
            if _scalar_is_zero(value):
                continue
            row = index.get(key)
            if row is None:
                return None
            b[row] = value
        b = self._convert(b)
        if b is None:
            return None
        _coords, m = self._reduce(b, wanted)
        if b:
            return None
        coeffs = [0] * self.count
        for i, v in m.items():
            coeffs[i] = self._lift(-v)
        return coeffs

    def reduce_vector_residual(self, vec):
        index = self.keys
        if index is None or not isinstance(vec, dict) or self.columns is not None:
            return None
        b = {}
        extra = []
        for key, value in vec.items():
            if _scalar_is_zero(value):
                continue
            row = index.get(key)
            if row is None:
                extra.append(value)
                continue
            b[row] = value
        b = self._convert(b)
        if b is None:
            return None
        _coords, m = self._reduce(b)
        coeffs = [0] * self.count
        for i, v in m.items():
            coeffs[i] = self._lift(-v)
        residual = [self._lift(v) for v in b.values()] + extra
        return coeffs, residual

    def _rows_of(self, vec, extend):
        index = self.keys
        b = {}
        for key, value in vec.items():
            if _scalar_is_zero(value):
                continue
            row = index.get(key)
            if row is None:
                if not extend:
                    return None
                row = len(index)
                index[key] = row
            b[row] = value
        self.dim = len(index)
        return b

    def insert_vector(self, vec, pivot_hint=None):
        if self.keys is None or not isinstance(vec, dict) or self.columns is not None:
            return None
        b = self._rows_of(vec, True)
        if not b:
            return False
        b = self._convert(b)
        if b is None:
            return None
        if self.membership_only:
            hint = None if pivot_hint is None else self.keys.get(pivot_hint)
            return self._insert_echelon(b, hint)
        return self._insert(b)

    def in_span(self, vec):
        if self.keys is None or not isinstance(vec, dict) or self.columns is not None:
            return None
        b = self._rows_of(vec, False)
        if b is None:
            return False
        if not b:
            return True
        b = self._convert(b)
        if b is None:
            return None
        if self.membership_only:
            self._reduce_echelon(b)
        else:
            self._reduce(b)
        return not b

    def pivot_keys(self):
        inverse = {row: key for key, row in self.keys.items()}
        return [inverse[p] for p in self.pivots]

    def _reduce_echelon(self, b):
        rows = self.rows
        pivots = self.pivots
        for a in range(len(rows)):
            c = b.get(pivots[a])
            if c is not None:
                self._axpy(b, c, rows[a])
        return b

    def _insert_echelon(self, b, pivot_hint=None):
        self._reduce_echelon(b)
        if not b:
            return False
        p = pivot_hint if pivot_hint is not None and pivot_hint in b else self._pick_pivot(b)
        piv = b[p]
        if self.numeric:
            if piv != 1:
                inv = Fraction(1) / piv
                b = {j: v * inv for j, v in b.items()}
        else:
            divisor = self._field.pivot_divisor(piv)
            if divisor is not None:
                self.divisors.append(divisor)
            if not piv.is_one:
                b = {j: v / piv for j, v in b.items()}
        self.rows.append(b)
        self.pivots.append(p)
        self.count += 1
        return True

    @classmethod
    def build(cls, elems):
        if not isinstance(elems, (list, tuple)) or not elems:
            return None
        if (
            get_dgcv_settings_registry().get("use_rank_basis_extraction", True)
            is not True
        ):
            return None
        alg = getattr(elems[0], "algebra", None)
        dim = getattr(alg, "dimension", None)
        if alg is None or not isinstance(dim, int):
            return None
        numeric = True
        vectors = []
        for elem in elems:
            cd = _elem_view(elem, alg)
            if cd is None:
                return None
            vec = {}
            for row, value in cd.items():
                if not isinstance(row, int) or row < 0 or row >= dim:
                    return None
                if _scalar_is_zero(value):
                    continue
                fr = _exact_fraction(value) if numeric else None
                if fr is None:
                    numeric = False
                vec[row] = value
            vectors.append(vec)
        field = engine_capability("span_field")
        if field is None and not numeric:
            return None
        if field is not None and not numeric:
            field = field.fresh(vectors)
        solver = cls(alg, dim, numeric)
        solver._field = field
        converted = []
        for vec in vectors:
            b = solver._convert(vec)
            if b is None:
                return None
            converted.append(b)
        order = sorted(
            range(len(converted)),
            key=lambda i: (len(converted[i]), solver._weight(converted[i])),
        )
        for i in order:
            if not solver._insert(converted[i], i):
                return None
        solver.count = len(converted)
        return solver

    def _weight(self, b):
        if self.numeric:
            return 0
        return self._field.weight(b)

    def _convert(self, vec):
        out = {}
        if self.numeric:
            for row, value in vec.items():
                fr = _exact_fraction(value)
                if fr is None:
                    return None
                if fr:
                    out[row] = fr
            return out
        return self._field.convert(vec)

    def _vector_of(self, elem):
        cd = _elem_view(elem, self.alg)
        if cd is None:
            return None
        vec = {}
        for row, value in cd.items():
            if not isinstance(row, int) or row < 0 or row >= self.dim:
                return None
            if not _scalar_is_zero(value):
                vec[row] = value
        return self._convert(vec)

    def _is_zero(self, e):
        return not e if self.numeric else e.is_zero

    def _axpy(self, target, c, source):
        for j, v in source.items():
            old = target.get(j)
            new = old - c * v if old is not None else -(c * v)
            if self._is_zero(new):
                if old is not None:
                    del target[j]
            else:
                target[j] = new

    def _reduce(self, b, wanted=None):
        coords = []
        m = {}
        pivots = self.pivots
        index = self._pivot_index
        if index is None or len(index) != len(pivots):
            index = self._pivot_index = {p: a for a, p in enumerate(pivots)}
        hits = [index[p] for p in b if p in index]
        if len(hits) > 1:
            hits.sort()
        if self.numeric:
            for a in hits:
                c = b.get(pivots[a])
                if c is None:
                    continue
                coords.append((a, c))
                self._axpy(b, c, self.rows[a])
                self._axpy(m, c, self.trans[a])
            return coords, m
        pending = {}
        for a in hits:
            c = b.get(pivots[a])
            if c is None:
                continue
            coords.append((a, c))
            self._axpy(b, c, self.rows[a])
            for i, v in self.trans[a].items():
                if wanted is not None and i not in wanted:
                    continue
                lst = pending.get(i)
                if lst is None:
                    pending[i] = [(c, v)]
                else:
                    lst.append((c, v))
        for i, lst in pending.items():
            total = self._field.msum(lst)
            if total is None:
                total = None
                for c, v in lst:
                    t = c * v
                    total = t if total is None else total + t
            if not total.is_zero:
                m[i] = -total
        return coords, m

    def _pick_pivot(self, b):
        best = None
        best_key = None
        for j, e in b.items():
            if self.numeric:
                key = (0, j)
            else:
                key = self._field.pivot_key(e) + (j,)
            if best_key is None or key < best_key:
                best_key = key
                best = j
        return best

    def _insert(self, b, idx=None):
        _coords, m = self._reduce(b)
        if not b:
            return False
        p = self._pick_pivot(b)
        piv = b[p]
        if idx is None:
            idx = self.count
            self.count += 1
        one = Fraction(1) if self.numeric else None
        if self.numeric:
            m[idx] = one
            if piv != 1:
                inv = one / piv
                b = {j: v * inv for j, v in b.items()}
                m = {i: v * inv for i, v in m.items()}
        else:
            field = self._field
            m[idx] = field.one
            divisor = field.pivot_divisor(piv)
            if divisor is not None:
                self.divisors.append(divisor)
            if not piv.is_one:
                b = {j: v / piv for j, v in b.items()}
                m = {i: v / piv for i, v in m.items()}
        for a in range(len(self.rows)):
            row = self.rows[a]
            c = row.get(p)
            if c is None:
                continue
            self._axpy(row, c, b)
            self._axpy(self.trans[a], c, m)
        self.rows.append(b)
        self.trans.append(m)
        self.pivots.append(p)
        return True

    def _singularities(self):
        out = self.divisors[self._reported :]
        self._reported = len(self.divisors)
        return out

    def _lift(self, value):
        field = self._field
        if self.numeric:
            return _span_scalar(value) if field is None else field.lift_number(value)
        return field.lift(value)

    def reduce(self, newE, return_decomp_coeffs=False, surface_singularities=False):
        if self.columns is not None:
            return None
        b = self._vector_of(newE)
        if b is None:
            return None
        coords, m = self._reduce(b)
        sing = self._singularities() if surface_singularities else None
        if b:
            if return_decomp_coeffs:
                return (True, [], sing) if surface_singularities else (True, [])
            return (True, sing) if surface_singularities else True
        if not return_decomp_coeffs:
            return (False, sing) if surface_singularities else False
        zero = 0
        coeffs = {idx: zero for idx in range(self.count)}
        for i, v in m.items():
            coeffs[i] = self._lift(-v)
        if surface_singularities:
            return (False, [coeffs], sing)
        return (False, [coeffs])

    def reduce_sparse(self, vec, surface_singularities=False):
        if self.columns is not None:
            return None
        dim = self.dim
        b = {}
        for row, value in vec.items():
            if not isinstance(row, int) or row < 0 or row >= dim:
                return None
            if not _scalar_is_zero(value):
                b[row] = value
        b = self._convert(b)
        if b is None:
            return None
        _coords, m = self._reduce(b)
        sing = self._singularities() if surface_singularities else None
        if b:
            return True, {}, sing
        coeffs = {}
        for i in sorted(m):
            coeffs[i] = self._lift(-m[i])
        return False, coeffs, sing

    def extend(self, newE, surface_singularities=False):
        if self.columns is not None:
            return None
        b = self._vector_of(newE)
        if b is None:
            return None
        added = self._insert(b)
        if surface_singularities:
            return added, self._singularities()
        return added


def _coefficient_matrix(prepared):
    if not prepared:
        return None
    alg = getattr(prepared[0][1], "algebra", None)
    dim = getattr(alg, "dimension", None)
    if alg is None or not isinstance(dim, int):
        return _decline("no_algebra_dimension")
    entries = {}
    for col, (_, elem) in enumerate(prepared):
        cd = getattr(elem, "coeff_dict", None)
        if not isinstance(cd, dict):
            return _decline("no_coeff_dict")
        if getattr(elem, "algebra", None) is not alg:
            return _decline("mixed_algebras")
        for row, value in cd.items():
            if not isinstance(row, int):
                return _decline("non_integer_key")
            if row < 0 or row >= dim:
                return _decline("key_out_of_range")
            if not _scalar_is_zero(value):
                entries[(row, col)] = value
    return matrix_dgcv(entries, shape=(dim, len(prepared)))


def _span_pick(prepared, surface_singularities):
    basis_pick = engine_capability("basis_pick")
    if basis_pick is not None:
        kept = basis_pick(prepared)
        if kept is not None:
            if not surface_singularities:
                return kept, []
            solver = _span_solver.build([prepared[p][1] for p in kept]) if kept else None
            if solver is not None or not kept:
                return kept, (solver._singularities() if solver is not None else [])
    solver = None
    pivots = []
    divisors = []
    for pos, (_, elem) in enumerate(prepared):
        if solver is None:
            cd = getattr(elem, "coeff_dict", None)
            if not isinstance(cd, dict) or all(_scalar_is_zero(v) for v in cd.values()):
                continue
            solver = _span_solver.build([elem])
            if solver is None:
                return None
            added = True
        else:
            added = solver.extend(elem)
            if added is None:
                return None
        if added:
            pivots.append(pos)
    if solver is not None and surface_singularities:
        divisors = solver._singularities()
    return pivots, divisors


def _extract_basis_by_rank(
    element_list,
    ALBS=False,
    print_solve_stats=False,
    method="linsolve",
    _solve_variables=None,
    return_indices=False,
    surface_singularities=False,
    simplify_singularities=None,
    force_heavy_solve=False,
):
    if not isinstance(element_list, (list, tuple)):
        element_list = list(element_list)

    prepared = []
    sing = []
    for idx, elem in enumerate(element_list):
        if _scalar_is_zero(elem):
            continue
        if ALBS is True:
            scaled = _elem_scale(elem, surface_singularities=surface_singularities)
            if surface_singularities:
                elem, new_sing = scaled
                sing = _ordered_union(sing, new_sing)
            else:
                elem = scaled
        prepared.append((idx, elem))

    picked = _span_pick(prepared, surface_singularities)
    if picked is not None:
        _rank_extraction_stats["fast_path"] += 1
        pivots, divisors = picked
        basis = [prepared[p][1] for p in pivots]
        idxs = [prepared[p][0] for p in pivots] if return_indices else None
        if surface_singularities:
            sing = _ordered_union(sing, divisors)
            return (basis, idxs, sing) if return_indices else (basis, sing)
        return (basis, idxs) if return_indices else basis

    mat = _coefficient_matrix(prepared)
    if mat is None:
        return _extract_basis_incremental(
            element_list,
            ALBS=ALBS,
            print_solve_stats=print_solve_stats,
            method=method,
            _solve_variables=_solve_variables,
            return_indices=return_indices,
            surface_singularities=surface_singularities,
            simplify_singularities=simplify_singularities,
            force_heavy_solve=force_heavy_solve,
        )

    _rank_extraction_stats["fast_path"] += 1
    if surface_singularities:
        pivots, divisors = mat.pivot_columns(record_divisors=True)
        sing = _ordered_union(sing, divisors)
    else:
        pivots = mat.pivot_columns()
    basis = [prepared[p][1] for p in pivots]
    idxs = [prepared[p][0] for p in pivots] if return_indices else None

    if surface_singularities:
        return (basis, idxs, sing) if return_indices else (basis, sing)
    return (basis, idxs) if return_indices else basis


def _extract_basis(
    element_list,
    ALBS=False,
    print_solve_stats=False,
    method="linsolve",
    _solve_variables=None,
    return_indices=False,
    surface_singularities=False,
    simplify_singularities=None,
    force_heavy_solve=False,
    use_rank=None,
):
    if use_rank is None:
        use_rank = (
            get_dgcv_settings_registry().get("use_rank_basis_extraction", True) is True
        )
    impl = _extract_basis_by_rank if use_rank else _extract_basis_incremental
    return impl(
        element_list,
        ALBS=ALBS,
        print_solve_stats=print_solve_stats,
        method=method,
        _solve_variables=_solve_variables,
        return_indices=return_indices,
        surface_singularities=surface_singularities,
        simplify_singularities=simplify_singularities,
        force_heavy_solve=force_heavy_solve,
    )


def _extract_basis_incremental(
    element_list,
    ALBS=False,
    print_solve_stats=False,
    method="linsolve",
    _solve_variables=None,
    return_indices=False,
    surface_singularities=False,
    simplify_singularities=None,
    force_heavy_solve=False,
):
    if not isinstance(element_list, (list, tuple)):
        element_list = list(element_list)
    basis = []
    idxs = [] if return_indices else None
    sing = []
    if _solve_variables is None and len(element_list) > 0:
        _solve_variables = _fresh_solve_variables(len(element_list))
    for i, newE in enumerate(element_list):
        old_len = len(basis)
        basis = _basis_builder(
            basis,
            newE,
            ALBS=ALBS,
            print_solve_stats=print_solve_stats,
            method=method,
            _solve_variables=_solve_variables,
            surface_singularities=surface_singularities,
            simplify_singularities=simplify_singularities,
            force_heavy_solve=force_heavy_solve,
        )

        if surface_singularities:
            basis, new_sing = basis
            sing = _ordered_union(sing, new_sing)

        if return_indices and len(basis) == old_len + 1:
            idxs.append(i)
    if surface_singularities:
        out = (basis, idxs, sing) if return_indices else (basis, sing)
    else:
        out = (basis, idxs) if return_indices else basis
    return out
