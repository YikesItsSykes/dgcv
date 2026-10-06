from __future__ import annotations

from ..._aux._backends._polynomials import expr_union_primitives
from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    get_free_symbols,
    simplify,
    subs,
)
from ..._aux._utilities._config import dgcv_warning, get_dgcv_settings_registry
from ..._aux._utilities._misc import linear_combination, zip_sum
from ..._aux._vmf._safeguards import get_dgcv_category
from ..._aux._vmf.vmf import order_coordinates
from ...core.solvers import solve_dgcv
from .heating import _timed_progress_call
from .util import (
    _indep_check,
    _solve_weight_kwargs,
    _span_solver,
    decompose_semisimple_algebra,
)


def _ldc(
    target_alg,
    decompose_semisimple_fully=False,
    _bust_cache=False,
    assume_Lie_algebra=False,
    verbose=False,
    surface_singularities=None,
    simplify_singularities=None,
    force_heavy_solve=False,
    _timed_reporting: bool | None = None,
    _reporting_threshold_s: float = 10,
    _progress_message: str | None = None,
    _on_timed_update=None,
):
    timed = bool(_timed_reporting) if _timed_reporting is not None else False
    threshold = float(_reporting_threshold_s)

    def _time_call(fn, step_desc: str, continue_desc: str | None):
        return _timed_progress_call(
            fn,
            timed=timed,
            threshold_s=threshold,
            step_desc=step_desc,
            continue_desc=continue_desc,
            progress_message=None,
            _on_timed_update=_on_timed_update,
        )

    if _bust_cache:
        target_alg._radical_cache = None
        target_alg._derived_series_cache = None
        target_alg._lower_central_series_cache = None
        target_alg._derived_subalg_cache = None
        target_alg._derived_subalg_lazy_cache = None
    if surface_singularities is None:
        surface_singularities = True if target_alg._parameters else False
    surface_singularities = bool(surface_singularities)
    if surface_singularities:
        sing = []
    if target_alg._Levi_deco_cache is None:
        if target_alg._educed_properties.get("special_type", None) in {
            "simple",
            "semisimple",
        }:
            target_alg._Levi_deco_cache = {
                "LD_components": (target_alg, target_alg.subalgebra([])),
                "simple_ideals": None,
            }
        elif target_alg._educed_properties.get("special_type", None) in {
            "nilpotent",
            "solvable",
            "abelian",
        }:
            target_alg._Levi_deco_cache = {
                "LD_components": (target_alg.subalgebra([]), target_alg),
                "simple_ideals": None,
            }
        else:
            if verbose is True:
                print("Deriving (or retrieving) maximal solvable ideal...")

            rad = _time_call(
                lambda: target_alg.radical(
                    assume_Lie_algebra=assume_Lie_algebra,
                    surface_singularities=surface_singularities,
                    simplify_singularities=simplify_singularities,
                    force_heavy_solve=force_heavy_solve,
                ),
                "deriving the maximal solvable ideal",
                "compute the max. solvable ideal's derived series",
            )
            if surface_singularities:
                sing += getattr(target_alg, "_singularities", {}).get("radical", [])
                new_sing = target_alg._singularities.get("LD", []) + [
                    v for v in sing if get_free_symbols(v)
                ]
                if get_dgcv_settings_registry().get(
                    "simplify_singularity_ideals_by_default", True
                ):
                    new_sing = expr_union_primitives(
                        new_sing,
                        order_coordinates(target_alg._parameters),
                        process_rationals=True,
                        fail_quietly=True,
                    )
                target_alg._singularities["LD"] = new_sing
            if len(rad.basis) > 0:
                if verbose is True:
                    print(
                        "Finding a semisimple complement to the max. solvable ideal..."
                    )

                rad_seq = _time_call(
                    lambda: rad.derived_series(
                        align_nested_bases=True,
                        surface_singularities=surface_singularities,
                        simplify_singularities=simplify_singularities,
                        force_heavy_solve=force_heavy_solve,
                    ),
                    "computing the max. solvable ideal's derived series",
                    "compute a semisimple complement to the maximal solvable ideal",
                )
                if surface_singularities:
                    sing += getattr(rad, "_singularities", {}).get(
                        "derived_series", set()
                    )
                    new_sing = target_alg._singularities.get("LD", []) + [
                        v for v in sing if get_free_symbols(v)
                    ]
                    if get_dgcv_settings_registry().get(
                        "simplify_singularity_ideals_by_default", True
                    ):
                        new_sing = expr_union_primitives(
                            new_sing,
                            order_coordinates(target_alg._parameters),
                            process_rationals=True,
                            fail_quietly=True,
                        )
                    target_alg._singularities["LD"] = new_sing

                def _compute_complement():
                    local_rad_seq = list(rad_seq) if rad_seq else []
                    if local_rad_seq and local_rad_seq[-1] == []:
                        local_rad_seq = local_rad_seq[:-1]  ###!!! note convention
                    local_rad_seq.append([])

                    discrep = target_alg.dimension - len(local_rad_seq[0])
                    naiveBasis = []
                    augment_NB = list(local_rad_seq[0])
                    solver = None
                    if augment_NB and not force_heavy_solve:
                        solver = _span_solver.build(
                            [
                                e.ambient_rep
                                if get_dgcv_category(e) == "subalgebra_element"
                                and getattr(e.algebra, "ambient", None) is target_alg
                                else e
                                for e in augment_NB
                            ]
                        )
                    for elem in target_alg.basis:
                        if len(naiveBasis) == discrep:
                            break
                        indep = None
                        if solver is None and not augment_NB and not force_heavy_solve:
                            solver = _span_solver.build([elem])
                            if solver is not None:
                                indep = (True, []) if surface_singularities else True
                        elif solver is not None:
                            indep = solver.extend(
                                elem, surface_singularities=surface_singularities
                            )
                        if indep is None:
                            solver = None
                            indep = _indep_check(
                                augment_NB,
                                elem,
                                surface_singularities=surface_singularities,
                                force_heavy_solve=force_heavy_solve,
                            )
                        if surface_singularities:
                            indep, sing = indep
                            new_sing = target_alg._singularities.get("LD", []) + [
                                v for v in sing if get_free_symbols(v)
                            ]
                            if get_dgcv_settings_registry().get(
                                "simplify_singularity_ideals_by_default", True
                            ):
                                new_sing = expr_union_primitives(
                                    new_sing,
                                    order_coordinates(target_alg._parameters),
                                    process_rationals=True,
                                    fail_quietly=True,
                                )
                            target_alg._singularities["LD"] = new_sing
                        if indep:
                            augment_NB.append(elem)
                            naiveBasis.append(elem)
                    ss_dim = len(naiveBasis)

                    for idx in range(len(local_rad_seq)):
                        if idx == len(local_rad_seq) - 1:
                            compare_set = local_rad_seq[idx]
                            quot_set = []
                            rad_discrep = len(local_rad_seq[idx])
                        else:
                            rad_discrep = len(local_rad_seq[idx]) - len(
                                local_rad_seq[idx + 1]
                            )
                            compare_set = local_rad_seq[idx][:rad_discrep]
                            quot_set = local_rad_seq[idx][rad_discrep:]
                        compLen = len(compare_set)

                        variables = []
                        basis_modifiers = []
                        for count in range(len(naiveBasis)):
                            if compLen > 0:
                                w_sum, w_vars = linear_combination(
                                    compare_set,
                                    prefix=f"_v_{count}_",
                                    _disposable=True,
                                )
                                variables += w_vars
                                basis_modifiers.append(w_sum)
                            else:
                                basis_modifiers.append(0 * naiveBasis[0])

                        leading_coeffs = {}
                        trailing_coeffs = {}
                        eqns = []
                        level_basis = naiveBasis + local_rad_seq[idx]
                        level_solver = (
                            None if force_heavy_solve else _span_solver.build(level_basis)
                        )
                        for idx1 in range(ss_dim):
                            for idx2 in range(idx1 + 1, ss_dim):
                                w1, w2 = naiveBasis[idx1], naiveBasis[idx2]
                                lb = w1 * w2
                                surfacing = (
                                    True
                                    if target_alg._parameters or surface_singularities
                                    else False
                                )
                                lb_decomp = None
                                if level_solver is not None:
                                    lb_decomp = level_solver.reduce(
                                        lb,
                                        return_decomp_coeffs=True,
                                        surface_singularities=surfacing,
                                    )
                                reduced = lb_decomp is not None
                                if not reduced:
                                    lb_decomp = _indep_check(
                                        level_basis,
                                        lb,
                                        return_decomp_coeffs=True,
                                        surface_singularities=surfacing,
                                        force_heavy_solve=force_heavy_solve,
                                    )
                                if (
                                    lb_decomp[0] is True
                                    and not force_heavy_solve
                                    and not reduced
                                ):
                                    dgcv_warning(
                                        "The Levi decomposition algorithm encountered a bug caused by solver failing to recognize a zero. Retrying now with the heavier solve algorithm.",
                                        wc_label="debug_log",
                                    )
                                    lb_decomp = _indep_check(
                                        level_basis,
                                        lb,
                                        return_decomp_coeffs=True,
                                        surface_singularities=surfacing,
                                        _force_eqn_simiplify=True,
                                        force_heavy_solve=True,
                                    )

                                if surfacing:
                                    new_sing = target_alg._singularities.get(
                                        "LD", []
                                    ) + [v for v in lb_decomp[2] if get_free_symbols(v)]
                                    if get_dgcv_settings_registry().get(
                                        "simplify_singularity_ideals_by_default",
                                        True,
                                    ):
                                        new_sing = expr_union_primitives(
                                            new_sing,
                                            order_coordinates(target_alg._parameters),
                                            process_rationals=True,
                                            fail_quietly=True,
                                        )
                                    target_alg._singularities["LD"] = new_sing
                                if lb_decomp[0] is True:
                                    raise RuntimeError(
                                        "the dgcv Levi decomposition algorithm could "
                                        "not express a bracket of complement basis "
                                        f"elements {idx1} and {idx2} within the span "
                                        f"of the naive complement and level {idx} of "
                                        f"{len(local_rad_seq) - 1} of the radical's "
                                        "derived series. Either the linear solver "
                                        "failed to recognize a vanishing expression, "
                                        "or an earlier step produced a naive "
                                        "complement that does not complement the "
                                        f"radical (complement dimension {ss_dim}, "
                                        f"expected {discrep}; comparison set size "
                                        f"{compLen}, level size {rad_discrep})."
                                    )
                                lb_decomp = lb_decomp[1][0]
                                leading_coeffs[(idx1, idx2)] = [
                                    lb_decomp.get(idx, 0) for idx in range(ss_dim)
                                ]
                                trailing_coeffs[(idx1, idx2)] = [
                                    lb_decomp.get(idx, 0)
                                    for idx in range(ss_dim, ss_dim + compLen)
                                ]

                        for idxs in leading_coeffs:
                            oldV_sum = zip_sum(trailing_coeffs[idxs], compare_set)
                            vTerms_sum = -zip_sum(leading_coeffs[idxs], basis_modifiers)
                            newV = (
                                naiveBasis[idxs[0]] * basis_modifiers[idxs[1]]
                                - naiveBasis[idxs[1]] * basis_modifiers[idxs[0]]
                            )
                            qTerms_sum, t_vars = linear_combination(
                                quot_set,
                                prefix=f"tv_{idxs[0]}_{idxs[1]}_",
                                _disposable=True,
                            )
                            variables += t_vars
                            eqns.append(oldV_sum + vTerms_sum + qTerms_sum + newV)
                        if force_heavy_solve:
                            eqns = [simplify(eqn) for eqn in eqns]
                        solve_kwargs = _solve_weight_kwargs(
                            force_heavy_solve,
                            surface_singularities,
                            simplify_singularities,
                        )
                        if surface_singularities:
                            sol, _ = solve_dgcv(eqns, variables, **solve_kwargs)
                        else:
                            sol = solve_dgcv(eqns, variables, **solve_kwargs)
                        if len(sol) == 0:
                            if not all(_scalar_is_zero(eqn) for eqn in eqns):
                                dgcv_warning(
                                    f"eqn: {eqns},\\n variables{variables},\\n sol: {sol}",
                                    wc_label="debug_log",
                                )
                                raise RuntimeError(
                                    "solver failed during the dgcv Levi decomposition algorithm."
                                )
                            new_basis = list(naiveBasis)
                        else:
                            new_basis = [
                                (w + v).subs(sol[0])
                                for w, v in zip(naiveBasis, basis_modifiers)
                            ]
                        free_variables = set()
                        for nb in new_basis:
                            for j in nb.coeff_dict.values():
                                free_variables |= set(get_free_symbols(j))
                        free_variables = {x for x in free_variables if x in variables}
                        if len(free_variables) > 0:
                            zeroing = {v: 0 for v in free_variables}
                            target = next(iter(free_variables))
                            new_basis = [
                                subs(bv, {**zeroing, target: 1}) for bv in new_basis
                            ]
                        if force_heavy_solve:
                            new_basis = [simplify(bv) for bv in new_basis]
                        naiveBasis = new_basis
                    return target_alg.ambient.subalgebra(
                        naiveBasis, span_warning=True, simplify_basis=True
                    )

                Levi_component = _time_call(
                    _compute_complement,
                    "computing a semisimple complement to the max. solvable ideal",
                    "decompose the semisimple component into simple ideals"
                    if decompose_semisimple_fully
                    else _progress_message,
                )
            else:
                Levi_component = target_alg

            target_alg._Levi_deco_cache = {
                "LD_components": (Levi_component, rad),
                "simple_ideals": None,
            }

    if (
        decompose_semisimple_fully is True
        and target_alg._Levi_deco_cache.get("LD_components", None) is not None
        and target_alg._Levi_deco_cache.get("simple_ideals", 1) is None
    ):
        if verbose is True:
            print("Decomposing semisimple subalgebra into simple subalgebras...")

        Levi_component, rad = target_alg._Levi_deco_cache.get("LD_components", None)

        def _decompose_semisimple():
            simples = decompose_semisimple_algebra(
                Levi_component,
                format_as_lists_of_elements=True,
                surface_singularities=surface_singularities,
                simplify_singularities=simplify_singularities,
                return_centroid_types=True,
            )
            if surface_singularities:
                simples, centroid_types, sing = simples
            else:
                simples, centroid_types = simples
            new_basis = []
            simple_ideals = []
            for comp, centroid_type in zip(simples, centroid_types):
                new_basis += comp
                ideal = Levi_component.subalgebra(comp, simplify_basis=True)
                if centroid_type is not None:
                    ideal._verified_ideal = True
                    ideal._centroid_type = centroid_type
                simple_ideals.append(ideal)
            new_Levi = Levi_component.subalgebra(new_basis)
            if surface_singularities:
                return new_Levi, tuple(simple_ideals), sing
            return new_Levi, tuple(simple_ideals)

        out = _time_call(
            _decompose_semisimple,
            "decomposing algebra into simple ideals",
            _progress_message,
        )
        if surface_singularities:
            new_Levi, simple_ideals, sing = out
            new_sing = target_alg._singularities.get("simple_ideals", []) + [
                v for v in sing if get_free_symbols(v)
            ]
            if get_dgcv_settings_registry().get(
                "simplify_singularity_ideals_by_default", True
            ):
                new_sing = expr_union_primitives(
                    new_sing,
                    order_coordinates(target_alg._parameters),
                    process_rationals=True,
                    fail_quietly=True,
                )
            target_alg._singularities["simple_ideals"] = new_sing
        else:
            new_Levi, simple_ideals = out
        target_alg._Levi_deco_cache["LD_components"] = (new_Levi, rad)
        target_alg._Levi_deco_cache["simple_ideals"] = simple_ideals

    return target_alg._Levi_deco_cache.get("LD_components", None)
