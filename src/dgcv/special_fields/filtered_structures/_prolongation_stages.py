from __future__ import annotations

from itertools import count

from ..._aux._backends._polynomials import expr_union_primitives
from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    clear_denominators,
    get_free_symbols,
    subs,
)
from ..._aux._backends._types_and_constants import symbol
from ..._aux._utilities._misc import linear_combination
from ..._aux._vmf._safeguards import create_key, get_dgcv_category
from ...core.solvers import solve_dgcv
from ._tensor_products import _fast_tensor_products


class _symbol_prolongation_stages:
    def _prolongation_ambient_basis(self, levels, height, use_generators=None):
        get_temp_id = count(-2, -1).__next__
        for temp_idx in [idx for idx in self._aliasing if idx < 0]:
            del self._aliasing[temp_idx]

        k_cache = {}

        def k_data(k, kidx, kdeg):
            cache_key = (kdeg, kidx)
            data = k_cache.get(cache_key)
            if data is None:
                data = _fast_tensor_products(k)
                k_cache[cache_key] = data
            return data

        def stamp(j, k, jidx, kidx, jdeg, kdeg):
            new_idx = get_temp_id()
            ftk = k_data(k, kidx, kdeg)
            obj = _fast_tensor_products({(new_idx,): 1}, _atomic_index=new_idx)
            self._aliasing[new_idx] = {"operator": ftk @ j}
            return obj

        if use_generators is None:
            use_generators = self._GLA_generators is not None
        if not use_generators:
            ambient_basis = []
            for weight in self.negWeights:
                kdeg = height + 1 + weight
                ambient_basis += [
                    stamp(j, k, jidx, kidx, weight, kdeg)
                    for jidx, j in enumerate(self.GLA_levels[weight])
                    for kidx, k in enumerate(levels[kdeg])
                ]
        else:
            gens = self._GLA_generators
            s_atoms = gens["s_atoms"]
            ambient_basis = []
            for weight, positions in gens["s_generators"].items():
                kdeg = height + 1 + weight
                for s in positions:
                    for k in levels[kdeg]:
                        new_idx = get_temp_id()
                        self._aliasing[new_idx] = {
                            "operator": _fast_tensor_products(k) @ s_atoms[s]
                        }
                        ambient_basis.append(
                            _fast_tensor_products({(new_idx,): 1}, _atomic_index=new_idx)
                        )
        return ambient_basis

    def _characteristic_space_reduction(
        self,
        new_level,
        expansions,
        levels,
        height,
        with_characteristic_space_reductions,
        surface_singularities,
        simplify_pivots,
        simplify_ideals,
        solve_method,
    ):
        if with_characteristic_space_reductions is True:
            if expansions is None:
                expansions = [self._aliased_expansion(el) for el in new_level]
            if height == -1:
                z_level = expansions
            else:
                z_level = [self._aliased_expansion(el) for el in levels[0]]
        else:
            z_level = []
        if len(new_level) > 0 and len(z_level) > 0:
            stabilized = False
            z_level_tp = [dzElem._convert_to_tp() for dzElem in z_level]
            while stabilized is False:
                ambient_basis = expansions
                ambient_alias = new_level
                varLabel = create_key(prefix="_cv")
                tVars = [symbol(f"{varLabel}{j}") for j in range(len(ambient_basis))]
                solVars = list(tVars)
                general_elem = _fast_tensor_products._dgcv_multiadd_scaled(
                    [(tVars[j], ambient_basis[j]) for j in range(len(tVars))]
                )
                eqns = []
                general_elem_tp = general_elem._convert_to_tp()
                for idx, dzElem in enumerate(z_level_tp):
                    varLabel2 = varLabel + f"{idx}_"
                    vars2 = [
                        symbol(f"{varLabel2}{j}") for j in range(len(ambient_basis))
                    ]
                    solVars += vars2
                    general_elem2 = _fast_tensor_products._dgcv_multiadd_scaled(
                        [(vars2[j], ambient_basis[j]) for j in range(len(tVars))]
                    )._convert_to_tp()

                    commutator = general_elem_tp * dzElem - general_elem2
                    if get_dgcv_category(commutator) == "tensorProduct":
                        eqns += list(commutator.coeff_dict.values())
                    elif get_dgcv_category(commutator) == "algebra_element":
                        eqns += commutator.coeffs
                if surface_singularities:
                    solution, sing = solve_dgcv(
                        eqns,
                        solVars,
                        method=solve_method,
                        return_divisors=True,
                        pass_to_symbolic_engine=False,
                        simplify_pivots=simplify_pivots,
                        simplify_result=False,
                    )
                    self._singularities["prolongation"] = expr_union_primitives(
                        list(self._singularities.get("prolongation", []))
                        + [v for v in sing if get_free_symbols(v)],
                        self._parameters,
                        process_rationals=True,
                        fail_quietly=True,
                        bypass=not simplify_ideals,
                    )
                else:
                    solution = solve_dgcv(
                        eqns, solVars, method=solve_method, simplify_result=False
                    )
                if len(solution) == 0:
                    raise RuntimeError(
                        f"`Tanaka_symbol.prolongation` failed at a step where a symbolic solver (e.g., sympy.solve if using the default sympy) was being applied. The equation system was {eqns} w.r.t. {solVars}"
                    )
                solution = solution[0]
                solCoeffs = [solution.get(j, 0) for j in tVars]

                free_variables = tuple(
                    set.union(
                        set(),
                        *[get_free_symbols(j) - self._parameters for j in solCoeffs],
                    )
                )
                filtered_vectors = []
                zeroingDict = {other_var: 0 for other_var in free_variables}
                for var in free_variables:
                    basis_element = [subs(j, zeroingDict | {var: 1}) for j in solCoeffs]
                    filtered_vectors.append(clear_denominators(basis_element))

                new_basis = []
                new_alias = []
                for coeffs in filtered_vectors:
                    new_basis.append(
                        _fast_tensor_products._dgcv_multiadd_scaled(
                            [
                                (coeffs[j], ambient_basis[j])
                                for j in range(len(ambient_basis))
                            ]
                        )
                    )
                    new_alias.append(
                        _fast_tensor_products._dgcv_multiadd_scaled(
                            [
                                (coeffs[j], ambient_alias[j])
                                for j in range(len(ambient_alias))
                            ]
                        )
                    )
                if len(new_basis) == 0:
                    new_level = []
                    expansions = []
                    stabilized = True
                elif len(new_basis) < len(new_level):
                    new_level = new_alias
                    expansions = new_basis
                else:
                    new_level = new_alias
                    expansions = new_basis
                    stabilized = True
        return new_level, expansions

    def _recoordinatize_DS_components(
        self, DS_records, height, atomized_level, expansions, solve_method
    ):
        for record in DS_records:
            if height + 1 > record.cap:
                continue
            component = record.component(height + 1)
            if component is None or len(component.spanners) == 0:
                continue
            if len(atomized_level) == 0:
                component.spanners = []
                component.coords = []
                continue
            dsGE, dsVars = linear_combination(component.spanners)
            if expansions is None:
                expansions = [self._aliased_expansion(el) for el in atomized_level]
            lvlGE, lvlVars = linear_combination(expansions)
            sanVars = list(dsVars) + list(lvlVars)
            residual = dsGE - lvlGE
            sanEqns = list(getattr(residual, "coeff_dict", {}).values())
            if len(sanEqns) == 0:
                continue
            sanSol = solve_dgcv(
                sanEqns, sanVars, method=solve_method, simplify_result=False
            )
            if len(sanSol) == 0:
                component.spanners = []
                component.coords = []
                continue
            sanSol = sanSol[0]
            terse = lvlVars[0] * atomized_level[0]
            for coeff, atom in zip(lvlVars[1:], atomized_level[1:]):
                terse = terse + coeff * atom
            terse = subs(terse, sanSol)
            fv_possibles = set(sanVars)
            fv = set()
            for variable in lvlVars:
                fv |= get_free_symbols(sanSol.get(variable, variable))
            zeroing = {v: 0 for v in fv if v in fv_possibles}
            coords = []
            for v in zeroing:
                el = subs(terse, {**zeroing, v: 1})
                if _scalar_is_zero(el):
                    continue
                coords.append(el)
            component.coords = coords
            component.spanners = [self._aliased_expansion(el) for el in coords]
