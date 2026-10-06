from __future__ import annotations

from contextlib import nullcontext

from ..._aux._backends._engine import engine_capability
from ..._aux._backends._polynomials import expr_union_primitives
from ..._aux._backends._symbolic_router import _scalar_is_zero, get_free_symbols, subs
from ..._aux._utilities._config import dgcv_warning
from ..._aux._utilities._misc import linear_combination
from ..._aux._vmf._safeguards import get_dgcv_category
from ...algebras import _extract_basis
from ...core.solvers import solve_dgcv
from ._tensor_products import _fast_tensor_products


def _loop_sum(a, b, p2):
    if not isinstance(a, _fast_tensor_products):
        return a + b - p2
    acc = dict(a.coeff_dict)
    deg = a.degree
    for term, negate in ((b, False), (p2, True)):
        if _scalar_is_zero(term):
            continue
        if isinstance(term, _fast_tensor_products):
            for k, v in term.coeff_dict.items():
                deg = max(len(k), deg)
                acc[k] = acc.get(k, 0) + (-v if negate else v)
            continue
        if (
            get_dgcv_category(term) == "algebra_element"
            and term.algebra.simplify_products_by_default is not True
        ):
            cd = term.coeff_dict
            for idx in sorted(cd):
                v = cd[idx]
                if _scalar_is_zero(v):
                    continue
                deg = max(1, deg)
                acc[(idx,)] = acc.get((idx,), 0) + (-v if negate else v)
            continue
        out = _fast_tensor_products(acc, a.algebra, _validated=deg)
        return (out - p2) if negate else (out + b - p2)
    return _fast_tensor_products(acc, a.algebra, _validated=deg)


class _symbol_prolongation_step:
    def _fast_prolong_by_1(
        self,
        levels,
        height,
        alias_counter,
        surface_singularities,
        simplify_pivots,
        simplify_ideals,
        solve_method,
        with_characteristic_space_reductions=False,
        DS_records=None,
        absorb_DS=False,
        generic=None,
    ):  # height must match levels structure
        # `surface_singularities`, `simplify_pivots`, `simplify_ideals` and
        # `solve_method` arrive already resolved from `prolong`
        get_alias_id = alias_counter

        if DS_records is None:
            DS_records = []
        ADS = absorb_DS is True
        if (
            self.assume_FGLA
            and len(levels[height]) == 0
            or (
                self._GLA_generators is not None
                and all(
                    len(levels[height - j]) == 0
                    for j in range(-min(self._GLA_generators.get("generators", levels)))
                )
                and min(self._GLA_generators.get("generators", levels)) >= -1 - height
            )
            or min(j for j in levels) >= -1 - height
            and all(
                len(levels[height - j]) == 0 for j in range(-min(j for j in levels))
            )
        ):  # stability check
            new_levels = levels
            new_levels._set_index_thr(height)
            stable = True
        else:
            use_generators = (
                self._compress_equation_systems
                and self._GLA_generators is not None
                and with_characteristic_space_reductions is not True
                and not ADS
                and all(record.cap < height + 1 for record in DS_records)
                and all(
                    "operator" in self._aliasing[atom._atomic_index]
                    or "operator_S" in self._aliasing[atom._atomic_index]
                    for w in levels
                    if w >= 0
                    for atom in levels[w]
                )
            )
            if use_generators:
                for record in DS_records:
                    source_bound = record.cap - height - 1
                    for (low, high) in record.components:
                        if high >= 0 or low > source_bound:
                            continue
                        if min(high, source_bound) + height + 1 >= 0:
                            use_generators = False
                            dgcv_warning(
                                "Precomputed generators are not used for a prolongation step whose distinguished-subspace constraints reach nonnegative weights; the general algorithm runs for this step.",
                                wc_label="debug_log",
                            )
            ambient_basis = self._prolongation_ambient_basis(
                levels, height, use_generators
            )

            empty_ambient = len(ambient_basis) == 0
            if empty_ambient:
                ambient_basis = [0 * self.basis[0]]

            general_elem_terse, tVars = linear_combination(ambient_basis)
            general_elem = self._aliased_expansion(general_elem_terse, partial=True)

            eqns = []
            esVars = list(tVars)
            products = {}
            expansions = {}

            def _accumulate(expr):
                if _scalar_is_zero(expr):
                    return
                if get_dgcv_category(expr) in {
                    "fastTensorProduct",
                    "tensorProduct",
                    "algebra_element",
                    "subalgebra_element",
                }:
                    eqns.extend(expr.coeff_dict.values())
                else:
                    dgcv_warning(
                        f"The constraint value {expr} is outside of expected class. Recieved type: {type(expr)}",
                        wc_label="debug_log",
                    )

            if use_generators:
                self._generator_equations(
                    general_elem, height, DS_records, esVars, _accumulate
                )
            if len(DS_records) > 0 and not use_generators:
                ambGE = None
                for record in DS_records:
                    source_bound = record.cap - height - 1
                    for (low, high), source in record.components.items():
                        if high >= 0 or low > source_bound:
                            continue
                        if low == high:
                            target = record.component(low + height + 1)
                            dsSpanners = target.spanners if target else []
                            sources = source.spanners
                        else:
                            dsSpanners = record.target_spanners(
                                low + height + 1,
                                min(high, source_bound) + height + 1,
                            )
                            sources = (
                                source.spanners
                                if high <= source_bound
                                else source.truncated_spanners(source_bound)
                            )
                        if min(high, source_bound) + height + 1 < 0:
                            operator = general_elem
                        else:
                            if ambGE is None:
                                ambGE = self._aliased_expansion(general_elem_terse)
                            operator = ambGE
                        for elem in sources:
                            if dsSpanners:
                                newGE, newVars = linear_combination(dsSpanners)
                                esVars += newVars
                                _accumulate(operator * elem + newGE)
                            else:
                                _accumulate(operator * elem)

            def _ge_product(t):
                out = products.get(id(t))
                if out is None:
                    out = general_elem * t
                    products[id(t)] = out
                return out

            def _ge_expansion(t):
                out = expansions.get(id(t))
                if out is None:
                    out = self._aliased_expansion(_ge_product(t), partial=True)
                    expansions[id(t)] = out
                return out

            if use_generators:
                test_commutators = []
            elif generic is not None:
                test_commutators = self._generic_test_commutators(generic)
            else:
                test_commutators = self.test_commutators
            for triple in test_commutators:
                t0, t1, t2 = triple[0], triple[1], triple[2]
                a = _ge_expansion(t0) * t1
                b = t0 * _ge_expansion(t1)
                if get_dgcv_category(t2) == "algebra_element" and not t2.coeff_dict:
                    p2 = 0
                else:
                    p2 = _ge_product(t2)
                _accumulate(_loop_sum(a, b, p2))

            if eqns == [0] or eqns == []:
                solution = [{}]
            else:
                guard = generic.constants.pivots() if generic is not None else nullcontext()
                if surface_singularities:
                    with guard:
                        solution, sing = solve_dgcv(
                            eqns,
                            esVars,
                            method=solve_method,
                            return_divisors=True,
                            pass_to_symbolic_engine=False,
                            simplify_pivots=simplify_pivots,
                            simplify_result=False,
                        )
                    if generic is not None:
                        sing = [generic.constants.expand(v) for v in sing]

                    self._singularities["prolongation"] = expr_union_primitives(
                        list(self._singularities.get("prolongation", []))
                        + [v for v in sing if get_free_symbols(v)],
                        self._parameters,
                        process_rationals=True,
                        fail_quietly=True,
                        bypass=not simplify_ideals,
                    )

                else:
                    with guard:
                        solution = solve_dgcv(
                            eqns, esVars, method=solve_method, simplify_result=False
                        )

            if len(solution) == 0:
                dgcv_warning(
                    f"At breakpoint in prolongation algorithm: The equation system was {eqns} w.r.t. {esVars}; return solution data was {solution}",
                    wc_label="debug_log",
                )
                raise RuntimeError(
                    "`Tanaka_symbol.prolongation` failed at a step where a symbolic solver (e.g., sympy.solve if using the default sympy) was being applied."
                )
            solution = solution[0]
            el_sol = subs(general_elem_terse, solution)
            if not isinstance(el_sol, _fast_tensor_products):
                el_sol = _fast_tensor_products(el_sol)

            fv_possibles = set(esVars)
            fv = set()
            for variable in tVars:
                fv |= get_free_symbols(solution.get(variable, variable))
            new_level = []
            zeroing = {} if empty_ambient else {v: 0 for v in fv if v in fv_possibles}
            linear_one_hot = engine_capability("linear_one_hot")
            parts = (
                linear_one_hot(el_sol.coeff_dict, list(zeroing))
                if linear_one_hot is not None
                else None
            )
            if parts is not None:
                for part in parts:
                    new_level.append(
                        _fast_tensor_products(part, el_sol.algebra, _validated=el_sol.degree)
                        if part
                        else _fast_tensor_products({tuple(): 0}, el_sol.algebra, _validated=0)
                    )
            else:
                for v in zeroing:
                    basis_element = subs(el_sol, {**zeroing, v: 1})
                    new_level.append(basis_element)

            completed = {}
            expansions = None
            if use_generators:
                for el in new_level:
                    completed[id(el)] = self._complete_S(el)
            if ADS is True:
                absorbed = []
                for record in DS_records:
                    component = record.component(height + 1)
                    if component is not None:
                        absorbed += list(component.spanners)
                if absorbed:
                    expansions = [self._aliased_expansion(el) for el in new_level]
                    _, kept_idxs = _extract_basis(
                        expansions + absorbed, return_indices=True
                    )
                    offset = len(expansions)
                    kept = [absorbed[i - offset] for i in kept_idxs if i >= offset]
                    new_level = new_level + kept
                    expansions = expansions + kept

            new_level, expansions = self._characteristic_space_reduction(
                new_level,
                expansions,
                levels,
                height,
                with_characteristic_space_reductions,
                surface_singularities,
                simplify_pivots,
                simplify_ideals,
                solve_method,
            )
            atomized_level = []
            for position, el in enumerate(new_level):
                new_idx = get_alias_id()
                expanded = expansions[position] if expansions is not None else None
                if use_generators:
                    alias_data = {"operator_S": completed[id(el)], "_pending_S": True}
                elif expanded is not None and el is expanded:
                    alias_data = {
                        "expanded": expanded
                        if isinstance(expanded, _fast_tensor_products)
                        else _fast_tensor_products(expanded)
                    }
                else:
                    alias_data = {
                        "operator": self._aliased_expansion(el, partial=True)
                    }
                    if expanded is not None:
                        alias_data["expanded"] = (
                            expanded
                            if isinstance(expanded, _fast_tensor_products)
                            else _fast_tensor_products(expanded)
                        )
                self._aliasing[new_idx] = alias_data
                atom = _fast_tensor_products({(new_idx,): 1}, _atomic_index=new_idx)
                atomized_level.append(atom)
            new_level = atomized_level

            self._recoordinatize_DS_components(
                DS_records, height, atomized_level, expansions, solve_method
            )

            new_levels = self._GLA_structure(
                levels | {height + 1: new_level}, levels.index_threshold
            )
            stable = False
        return new_levels, stable, DS_records
