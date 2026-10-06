from __future__ import annotations

from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    is_zero_knowing_zero_is_expected,
)
from ..._aux._backends._types_and_constants import _disposable_symbols
from ..._aux._utilities._misc import linear_combination
from ..._aux._vmf._safeguards import create_key
from ...algebras import _extract_basis
from ...core.solvers import solve_knowing_solution_exists
from ...core.vector_fields_and_differential_forms.decomposition import (
    _decompose_prepared,
    _prepared_span,
)
from ._tensor_products import _fast_tensor_products


def _column_vector(column):
    return column


def _sparse_row(vec):
    if isinstance(vec, dict):
        return vec
    return {i: c for i, c in enumerate(vec) if not _scalar_is_zero(c)}


class _symbol_brackets:
    def _alias_expansion(self, alias):
        expanded = alias.get("expanded")
        if expanded is None:
            if alias.get("_pending_S"):
                self._materialize_S_atom(alias)
                expanded = alias.get("expanded")
                if expanded is not None:
                    return expanded
            operator = alias.get("operator")
            if operator is None:
                return None
            expanded = self._expand_operator(operator)
            alias["expanded"] = expanded
        return expanded

    def _expand_operator(self, operator):
        neg = self.negativePart
        neg_dim = neg.dimension
        aliasing = self._aliasing
        alg = operator.algebra if operator.algebra is not None else neg
        pairs = []
        leftovers = {}
        deg = 0
        for key, c in operator.coeff_dict.items():
            alias = None
            if len(key) == 2 and key[0] >= neg_dim:
                alias = aliasing.get(key[0])
            if alias is not None and alias.get("_pending_S"):
                self._materialize_S_atom(alias)
            image = None if alias is None else self._alias_expansion(alias)
            if image is None:
                leftovers[key] = c
                deg = max(deg, len(key))
                continue
            tail = key[1:]
            pairs.append(
                (
                    c,
                    _fast_tensor_products(
                        {k + tail: v for k, v in image.coeff_dict.items()},
                        image.algebra if image.algebra is not None else alg,
                        _validated=image.degree + 1,
                    ),
                )
            )
        start = (
            _fast_tensor_products(leftovers, alg, _validated=deg)
            if leftovers
            else _fast_tensor_products(dict(), alg, _validated=0)
        )
        if not pairs:
            return start
        out = _fast_tensor_products._dgcv_multiadd_scaled(pairs, start)
        if out.algebra is None:
            out = _fast_tensor_products(out.coeff_dict, alg, _validated=out.degree)
        return out

    def _aliased_expansion(self, ftp: _fast_tensor_products, partial=False):
        if not isinstance(ftp, _fast_tensor_products):
            return ftp
        aliasing = self._aliasing
        alias = aliasing.get(ftp._atomic_index)
        if alias is not None:
            if alias.get("_pending_S"):
                self._materialize_S_atom(alias)
            if partial:
                operator = alias.get("operator")
                if operator is not None:
                    return operator
            expanded = self._alias_expansion(alias)
            return ftp if expanded is None else expanded

        pairs = []
        leftovers = {}
        for k, v in ftp.coeff_dict.items():
            new_term = None
            if len(k) == 1:
                alias = aliasing.get(k[0])
                if alias is not None:
                    if alias.get("_pending_S"):
                        self._materialize_S_atom(alias)
                    if partial:
                        new_term = alias.get("operator")
                    if new_term is None:
                        new_term = self._alias_expansion(alias)
            if new_term:
                pairs.append((v, new_term))
            else:
                leftovers[k] = v
        if not pairs:
            return _fast_tensor_products(leftovers) if leftovers else 0
        return _fast_tensor_products._dgcv_multiadd_scaled(
            pairs, _fast_tensor_products(leftovers) if leftovers else 0
        )

    def _negative_basis_positions(self):
        positions = dict()
        for weight, level in self.levels.items():
            if weight < 0:
                for s, elem in enumerate(level):
                    positions[(weight, s)] = self.negativePart.basis.index(elem)
        return positions

    def _decompose_in_level(self, elem, weight, neg_positions, try_hard=False):
        level = self.levels[weight]
        if len(level) == 0:
            if is_zero_knowing_zero_is_expected(elem):
                return []
            return None
        if weight < 0:
            coeffs = getattr(elem, "coeffs", None)
            if coeffs is not None and len(coeffs) == self.negativePart.dimension:
                return [coeffs[neg_positions[(weight, t)]] for t in range(len(level))]
        prepared = _decompose_prepared(elem, level)
        if prepared is not None:
            return prepared
        general_elem, tVars = linear_combination(level, _disposable=True)
        eqns = [elem - general_elem]
        sol = solve_knowing_solution_exists(
            eqns,
            tVars,
            try_hard=try_hard is True,
            method="linear_parametric" if self._parameters else "linear",
            simplify_result=False,
        )
        if len(sol) == 0:
            return None
        return [sol[0].get(var, var) for var in tVars]

    def _derive_nonneg_hom(self, assume_independent=False):
        """
        For reprocessing nonnegative symbol data.

        Each level is decomposed degree by degree into its action on the
        negative part, reduced to a linearly independent spanning set unless
        that is assumed, and registered in the aliasing registry.

        Parameters
        ----------
        assume_independent : bool, default False
            Skip the basis extraction, taking the supplied spanning sets to be
            linearly independent already.

        Returns
        -------
        dict
            Maps `(weight, position)` to `{(jidx, kidx, jdeg, kdeg): coeff}`,
            recording where each nonnegative level element sends each negative
            basis element.

        Raises
        ------
        ValueError
            If some level element does not map the negative part into lower
            levels, so the supplied nonnegative data is not an algebraic
            prolongation of the levels below it.
        """
        self._nonneg_atoms = dict()
        if len(self.nonneg_levels) == 0:
            return dict()
        self.nonneg_levels = {w: list(v) for w, v in self.nonneg_levels.items()}
        neg_positions = self._negative_basis_positions()
        neg_coords = sorted(neg_positions)
        next_idx = max(
            [sum(len(level) for w, level in self.levels.items() if w < 0)]
            + [idx + 1 for idx in self._aliasing if idx >= 0]
        )
        forms = dict()
        reduced = False
        for weight in sorted(w for w in self.levels if w >= 0):
            decomps = []
            operators = []
            stored_flags = []
            for position, elem in enumerate(self.levels[weight]):
                decomp = (
                    None
                    if reduced
                    else self._stored_hom_decomp(elem, weight, neg_positions)
                )
                stored_flags.append(decomp is not None)
                if decomp is None:
                    decomp = dict()
                    for jdeg, jidx in neg_coords:
                        kdeg = weight + jdeg
                        image = elem * self.levels[jdeg][jidx]
                        vec = self._decompose_in_level(image, kdeg, neg_positions)
                        if vec is None:
                            vec = self._decompose_in_level(
                                image, kdeg, neg_positions, try_hard=True
                            )
                        if vec is None:
                            raise ValueError(
                                f"`Tanaka_symbol` received nonnegative level data that is not admissible. Element {position + 1} of degree {weight} maps a degree {jdeg} basis element outside of degree {kdeg}, so degree {weight} is not contained in the algebraic prolongation of the levels preceding it."
                            )
                        for kidx, c in enumerate(vec):
                            if not _scalar_is_zero(c):
                                decomp[(jidx, kidx, jdeg, kdeg)] = c
                decomps.append(decomp)
                operators.append(self._hom_operator_form(decomp, neg_positions))
            keep = list(range(len(decomps)))
            if assume_independent is not True and len(decomps) > 0:
                live = [n for n in keep if len(operators[n].coeff_dict) > 0]
                if len(live) == 0:
                    keep = []
                else:
                    _, kept = _extract_basis(
                        [operators[n] for n in live], return_indices=True
                    )
                    keep = sorted(live[m] for m in kept)
                if len(keep) != len(decomps):
                    reduced = True
            self.levels[weight] = [self.levels[weight][n] for n in keep]
            if weight in self.nonneg_levels:
                self.nonneg_levels[weight] = list(self.levels[weight])
            for position, n in enumerate(keep):
                self._nonneg_atoms[(weight, position)] = next_idx
                alias = {"operator": operators[n], "hom": decomps[n]}
                if not stored_flags[n]:
                    alias["expanded"] = _fast_tensor_products(
                        self.levels[weight][position]
                    )
                self._aliasing[next_idx] = alias
                forms[(weight, position)] = decomps[n]
                next_idx += 1
        self.dimension = sum(len(level) for level in self.levels.values())
        return forms

    def _hom_operator_form(self, decomp, neg_positions):
        terms = []
        for (jidx, kidx, jdeg, kdeg), c in decomp.items():
            if kdeg < 0:
                image = _fast_tensor_products(self.levels[kdeg][kidx])
            else:
                idx = self._nonneg_atoms[(kdeg, kidx)]
                image = _fast_tensor_products({(idx,): 1}, _atomic_index=idx)
            terms.append((c, image @ self.levels[jdeg][jidx]))
        if len(terms) == 0:
            return _fast_tensor_products(dict(), self.negativePart, _validated=0)
        return _fast_tensor_products._dgcv_multiadd_scaled(terms)

    def _stored_hom_decomp(self, elem, weight, neg_positions):
        stored = getattr(elem, "_properties", dict()).get("_hom_decomp")
        if not stored:
            return None
        decomp = dict()
        for key, c in stored.items():
            jidx, kidx, jdeg, kdeg = key
            if (
                kdeg != weight + jdeg
                or (jdeg, jidx) not in neg_positions
                or kidx >= len(self.levels[kdeg])
            ):
                return None
            decomp[key] = c
        return decomp

    def _alias_hom_decomp(self, alias, weight, atom_positions, neg_lookup, neg_dim):
        if alias is None:
            return None
        stored = alias.get("hom")
        if stored:
            return dict(stored)
        operator = alias.get("operator")
        if operator is None:
            return None
        decomp = dict()
        for key, v in operator.coeff_dict.items():
            if len(key) != 2:
                return None
            source = neg_lookup.get(key[1])
            if source is None:
                return None
            jdeg, jidx = source
            if key[0] < neg_dim:
                target = neg_lookup.get(key[0])
            else:
                target = atom_positions.get(key[0])
            if target is None:
                return None
            kdeg, kidx = target
            if kdeg != weight + jdeg:
                return None
            entry = (jidx, kidx, jdeg, kdeg)
            decomp[entry] = decomp.get(entry, 0) + v
        return {k: v for k, v in decomp.items() if not _scalar_is_zero(v)}

    def _mixed_action_table(self, neg_positions, neg_coords, try_hard=False):
        table = dict()
        for weight, level in self.levels.items():
            if weight < 0:
                continue
            for m, elem in enumerate(level):
                decomp = getattr(elem, "_properties", dict()).get("_hom_decomp")
                rows = None
                if decomp:
                    rows = dict()
                    for key, c in decomp.items():
                        jidx, kidx, jdeg, kdeg = key
                        if (
                            kdeg != weight + jdeg
                            or (jdeg, jidx) not in neg_positions
                            or kidx >= len(self.levels[kdeg])
                        ):
                            rows = None
                            break
                        vec = rows.setdefault((jdeg, jidx), dict())
                        vec[kidx] = vec.get(kidx, 0) + c
                    if rows is not None:
                        rows = {
                            coord: {r: vec[r] for r in sorted(vec)}
                            for coord, vec in rows.items()
                        }
                if rows is None:
                    rows = dict()
                    for coord in neg_coords:
                        vec = self._decompose_in_level(
                            elem * self.levels[coord[0]][coord[1]],
                            weight + coord[0],
                            neg_positions,
                            try_hard,
                        )
                        if vec is None:
                            return None
                        rows[coord] = _sparse_row(vec)
                for coord in neg_coords:
                    if coord not in rows:
                        rows[coord] = dict()
                table[(weight, m)] = rows
        return table

    def _apply_bracket(self, source, vec, weight, action, memo):
        result = dict()
        for q, c in vec.items():
            if _scalar_is_zero(c):
                continue
            if weight < 0:
                contrib = action[source][(weight, q)]
            else:
                contrib = memo.get((source, (weight, q)))
            if contrib is None:
                return None
            for r, value in contrib.items():
                if type(value) is int:
                    if value == 0:
                        continue
                    if value == 1:
                        result[r] = result.get(r, 0) + c
                        continue
                result[r] = result.get(r, 0) + c * value
        return result

    def _action_columns(self, total, neg_coords, action):
        columns = []
        for m in range(len(self.levels[total])):
            col = dict()
            offset = 0
            for coord in neg_coords:
                for r, value in action[(total, m)][coord].items():
                    col[offset + r] = value
                offset += len(self.levels[total + coord[0]])
            columns.append(col)
        return columns

    def _jacobi_bracket(
        self, first, second, total, neg_coords, columns, action, memo, try_hard=False
    ):
        rhs = dict()
        offset = 0
        for coord in neg_coords:
            left = self._apply_bracket(
                first, action[second][coord], second[0] + coord[0], action, memo
            )
            right = self._apply_bracket(
                second, action[first][coord], first[0] + coord[0], action, memo
            )
            if left is None or right is None:
                return None
            for r, value in left.items():
                rhs[offset + r] = value
            for r, value in right.items():
                key = offset + r
                rhs[key] = rhs[key] - value if key in rhs else -value
            offset += len(self.levels[total + coord[0]])
        if len(columns) == 0:
            if all(is_zero_knowing_zero_is_expected(c) for c in rhs.values()):
                return dict()
            return None
        solver = _prepared_span(columns, _column_vector)
        if solver is not None:
            coords = solver.reduce_vector(rhs)
            if coords is not None:
                return _sparse_row(coords)
        varLabel = create_key(prefix="_ja")
        tVars = _disposable_symbols(varLabel, len(columns))
        rows = set(rhs)
        for col in columns:
            rows.update(col)
        eqns = [
            sum(tVars[m] * columns[m].get(r, 0) for m in range(len(columns)))
            - rhs.get(r, 0)
            for r in sorted(rows)
        ]
        sol = solve_knowing_solution_exists(
            eqns,
            tVars,
            try_hard=try_hard is True,
            method="linear_parametric" if self._parameters else "linear",
            simplify_result=False,
        )
        if len(sol) == 0:
            return None
        return _sparse_row([sol[0].get(var, var) for var in tVars])

    def _bracket_memo(
        self, neg_positions, neg_coords, action, jacobi_threshold, try_hard=False
    ):
        memo = dict()
        nonneg = [w for w in self.levels if w >= 0 and len(self.levels[w]) > 0]
        if len(nonneg) == 0:
            return memo
        for total in range(2 * max(nonneg) + 1):
            columns = None
            if total % 2 == 0:
                half = total // 2
                for s in range(len(self.levels[half])):
                    memo[((half, s), (half, s))] = dict()
            for w1 in range(total // 2 + 1):
                w2 = total - w1
                L1, L2 = self.levels[w1], self.levels[w2]
                if len(L1) == 0 or len(L2) == 0:
                    continue
                for s1 in range(len(L1)):
                    for s2 in range(len(L2)):
                        if w1 == w2 and s2 <= s1:
                            continue
                        if total <= jacobi_threshold:
                            vec = self._decompose_in_level(
                                L1[s1] * L2[s2], total, neg_positions, try_hard
                            )
                            if vec is not None:
                                vec = _sparse_row(vec)
                        else:
                            if columns is None:
                                columns = self._action_columns(
                                    total, neg_coords, action
                                )
                            vec = self._jacobi_bracket(
                                (w1, s1),
                                (w2, s2),
                                total,
                                neg_coords,
                                columns,
                                action,
                                memo,
                                try_hard,
                            )
                        if vec is None:
                            return None
                        memo[((w1, s1), (w2, s2))] = vec
                        memo[((w2, s2), (w1, s1))] = {r: -c for r, c in vec.items()}
        return memo
