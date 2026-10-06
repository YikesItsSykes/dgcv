from __future__ import annotations

from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    is_zero_knowing_zero_is_expected,
)
from ..._aux._utilities._config import dgcv_warning
from ..._aux._utilities._misc import linear_combination
from ..._aux._vmf._safeguards import retrieve_passkey
from ...core.arrays import array_dgcv, freeze_matrix, matrix_dgcv
from ...core.solvers import solve_knowing_solution_exists
from ._brackets import _sparse_row


class _symbol_export:
    def export_algebra_data(
        self,
        preserve_negative_part_basis=True,
        _internal_call_lock=None,
        try_hard=False,
        jacobi_threshold=None,
    ):
        """
        recommended to set jacobi_threshold = 1, although it defaults to 0.
        """
        grading_vec = []
        indexBands = dict()
        dimen = 0
        complimentWeights = {}
        permutation = []
        for weight, level in self.levels.items():
            if (
                preserve_negative_part_basis
                and weight < 0
                and not (level == [0] or level is None)
            ):
                permutation += [self.negativePart.basis.index(elem) for elem in level]
            lLength = 0 if (level == [0] or level is None) else len(level)
            nextDim = dimen + lLength
            for j in range(dimen, nextDim):
                indexBands[j] = (weight, j - dimen)
            complimentWeights[weight] = (dimen, self.dimension - nextDim)
            dimen = nextDim
            grading_vec += [weight] * lLength
        if preserve_negative_part_basis:
            invPerm = [permutation.index(j) for j in range(len(permutation))]
            tail = list(range(self.negativePart.dimension, self.dimension))
            back = invPerm + tail
            forward = permutation + tail

        def flatToLayered(idx):
            return indexBands[idx]

        neg_positions = None
        action = None
        table_bracket = None
        if jacobi_threshold is not None:
            neg_positions = self._negative_basis_positions()
            neg_coords = sorted(neg_positions)
            action = self._mixed_action_table(neg_positions, neg_coords, try_hard)
            if action is not None:
                table_bracket = self._bracket_memo(
                    neg_positions, neg_coords, action, jacobi_threshold, try_hard
                )
            if table_bracket is None:
                action = None

        def table_decomp(w1, sId1, w2, sId2, try_hard=False):
            if w1 < 0 and w2 < 0:
                vec = self._decompose_in_level(
                    self.levels[w1][sId1] * self.levels[w2][sId2],
                    w1 + w2,
                    neg_positions,
                    try_hard,
                )
                return None if vec is None else _sparse_row(vec)
            if w2 < 0:
                return action[(w1, sId1)][(w2, sId2)]
            if w1 < 0:
                return {r: -c for r, c in action[(w2, sId2)][(w1, sId1)].items()}
            return table_bracket.get(((w1, sId1), (w2, sId2)))

        def bracket_decomp(idx1, idx2, try_hard=False):
            w1, sId1 = flatToLayered(idx1)
            w2, sId2 = flatToLayered(idx2)
            newWeight = w1 + w2
            if table_bracket is not None:
                coeffVec = table_decomp(w1, sId1, w2, sId2, try_hard)
                if coeffVec is None:
                    return "NoSol"
                if len(coeffVec) == 0:
                    return dict()
                offset = complimentWeights[newWeight][0]
                return {offset + r: c for r, c in coeffVec.items()}
            newElem = (
                (self.levels[w1][sId1]) * (self.levels[w2][sId2])
            )  ###!!! review for ambient_rep requirements
            if self.levels[newWeight] is not None:
                ambient_basis = [
                    j for j in self.levels[newWeight]
                ]  ###!!! review for ambient_rep requirements
            nLDim = (
                0
                if (ambient_basis is None or ambient_basis == [0])
                else len(ambient_basis)
            )
            if nLDim == 0:
                if is_zero_knowing_zero_is_expected(newElem):
                    return dict()
                else:
                    return "NoSol"
            general_elem, tVars = linear_combination(ambient_basis, _disposable=True)
            eqns = [newElem - general_elem]
            sol = solve_knowing_solution_exists(
                eqns,
                tVars,
                try_hard=bool(try_hard),
                method="linear_parametric" if self._parameters else "linear",
                simplify_result=False,
            )
            if len(sol) == 0:
                return "NoSol"
            offset = complimentWeights[newWeight][0]
            return {
                offset + i: sol[0].get(var, var) for i, var in enumerate(tVars)
            }

        str_data = array_dgcv(
            dict(),
            shape=(dimen, dimen),
            null_return=freeze_matrix(matrix_dgcv.zeros(dimen, 1)),
        )
        for j in range(dimen):
            for k in range(j + 1, dimen):
                bracket_data = bracket_decomp(k, j, try_hard=try_hard)
                if bracket_data == "NoSol":
                    if not try_hard:
                        dgcv_warning(
                            "Failed to find prolongation algebraic structure without heavier solves. Retrying",
                            wc_label="debug_log",
                        )
                        bracket_data = bracket_decomp(k, j, try_hard=True)
                    if bracket_data == "NoSol":
                        warningStr = f"due to failure to confirm if the symbol data is closed under brackets between basis elements {j} and {k}."
                        if _internal_call_lock != retrieve_passkey():
                            dgcv_warning(
                                "Unable to extract algebra structure, "
                                + warningStr
                                + " So `None` was returned by `export_algebra_data`."
                            )
                            return None
                        return (
                            "Unable to extract algebra structure from `Tanaka_symbol` object, "
                            + warningStr
                        )
                new_mat = matrix_dgcv(bracket_data, shape=(dimen, 1))
                if new_mat:
                    str_data[(k, j)] = new_mat
                    str_data[(j, k)] = -new_mat

        if preserve_negative_part_basis:

            def permute_structure_data(SD, perm):
                d = SD.shape[0]
                new_sd = dict()
                perm = list(perm)
                sddd = SD._data
                for idx, v in sddd.items():
                    i, j = SD._unspool(idx)
                    new_key = (perm[i], perm[j])
                    inner_shp = (d, 1)
                    for k, value in v._data.items():
                        if not _scalar_is_zero(value):
                            if new_key in new_sd:
                                new_sd[new_key][perm[k]] = value
                            else:
                                new_sd[new_key] = matrix_dgcv(
                                    {perm[k]: value}, shape=inner_shp
                                )

                return array_dgcv(
                    new_sd,
                    shape=(d, d),
                    null_return=freeze_matrix(matrix_dgcv.zeros(d, 1)),
                )

            return {
                "structure_data": permute_structure_data(str_data, forward),
                "grading": [
                    (grading_vec[back[j]] if j < len(back) else grading_vec[j])
                    for j in range(self.dimension)
                ],
            }
        return {"structure_data": str_data, "grading": [grading_vec]}
