from __future__ import annotations

from ..._aux._backends._engine import engine_capability
from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    get_free_symbols,
    simplify,
    subs,
)
from ..._aux._backends._types_and_constants import rational
from ..._aux._utilities._config import dgcv_warning
from ..._aux._utilities._misc import linear_combination
from ..._aux._vmf._safeguards import query_dgcv_categories, retrieve_passkey
from ...algebras import algebra_class, createAlgebra
from ...core.arrays import array_dgcv, freeze_matrix, matrix_dgcv
from ...core.base import dgcv_class
from ...core.vector_fields_and_differential_forms import LieDerivative, decompose
from ...core.vector_fields_and_differential_forms.decomposition import (
    _decompose_prepared,
    _prepared_prefix_spans,
)
from ._distribution import distribution
from ._symbol import Tanaka_symbol


class filtration_class(dgcv_class):
    def __init__(
        self,
        spanning_vf_sets: list[list],
        assume_spanning_sections_linearly_indep=False,
    ):
        def sublist_check(sl):
            return isinstance(sl, (list, set, tuple)) and all(
                query_dgcv_categories(vf, {"vector_field"}) for vf in sl
            )

        if not isinstance(spanning_vf_sets, (list, set, tuple)) or not all(
            sublist_check(sl) for sl in spanning_vf_sets
        ):
            raise TypeError(
                "`filtration init expects `spanning_vf_sets` to be a list of lists of vector fields.`"
            )
        spanning_vf_sets = list(spanning_vf_sets)
        self.depth = len(spanning_vf_sets)
        if len(spanning_vf_sets) == 0:
            spanning_vf_sets = [[]]
        distros = [
            distribution(
                spanning_vf_set=spanning_vf_sets[0],
                assume_spanning_sections_linearly_indep=assume_spanning_sections_linearly_indep,
            )
        ]
        for sl in spanning_vf_sets[1:]:
            distros.append(
                distribution(
                    list(distros[-1].vf_basis) + list(sl),
                    assume_spanning_sections_linearly_indep=assume_spanning_sections_linearly_indep,
                )
            )
        self.distributions = tuple(distros)
        self.growth_vector = tuple(distro.rank for distro in self.distributions)
        self._frame_torsion = None
        self._graded_frame_torsion = None
        self._singularities = dict()
        super().__init__()

    @property
    def vf_basis(self):
        return self.distributions[-1].vf_basis if self.depth > 0 else []

    @property
    def associated_graded_bases(self):
        vfb = self.vf_basis
        out = [[]]
        level = 0
        for idx, vf in enumerate(vfb):
            if idx < self.growth_vector[level]:
                out[-1].append(vf)
                continue
            gap = min(
                gvidx - level
                for gvidx in range(self.depth)
                if self.growth_vector[gvidx] > self.growth_vector[level]
            )
            level += gap
            if gap > 1:
                out += [[] for _ in range(gap - 1)]
            out.append([vf])
        return out

    @property
    def frame_torsion(self):
        if self._frame_torsion is None:
            vfb = self.vf_basis
            dim = len(vfb)
            ft = array_dgcv(
                dict(),
                shape=(dim, dim),
                null_return=freeze_matrix(matrix_dgcv.zeros(dim, 1)),
            )
            combo = linear_combination(vfb, _disposable=True) if dim else None
            for c1, vf1 in enumerate(vfb):
                for c, vf2 in enumerate(vfb[c1 + 1 :]):
                    c2 = c1 + 1 + c
                    bracket = LieDerivative(vf1, vf2)
                    coeffs = decompose(
                        bracket, vfb, assume_basis=True, _generic_combination=combo
                    )[0]
                    if len(coeffs) != dim:
                        print(type(bracket))
                        raise ValueError(
                            f"The filtration's largest level is not involutive. VFs in indices {c1} and {c2} fail bracket closure."
                        )
                    coeffs = matrix_dgcv(coeffs)
                    ft[c1, c2] = coeffs
                    ft[c2, c1] = -coeffs
            self._frame_torsion = ft
        return self._frame_torsion

    @property
    def associated_graded_frame_torsion(self):
        if self._graded_frame_torsion is None:
            ft = self._frame_torsion
            vfb = self.vf_basis
            dim = ft.shape[0] if ft is not None else len(vfb)
            nft = array_dgcv(
                dict(),
                shape=(dim, dim),
                null_return=freeze_matrix(matrix_dgcv.zeros(dim, 1)),
            )

            def find_level(idx):
                for level, ldim in enumerate(self.growth_vector):
                    if idx < ldim:
                        return level + 1
                return self.depth

            def trim(clist, level, c1, c2):
                if level > self.depth:
                    return matrix_dgcv({}, shape=(dim, 1))
                l_idx = level - 1
                if l_idx == 0:
                    ld, ld_inc = 0, self.growth_vector[l_idx]
                else:
                    ld, ld_inc = (
                        self.growth_vector[l_idx - 1],
                        self.growth_vector[l_idx],
                    )
                part2, part3 = clist[ld:ld_inc], clist[ld_inc:]
                if any(not _scalar_is_zero(coef) for coef in part3):
                    if any(not _scalar_is_zero(simplify(coef)) for coef in part3):
                        raise ValueError(
                            f"The filtration is not compatible with Lie brackets, i.e., [F_i,F_j] is not in F_{{i+j}} for i={-find_level(c1)} and j={-find_level(c2)}. The problem occurs at index pair {(c1, c2)}."
                        )
                out = matrix_dgcv(
                    {idx + ld: coef for idx, coef in enumerate(part2)}, shape=(dim, 1)
                )
                return out

            if ft is not None:
                for k, v in ft._data.items():
                    c1, c2 = ft._unspool(k)
                    if c1 > c2:
                        continue
                    nc = trim(v, find_level(c1) + find_level(c2), c1, c2)
                    nft[c1, c2] = nc
                    nft[c2, c1] = -nc
            else:
                combo = None
                levels = [find_level(idx) for idx in range(dim)]
                dependence = engine_capability("apply_dependence")
                deps = (
                    [dependence(vf) for vf in vfb] if dependence is not None else None
                )
                _prepared_prefix_spans(vfb, self.growth_vector)
                for c1, vf1 in enumerate(vfb):
                    d1 = deps[c1] if deps is not None else None
                    for c, vf2 in enumerate(vfb[c1 + 1 :]):
                        c2 = c1 + 1 + c
                        level = levels[c1] + levels[c2]
                        if level > self.depth:
                            nc = matrix_dgcv({}, shape=(dim, 1))
                        else:
                            l_idx = level - 1
                            ld = 0 if l_idx == 0 else self.growth_vector[l_idx - 1]
                            ld_inc = self.growth_vector[l_idx]
                            d2 = deps[c2] if deps is not None else None
                            if (
                                d1 is not None
                                and d2 is not None
                                and d1[0].isdisjoint(d2[1])
                                and d2[0].isdisjoint(d1[1])
                            ):
                                nc = matrix_dgcv(
                                    {idx: 0 for idx in range(ld, ld_inc)},
                                    shape=(dim, 1),
                                )
                                nft[c1, c2] = nc
                                nft[c2, c1] = -nc
                                continue
                            bracket = LieDerivative(vf1, vf2)
                            block = _decompose_prepared(
                                bracket, vfb[:ld_inc], range(ld, ld_inc)
                            )
                            if block is not None:
                                nc = matrix_dgcv(
                                    {
                                        idx + ld: coef
                                        for idx, coef in enumerate(block[ld:ld_inc])
                                    },
                                    shape=(dim, 1),
                                )
                            else:
                                if combo is None and dim:
                                    combo = linear_combination(vfb, _disposable=True)
                                coeffs = decompose(
                                    bracket,
                                    vfb,
                                    assume_basis=True,
                                    _generic_combination=combo,
                                    _coordinate_indices=range(ld, dim),
                                )[0]
                                if len(coeffs) != dim:
                                    raise ValueError(
                                        f"The filtration's largest level is not involutive. VFs in indices {c1} and {c2} fail bracket closure."
                                    )
                                nc = trim(coeffs, level, c1, c2)
                        nft[c1, c2] = nc
                        nft[c2, c1] = -nc
            self._graded_frame_torsion = nft
        return self._graded_frame_torsion

    def nilpotent_approximation(
        self,
        approximation_point=None,
        label=None,
        basis_labels=None,
        exclude_from_VMF: bool = None,
        return_created_object: bool = True,
        Tanaka_symbol_format: bool = False,
        randomize_approximation_point=False,
        **kwargs,
    ) -> algebra_class | Tanaka_symbol:
        if exclude_from_VMF is None:
            exclude_from_VMF = Tanaka_symbol_format
        if exclude_from_VMF or Tanaka_symbol_format:
            return_created_object = True
        agft = self.associated_graded_frame_torsion
        if randomize_approximation_point:
            coordinates = get_free_symbols(agft)
            from random import randint

            approximation_point = dict()
            for var in coordinates:
                in1 = randint(1, 20)
                in2 = randint(in1 + 1, in1 + 20)
                ins = [in1, in2]
                idx = randint(0, 1)
                approximation_point[var] = rational(ins[idx], ins[1 - idx])
            # Add plain text printing
            from ..._aux.printing.printing._dgcv_display import (
                LaTeX_eqn_system,
                show,
            )

            print("Evaluating nilpotent approximation at the randomly chosen point:")
            show(LaTeX_eqn_system(approximation_point, one_line=True))
        if approximation_point is not None:
            if not isinstance(approximation_point, dict):
                dgcv_warning(
                    "approximation_point was given in an unsuported format, and was ignored."
                )
            else:
                agft = subs(agft, approximation_point)
        grading = []
        for level, ldim in enumerate(self.growth_vector):
            if level > 0:
                ldim = ldim - self.growth_vector[level - 1]
            grading += [-level - 1 for _ in range(ldim)]
        if label is None:
            if basis_labels is not None:
                dgcv_warning(
                    "`basis_labels` was provided but no `label` was provided; `basis_labels` is ignored."
                )
            printWarning = (
                "This algebra was initialized via `filtration_class.nilpotent_approximation` with no label; "
                "automatic labels were assigned. Provide `label=...` (and optionally `basis_labels=...`) to control labeling, "
                "or use exclude_from_VMF=True to suppress warnings."
            )
            childPrintWarning = (
                "This algebraElement's parent algebra was initialized via `filtration_class.nilpotent_approximation` with no label; "
                "automatic labels were assigned."
            )
            exclusionPolicy = retrieve_passkey() if exclude_from_VMF is True else None
            outalg = algebra_class(
                agft,
                grading=[grading],
                assume_skew=True,
                _callLock=retrieve_passkey(),
                _print_warning=printWarning,
                _child_print_warning=childPrintWarning,
                _exclude_from_VMF=exclusionPolicy,
            )
        else:
            outalg = createAlgebra(
                agft,
                label,
                basis_labels=basis_labels,
                grading=[grading],
                assume_skew=True,
                return_created_object=True,
                forgo_vmf_registry=exclude_from_VMF,
            )
        if return_created_object:
            if Tanaka_symbol_format:
                return Tanaka_symbol(outalg)
            return outalg
