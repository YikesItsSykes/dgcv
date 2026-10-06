from __future__ import annotations

import gc
import numbers
from contextlib import nullcontext
from itertools import count
from typing import TYPE_CHECKING

from ..._aux._utilities._config import dgcv_warning, get_dgcv_settings_registry
from ..._aux._vmf._safeguards import retrieve_passkey
from ...core.tensors import lazy_tensorProduct
from ._ds import _DS_component, _DS_record
from ._tensor_products import _fast_tensor_products, _hom_id_from_decomp

if TYPE_CHECKING:
    from ._symbol import Tanaka_symbol


class _symbol_prolongation:
    def prolong(
        self,
        iterations: int,
        return_symbol: bool = True,
        report_progress: bool = False,
        report_progress_and_return_nothing: bool = False,
        with_characteristic_space_reductions: bool = None,
        absorb_distinguished_subspaces: bool = False,
        surface_singularities: bool = None,
        simplify_singularities: bool = None,
        max_report_columns: int = 13,
        solve_method: str = None,
    ) -> Tanaka_symbol:
        """
        Computes a number of prolongations above the highest level stored in the current Tanaka_symbol data (starting with the first nonnegative integer, so if highest level is -2, it starts with 0, e.g.), up to the number given in `iterations`. Will stop earlier if stabelization is detected.

        New prolongation levels are stored in the symbols internal data. Can alternatively return a new Tanaka_symbol object with the new prolongation levels stored in its data.

        parameters:
        -----------
            iterations: int
        Determines how many prolongation levels to compute

            return_symbol: bool (optional, default=True)
        If True, the algorithm will return a new Tanaka_symbol object with the new prolongation levels stored in its data. If False returns a dictionary whose keys are level weights and values are corresponding prolongation levels (including all initial levels in the symbol data).

            report_progress:
        Reports progress summaries as prolongations are computed in printed output

            report_progress_and_return_nothing: bool (optional, default=False)
        If True, overides `return_symbol` and `surface_singularities`, reports progress summaries of prolongation in printed output, and returns nothing further.

            with_characteristic_space_reductions: bool (defaults to True if relevant - False if theoretically irrelivant)
        If True and `distinguished_subspaces` was registered in the Tanaka_symbol obj creationg then the prolongation algorithm reduces computed prolongation subspaces to the maximal subspaces allowing the result to be a Lie algebra. This is relavent because if any subspace in distinguished_subspaces has elements of nonnegative weight then the standard subspace-preserving prolongation need not yield an algebra. While this reduction can be geometrically motivated, omitting it permits a less demanding algorithm that will produce a larger prolongation space. A typical application for omitting it would be confirming finite prolongation dimensionality with a less demanding algorithm.

            absorb_distinguished_subspaces:bool (default=False)
        This is relevant when distinguished_subspaces include weighted components of higher weight than the given levels data. In this case those higher weighted components are interpreted as subspaces in a tensor algebra where the theoretical prolongation lives, and on which it also operates. The prolongation algorithm proceeds respecting these in the sense that it honors the axiom that distinguished subspaces are invariant under the action of computed prolongation levels. When absorb_distinguished_subspaces=True, then higher weighted levels are artificially added to computed prolongation levels of the same weight. Setting this to True is only natural in niche geometry motivated contexts, so leaving it as False is likely correct if unsure.

            surface_singularities:bool (optional, default=None)
        If paramater space is empty then this is irrelevant for the prolongation algorithm, but affects the output signature. If True (and not overridden by report_progress_and_return_nothing), additionally returns a list of functions whose zeros are singularities in parameter space found during prolongation. i.e., prolongation will be correct for parameters outside of these singularities. If not set to anything (i.e., left as default None) then when relevant all singularities are still found and saved internally. If set to False then tracking of singularities will be skipped, which may improve the algorithm speed slightly. Note that skipping the tracking does not make the computation parameter-safe: the linear solver still pivots on parameter-dependent coefficients, so the result remains valid for generic parameters, just without a record of where it fails.

            simplify_singularities:bool (optional, default=None)
        Controls simplification of the pivots and singularity ideals produced while tracking singularities. Has no effect when singularity tracking is off (i.e. when surface_singularities=False or the parameter space is empty). Defaults to the `simplify_singularity_ideals_by_default` setting.

            max_report_columns: int (positive)
        This controls the maximum number of columns displayed in the output tables when report_progress is true.

            solve_method: str (optional, default=None)
        The `method` forwarded to every equation solve in the prolongation algorithm. Left as None, it resolves to "linear", except when the symbol carries parameters and singularity tracking is off, in which case it resolves to "linear_parametric" - that case solves systems whose coefficients are unreduced parameter-dependent expressions, which the parametric dispatch handles better. Set it to any method `solve_dgcv` accepts to override uniformly. While singularity tracking is on, a non-linear method is rejected with a warning and replaced by "linear", since divisor collection requires a linear method.

        Under the builtin symbolic engine, a symbol with parameters is prolonged on generic structure constants (each distinct primitive structure constant replaced by a fresh symbol, pivots verified nonzero by expansion, results expanded back at the end) unless `set_dgcv_settings(generic_structure_constants=False)`; distinguished-subspace absorption, characteristic-space reductions and compressed equation systems take the direct algorithm.

        returns:
        --------
            A Tanaka_symbol object with new prolongation levels stored in its data. Returns a dictionary of unformatted prolongation level data if return_symbol=False.

            Additionally returns a list of singularities in parameter space if surface_singularities
        """
        if (
            not isinstance(max_report_columns, numbers.Integral)
            or max_report_columns < 5
        ):
            max_report_columns = 5
        subspace_data = [
            _DS_record(
                {
                    k: _DS_component(
                        list(c.spanners),
                        None if c.parts is None else list(c.parts),
                    )
                    for k, c in record.components.items()
                },
                record.display,
                cap=record.cap,
            )
            for record in self._DS_records
        ]
        if absorb_distinguished_subspaces is True:
            if len(self._inadmissible_DS) > 0:
                dgcv_warning(
                    "Some of the distinguished subspaces supplied to this symbol were found inadmissible during preprocessing, as reported by warnings raised at construction. Those subspaces constrain nothing and are not absorbed into the prolongation."
                )
        if with_characteristic_space_reductions is None:
            with_characteristic_space_reductions = (
                self._default_to_characteristic_space_reductions
            )
        if report_progress_and_return_nothing is True:
            report_progress = True
        if not isinstance(iterations, numbers.Integral) or iterations < 1:
            dgcv_warning(
                "`prolong` expects `iterations` to be a positive int. So no prolongation was performed."
            )
            if return_symbol:
                return self
            return
        has_parameters = len(self._parameters) > 0
        track_singularities = has_parameters and surface_singularities is not False
        if simplify_singularities is None:
            simplify_pivots = track_singularities
            simplify_ideals = get_dgcv_settings_registry().get(
                "simplify_singularity_ideals_by_default", True
            )
        else:
            simplify_pivots = simplify_singularities
            simplify_ideals = simplify_singularities
        if solve_method is None:
            solve_method = (
                "linear_parametric"
                if has_parameters and not track_singularities
                else "linear"
            )
        elif track_singularities and solve_method not in _solve_methods:
            dgcv_warning(
                "`prolong` requires a linear solve method while tracking parameter-space "
                f"singularities, so `solve_method={solve_method!r}` was replaced by 'linear'."
            )
            solve_method = "linear"
        levels = self.levels
        height = self.height
        stable = False
        if report_progress:
            prol_counter = 1

            def count_to_str(count):
                return f"{count}{'st' if count == 1 else 'nd' if count == 2 else 'rd' if count == 3 else 'th'}"

        levels = self._GLA_structure(
            {w: list(level) for w, level in levels.items()},
            initial_index=levels.index_threshold,
        )
        alias_counter = count(
            max(
                [sum(len(level) for w, level in levels.items() if w < 0)]
                + [idx + 1 for idx in self._aliasing if idx >= 0]
            )
        )
        get_alias_id = alias_counter.__next__
        generic = self._generic_mode(
            with_characteristic_space_reductions,
            absorb_distinguished_subspaces is True,
            subspace_data,
        )
        preprocess_noted = False
        for w in levels:
            if w >= 0:
                if not preprocess_noted and report_progress:
                    preprocess_noted = True
                    print(
                        "Preprocessing nonnegative-weighted components for the faster prolongation algorithm."
                    )

                reformatted_level = []
                for position, j in enumerate(levels[w]):
                    new_idx = self._nonneg_atoms.get((w, position))
                    if new_idx is None:
                        new_idx = get_alias_id()
                        alias_data = {"expanded": _fast_tensor_products(j)}
                        hom = self._nonneg_hom.get((w, position))
                        if hom:
                            alias_data["hom"] = hom
                        self._aliasing[new_idx] = alias_data
                    reformatted_level.append(
                        _fast_tensor_products({(new_idx,): 1}, _atomic_index=new_idx)
                    )
                levels[w] = reformatted_level
        swap = self._generic_structure(generic) if generic is not None else nullcontext()
        with swap:
            for j in range(iterations):
                if stable:
                    break
                gc.freeze()
                try:
                    levels, stable, subspace_data = self._fast_prolong_by_1(
                        levels,
                        height,
                        with_characteristic_space_reductions=with_characteristic_space_reductions,
                        DS_records=subspace_data,
                        absorb_DS=absorb_distinguished_subspaces is True,
                        surface_singularities=track_singularities,
                        simplify_pivots=simplify_pivots,
                        simplify_ideals=simplify_ideals,
                        solve_method=solve_method,
                        alias_counter=get_alias_id,
                        generic=generic,
                    )
                finally:
                    gc.unfreeze()
                if report_progress:
                    keys = list(levels.keys())
                    values = list(levels.values())
                    n_cols = len(keys)
                    elision_cell = " … "
                    elision_border = "     "

                    if n_cols > max_report_columns:
                        seg = max_report_columns // 2
                        display_keys = keys[:seg] + [None] + keys[-seg:]
                        display_values = values[:seg] + [None] + values[-seg:]
                    else:
                        display_keys = keys
                        display_values = values

                    max_len = max(
                        max(len(str(k)) for k in keys),
                        max(len(str(len(v))) for v in values),
                    )
                    is_elision = [item is None for item in display_keys]

                    def fmt_row(label, items):
                        cells = []
                        for item, elide in zip(items, is_elision):
                            cells.append(
                                elision_cell if elide else str(item).ljust(max_len)
                            )
                        return f"│ {label} │ " + " │ ".join(cells) + " │"

                    def fmt_border(left, mid, right, junction):
                        segments = []
                        for elide in is_elision:
                            segments.append(
                                elision_border if elide else "─" * (max_len + 2)
                            )
                        header_fill = "─" * len("Weights    │")
                        return f"{left}{header_fill}{mid}" + junction.join(segments) + right

                    weight_strs = [None if k is None else str(k) for k in display_keys]
                    dim_strs = [None if v is None else str(len(v)) for v in display_values]

                    print(f"After {count_to_str(prol_counter)} iteration:")
                    print(fmt_border("┌", "┬", "┐", "┬"))
                    print(fmt_row("Weights   ", weight_strs))
                    print(fmt_border("├", "┼", "┤", "┼"))
                    print(fmt_row("Dimensions", dim_strs))
                    print(fmt_border("└", "┴", "┘", "┴"))
                    prol_counter += 1
                height += 1
        if generic is not None:
            self._expand_generic_levels(generic)
        atom_positions = dict()
        for w in levels:
            if w >= 0:
                for c, j in enumerate(levels[w]):
                    idx = getattr(j, "_atomic_index", -1)
                    if idx >= 0:
                        atom_positions[idx] = (w, c)
        neg_lookup = {v: k for k, v in self._negative_basis_positions().items()}
        neg_dim = self.negativePart.dimension
        neg_basis = self.negativePart.basis
        card = getattr(
            getattr(neg_basis[0], "algebra", self.negativePart), "card", None
        )
        grading = self.negativePart.grading
        if isinstance(grading[0], (list, tuple)):
            grading = grading[0]
        for w in levels:
            if w >= 0:
                converted = []
                for c, j in enumerate(levels[w]):
                    alias = self._aliasing.get(getattr(j, "_atomic_index", -1))
                    if alias is not None and alias.get("_pending_S"):
                        self._materialize_S_atom(alias)
                    decomp = self._alias_hom_decomp(
                        alias,
                        w,
                        atom_positions,
                        neg_lookup,
                        neg_dim,
                    )
                    label = f"{self._plp}_{c + 1}__{{[{w}]}}"
                    if decomp and card is not None:
                        converted.append(
                            lazy_tensorProduct(
                                self._level_thunk(j),
                                card,
                                w,
                                grading=grading,
                                _hom_id=_hom_id_from_decomp(
                                    decomp, label, self._plp, self.levels
                                ),
                                _hom_decomp=dict(decomp),
                            )
                        )
                        continue
                    converted.append(
                        self._aliased_expansion(j)._convert_to_tp(
                            _hom_id_map=(self._plp, self.levels),
                            _hom_id_label=label,
                            _hom_id=[decomp, ""] if decomp else None,
                            _decomp_complete=True,
                        )
                    )
                levels[w] = converted
        if report_progress_and_return_nothing is not True:
            if return_symbol:
                from ._symbol import Tanaka_symbol

                new_nonneg_parts = []
                for key, value in levels.items():
                    if key >= 0:
                        new_nonneg_parts += value
                out = Tanaka_symbol(
                    self.negativePart,
                    new_nonneg_parts,
                    assume_FGLA=self.assume_FGLA,
                    precompute_generators=self._GLA_generators is not None,
                    compress_equation_systems=self._compress_equation_systems,
                    distinguished_subspaces=self.distinguished_subspaces,
                    assume_NNP_linear_indep=True,
                    index_threshold=levels.index_threshold,
                    _validated=retrieve_passkey(),
                    _internal_parameters=self._parameters,
                    _internal_singularities=self._singularities,
                )
                if surface_singularities is True:
                    return out, self._singularities.get("prolongation", set())
                return out
            else:
                if surface_singularities is True:
                    return levels, self._singularities.get("prolongation", set())
                return levels


    def _level_thunk(self, atom):
        def thunk():
            return self._aliased_expansion(atom)._convert_to_tp()

        return thunk


_solve_methods = ("linear", "linear_parametric", "linsolve")
