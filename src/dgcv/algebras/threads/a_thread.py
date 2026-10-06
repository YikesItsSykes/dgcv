from __future__ import annotations

import numbers

from ..._aux._backends._display_engine import is_rich_displaying_available
from ..._aux._utilities._config import (
    get_dgcv_settings_registry,
)
from ..._aux._vmf._safeguards import (
    get_dgcv_category,
)
from ..._aux.printing.printing._dgcv_display import show
from .classifying import (
    _certify_centroid_type,
    _finalize_real_form,
    _frozen_killing_form,
    _narrow_by_complex_type,
    _require_real_semisimple,
    approximate_rank,
    identify_complex_type,
    identify_real_form,
    is_compact_real_form,
    is_split_real_form,
    killing_inertia,
    root_length_profile,
)
from .heating import _summary_warm_caches, _timed_progress_call
from .introspection import (
    Jacobi_identities,
    _check_jacobi_identity,
    _check_skew_symmetric,
    _profile_structure_data,
    _require_lie_algebra,
    _warn_associativity_assumption,
    center,
    compute_derived_algebra,
    compute_simple_subalgebras,
    derived_series,
    is_abelian,
    is_Lie_algebra,
    is_lie_algebra,
    is_nilpotent,
    is_semisimple,
    is_simple,
    is_skew_symmetric,
    is_solvable,
    lower_central_series,
    radical,
    satisfies_jacobi_identity,
)
from .ld import _ldc
from .report_components import _alg_name_latex, _levi_terse
from .reporting import _summary_render_plain, _summary_render_rich
from .vs_thread import _vector_space_methods


class _algebra_methods(_vector_space_methods):
    """
    Shared method implementations for `dgcv` algebra-like classes.

    Notes
    -----
    Inheriting classes must provide `dimension`, `basis`, `structureData`,
    `structureDataDict`, `grading`, `_gradingNumber`, `_educed_properties`,
    `_parameters`, `_registered`, `ambient`, `_dgcv_category` and
    `_verbose_subject`.
    """

    _skew_symmetric_cache = None
    _jacobi_identity_cache = None
    _lie_algebra_cache = None
    _is_semisimple_cache = None
    _is_simple_cache = None
    _is_nilpotent_cache = None
    _is_solvable_cache = None
    _is_abelian_cache = None
    _killing_form = None
    _Levi_deco_cache = None
    _lower_central_series_cache = None
    _lower_central_series_terminated = None
    _lower_central_series_depth = None
    _derived_series_cache = None
    _derived_series_terminated = None
    _derived_series_depth = None
    _derived_subalg_cache = None
    _derived_subalg_lazy_cache = None
    _radical_cache = None
    _center_cache = None
    _grading_compatible = None
    _grading_report = None
    _rank_approximation = None
    _structure_data_slices = None
    _structure_rows_cache = None
    base_field = "complex"
    _structure_data_profile = None
    _verified_ideal = False
    _centroid_type = None
    _real_form_cache = None
    _root_length_profile_cache = None

    @property
    def _structure_rows(self):
        if self._structure_rows_cache is None:
            rows = dict()
            for (i, j, k), v in self.structureDataDict.items():
                row = rows.get((i, j))
                if row is None:
                    rows[(i, j)] = {k: v}
                else:
                    row[k] = v
            self._structure_rows_cache = rows
        return self._structure_rows_cache

    def _profile_structure_data(self):
        return _profile_structure_data(self)

    def _set_product_protocol(self):
        if self.simplify_products_by_default is None:
            number_type = numbers.Number
            if any(
                not isinstance(j, number_type) for j in self.structureDataDict.values()
            ):
                self.simplify_products_by_default = True
            else:
                self.simplify_products_by_default = False
        elif self.simplify_products_by_default is not True:
            self.simplify_products_by_default = False

    def is_real_structure_data(self):
        """
        Returns
        -------
        bool
            True if every structure constant can be validated as real.
        """
        return self._profile_structure_data().is_real

    def is_skew_symmetric(
        self,
        verbose=False,
        _return_proof_path=False,
        _ignore_caches=False,
        *,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
    ):
        """
        Checks if the algebra is skew-symmetric.
        """
        return is_skew_symmetric(
            self,
            verbose=verbose,
            _return_proof_path=_return_proof_path,
            _ignore_caches=_ignore_caches,
            _timed_reporting=_timed_reporting,
            _reporting_threshold_s=_reporting_threshold_s,
            _progress_message=_progress_message,
            _on_timed_update=_on_timed_update,
        )

    def _check_skew_symmetric(self):
        return _check_skew_symmetric(self)

    def satisfies_jacobi_identity(
        self,
        verbose=False,
        _return_proof_path=False,
        _ignore_caches=False,
        *,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
    ):
        """
        Checks if the algebra satisfies the Jacobi identity.
        Includes a warning for unregistered instances only if verbose=True.
        """
        return satisfies_jacobi_identity(
            self,
            verbose=verbose,
            _return_proof_path=_return_proof_path,
            _ignore_caches=_ignore_caches,
            _timed_reporting=_timed_reporting,
            _reporting_threshold_s=_reporting_threshold_s,
            _progress_message=_progress_message,
            _on_timed_update=_on_timed_update,
        )

    def Jacobi_identities(self):
        return Jacobi_identities(self)

    def _check_jacobi_identity(self):
        return _check_jacobi_identity(self)

    def _warn_associativity_assumption(self, method_name):
        """
        Issues a warning that the method assumes the algebra is associative.
        """
        _warn_associativity_assumption(self, method_name)

    def is_lie_algebra(self, verbose=False, return_bool=True):
        return is_lie_algebra(self, verbose=verbose, return_bool=return_bool)

    def is_Lie_algebra(
        self,
        verbose=False,
        return_bool=True,
        _return_proof_path=False,
        _ignore_caches=False,
        *,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
    ):
        return is_Lie_algebra(
            self,
            verbose=verbose,
            return_bool=return_bool,
            _return_proof_path=_return_proof_path,
            _ignore_caches=_ignore_caches,
            _timed_reporting=_timed_reporting,
            _reporting_threshold_s=_reporting_threshold_s,
            _progress_message=_progress_message,
            _on_timed_update=_on_timed_update,
        )

    def _require_lie_algebra(self, method_name):
        """
        Raise if algebra is not a Lie algebra
        """
        _require_lie_algebra(self, method_name)

    def is_semisimple(
        self,
        verbose=False,
        return_bool=True,
        _return_proof_path=False,
        _ignore_caches=False,
        *,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
    ):
        """
        Checks if the algebra is semisimple.
        Nothing is returned if return_bool=False is set.
        """
        return is_semisimple(
            self,
            verbose=verbose,
            return_bool=return_bool,
            _return_proof_path=_return_proof_path,
            _ignore_caches=_ignore_caches,
            _timed_reporting=_timed_reporting,
            _reporting_threshold_s=_reporting_threshold_s,
            _progress_message=_progress_message,
            _on_timed_update=_on_timed_update,
        )

    def is_simple(
        self,
        verbose=False,
        bypass_semisimple_check=False,
        _return_proof_path=False,
        _ignore_caches=False,
        *,
        surface_singularities=False,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
    ):
        return is_simple(
            self,
            verbose=verbose,
            bypass_semisimple_check=bypass_semisimple_check,
            _return_proof_path=_return_proof_path,
            _ignore_caches=_ignore_caches,
            surface_singularities=surface_singularities,
            _timed_reporting=_timed_reporting,
            _reporting_threshold_s=_reporting_threshold_s,
            _progress_message=_progress_message,
            _on_timed_update=_on_timed_update,
        )

    def is_nilpotent(
        self,
        *,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
        **kwargs,
    ):
        """
        Checks if the algebra is nilpotent.

        Returns
        -------
        bool
            True if the algebra is nilpotent, False otherwise.
        """
        return is_nilpotent(
            self,
            _timed_reporting=_timed_reporting,
            _reporting_threshold_s=_reporting_threshold_s,
            _progress_message=_progress_message,
            _on_timed_update=_on_timed_update,
            **kwargs,
        )

    def is_solvable(
        self,
        *,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
        **kwargs,
    ):
        """
        Checks if the algebra is solvable.

        Returns
        -------
        bool
            True if the algebra is solvable, False otherwise.
        """
        return is_solvable(
            self,
            _timed_reporting=_timed_reporting,
            _reporting_threshold_s=_reporting_threshold_s,
            _progress_message=_progress_message,
            _on_timed_update=_on_timed_update,
            **kwargs,
        )

    def is_abelian(
        self,
        *,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
        **kwargs,
    ):
        return is_abelian(
            self,
            _timed_reporting=_timed_reporting,
            _reporting_threshold_s=_reporting_threshold_s,
            _progress_message=_progress_message,
            _on_timed_update=_on_timed_update,
            **kwargs,
        )

    def compute_simple_subalgebras(
        self,
        verbose: bool = False,
        *,
        surface_singularities=False,
        _timed_reporting: bool | None = None,
        _reporting_threshold_s: float = 10,
        _progress_message: str | None = None,
        _on_timed_update=None,
    ):
        return compute_simple_subalgebras(
            self,
            verbose=verbose,
            surface_singularities=surface_singularities,
            _timed_reporting=_timed_reporting,
            _reporting_threshold_s=_reporting_threshold_s,
            _progress_message=_progress_message,
            _on_timed_update=_on_timed_update,
        )

    def compute_derived_algebra(self, lazy_structure=False):
        """
        Computes the derived algebra (commutator subalgebra) for Lie algebras.

        Returns
        -------
        algebra
            A new algebra instance representing the derived algebra.

        Raises
        ------
        ValueError
            If the algebra is not a Lie algebra or if the derived algebra cannot be computed.

        Notes
        -----
        - This method only applies to Lie algebras.
        - The derived algebra is generated by all products [x, y] = x * y, where * is the Lie bracket.
        """
        return compute_derived_algebra(self, lazy_structure=lazy_structure)

    def lower_central_series(
        self,
        max_depth=None,
        format_as_subalgebras=False,
        align_nested_bases=False,
        lazy_structure=False,
    ):
        """
        Computes the lower central series of the algebra (or given subalgebra).

        Parameters
        ----------
        max_depth : int, optional
            Maximum depth to compute the series. Defaults to the dimension of the algebra.

        Returns
        -------
        list of lists
            A list where each entry contains the basis for that level of the lower central series.

        Notes
        -----
        - The lower central series is defined as:
            g_1 = g,
            g_{k+1} = [g_k, g]
        """
        return lower_central_series(
            self,
            max_depth=max_depth,
            format_as_subalgebras=format_as_subalgebras,
            align_nested_bases=align_nested_bases,
            lazy_structure=lazy_structure,
        )

    def derived_series(
        self,
        max_depth=None,
        format_as_subalgebras=False,
        align_nested_bases=False,
        surface_singularities=False,
        simplify_singularities=None,
        force_heavy_solve=False,
        lazy_structure=False,
    ):
        """
        Computes the derived series of the algebra.

        Parameters
        ----------
        max_depth : int, optional
            Maximum depth to compute the series. Defaults to the dimension of the algebra.

        Returns
        -------
        list of lists
            A list where each entry contains the basis for that level of the derived series.

        Notes
        -----
        - The derived series is defined as:
            g^{(1)} = g,
            g^{(k+1)} = [g^{(k)}, g^{(k)}]
        """
        return derived_series(
            self,
            max_depth=max_depth,
            format_as_subalgebras=format_as_subalgebras,
            align_nested_bases=align_nested_bases,
            surface_singularities=surface_singularities,
            simplify_singularities=simplify_singularities,
            force_heavy_solve=force_heavy_solve,
            lazy_structure=lazy_structure,
        )

    def radical(
        self,
        assume_Lie_algebra=False,
        surface_singularities=False,
        simplify_singularities=None,
        force_heavy_solve=False,
        lazy_structure=False,
    ):
        return radical(
            self,
            assume_Lie_algebra=assume_Lie_algebra,
            surface_singularities=surface_singularities,
            simplify_singularities=simplify_singularities,
            force_heavy_solve=force_heavy_solve,
            lazy_structure=lazy_structure,
        )

    def simple_subalgebras(
        self,
        assume_Lie_algebra=False,
        verbose=False,
        surface_singularities=None,
        simplify_singularities=None,
        force_heavy_solve=False,
    ):
        _ = self.Levi_decomposition(
            decompose_semisimple_fully=True,
            assume_Lie_algebra=assume_Lie_algebra,
            verbose=verbose,
            surface_singularities=surface_singularities,
            simplify_singularities=simplify_singularities,
            force_heavy_solve=force_heavy_solve,
        )
        return self._Levi_deco_cache["simple_ideals"]

    def Levi_decomposition(
        self,
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
        return _ldc(
            target_alg=self,
            decompose_semisimple_fully=decompose_semisimple_fully,
            _bust_cache=_bust_cache,
            assume_Lie_algebra=assume_Lie_algebra,
            verbose=verbose,
            surface_singularities=surface_singularities,
            simplify_singularities=simplify_singularities,
            force_heavy_solve=force_heavy_solve,
            _timed_reporting=_timed_reporting,
            _reporting_threshold_s=_reporting_threshold_s,
            _progress_message=_progress_message,
            _on_timed_update=_on_timed_update,
        )

    def Levi_decomposition_report(
        self,
        decompose_semisimple_fully=True,
        generate_partial_report=False,
        generate_full_report=True,
        **kwargs,
    ):
        """
        One-line plain-text summary of the Levi decomposition.

        Parameters
        ----------
        decompose_semisimple_fully : bool, default True
            Split the semisimple part into simple ideals, so that they are
            named individually rather than by dimension.
        **kwargs
            Forwarded to `Levi_decomposition`.

        Returns
        -------
        str
            For example `'su(2)+sl(2,R)'`, or `'not a Lie algebra'`.

        Notes
        -----
        Names the ideals from cached invariants only, so it reports the
        complexification unless the Killing signature is already available.
        Run `summary(generate_full_report=True)` or `identify_real_form`
        first for real-form names.
        """
        if self.is_Lie_algebra(verbose=False) is not True:
            return "not a Lie algebra"
        _ = self.summary(
            plain_text=True,
            return_displayable=True,
            generate_partial_report=generate_partial_report,
            generate_full_report=generate_full_report,
        )
        if not (generate_partial_report or generate_full_report):
            self.Levi_decomposition(
                decompose_semisimple_fully=decompose_semisimple_fully,
                verbose=False,
                **kwargs,
            )
        return _levi_terse(self)

    def center(
        self,
        surface_singularities: bool = None,
        simplify_singularities: bool = None,
        format_as_subalgebra=True,
    ):
        return center(
            self,
            surface_singularities=surface_singularities,
            simplify_singularities=simplify_singularities,
            format_as_subalgebra=format_as_subalgebra,
        )

    def approximate_rank(
        self,
        check_semisimple=False,
        assume_semisimple=False,
        _use_cache=False,
        surface_singularities=False,
        simplify_singularities=None,
    ):
        return approximate_rank(
            self,
            check_semisimple=check_semisimple,
            assume_semisimple=assume_semisimple,
            _use_cache=_use_cache,
            surface_singularities=surface_singularities,
            simplify_singularities=simplify_singularities,
        )

    def _require_real_semisimple(
        self,
        method_name,
        assume_semisimple=False,
        assume_simple=False,
        assume_real=False,
        assume_Lie_algebra=False,
    ):
        return _require_real_semisimple(
            self,
            method_name,
            assume_semisimple=assume_semisimple,
            assume_simple=assume_simple,
            assume_real=assume_real,
            assume_Lie_algebra=assume_Lie_algebra,
        )

    def _frozen_killing_form(self, assume_Lie_algebra=False):
        return _frozen_killing_form(self, assume_Lie_algebra=assume_Lie_algebra)

    def killing_inertia(
        self,
        signature_only: bool = False,
        assume_semisimple=False,
        assume_simple=False,
        assume_real=False,
        assume_Lie_algebra=False,
    ):
        """
        Sylvester inertia of the Killing form.

        Parameters
        ----------
        signature_only : bool, default False
            Return the signature `p - n` in place of the full inertia.
        assume_semisimple : bool, default False
            Skip the semisimplicity check.
        assume_simple : bool, default False
            Skip the semisimplicity check. Does not imply `assume_real`.
        assume_real : bool, default False
            Skip the check that the structure constants are real.
        assume_Lie_algebra : bool, default False
            Skip the Lie algebra check.

        Returns
        -------
        tuple of int or int
            `(p, n, z)`, the counts of positive, negative, and zero
            eigenvalues, or `p - n` when `signature_only` is set.

        Raises
        ------
        IndeterminateSignError
            If a pivot sign cannot be certified, as happens when the
            structure constants carry parameters.
        """
        return killing_inertia(
            self,
            signature_only=signature_only,
            assume_semisimple=assume_semisimple,
            assume_simple=assume_simple,
            assume_real=assume_real,
            assume_Lie_algebra=assume_Lie_algebra,
        )

    def is_compact_real_form(
        self,
        assume_semisimple=False,
        assume_simple=False,
        assume_real=False,
        assume_Lie_algebra=False,
    ):
        """
        Tests whether the algebra is a compact real form.

        Parameters
        ----------
        assume_semisimple : bool, default False
            Skip the semisimplicity check.
        assume_simple : bool, default False
            Skip the semisimplicity check. Does not imply `assume_real`.
        assume_real : bool, default False
            Skip the check that the structure constants are real.
        assume_Lie_algebra : bool, default False
            Skip the Lie algebra check.

        Returns
        -------
        bool

        Notes
        -----
        Equivalent to negative definiteness of the Killing form, so the
        verdict is meaningful for compact semisimple algebras and not only
        simple ones.
        """
        return is_compact_real_form(
            self,
            assume_semisimple=assume_semisimple,
            assume_simple=assume_simple,
            assume_real=assume_real,
            assume_Lie_algebra=assume_Lie_algebra,
        )

    def is_split_real_form(
        self,
        assume_semisimple=False,
        assume_simple=False,
        assume_real=False,
        assume_Lie_algebra=False,
    ):
        """
        Tests whether the algebra is a split real form.

        Parameters
        ----------
        assume_semisimple : bool, default False
            Skip the semisimplicity check.
        assume_simple : bool, default False
            Skip the semisimplicity check. Does not imply `assume_real`.
        assume_real : bool, default False
            Skip the check that the structure constants are real.
        assume_Lie_algebra : bool, default False
            Skip the Lie algebra check.

        Returns
        -------
        bool

        Notes
        -----
        Compares the Killing signature against `approximate_rank`, which
        bounds the rank from above, so a False verdict can be an artifact of
        an overestimated rank.
        """
        return is_split_real_form(
            self,
            assume_semisimple=assume_semisimple,
            assume_simple=assume_simple,
            assume_real=assume_real,
            assume_Lie_algebra=assume_Lie_algebra,
        )

    def _certify_centroid_type(self):
        return _certify_centroid_type(self)

    def root_length_profile(
        self,
        assume_simple=False,
        assume_Lie_algebra=False,
        attempts=4,
        _use_cache=True,
    ):
        """
        Killing-normalized squared root lengths of the complexification.

        Parameters
        ----------
        assume_simple : bool, default False
            Skip the simplicity check
        assume_Lie_algebra : bool, default False
            Skip the Lie algebra check
        attempts : int, default 4
            Number of random elements tried before giving up
        _use_cache : bool, default True

        Returns
        -------
        RealFormReport or None
            `None` when no attempt produced a Cartan subalgebra with a
            rational length spectrum, e.g., when parameters are present.
        """
        return root_length_profile(
            self,
            assume_simple=assume_simple,
            assume_Lie_algebra=assume_Lie_algebra,
            attempts=attempts,
            _use_cache=_use_cache,
        )

    def identify_complex_type(
        self,
        assume_simple=False,
        assume_Lie_algebra=False,
        attempts=4,
        _use_cache=True,
    ):
        """
        Dynkin type of the complexification.

        Parameters
        ----------
        assume_simple : bool, default False
            Skip the simplicity check
        assume_Lie_algebra : bool, default False
            Skip the Lie algebra check
        attempts : int, default 4
            Number of random elements tried before giving up.
        _use_cache : bool, default True

        Returns
        -------
        str or None
            A string tag such as `'B3'` or `'A1+A1'`. `None`
            when the measurement failed or matched no type.
        """
        return identify_complex_type(
            self,
            assume_simple=assume_simple,
            assume_Lie_algebra=assume_Lie_algebra,
            attempts=attempts,
            _use_cache=_use_cache,
        )

    def _narrow_by_complex_type(self, report):
        return _narrow_by_complex_type(self, report)

    def _finalize_real_form(self, report, tier2):
        return _finalize_real_form(self, report, tier2)

    def identify_real_form(
        self,
        assume_simple=False,
        assume_real=False,
        assume_Lie_algebra=False,
        certify_centroid=None,
        tier2=False,
        _use_cache=True,
    ):
        """
        Identifies the real form of a simple Lie algebra

        Parameters
        ----------
        assume_simple : bool, default False
            Skip the semisimplicity and simplicity checks. Does not imply
            `assume_real`.
        assume_real : bool, default False
            Skip the check that the structure constants are real.
        assume_Lie_algebra : bool, default False
            Skip the Lie algebra check.
        certify_centroid : bool, optional
            Whether to compute the centroid verdict separating an absolutely
            simple algebra from a realification
        tier2 : bool, default False
            Whether to resolve a candidate set by measuring the root lengths
            of the complexification
        _use_cache : bool, default True

        Returns
        -------
        RealFormReport
            Candidate real forms together with the invariants they were drawn
            from. More than one candidate means the invariants collide and
            Tier 2 identification is needed to separate them.
        """
        return identify_real_form(
            self,
            assume_simple=False,
            assume_real=False,
            assume_Lie_algebra=False,
            certify_centroid=None,
            tier2=False,
            _use_cache=True,
        )

    def summary(
        self,
        generate_full_report: bool = False,
        generate_partial_report: bool = False,
        theme=None,
        use_latex=None,
        *,
        plain_text: bool = False,
        return_displayable: bool = False,
        show_singularities: bool | None = None,
        interrupt_to_partial_report: bool = True,
        force_heavy_solve: bool = False,
        elide_large_output: bool = True,
        _reporting_threshold_s: float = 7.0,
        **kwargs,
    ):
        dgcvSR = get_dgcv_settings_registry()

        if not isinstance(theme, str):
            theme = kwargs.get("style", None)
            if theme is None:
                theme = dgcvSR.get("theme", "dark")
        if use_latex is None:
            use_latex = dgcvSR.get("use_latex")

        if (plain_text is False) and (not is_rich_displaying_available()):
            plain_text = True

        extra_support_for_math_in_tables = bool(
            dgcvSR.get("extra_support_for_math_in_tables")
        )

        subAlg = get_dgcv_category(self) == "subalgebra"
        parentAlg = self.ambient

        if use_latex and not plain_text:
            algebra_name, algebra_name_cap = _alg_name_latex(parentAlg)
        else:
            algebra_name = (
                parentAlg.label if getattr(parentAlg, "label", None) else "the algebra"
            )
            algebra_name_cap = (
                parentAlg.label if getattr(parentAlg, "label", None) else "The algebra"
            )

        reporting = bool(generate_full_report or generate_partial_report)
        threshold = float(_reporting_threshold_s)
        updates_printed = 0
        interrupted = False

        def _on_update():
            nonlocal updates_printed
            updates_printed += 1

        if reporting:
            try:
                _summary_warm_caches(
                    self,
                    subAlg=subAlg,
                    reporting_threshold_s=threshold,
                    progress_message="finish building the summary",
                    full=generate_full_report,
                    force_heavy_solve=force_heavy_solve,
                    _on_timed_update=_on_update,
                )
            except KeyboardInterrupt:
                if interrupt_to_partial_report is False:
                    raise
                interrupted = True
                updates_printed += 1
                print(
                    "\nInterrupted. Rendering the report from results computed so far. "
                    "Results already cached are retained, so re-running summary resumes "
                    "from where this left off."
                )

        report_full = generate_full_report and not interrupted

        if plain_text:
            out = _timed_progress_call(
                lambda: _summary_render_plain(
                    parentAlg,
                    self,
                    subAlg=subAlg,
                    algebra_name=algebra_name,
                    algebra_name_cap=algebra_name_cap,
                    show_singularities=show_singularities,
                    elide_large_output=elide_large_output,
                ),
                timed=reporting,
                threshold_s=threshold,
                step_desc="rendering the summary",
                continue_desc=None,
                progress_message=None,
                _on_timed_update=_on_update,
            )
            if updates_printed:
                print()
            if return_displayable:
                return out
            print(out)
            return

        out = _timed_progress_call(
            lambda: _summary_render_rich(
                refAlg=self,
                subAlg=subAlg,
                algebra_name=algebra_name,
                algebra_name_cap=algebra_name_cap,
                style=theme,
                use_latex=use_latex,
                extra_support_for_math_in_tables=extra_support_for_math_in_tables,
                show_singularities=show_singularities,
                full=report_full,
                elide_large_output=elide_large_output,
            ),
            timed=reporting,
            threshold_s=threshold,
            step_desc="rendering the summary",
            continue_desc=None,
            progress_message=None,
            _on_timed_update=_on_update,
        )
        if updates_printed:
            print()
        if return_displayable:
            return out
        show(out)

    def _structure_data_slice(self, idx):
        slices = self._structure_data_slices
        if slices is None:
            slices = dict()
            for (i, j, k), v in self.structureDataDict.items():
                slot = slices.get(i)
                if slot is None:
                    slices[i] = {(j, k): v}
                else:
                    slot[(j, k)] = v
            self._structure_data_slices = slices
        slot = slices.get(idx)
        return dict(slot) if slot else dict()

    def _weight_coordinates(self, element):
        if (
            get_dgcv_category(element) == "subalgebra_element"
            and element.algebra != self
            and element.algebra.ambient == self
        ):
            element = element.ambient_rep
        if get_dgcv_category(element) not in {"algebra_element", "subalgebra_element"}:
            raise TypeError(
                f"Input to `check_element_weight` must be an algebra element belonging to the {self._dgcv_category} instance whose `check_element_weight` is being called."
            ) from None
        if element.algebra != self:
            raise TypeError(
                f"Input to `check_element_weight` must be an algebra element belonging to the {self._dgcv_category} instance whose `check_element_weight` is being called."
            ) from None
        return element.coeff_dict
