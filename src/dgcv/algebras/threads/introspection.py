from __future__ import annotations

from dataclasses import dataclass

from ..._aux._backends._engine import engine_capability
from ..._aux._backends._polynomials import (
    expr_union_primitives,
)
from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    conjugate,
    get_free_symbols,
    simplify,
    subs,
)
from ..._aux._utilities._config import (
    dgcv_warning,
    dgcvDeprecationWarning,
    get_dgcv_settings_registry,
)
from ..._aux._utilities._misc import linear_combination
from ..._aux._vmf._safeguards import (
    get_dgcv_category,
    retrieve_passkey,
)
from ..._aux._vmf.vmf import clearVar, listVar, order_coordinates, vmf_lookup
from ...core.arrays import freeze_matrix, matrix_dgcv
from ..aec import algebra_element_class
from ...core.conversions.conversions import cleanUpConjugation
from ...core.solvers import solve_dgcv
from .heating import _timed_progress_call
from .util import (
    _indep_check,
    _solve_weight_kwargs,
    _span_solver,
    killingForm,
)


def _profile_structure_data(target_alg):
    if target_alg._structure_data_profile is not None:
        return target_alg._structure_data_profile
    sub_types = {
        vmf_lookup(param).get("sub_type") for param in (target_alg._parameters or ())
    }
    if sub_types <= {"real", "imag", "holo", "anti"}:
        mode = "symbolic"
    elif not {"holo", "anti"} <= sub_types:
        mode = "plain"
    else:
        mode = "cleanup"
    witnesses = []
    for key, value in target_alg.structureDataDict.items():
        if mode == "symbolic":
            conjugated = conjugate(value, symbolic=True)
        elif mode == "plain":
            conjugated = conjugate(value, symbolic=False)
        else:
            conjugated = cleanUpConjugation(conjugate(value, symbolic=False))
        diff = value - conjugated
        if _scalar_is_zero(diff):
            continue
        if get_free_symbols(diff) and _scalar_is_zero(simplify(diff)):
            continue
        witnesses.append(key)
    profile = _StructureDataProfile(
        is_real=not witnesses,
        realness_witnesses=tuple(witnesses),
    )
    target_alg._structure_data_profile = profile
    return profile


def is_skew_symmetric(
    target_alg,
    verbose=False,
    _return_proof_path=False,
    _ignore_caches=False,
    *,
    _timed_reporting: bool | None = None,
    _reporting_threshold_s: float = 10,
    _progress_message: str | None = None,
    _on_timed_update=None,
):
    if verbose and not target_alg._registered:
        if target_alg.ambient._callLock == retrieve_passkey() and isinstance(
            target_alg.ambient._print_warning, str
        ):
            print(target_alg.ambient._print_warning)
        else:
            print(
                "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
            )

    educed = target_alg._educed_properties.get("is_skew", None)
    if isinstance(educed, str) and _ignore_caches is False:
        t_message = educed
        target_alg._skew_symmetric_cache = (True, None)
    else:
        t_message = ""

    timed = bool(_timed_reporting) if _timed_reporting is not None else False

    cached = target_alg._skew_symmetric_cache
    if cached is not None and _ignore_caches is False:
        result, failure = cached
    else:
        result, failure = _timed_progress_call(
            target_alg._check_skew_symmetric,
            timed=timed,
            threshold_s=float(_reporting_threshold_s),
            step_desc="checking skew symmetry of the structure constants",
            continue_desc=_progress_message,
            progress_message=None,
            _on_timed_update=_on_timed_update,
        )
        target_alg._skew_symmetric_cache = (result, failure)

    if verbose and not timed:
        if result:
            print(f"{target_alg._verbose_subject()} is skew-symmetric.")
        else:
            i, j, k = failure
            print(
                f"Skew symmetry fails for basis elements {i}, {j}, at coefficient index {k}."
            )
    if _return_proof_path is True:
        return result, t_message
    return result


def _check_skew_symmetric(target_alg):
    sdd = target_alg.structureDataDict
    candidates = {(i, j, k) if i <= j else (j, i, k) for i, j, k in sdd}
    for i, j, k in sorted(candidates):
        expr = sdd.get((i, j, k), 0) + sdd.get((j, i, k), 0)
        if _scalar_is_zero(expr):
            continue
        if get_free_symbols(expr) and _scalar_is_zero(simplify(expr)):
            continue
        return False, (i, j, k)
    return True, None


def satisfies_jacobi_identity(
    target_alg,
    verbose=False,
    _return_proof_path=False,
    _ignore_caches=False,
    *,
    _timed_reporting: bool | None = None,
    _reporting_threshold_s: float = 10,
    _progress_message: str | None = None,
    _on_timed_update=None,
):
    if verbose and not target_alg._registered:
        if target_alg.ambient._callLock == retrieve_passkey() and isinstance(
            target_alg.ambient._print_warning, str
        ):
            print(target_alg.ambient._print_warning)
        else:
            print(
                "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
            )

    if (
        isinstance(target_alg._educed_properties.get("satisfies_Jacobi_ID", None), str)
        and _ignore_caches is False
    ):
        t_message = target_alg._educed_properties.get("satisfies_Jacobi_ID", None)
        target_alg._jacobi_identity_cache = (True, None)
    else:
        t_message = ""

    timed = bool(_timed_reporting) if _timed_reporting is not None else False
    threshold = float(_reporting_threshold_s)

    if target_alg._jacobi_identity_cache is None or _ignore_caches is True:
        result, fail_list = _timed_progress_call(
            target_alg._check_jacobi_identity,
            timed=timed,
            threshold_s=threshold,
            step_desc="checking the Jacobi identity",
            continue_desc=_progress_message,
            progress_message=None,
            _on_timed_update=_on_timed_update,
        )
        target_alg._jacobi_identity_cache = (result, fail_list)
    else:
        result, fail_list = target_alg._jacobi_identity_cache

    if verbose and not timed:
        if result:
            print(f"{target_alg._verbose_subject()} satisfies the Jacobi identity.")
        else:
            print(f"Jacobi identity fails for the following triples: {fail_list}")

    if _return_proof_path is True:
        return result, t_message
    return result


def Jacobi_identities(target_alg):
    skew, dim, basis = (
        target_alg.is_skew_symmetric(),
        target_alg.dimension,
        target_alg.basis,
    )
    JI_list = []
    for i in range(dim):
        lower_j = i + 1 if skew else 0
        for j in range(lower_j, dim):
            lower_k = j + 1 if skew else 0
            for k in range(lower_k, dim):
                ai, aj, ak = basis[i], basis[j], basis[k]
                JI_list.append(ai * aj * ak + aj * ak * ai + ak * ai * aj)
    return JI_list


def _check_jacobi_identity(target_alg):
    skew, dim = target_alg.is_skew_symmetric(), target_alg.dimension
    rows = target_alg._structure_rows
    if skew:
        candidates = set()
        for a, b in rows:
            lo, hi = (a, b) if a < b else (b, a)
            if lo == hi:
                continue
            for c in range(dim):
                if c == lo or c == hi:
                    continue
                candidates.add(tuple(sorted((lo, hi, c))))
        triples = sorted(candidates)
    else:
        triples = (
            (i, j, k)
            for i in range(dim)
            for j in range(dim)
            for k in range(dim)
            if (i, j) in rows or (j, k) in rows or (k, i) in rows
        )
    fail_list = []
    for i, j, k in triples:
        acc = dict()
        for a, b, c in ((i, j, k), (j, k, i), (k, i, j)):
            left = rows.get((a, b))
            if not left:
                continue
            for m, cm in left.items():
                right = rows.get((m, c))
                if not right:
                    continue
                for n, cn in right.items():
                    acc[n] = acc.get(n, 0) + cm * cn
        for expr in acc.values():
            if _scalar_is_zero(expr):
                continue
            if get_free_symbols(expr) and _scalar_is_zero(simplify(expr)):
                continue
            fail_list.append((i, j, k))
            break
    if fail_list:
        return False, fail_list
    return True, None


def _warn_associativity_assumption(target_alg, method_name):
    dgcv_warning(
        f"{method_name} assumes the algebra is associative. "
        "If it is not then unexpected results may occur."
    )


def is_lie_algebra(target_alg, verbose=False, return_bool=True):
    dgcv_warning(
        "`is_lie_algebra` has been deprecated as part of the shift toward standardized naming conventions in the `dgcv` library.",
        dgcvDeprecationWarning,
        stacklevel=2,
        old_kw="is_lie_algebra",
        new_kw="is_Lie_algebra",
        sunset="2026",
    )
    return target_alg.is_Lie_algebra(verbose=verbose, return_bool=return_bool)


def is_Lie_algebra(
    target_alg,
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
    if verbose and not target_alg._registered:
        if target_alg.ambient._callLock == retrieve_passkey() and isinstance(
            target_alg.ambient._print_warning, str
        ):
            print(target_alg.ambient._print_warning)
        else:
            print(
                "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
            )

    if isinstance(target_alg._educed_properties.get("is_Lie_algebra", None), str):
        t_message = target_alg._educed_properties.get("is_Lie_algebra", None)
        target_alg._lie_algebra_cache = True
        target_alg._jacobi_identity_cache = (True, None)
        target_alg._skew_symmetric_cache = (True, None)
    else:
        t_message = ""

    timed = bool(_timed_reporting) if _timed_reporting is not None else False
    threshold = float(_reporting_threshold_s)

    if target_alg._lie_algebra_cache is not None and _ignore_caches is False:
        if verbose and not timed:
            print(
                f"Cached result: Previously verified "
                f"{target_alg._verbose_subject()} is"
                f"{'' if target_alg._lie_algebra_cache else ' not'} a Lie algebra."
            )
        if _return_proof_path is True:
            return target_alg._lie_algebra_cache, t_message
        return target_alg._lie_algebra_cache

    ok_skew = target_alg.is_skew_symmetric(
        verbose=verbose,
        _ignore_caches=_ignore_caches,
        _timed_reporting=timed,
        _reporting_threshold_s=threshold,
        _progress_message="check the Jacobi identity",
        _on_timed_update=_on_timed_update,
    )
    if not ok_skew:
        target_alg._lie_algebra_cache = False
        if return_bool is True:
            if _return_proof_path is True:
                return False, t_message
            return False
        return

    ok_jacobi = target_alg.satisfies_jacobi_identity(
        verbose=verbose,
        _ignore_caches=_ignore_caches,
        _timed_reporting=timed,
        _reporting_threshold_s=threshold,
        _progress_message=_progress_message,
        _on_timed_update=_on_timed_update,
    )
    if not ok_jacobi:
        target_alg._lie_algebra_cache = False
        if return_bool is True:
            if _return_proof_path is True:
                return False, t_message
            return False
        return

    if target_alg._lie_algebra_cache is None or _ignore_caches is True:
        target_alg._lie_algebra_cache = True

    if verbose and not timed:
        print(f"{target_alg._verbose_subject()} is a Lie algebra.")

    if return_bool is True:
        if _return_proof_path is True:
            return target_alg._lie_algebra_cache, t_message
        return target_alg._lie_algebra_cache


def _require_lie_algebra(target_alg, method_name):
    if not target_alg.is_Lie_algebra():
        raise ValueError(
            f"{method_name} can only be applied to Lie algebras."
        ) from None


def is_semisimple(
    target_alg,
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
    if verbose and not target_alg._registered:
        if target_alg.ambient._callLock == retrieve_passkey() and isinstance(
            target_alg.ambient._print_warning, str
        ):
            print(target_alg.ambient._print_warning)
        else:
            print(
                "Warning: This algebra instance is unregistered. Initialize algebra objects with createFiniteAlg instead to register them."
            )

    if (
        isinstance(target_alg._educed_properties.get("is_simple", None), str)
        and _ignore_caches is False
    ):
        t_message = target_alg._educed_properties.get("is_simple", None)
        target_alg._is_simple_cache = True
        target_alg._is_semisimple_cache = True
        target_alg._educed_properties["special_type"] = "simple"
        target_alg._is_nilpotent_cache = False
        target_alg._is_solvable_cache = False
    elif (
        isinstance(target_alg._educed_properties.get("is_semisimple", None), str)
        and _ignore_caches is False
    ):
        t_message = target_alg._educed_properties.get("is_semisimple", None)
        target_alg._is_semisimple_cache = True
        target_alg._educed_properties["special_type"] = (
            target_alg._educed_properties.get("special_type", None) or "semisimple"
        )
        target_alg._is_nilpotent_cache = False
        target_alg._is_solvable_cache = False
    else:
        t_message = ""

    timed = bool(_timed_reporting) if _timed_reporting is not None else False
    threshold = float(_reporting_threshold_s)

    if target_alg._is_semisimple_cache is None and _ignore_caches is False:
        if target_alg._is_simple_cache is True:
            target_alg._is_semisimple_cache = True
            target_alg._is_solvable_cache = False
            target_alg._is_abelian_cache = False
            target_alg._is_nilpotent_cache = False
        elif target_alg._Levi_deco_cache is not None:
            LC, MSI = target_alg._Levi_deco_cache["LD_components"]
            if getattr(MSI, "dimension", None) == 0 and target_alg.dimension > 0:
                target_alg._is_semisimple_cache = True
                target_alg._is_solvable_cache = False
                target_alg._is_abelian_cache = False
                target_alg._is_nilpotent_cache = False
            elif getattr(MSI, "dimension", None) != 0:
                target_alg._is_semisimple_cache = False
                target_alg._is_simple_cache = False
                if getattr(LC, "dimension", None) == 0:
                    target_alg._is_solvable_cache = True
                    target_alg._educed_properties["special_type"] = "solvable"
                else:
                    target_alg._is_solvable_cache = False
                    target_alg._is_nilpotent_cache = False
                    target_alg._is_abelian_cache = False

    if target_alg._is_semisimple_cache is not None and _ignore_caches is False:
        if verbose and not timed:
            print(
                f"Cached result: Previously verified "
                f"{target_alg._verbose_subject()} is"
                f"{'' if target_alg._is_semisimple_cache else ' not'} a semisimple Lie algebra."
            )
        if return_bool is True:
            if _return_proof_path is True:
                return target_alg._is_semisimple_cache, t_message
            return target_alg._is_semisimple_cache
        if _return_proof_path is True:
            return t_message
        return

    ok_lie = target_alg.is_Lie_algebra(
        verbose=verbose,
        _ignore_caches=_ignore_caches,
        _timed_reporting=timed,
        _reporting_threshold_s=threshold,
        _progress_message=_progress_message,
        _on_timed_update=_on_timed_update,
    )

    if not ok_lie:
        target_alg._is_semisimple_cache = False
        if return_bool is True:
            if _return_proof_path is True:
                return False, "not a Lie algebra"
            return False
        if _return_proof_path is True:
            return "not a Lie algebra"
        return

    def _killing_det():
        if target_alg._killing_form is None:
            target_alg._killing_form = freeze_matrix(killingForm(target_alg))
        kf = target_alg._killing_form
        generic_det = engine_capability("generic_det")
        if generic_det is not None:
            probe = generic_det(kf)
            if probe is not None:
                return probe
        return simplify(kf.det())

    det = _timed_progress_call(
        _killing_det,
        timed=timed,
        threshold_s=threshold,
        step_desc="computing determinant of the Killing form",
        continue_desc=_progress_message,
        progress_message=None,
        _on_timed_update=_on_timed_update,
    )

    ###!!! performan vs. accuracy flag: heavier zero-test falling back to literal
    iz = getattr(det, "is_zero", None)
    if iz is True:
        det_is_zero = True
    elif callable(iz):
        try:
            det_is_zero = bool(iz())
        except Exception:
            det_is_zero = _scalar_is_zero(det)
    else:
        det_is_zero = _scalar_is_zero(det)

    det_is_nonzero = not det_is_zero

    if det_is_nonzero:
        target_alg._is_semisimple_cache = True
        target_alg._educed_properties["special_type"] = "semisimple"
        target_alg._is_nilpotent_cache = False
        target_alg._is_solvable_cache = False
    else:
        target_alg._is_semisimple_cache = False
        target_alg._is_simple_cache = False

    if verbose and not timed:
        print(
            f"{target_alg._verbose_subject()} is"
            f"{'' if det_is_nonzero else ' not'} semisimple."
        )

    if return_bool is True:
        if _return_proof_path is True:
            return det_is_nonzero, t_message
        return det_is_nonzero


def is_simple(
    target_alg,
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
    if isinstance(target_alg._educed_properties.get("is_simple", None), str):
        t_message = target_alg._educed_properties.get("is_simple", None)
        target_alg._is_simple_cache = True
        target_alg._is_semisimple_cache = True
        target_alg._educed_properties["special_type"] = "simple"
        target_alg._is_nilpotent_cache = False
        target_alg._is_solvable_cache = False
    else:
        t_message = ""

    timed = bool(_timed_reporting) if _timed_reporting is not None else False
    threshold = float(_reporting_threshold_s)

    if bypass_semisimple_check is False and target_alg._is_semisimple_cache is None:
        target_alg.is_semisimple(
            verbose=verbose,
            _ignore_caches=_ignore_caches,
            _timed_reporting=timed,
            _reporting_threshold_s=threshold,
            _progress_message=_progress_message,
            _on_timed_update=_on_timed_update,
        )

    if target_alg._is_simple_cache is None:
        target_alg.compute_simple_subalgebras(
            verbose=verbose,
            surface_singularities=surface_singularities,
            _timed_reporting=timed,
            _reporting_threshold_s=threshold,
            _progress_message=_progress_message,
            _on_timed_update=_on_timed_update,
        )

        if target_alg._Levi_deco_cache["LD_components"][1].dimension == 0:
            target_alg._is_semisimple_cache = True
            target_alg._is_nilpotent_cache = False
            target_alg._is_solvable_cache = False
            if len(target_alg._Levi_deco_cache["simple_ideals"]) == 1:
                target_alg._is_simple_cache = True
                target_alg._educed_properties["special_type"] = "simple"
            else:
                target_alg._is_simple_cache = False
                target_alg._educed_properties["special_type"] = "semisimple"
        else:
            target_alg._is_semisimple_cache = False
            target_alg._is_simple_cache = False
            if target_alg._Levi_deco_cache["LD_components"][0].dimension == 0:
                target_alg._is_solvable_cache = True
                if target_alg._educed_properties.get("special_type", None) is None:
                    target_alg._educed_properties["special_type"] = "solvable"

    if _return_proof_path is True:
        return target_alg._is_simple_cache, t_message
    return target_alg._is_simple_cache


def is_nilpotent(
    target_alg,
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
    if kwargs:
        dgcv_warning(
            f"`{type(target_alg).__name__}.is_nilpotent` received unexpected keyword "
            f"argument(s) {sorted(kwargs)}, which were ignored."
        )
    if target_alg._is_nilpotent_cache is None and target_alg._is_abelian_cache is True:
        target_alg._is_nilpotent_cache = True
    if target_alg._is_nilpotent_cache is None:
        _timed_progress_call(
            target_alg.lower_central_series,
            timed=bool(_timed_reporting) if _timed_reporting is not None else False,
            threshold_s=float(_reporting_threshold_s),
            step_desc="computing the lower central series",
            continue_desc=_progress_message,
            progress_message=None,
            _on_timed_update=_on_timed_update,
        )
        if getattr(target_alg, "_lower_central_series_terminated", None) is True:
            target_alg._is_nilpotent_cache = True
            target_alg._educed_properties["special_type"] = "nilpotent"
            target_alg._is_semisimple_cache = False
            target_alg._is_simple_cache = False
        else:
            target_alg._is_nilpotent_cache = False
            target_alg._is_abelian_cache = False
    return target_alg._is_nilpotent_cache


def is_solvable(
    target_alg,
    *,
    _timed_reporting: bool | None = None,
    _reporting_threshold_s: float = 10,
    _progress_message: str | None = None,
    _on_timed_update=None,
    **kwargs,
):
    if kwargs:
        dgcv_warning(
            f"`{type(target_alg).__name__}.is_solvable` received unexpected keyword "
            f"argument(s) {sorted(kwargs)}, which were ignored."
        )
    if target_alg._is_solvable_cache is None:
        if (
            target_alg._is_nilpotent_cache is None
            or target_alg._is_nilpotent_cache is False
        ):
            _timed_progress_call(
                target_alg.derived_series,
                timed=bool(_timed_reporting) if _timed_reporting is not None else False,
                threshold_s=float(_reporting_threshold_s),
                step_desc="computing the derived series",
                continue_desc=_progress_message,
                progress_message=None,
                _on_timed_update=_on_timed_update,
            )
            if getattr(target_alg, "_derived_series_terminated", None) is True:
                target_alg._is_solvable_cache = True
                target_alg._is_semisimple_cache = False
                target_alg._is_simple_cache = False
                target_alg._educed_properties["special_type"] = "solvable"
            else:
                target_alg._is_solvable_cache = False
                target_alg._is_abelian_cache = False
                target_alg._is_nilpotent_cache = False
        else:
            target_alg._is_solvable_cache = target_alg._is_nilpotent_cache
    return target_alg._is_solvable_cache


def is_abelian(
    target_alg,
    *,
    _timed_reporting: bool | None = None,
    _reporting_threshold_s: float = 10,
    _progress_message: str | None = None,
    _on_timed_update=None,
    **kwargs,
):
    if kwargs:
        dgcv_warning(
            f"`{type(target_alg).__name__}.is_abelian` received unexpected keyword "
            f"argument(s) {sorted(kwargs)}, which were ignored."
        )
    if target_alg._is_abelian_cache is None:
        if target_alg._educed_properties.get("special_type", None) == "abelian":
            target_alg._is_abelian_cache = True
            target_alg._is_nilpotent_cache = True
            target_alg._is_solvable_cache = True
            target_alg._is_semisimple_cache = False
            target_alg._is_simple_cache = False
        else:
            target_alg._is_abelian_cache = _timed_progress_call(
                lambda: all(
                    _scalar_is_zero(elem)
                    for elem in target_alg.structureDataDict.values()
                ),
                timed=bool(_timed_reporting) if _timed_reporting is not None else False,
                threshold_s=float(_reporting_threshold_s),
                step_desc="checking whether every structure constant vanishes",
                continue_desc=_progress_message,
                progress_message=None,
                _on_timed_update=_on_timed_update,
            )
            if target_alg._is_abelian_cache is True:
                target_alg._educed_properties["special_type"] = "abelian"
                target_alg._is_nilpotent_cache = True
                target_alg._is_solvable_cache = True
                target_alg._is_semisimple_cache = False
                target_alg._is_simple_cache = False
    return target_alg._is_abelian_cache


def compute_simple_subalgebras(
    target_alg,
    verbose: bool = False,
    *,
    surface_singularities=False,
    _timed_reporting: bool | None = None,
    _reporting_threshold_s: float = 10,
    _progress_message: str | None = None,
    _on_timed_update=None,
):
    timed = bool(_timed_reporting) if _timed_reporting is not None else False
    threshold = float(_reporting_threshold_s)
    target_alg.Levi_decomposition(
        decompose_semisimple_fully=True,
        verbose=verbose,
        _timed_reporting=timed,
        _reporting_threshold_s=threshold,
        _progress_message=_progress_message,
        _on_timed_update=_on_timed_update,
        surface_singularities=surface_singularities,
    )
    return target_alg._Levi_deco_cache["simple_ideals"]


def _cache_derived_algebra(target_alg, spanners, lazy):
    if lazy:
        if target_alg._derived_subalg_lazy_cache is None:
            target_alg._derived_subalg_lazy_cache = target_alg.subalgebra(
                spanners, span_warning=False, simplify_basis=True, lazy_structure=True
            )
        return target_alg._derived_subalg_lazy_cache
    if target_alg._derived_subalg_cache is None:
        lazy_cached = target_alg._derived_subalg_lazy_cache
        if lazy_cached is not None:
            lazy_cached._materialize_structure_data()
            target_alg._derived_subalg_cache = lazy_cached
        else:
            target_alg._derived_subalg_cache = target_alg.subalgebra(
                spanners, span_warning=False, simplify_basis=True
            )
    return target_alg._derived_subalg_cache


def _derived_algebra(target_alg, lazy):
    cached = (
        target_alg._derived_subalg_lazy_cache if lazy else target_alg._derived_subalg_cache
    )
    if cached is not None:
        return cached
    if not lazy and target_alg._derived_subalg_lazy_cache is not None:
        return _cache_derived_algebra(target_alg, None, False)
    if lazy and target_alg._derived_subalg_cache is not None:
        return target_alg._derived_subalg_cache
    commutators = []
    basis = target_alg.basis
    dim = len(basis)
    skew = target_alg.is_skew_symmetric()
    for j in range(dim):
        el1 = basis[j]
        lIdx = j + 1 if skew else 0
        for k in range(lIdx, dim):
            commutators.append(el1 * basis[k])
    return _cache_derived_algebra(target_alg, commutators, lazy)


def compute_derived_algebra(target_alg, lazy_structure=False):
    target_alg._set_product_protocol()

    ###!!!
    # target_alg._require_lie_algebra("compute_derived_algebra")

    return _derived_algebra(target_alg, bool(lazy_structure))


def _align_against_fixed(compare_set, old_level, new_level, discrep):
    solver = _span_solver.build(list(compare_set)) if compare_set else None
    for idx2 in range(len(old_level)):
        if discrep == 0:
            break
        elem = old_level[-1 - idx2]
        indep = solver.reduce(elem) if solver is not None else None
        if indep is None:
            indep = _indep_check(compare_set, elem)
        if indep:
            new_level.insert(0, elem)
            discrep += -1
    return new_level


def _align_extending(old_level, new_level, discrep, force_heavy_solve=False):
    solver = None
    if new_level and not force_heavy_solve:
        solver = _span_solver.build(list(new_level))
    for idx2 in range(len(old_level)):
        if discrep == 0:
            break
        elem = old_level[-1 - idx2]
        indep = None
        if solver is None and not new_level and not force_heavy_solve:
            solver = _span_solver.build([elem])
            if solver is not None:
                indep = True
        elif solver is not None:
            indep = solver.extend(elem)
        if indep is None:
            solver = None
            indep = _indep_check(
                new_level,
                elem,
                force_heavy_solve=force_heavy_solve,
            )
        if indep:
            new_level.insert(0, elem)
            discrep += -1
    return new_level


def lower_central_series(
    target_alg,
    max_depth=None,
    format_as_subalgebras=False,
    align_nested_bases=False,
    lazy_structure=False,
):
    target_alg._set_product_protocol()
    lazy = getattr(target_alg, "_lazy_structure_data", False)
    scoped_basis = list(
        target_alg.basis_in_ambient_alg if lazy else target_alg.basis
    )
    scope = target_alg.ambient if lazy else target_alg
    requested_depth = (
        max(target_alg.dimension, 1) if max_depth is None else int(max_depth)
    )
    cached_depth = getattr(target_alg, "_lower_central_series_depth", None)
    cache_usable = target_alg._lower_central_series_cache is not None and (
        getattr(target_alg, "_lower_central_series_terminated", None) is True
        or (cached_depth is not None and cached_depth >= requested_depth)
    )
    if not cache_usable:
        series = []
        current_basis = scoped_basis
        previous_length = len(current_basis)
        terminated = False

        for _ in range(requested_depth):
            series.append(current_basis)

            lower_central = []
            for el1 in current_basis:
                for el2 in scoped_basis:
                    commutator = el1 * el2
                    lower_central.append(commutator)
            independent_generators = scope.filter_independent_elements(
                lower_central, apply_light_basis_simplification=True
            )
            if len(independent_generators) == 0:
                if len(scoped_basis) > 0:
                    series.append([])
                terminated = True
                break
            if len(independent_generators) == previous_length:
                break
            current_basis = independent_generators
            previous_length = len(independent_generators)
        if len(series) > 1:
            _cache_derived_algebra(target_alg, series[1], bool(lazy_structure))
        target_alg._lower_central_series_cache = (
            series,
            False,
        )  # series, alignment bool
        target_alg._lower_central_series_terminated = terminated
        target_alg._lower_central_series_depth = requested_depth
    if (
        align_nested_bases is True
        and target_alg._lower_central_series_cache[1] is False
    ):
        if len(target_alg._lower_central_series_cache[0]) > 0 and get_dgcv_category(
            target_alg._lower_central_series_cache[0][0]
        ) in {"algebra", "subalgebra"}:
            ser = [list(alg.basis) for alg in target_alg._lower_central_series_cache[0]]
        else:
            ser = target_alg._lower_central_series_cache[0]
        new_series = [ser[-1]]
        depth = len(ser)
        for idx in range(1, depth):
            old_level = ser[depth - 1 - idx]
            discrep = len(old_level) - len(ser[depth - idx])
            new_level = _align_against_fixed(
                ser[depth - idx], old_level, list(new_series[0]), discrep
            )
            new_series.insert(0, new_level)
        target_alg._lower_central_series_cache = (
            new_series,
            True,
        )  # series, alignment bool
    if format_as_subalgebras:
        if len(target_alg._lower_central_series_cache[0]) > 0 and isinstance(
            target_alg._lower_central_series_cache[0][0], list
        ):
            target_alg._lower_central_series_cache = (
                [
                    target_alg.subalgebra(sa, span_warning=False)
                    for sa in target_alg._lower_central_series_cache[0]
                ],
                target_alg._lower_central_series_cache[1],
            )
        returnSer = target_alg._lower_central_series_cache[0]
    else:
        if len(target_alg._lower_central_series_cache[0]) > 0 and get_dgcv_category(
            target_alg._lower_central_series_cache[0][0]
        ) in {"algebra", "subalgebra"}:
            returnSer = [
                list(alg.basis) for alg in target_alg._lower_central_series_cache[0]
            ]
        else:
            returnSer = target_alg._lower_central_series_cache[0]
    return returnSer


def derived_series(
    target_alg,
    max_depth=None,
    format_as_subalgebras=False,
    align_nested_bases=False,
    surface_singularities=False,
    simplify_singularities=None,
    force_heavy_solve=False,
    lazy_structure=False,
):
    target_alg._set_product_protocol()
    lazy = getattr(target_alg, "_lazy_structure_data", False)
    scoped_basis = list(
        target_alg.basis_in_ambient_alg if lazy else target_alg.basis
    )
    scope = target_alg.ambient if lazy else target_alg
    requested_depth = (
        max(target_alg.dimension, 1) if max_depth is None else int(max_depth)
    )
    cached_depth = getattr(target_alg, "_derived_series_depth", None)
    cached_heavy = getattr(target_alg, "_derived_series_heavy", False)
    cache_usable = (
        target_alg._derived_series_cache is not None
        and (cached_heavy or not force_heavy_solve)
        and (
            getattr(target_alg, "_derived_series_terminated", None) is True
            or (cached_depth is not None and cached_depth >= requested_depth)
        )
    )
    if not cache_usable:
        series = []
        current_basis = scoped_basis
        previous_length = len(current_basis)
        total_sing = []
        terminated = False
        for _ in range(requested_depth):
            series.append(list(current_basis))

            derived = []
            level_len = len(current_basis)
            for count in range(level_len):
                el1 = current_basis[count]
                start = count + 1 if target_alg.is_skew_symmetric() else 0
                for idx2 in range(start, level_len):
                    derived.append(el1 * current_basis[idx2])
            out = scope.filter_independent_elements(
                derived,
                apply_light_basis_simplification=True,
                surface_singularities=surface_singularities,
                simplify_singularities=simplify_singularities,
                force_heavy_solve=force_heavy_solve,
            )
            if surface_singularities:
                independent_generators, sing = out
                total_sing += sing
            else:
                independent_generators = out
            if len(independent_generators) == 0:
                if len(scoped_basis) > 0:
                    series.append([])
                terminated = True
                break
            if len(independent_generators) == previous_length:
                break

            if force_heavy_solve:
                independent_generators = [
                    simplify(gen) for gen in independent_generators
                ]
            current_basis = list(independent_generators)
            previous_length = len(independent_generators)
        if surface_singularities:
            if get_dgcv_settings_registry().get(
                "simplify_singularity_ideals_by_default", True
            ):
                target_alg._singularities["derived_series"] = expr_union_primitives(
                    [v for v in total_sing if get_free_symbols(v)],
                    order_coordinates(target_alg._parameters),
                    process_rationals=True,
                    fail_quietly=True,
                )
            else:
                target_alg._singularities["derived_series"] = [
                    v for v in total_sing if get_free_symbols(v)
                ]
        if len(series) > 1:
            _cache_derived_algebra(target_alg, series[1], bool(lazy_structure))
        target_alg._derived_series_cache = (series, False)  # series, alignment bool
        target_alg._derived_series_terminated = terminated
        target_alg._derived_series_depth = requested_depth
        target_alg._derived_series_heavy = bool(force_heavy_solve)
    if align_nested_bases is True and target_alg._derived_series_cache[1] is False:
        if len(target_alg._derived_series_cache[0]) > 0 and get_dgcv_category(
            target_alg._derived_series_cache[0][0]
        ) in {"algebra", "subalgebra"}:
            ser = [list(alg.basis) for alg in target_alg._derived_series_cache[0]]
        else:
            ser = target_alg._derived_series_cache[0]
        depth = len(ser)
        new_series = [] if depth == 0 else [ser[-1]]
        build_step = 1
        if (
            len(new_series) == 1
            and len(new_series[0]) == 1
            and _scalar_is_zero(new_series[0][0])
        ):
            new_series.insert(0, ser[-2])
            build_step = 2
        for idx in range(build_step, depth):
            old_level = ser[depth - 1 - idx]
            discrep = len(old_level) - len(ser[depth - idx])
            new_level = _align_extending(
                old_level, list(new_series[0]), discrep, force_heavy_solve
            )
            new_series.insert(0, new_level)
        target_alg._derived_series_cache = (new_series, True)  # series, alignment bool
    if format_as_subalgebras:
        if len(target_alg._derived_series_cache[0]) > 0 and isinstance(
            target_alg._derived_series_cache[0][0], list
        ):
            target_alg._derived_series_cache = (
                [
                    target_alg.subalgebra(sa, span_warning=False)
                    for sa in target_alg._derived_series_cache[0]
                ],
                target_alg._derived_series_cache[1],
            )
        returnSer = target_alg._derived_series_cache[0]
    else:
        if len(target_alg._derived_series_cache[0]) > 0 and get_dgcv_category(
            target_alg._derived_series_cache[0][0]
        ) in {"algebra", "subalgebra"}:
            returnSer = [list(alg.basis) for alg in target_alg._derived_series_cache[0]]
        else:
            returnSer = target_alg._derived_series_cache[0]
    return returnSer


def radical(
    target_alg,
    assume_Lie_algebra=False,
    surface_singularities=False,
    simplify_singularities=None,
    force_heavy_solve=False,
    lazy_structure=False,
):
    if (
        target_alg._radical_cache is not None
        and force_heavy_solve
        and not getattr(target_alg, "_radical_heavy", False)
    ):
        target_alg._radical_cache = None
    if target_alg._radical_cache is None and target_alg.dimension == 0:
        target_alg._radical_cache = target_alg.subalgebra([], span_warning=False)
        target_alg._radical_heavy = True
    elif target_alg._radical_cache is None:
        da = _derived_algebra(target_alg, True)
        genElem, variables = linear_combination(
            target_alg.basis_in_ambient_alg, _disposable=True
        )
        amb = target_alg.ambient
        if amb._killing_form is None:
            amb._killing_form = freeze_matrix(
                killingForm(amb, assume_Lie_algebra=assume_Lie_algebra)
            )
        amb_dim = amb.dimension
        kf_gen = amb._killing_form * matrix_dgcv(genElem.coeff_dict, shape=(amb_dim, 1))
        eqns = [
            (matrix_dgcv(elem.coeff_dict, shape=(1, amb_dim)) * kf_gen)[0]
            for elem in da.basis_in_ambient_alg
        ]
        solve_kwargs = _solve_weight_kwargs(
            force_heavy_solve, surface_singularities, simplify_singularities
        )
        if surface_singularities:
            sol, singularities = solve_dgcv(eqns, variables, **solve_kwargs)
        else:
            sol = solve_dgcv(eqns, variables, **solve_kwargs)
        if len(sol) == 0:
            raise RuntimeError("failed to compute radical.")
        else:
            genSol = subs(genElem, sol[0])
            if surface_singularities:
                sing = [subs(v, sol[0]) for v in singularities]
                sing = [v for v in sing if get_free_symbols(v)]
                if get_dgcv_settings_registry().get(
                    "simplify_singularity_ideals_by_default", True
                ):
                    sing = expr_union_primitives(
                        sing,
                        order_coordinates(target_alg._parameters),
                        process_rationals=True,
                        fail_quietly=True,
                    )
                target_alg._singularities["radical"] = sing
        freeVars = get_free_symbols(genSol)
        if target_alg._parameters:
            freeVars = {v for v in freeVars if v not in target_alg._parameters}
        if len(freeVars) != 0:
            freeVars = sorted(freeVars, key=str)
            radSpanners = None
            linear_one_hot = engine_capability("linear_one_hot")
            if linear_one_hot is not None:
                parts = linear_one_hot(genSol.coeff_dict, freeVars)
                if parts is not None:
                    radSpanners = [
                        algebra_element_class(genSol.algebra, part, genSol.valence)
                        for part in parts
                    ]
            if radSpanners is None:
                zeroing = {v: 0 for v in freeVars}
                radSpanners = [genSol.subs({**zeroing, var: 1}) for var in freeVars]
        else:
            radSpanners = []
        if force_heavy_solve:
            radSpanners = [simplify(sp) for sp in radSpanners]
        target_alg._radical_cache = target_alg.subalgebra(
            radSpanners, span_warning=False, lazy_structure=lazy_structure
        )
        target_alg._radical_heavy = bool(force_heavy_solve)
        clearVar(*listVar(temporary_only=True), report=False)
    return target_alg._radical_cache


def center(
    target_alg,
    surface_singularities: bool = None,
    simplify_singularities: bool = None,
    format_as_subalgebra=True,
):
    if surface_singularities is None:
        surface_singularities = True if target_alg._parameters else False

    if target_alg._center_cache is None:
        dim_bound = target_alg.dimension
        ldc = target_alg._Levi_deco_cache
        if isinstance(ldc, dict):
            _, MSI = target_alg._Levi_deco_cache["LD_components"]
            if MSI.dimension == 0:
                dim_bound = 0
            else:
                spanners = [elem.ambient_rep for elem in MSI.basis]
        else:
            spanners = target_alg.basis
        if (
            dim_bound == 0
            or target_alg._is_semisimple_cache is True
            or target_alg._is_simple_cache is True
        ):
            target_alg._center_cache = target_alg.subalgebra([])
            if format_as_subalgebra:
                return target_alg._center_cache
            return target_alg._center_cache.basis

        gene, variables = linear_combination(spanners, _disposable=True)
        eqns = [gene * elem for elem in target_alg.basis]
        if not target_alg.is_skew_symmetric():
            eqns += [elem * gene for elem in target_alg.basis]
        if surface_singularities is True:
            sol, sing = solve_dgcv(
                eqns,
                variables,
                method="linsolve",
                return_divisors=True,
                pass_to_symbolic_engine=False,
                simplify_pivots=simplify_singularities
                if simplify_singularities is not None
                else True,
                simplify_result=False,
            )
            if not sol:
                raise RuntimeError("failed to compute the center.") from None
            sol = sol[0]
            if get_dgcv_settings_registry().get(
                "simplify_singularity_ideals_by_default", True
            ):
                target_alg._singularities["center"] = expr_union_primitives(
                    [v for v in sing if get_free_symbols(v)],
                    order_coordinates(target_alg._parameters),
                    process_rationals=True,
                    fail_quietly=True,
                )
            else:
                target_alg._singularities["center"] = [
                    v for v in sing if get_free_symbols(v)
                ]
        else:
            sol = solve_dgcv(eqns, variables, method="linsolve", simplify_result=False)
            if not sol:
                raise RuntimeError("failed to compute the center.") from None
            sol = sol[0]
        gsol = subs(gene, sol)
        fv = set()
        vset = set(variables)
        for v in variables:
            fv |= {x for x in get_free_symbols(sol.get(v)) if x in vset}
        if len(fv) == 0:
            target_alg._center_cache = target_alg.subalgebra([])
        else:
            fv = sorted(fv, key=str)
            zeroing = {v: 0 for v in fv}
            target_alg._center_cache = target_alg.subalgebra(
                [subs(gsol, {**zeroing, v: 1}) for v in fv]
            )
    if format_as_subalgebra:
        return target_alg._center_cache
    return target_alg._center_cache.basis


###!!! for a future update with SD property aware optimizations:
@dataclass(frozen=True, slots=True)
class _StructureDataProfile:
    """
    Stores scan of an algebra's structure-constant data.
    """

    is_real: bool
    realness_witnesses: tuple
