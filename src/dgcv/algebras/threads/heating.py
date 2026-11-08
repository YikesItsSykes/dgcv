from __future__ import annotations

from ..._aux._backends._polynomials import (
    expr_union_primitives,
)
from ..._aux._backends._symbolic_router import (
    get_free_symbols,
)
from ..._aux._utilities._config import (
    get_dgcv_settings_registry,
)
from ..._aux._utilities._progress import _timed_progress_call
from ..._aux._vmf.vmf import order_coordinates
from .algebra_classifications import (
    complex_algebra_label,
    complex_type_candidates,
    real_form_candidates,
)


def _summary_warm_caches(
    refAlg,
    *,
    subAlg: bool,
    reporting_threshold_s: float = 10.0,
    progress_message: str | None = None,
    full=False,
    force_heavy_solve: bool = False,
    _on_timed_update=None,
):
    thr = float(reporting_threshold_s)
    heavy = bool(force_heavy_solve)

    def _timed_kwargs(continue_desc):
        return {
            "_timed_reporting": True,
            "_reporting_threshold_s": thr,
            "_progress_message": continue_desc,
            "_on_timed_update": _on_timed_update,
        }

    def _timed_step(fn, step_desc, continue_desc):
        try:
            return _timed_progress_call(
                fn,
                timed=True,
                threshold_s=thr,
                step_desc=step_desc,
                continue_desc=continue_desc,
                progress_message=None,
                _on_timed_update=_on_timed_update,
            )
        except Exception:
            return None

    def _warm_ideal_ranks():
        ld = getattr(refAlg, "_Levi_deco_cache", None)
        simples = ld.get("simple_ideals", None) if isinstance(ld, dict) else None
        if not simples:
            return
        surfacing = bool(getattr(refAlg, "_parameters", None))
        total = len(simples)
        for idx, ideal in enumerate(simples, start=1):
            if full and getattr(ideal, "base_field", "complex") == "real":
                _timed_step(
                    lambda a=ideal: a.killing_inertia(
                        signature_only=True, assume_simple=True
                    ),
                    f"computing the Killing signature of simple ideal {idx} of {total}",
                    progress_message,
                )
            if getattr(ideal, "_rank_approximation", None) is None:
                out = _timed_step(
                    lambda a=ideal: a.approximate_rank(
                        _use_cache=True,
                        assume_semisimple=True,
                        surface_singularities=surfacing,
                    ),
                    f"estimating the rank of simple ideal {idx} of {total}",
                    progress_message
                    if idx == total
                    else "finish estimating the ranks of the simple ideals",
                )
                if surfacing and isinstance(out, tuple) and len(out) == 2:
                    try:
                        _merge_rank_singularities(ideal, refAlg, out[1])
                    except Exception:
                        pass
            if full and _ideal_label_is_ambiguous(ideal):
                _timed_step(
                    lambda a=ideal: a.root_length_profile(
                        assume_simple=True, assume_Lie_algebra=True
                    ),
                    f"measuring the root lengths of simple ideal {idx} of {total}",
                    progress_message,
                )

    is_lie = refAlg.is_Lie_algebra(
        verbose=False,
        **_timed_kwargs(progress_message),
    )

    if not is_lie:
        return

    recovered = False
    try:
        refAlg.Levi_decomposition(
            decompose_semisimple_fully=full,
            verbose=False,
            force_heavy_solve=heavy,
            **_timed_kwargs(progress_message),
        )
    except Exception:
        if not heavy and refAlg._parameters:
            print(
                "A decomposition subroutine failed. Retrying with a heavier "
                "solve algorithm."
            )
            try:
                refAlg.Levi_decomposition(
                    decompose_semisimple_fully=full,
                    verbose=False,
                    force_heavy_solve=True,
                    _bust_cache=True,
                    **_timed_kwargs(progress_message),
                )
                recovered = True
                heavy = True
            except Exception:
                recovered = False
        if not recovered:
            if subAlg and not refAlg._parameters:
                print(
                    "A decomposition subroutine failed; proceeding with a partial report."
                    "Currently, summary is not fully tested for subalgebras, and that may be the reason."
                    "Suggestion: convert to an algebra_class via the subalgebra copy method."
                )
            else:
                addon = (
                    ", likely due to a presence of parameters in the algebra structure equations which is not fully tested across the algebra_class methods"
                    if refAlg._parameters
                    else ""
                )
                print(
                    f"A decomposition subroutine failed{addon}; proceeding with a partial report."
                )

    rad = None
    try:
        ld = getattr(refAlg, "_Levi_deco_cache", None)
        comps = ld.get("LD_components", None) if isinstance(ld, dict) else None
        if isinstance(comps, (list, tuple)) and len(comps) > 1:
            rad = comps[1]
    except Exception:
        rad = None

    if rad is None:
        try:
            rad = getattr(refAlg, "_radical_cache", None)
        except Exception:
            rad = None

    if rad is not None and getattr(rad, "dimension", 0) != 0:
        _timed_step(
            lambda: rad.derived_series(force_heavy_solve=heavy),
            "computing the maximal solvable ideal's derived series",
            "compute the maximal solvable ideal's lower central series",
        )
        _timed_step(
            lambda: rad.lower_central_series(),
            "computing the maximal solvable ideal's lower central series",
            "compute the center" if full else progress_message,
        )
        if (
            getattr(rad, "dimension", None) == getattr(refAlg, "dimension", None)
            and refAlg._is_nilpotent_cache is None
        ):
            terminated = getattr(rad, "_lower_central_series_terminated", None)
            if terminated is True:
                refAlg._is_nilpotent_cache = True
                refAlg._educed_properties["special_type"] = "nilpotent"
                refAlg._is_semisimple_cache = False
                refAlg._is_simple_cache = False
            elif terminated is False:
                refAlg._is_nilpotent_cache = False
                refAlg._is_abelian_cache = False

    if full:
        _timed_step(
            lambda: refAlg.center(),
            "computing the center",
            progress_message,
        )

    try:
        abelian = refAlg.is_abelian(**_timed_kwargs(progress_message))
    except Exception:
        abelian = None

    if not abelian:
        try:
            is_ss = refAlg.is_semisimple(
                verbose=False,
                **_timed_kwargs(progress_message),
            )
        except Exception:
            is_ss = False

        if is_ss:
            try:
                refAlg.is_simple(
                    verbose=False,
                    **_timed_kwargs(progress_message),
                )
            except Exception:
                pass
        else:
            try:
                is_sol = refAlg.is_solvable(**_timed_kwargs(progress_message))
            except Exception:
                is_sol = False

            if is_sol:
                try:
                    refAlg.is_nilpotent(**_timed_kwargs(progress_message))
                except Exception:
                    pass

    _warm_ideal_ranks()



def _merge_rank_singularities(alg, refAlg, divisors):
    terms = [v for v in divisors if get_free_symbols(v)]
    if not terms:
        return
    hosts = (alg,) if alg is refAlg else (alg, refAlg)
    for host in hosts:
        merged = list(host._singularities.get("subalgebra_ranks", [])) + terms
        if get_dgcv_settings_registry().get(
            "simplify_singularity_ideals_by_default", True
        ):
            merged = expr_union_primitives(
                merged,
                order_coordinates(host._parameters),
                process_rationals=True,
                fail_quietly=True,
            )
        host._singularities["subalgebra_ranks"] = merged


def _ideal_label_is_ambiguous(alg):
    try:
        dim = int(getattr(alg, "dimension", None))
        rank = int(getattr(alg, "_rank_approximation", None))
    except (TypeError, ValueError):
        return False
    signature = _ideal_signature(alg)
    if signature is not None and getattr(alg, "base_field", "complex") != "complex":
        _, certain = real_form_candidates(
            dim,
            rank,
            signature,
            {"real": True, "complex": False}.get(
                getattr(alg, "_centroid_type", None), None
            ),
        )
        return not certain
    return len(complex_type_candidates(dim, rank)) != 1


def _ideal_signature(alg):
    if getattr(alg, "base_field", "complex") != "real":
        return None
    if getattr(alg, "_killing_form", None) is None:
        return None
    try:
        return alg.killing_inertia(signature_only=True, assume_simple=True)
    except Exception:
        return None


def _ideal_iso_label(alg, *, use_latex, rank=None, refAlg=None, return_scope=False):
    if rank is None:
        rank = getattr(alg, "_rank_approximation", None)
    root_profile = getattr(alg, "_root_length_profile_cache", None)
    c_type = None if root_profile is None else root_profile.complex_type
    return _simple_iso_label(
        getattr(alg, "dimension", None),
        rank,
        use_latex=use_latex,
        signature=_ideal_signature(alg),
        absolutely_simple={"real": True, "complex": False}.get(
            getattr(alg, "_centroid_type", None), None
        ),
        base_field=getattr(alg, "base_field", "complex"),
        complex_type=c_type,
        force_complexification=_ideal_over_complexification(alg, refAlg),
        return_scope=return_scope,
    )


def _simple_iso_label(
    dim,
    rank,
    *,
    use_latex: bool,
    signature=None,
    absolutely_simple=None,
    base_field=None,
    complex_type=None,
    force_complexification=False,
    return_scope=False,
):
    """
    Isomorphism-class label for a simple Lie algebra.

    Parameters
    ----------
    dim : int
    rank : int
    use_latex : bool
    signature : int, optional
        Killing signature. When supplied, real-form names are returned in
        place of a name for the complexification.
    absolutely_simple : bool, optional
        Centroid verdict, filtering realifications in or out.
    base_field : str, optional
        `'real'` allows real-form names, `'complex'` names the complex
        algebra itself, and `None` falls back to naming the complexification.
    complex_type : str, optional
        Measured Dynkin type of the complexification, used to resolve
        alternatives that the other invariants cannot separate. Ignored
        when it matches nothing.
    force_complexification : bool, default False
        Suppress real-form names, for a summand that was only ever
        identified inside the complexification.
    return_scope : bool, default False
        Also report what the label names.

    Returns
    -------
    str or None
        Alternatives are joined by ' or '. None when nothing matches. With
        `return_scope`, a pair `(label, scope)` instead, where `scope` is
        `'algebra'` when the label names the algebra itself and
        `'complexification'` when it names the complexification, and is
        `None` whenever the label is.
    """
    label = None
    scope = None

    if (
        force_complexification is False
        and signature is not None
        and base_field != "complex"
    ):
        records, _ = real_form_candidates(dim, rank, signature, absolutely_simple)
        if complex_type is not None:
            narrowed = tuple(r for r in records if r.complex_type == complex_type)
            if narrowed:
                records = narrowed
        if records:
            if use_latex:
                label = " or ".join(_real_form_latex(r.label) for r in records)
            else:
                label = " or ".join(r.label for r in records)
            scope = "algebra"

    if label is None:
        typ = _classify_simple_by_dim_rank(dim, rank)
        if typ is not None:
            tags = typ.split(" or ")
            if complex_type in tags:
                tags = [complex_type]
            if use_latex:
                label = " or ".join(_complex_type_latex(t) for t in tags)
            else:
                label = " or ".join(complex_algebra_label(t) or t for t in tags)
            scope = "algebra" if base_field == "complex" else "complexification"

    return (label, scope) if return_scope else label


def _classify_simple_by_dim_rank(dim, rank):
    """
    Return a Dynkin-type tag like 'A1', 'D4', 'G2', or 'B3 or C3', else None.
    """
    try:
        d = int(dim)
        r = int(rank)
    except Exception:
        return None

    tags = complex_type_candidates(d, r)
    if not tags:
        return None
    return " or ".join(tags)


def _real_form_latex(label):
    head = _latex_exceptional_alg_prefix.get(label[:2], None)
    if head is not None and label[2:3] == "(":
        return f"${head}{label[2:]}}}$"
    for prefix, tex in _latex_alg_labels:
        if label.startswith(prefix):
            tail = label[len(prefix) :]
            tail = tail.replace(",R)", r",\mathbb{R})").replace(",C)", r",\mathbb{C})")
            if tail.endswith("_R"):
                tail = tail[:-2] + r"_{\mathbb{R}}"
            return f"${tex}{tail}$"
    return f"${label}$"


def _complex_type_latex(tag):
    head = _latex_exceptional_alg_prefix.get(tag, None)
    if head is not None:
        return f"${head}}}(\\mathbb{{C}})$"
    family = tag[:1]
    try:
        r = int(tag[1:])
    except ValueError:
        return rf"${tag}$"
    if family == "A":
        name, sub = "sl", r + 1
    elif family == "B":
        name, sub = "so", 2 * r + 1
    elif family == "C":
        name, sub = "sp", 2 * r
    else:
        name, sub = "so", 2 * r
    return rf"$\mathfrak{{{name}}}_{{{sub}}}(\mathbb{{C}})$"


_latex_alg_labels = (
    ("su*", r"\mathfrak{su}^{*}"),
    ("so*", r"\mathfrak{so}^{*}"),
    ("su", r"\mathfrak{su}"),
    ("sl", r"\mathfrak{sl}"),
    ("so", r"\mathfrak{so}"),
    ("sp", r"\mathfrak{sp}"),
)


_latex_exceptional_alg_prefix = {
    "G2": r"\mathfrak{g}_{2",
    "F4": r"\mathfrak{f}_{4",
    "E6": r"\mathfrak{e}_{6",
    "E7": r"\mathfrak{e}_{7",
    "E8": r"\mathfrak{e}_{8",
}


def _ideal_over_complexification(alg, refAlg=None):
    if getattr(alg, "base_field", "complex") != "real":
        return False
    params = getattr(alg, "_parameters", None)
    if not params and refAlg is not None:
        params = getattr(refAlg, "_parameters", None)
    if not params:
        return False
    return getattr(alg, "_centroid_type", None) is None
