"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv._aux._settings


---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import Literal

from dgcv import __version__

from ...core.base import dgcv_class
from .._backends._cls_coercion import (
    attach_legacy_sympy_converters,
    attach_sympy_hook,
    detach_legacy_sympy_converters,
    detach_sympy_hook,
)
from .._backends._engine import (
    engine_kind,
    invalidate_engine_cache,
    is_sage_available,
    is_sympy_available,
)
from .._backends._notebooks import invalidate_notebook_cache, is_ipython_available
from .._backends._types_and_constants import (
    engines_seen,
    invalidate_types_and_constants_cache,
    note_engine,
)
from .._backends._updates import needs_sympy_hook
from ._config import (
    dgcv_warning,
    dgcvDeprecationWarning,
    get_dgcv_settings_registry,
    get_variable_registry,
    on_sage_kernel_inference,
    vlp,
)

__all__ = ["set_dgcv_settings", "view_dgcv_settings", "reset_dgcv_settings"]


# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------
_UNSET = object()


def set_dgcv_settings(
    theme: str | None = None,
    format_displays: bool | None = None,
    use_latex: bool | None = None,
    print_style: Literal["readable", "literal"] | None = None,
    version_specific_defaults: str | None = None,
    ask_before_overwriting_objects_in_vmf: bool | None = None,
    forgo_warnings: bool | None = None,
    default_engine: Literal["sage", "sympy", "builtin"] | None = None,
    secondary_engine: Literal["sage", "sympy", "auto", "none"] | None = _UNSET,
    verbose_label_printing: bool | None = None,
    pass_solve_requests_to_symbolic_engine: bool | None = None,
    use_rank_basis_extraction: bool | None = None,
    DEBUG: bool | None = None,
    extra_support_for_math_in_tables: bool | None = None,
    preferred_variable_format: Literal["complex", "real"] | None = None,
    use_numeric_methods: bool | None = None,
    force_rich_display: bool | None = None,
    conjugation_prefix: str | None = None,
    fallback_conjugate_prefix: str | None = None,
    simplify_singularity_ideals_by_default: str | None = None,
    forgo_CAS_provenance_pruning: bool | None = None,
    forgo_builtin_probabilistic_shortcuts: bool | None = None,
    secondary_time_budget: float | None = _UNSET,
    generic_structure_constants: bool | None = _UNSET,
    **kwargs,
):
    if kwargs.pop("quick_notebook", False):
        set_dgcv_settings(
            format_displays=True,
            use_latex=True,
            print_style="readable",
            ask_before_overwriting_objects_in_vmf=False,
            forgo_warnings=True,
            extra_support_for_math_in_tables=True,
            force_rich_display=True,
        )
        print("Applied the `quick_notebook` settings.")
    dgcvSR = get_dgcv_settings_registry()

    # deprecated_keywords
    _depr_kw = "apply_awkward_workarounds_to_fix_VSCode_display_issues"
    if _depr_kw in kwargs:
        dgcv_warning(
            "The settings keyword "
            "`apply_awkward_workarounds_to_fix_VSCode_display_issues` is deprecated.",
            dgcvDeprecationWarning,
            stacklevel=2,
            old_kw="apply_awkward_workarounds_to_fix_VSCode_display_issues",
            new_kw="extra_support_for_math_in_tables",
            sunset="2027",
        )
        if extra_support_for_math_in_tables is None:
            extra_support_for_math_in_tables = kwargs.get(_depr_kw)
    missing = [key for key in kwargs if key != _depr_kw]
    if len(missing) > 0:
        plurality_verb = (
            ["", "is", "was"] if len(missing) == 1 else ["s", "are", "were"]
        )
        dgcv_warning(
            f"The keyword{plurality_verb[0]} {missing} {plurality_verb[1]} not recognized by `set_dgcv_settings` and {plurality_verb[2]} ignored.",
            stacklevel=2,
        )

    def _norm_v(v):
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        if s[0] in ("v", "V"):
            return "v" + s[1:]
        return "v" + s

    def _resolve_vscode_patch_value(v):
        if v == "infer":
            from ._config import environment_inference

            return environment_inference()
        if isinstance(v, bool):
            return v
        if v is None:
            return None
        dgcv_warning(
            "dgcv: extra_support_for_math_in_tables should be True/False or 'infer'; "
            f"coercing {v!r} to bool.",
            stacklevel=2,
        )
        return bool(v)

    def _set_engine_symbolic(new_engine):
        note_engine(engine_kind())
        if dgcvSR.get("default_symbolic_engine") != new_engine:
            dgcvSR["default_symbolic_engine"] = new_engine
            invalidate_engine_cache()
            note_engine(new_engine)

    def _apply_keyval(k, v):
        if k == "secondary_symbolic_engine":
            if v in ("sage", "sagemath"):
                if is_sage_available():
                    dgcvSR["secondary_symbolic_engine"] = "sage"
                else:
                    dgcv_warning(
                        "dgcv: requested secondary_engine='sage' but Sage is not available; "
                        "secondary_symbolic_engine was not changed.",
                        stacklevel=2,
                    )
            elif v == "sympy":
                if is_sympy_available():
                    dgcvSR["secondary_symbolic_engine"] = "sympy"
                else:
                    dgcv_warning(
                        "dgcv: requested secondary_engine='sympy' but SymPy is not available; "
                        "secondary_symbolic_engine was not changed.",
                        stacklevel=2,
                    )
            elif v in ("none", None):
                dgcvSR["secondary_symbolic_engine"] = None
            elif v == "auto":
                dgcvSR["secondary_symbolic_engine"] = "auto"
            else:
                dgcv_warning(
                    f"dgcv: unrecognized secondary_engine value {v!r}. "
                    "Supported options are 'sympy', 'sage', 'auto', and None. secondary_symbolic_engine was not changed.",
                    stacklevel=2,
                )
            return

        if k == "default_symbolic_engine":
            if v in ("sage", "sagemath"):
                if is_sage_available():
                    _set_engine_symbolic("sage")
                else:
                    dgcv_warning(
                        "dgcv: requested default_engine='sage' but Sage is not available; "
                        "default_symbolic_engine was not changed.",
                        stacklevel=2,
                    )
            elif v in ("sympy",):
                if is_sympy_available():
                    _set_engine_symbolic("sympy")
                else:
                    dgcv_warning(
                        "dgcv: requested default_engine='sympy' but SymPy is not available; "
                        "default_symbolic_engine was not changed.",
                        stacklevel=2,
                    )
            elif v in ("builtin",):
                _set_engine_symbolic("builtin")
            else:
                dgcv_warning(
                    f"dgcv: unrecognized default_engine value {v!r}. "
                    "Supported options are 'sympy', 'sage', and 'builtin'. Default_symbolic_engine was not changed.",
                    stacklevel=2,
                )
            return

        if k == "format_displays":
            new_val = bool(v)
            old_val = bool(dgcvSR.get("format_displays", False))
            dgcvSR["format_displays"] = new_val
            if new_val != old_val:
                invalidate_notebook_cache()
            if new_val is True:
                from ..printing.printing._dgcv_display import (
                    dgcv_init_printing,
                )

                dgcv_init_printing()
            return

        if k == "conjugation_prefix":
            if len(get_variable_registry().get("complex_variable_systems", {})) > 0:
                dgcv_warning(
                    "The default `conjugation_prefix` cannot be changed while complex "
                    "coordinate systems exist in the VMF. Clear such systems from the "
                    "VMF first, e.g., using `clear_vmf`. Recommend usage: set "
                    "`conjugation_prefix` only once at the start of a session with dgcv."
                )
                return
            if v == get_dgcv_settings_registry().get("fallback_conjugate_prefix"):
                newfallback = "anti_" if v != "anti_" else "BAR"
                dgcv_warning(
                    "The requested `conjugation_prefix` is already assigned to `fallback_conjugate_prefix`. "
                    f"To free the assignement, `fallback_conjugate_prefix` has been changed to {newfallback}. "
                )
                dgcvSR["fallback_conjugate_prefix"] = newfallback
            dgcvSR[k] = v
        if k == "fallback_conjugate_prefix":
            if len(get_variable_registry().get("complex_variable_systems", {})) > 0:
                dgcv_warning(
                    "The default `fallback_conjugate_prefix` cannot be changed while complex"
                    "coordinate systems exist in the VMF. Clear such systems from the"
                    "VMF first. Recommend usage: set `fallback_conjugate_prefix` only once"
                    "at the start of a session with dgcv."
                )
                return
            if v == get_dgcv_settings_registry.get("conjugation_prefix"):
                newfallback = "BAR" if v != "BAR" else "Banti_R"
                dgcv_warning(
                    "The requested `fallback_conjugate_prefix` is already assigned to `conjugation_prefix`."
                    f"To free the assignement, `conjugation_prefix` has been changed to {newfallback}."
                )
                dgcvSR["conjugation_prefix"] = newfallback
            dgcvSR[k] = v

        if k == "extra_support_for_math_in_tables":
            dgcvSR[k] = _resolve_vscode_patch_value(v)
            return

        if k == "verbose_label_printing":
            dgcvSR["verbose_label_printing"] = v
            if dgcvSR["verbose_label_printing"] is False:
                dgcvSR["VLP"] = vlp
            return

        dgcvSR[k] = v

    current_vsd = _norm_v(f"v{__version__}")

    requested_vsd = (
        _norm_v(version_specific_defaults)
        if version_specific_defaults is not None
        else None
    )

    if requested_vsd is not None:
        if requested_vsd != current_vsd:
            from dgcv._aux._backends._updates import defaults_for_version

            version_defaults = defaults_for_version(
                requested_vsd,
                current_version=__version__,
                vlp=vlp,
            )

            for k, v in version_defaults.items():
                if k == "version_specific_defaults":
                    continue
                _apply_keyval(k, v)

        dgcvSR["version_specific_defaults"] = requested_vsd

    effective_vsd = (
        requested_vsd
        if requested_vsd is not None
        else dgcvSR.get("version_specific_defaults")
    )

    if requested_vsd is not None and needs_sympy_hook(requested_vsd):
        attach_sympy_hook(dgcv_class)
    else:
        detach_sympy_hook(dgcv_class)

    if effective_vsd is not None and needs_sympy_hook(effective_vsd):
        attach_legacy_sympy_converters()
    else:
        detach_legacy_sympy_converters()

    if theme is not None:
        _apply_keyval("theme", theme)

    if use_latex is not None:
        _apply_keyval("use_latex", use_latex)

    if ask_before_overwriting_objects_in_vmf is not None:
        _apply_keyval(
            "ask_before_overwriting_objects_in_vmf",
            ask_before_overwriting_objects_in_vmf,
        )
    if simplify_singularity_ideals_by_default is not None:
        _apply_keyval(
            "simplify_singularity_ideals_by_default",
            simplify_singularity_ideals_by_default is True,
        )

    if forgo_warnings is not None:
        _apply_keyval("forgo_warnings", forgo_warnings)

    if default_engine is not None:
        engine = str(default_engine).lower()
        if engine in ("sage", "sagemath"):
            _apply_keyval("default_symbolic_engine", "sage")
        elif engine in ("sympy",):
            _apply_keyval("default_symbolic_engine", "sympy")
        elif engine in ("builtin",):
            _apply_keyval("default_symbolic_engine", "builtin")
        else:
            _apply_keyval("default_symbolic_engine", engine)

    if secondary_engine is not _UNSET:
        _apply_keyval(
            "secondary_symbolic_engine",
            None if secondary_engine is None else str(secondary_engine).lower(),
        )

    if format_displays is not None:
        _apply_keyval("format_displays", format_displays)

    if print_style is not None:
        if print_style in {"readable", "literal"}:
            _apply_keyval("print_style", print_style)

    if force_rich_display is not None:
        _apply_keyval("force_rich_display", force_rich_display)

        if force_rich_display:
            try:
                if not is_ipython_available():
                    dgcv_warning(
                        "force_rich_display=True was requested, but IPython does not "
                        "appear to be available. Some outputs may render as raw "
                        "HTML or unformatted objects.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            except Exception:
                dgcv_warning(
                    "force_rich_display=True was requested, but display environment "
                    "could not be verified. Some outputs may render as raw HTML "
                    "or unformatted objects.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    if preferred_variable_format is not None:
        if preferred_variable_format in {"real", "complex"}:
            _apply_keyval("preferred_variable_format", preferred_variable_format)

    if use_numeric_methods is not None:
        _apply_keyval("use_numeric_methods", use_numeric_methods)

    if conjugation_prefix is not None:
        if not isinstance(conjugation_prefix, str):
            dgcv_warning(
                "In dgcv settings, `conjugation_prefix` can only be set to a string value."
            )
        else:
            _apply_keyval("conjugation_prefix", conjugation_prefix)

    if fallback_conjugate_prefix is not None:
        if not isinstance(fallback_conjugate_prefix, str):
            dgcv_warning(
                "In dgcv settings, `fallback_conjugate_prefix` can only be set to a string value."
            )
        else:
            _apply_keyval("fallback_conjugate_prefix", fallback_conjugate_prefix)

    if extra_support_for_math_in_tables is not None:
        _apply_keyval(
            "extra_support_for_math_in_tables",
            extra_support_for_math_in_tables,
        )

    if verbose_label_printing is not None:
        _apply_keyval("verbose_label_printing", verbose_label_printing)

    if pass_solve_requests_to_symbolic_engine is not None:
        _apply_keyval(
            "pass_solve_requests_to_symbolic_engine",
            pass_solve_requests_to_symbolic_engine,
        )

    if use_rank_basis_extraction is not None:
        _apply_keyval("use_rank_basis_extraction", use_rank_basis_extraction)

    if DEBUG is not None:
        _apply_keyval("DEBUG", DEBUG)

    if forgo_CAS_provenance_pruning is not None:
        _apply_keyval(
            "forgo_CAS_provenance_pruning", bool(forgo_CAS_provenance_pruning)
        )
        invalidate_types_and_constants_cache()

    if forgo_builtin_probabilistic_shortcuts is not None:
        _apply_keyval(
            "forgo_builtin_probabilistic_shortcuts",
            bool(forgo_builtin_probabilistic_shortcuts),
        )

    if secondary_time_budget is not _UNSET:
        budget = None if secondary_time_budget is None else float(secondary_time_budget)
        _apply_keyval(
            "secondary_time_budget", budget if budget and budget > 0 else None
        )

    if generic_structure_constants is not _UNSET:
        _apply_keyval(
            "generic_structure_constants",
            None
            if generic_structure_constants is None
            else bool(generic_structure_constants),
        )

    _provenance_guard(dgcvSR)


_provenance_warned = set()


def _provenance_guard(dgcvSR):
    if dgcvSR.get("forgo_CAS_provenance_pruning", False):
        return
    kind = engine_kind()
    if kind == "sympy" and on_sage_kernel_inference():
        key = "sympy_on_sage_kernel"
        message = (
            "dgcv is running the sympy engine on a Sage kernel. Sage's preparser turns "
            "literals into Sage objects, which dgcv converts to sympy on entry (CAS "
            "provenance pruning). Set `default_engine='sage'` unless this notebook needs "
            "sympy, or call `set_dgcv_settings(forgo_CAS_provenance_pruning=True)` to switch "
            "the conversion off."
        )
    elif kind == "sage" and "sympy" in engines_seen():
        key = "sage_after_sympy"
        message = (
            "dgcv switched from the sympy engine to Sage in this session. sympy expressions "
            "created earlier are converted to Sage on entry (CAS provenance pruning). Call "
            "`set_dgcv_settings(forgo_CAS_provenance_pruning=True)` to switch the conversion "
            "off."
        )
    elif kind == "builtin" and (engines_seen() & {"sympy", "sage"}):
        key = "builtin_after_cas"
        message = (
            "dgcv switched to its builtin symbolic engine after a CAS engine was active in "
            "this session. sympy or Sage expressions are converted to builtin zero forms on "
            "entry (rational expressions only), but objects already registered in the "
            "variable management framework are not migrated; call `clear_vmf()` and "
            "recreate variables, or `set_dgcv_settings(forgo_CAS_provenance_pruning=True)` "
            "to switch the conversion off."
        )
    else:
        return
    if key in _provenance_warned:
        return
    _provenance_warned.add(key)
    dgcv_warning(message, stacklevel=3)


def _toggle_or_set_verbosity(setting=None):
    dgcvSR = get_dgcv_settings_registry()
    if "__" not in dgcvSR:
        dgcvSR["__"] = dict()
    if (setting is None and "verbose" in dgcvSR["__"]) or setting == 0:
        del dgcvSR["__"]["verbose"]
    else:
        dgcvSR["__"]["verbose"] = setting if setting is not None else 1


def view_dgcv_settings(verbose=False):
    settings = get_dgcv_settings_registry()
    if not settings:
        print("dgcv settings registry is empty.")
        return
    hidden = {"VLP", "__"}
    if verbose is not True:
        hidden = (
            hidden
            | {
                "numeric_error_thresholds",
                "default_numeric_engine",
                "DEBUG",
                "fallback_conjugate_prefix",
                "_solve_default",
            }
            | {key for key in settings.keys() if str(key).startswith("_")}
        )
    if not settings.get("forgo_CAS_provenance_pruning", False):
        hidden = hidden | {"forgo_CAS_provenance_pruning"}
    if not settings.get("forgo_builtin_probabilistic_shortcuts", False):
        hidden = hidden | {"forgo_builtin_probabilistic_shortcuts"}
    if settings.get("secondary_time_budget") is None:
        hidden = hidden | {"secondary_time_budget"}
    if settings.get("generic_structure_constants") is None:
        hidden = hidden | {"generic_structure_constants"}
    if settings.get("secondary_symbolic_engine") is None:
        hidden = hidden | {"secondary_symbolic_engine"}
    items = [(k, v) for k, v in settings.items() if k not in hidden]
    if settings.get("use_numeric_methods", False):
        items.append(("default_numeric_engine", settings.get("default_numeric_engine")))
        items.append(
            (
                "numeric_error_thresholds.abs_tolerance",
                settings.get("numeric_error_thresholds", {}).get("abs_tolerance"),
            )
        )
        items.append(
            (
                "numeric_error_thresholds.rel_tolerance",
                settings.get("numeric_error_thresholds", {}).get("rel_tolerance"),
            )
        )
        items.append(
            (
                "numeric_error_thresholds.policy",
                settings.get("numeric_error_thresholds", {}).get("policy"),
            )
        )
    if settings.get("fallback_conjugate_prefix", False):
        if settings.get("fallback_conjugate_prefix") != "anti_":
            items.append(
                ("fallback_conjugate_prefix", settings.get("fallback_conjugate_prefix"))
            )
    if not items:
        print("dgcv settings registry is empty.")
        return

    items = sorted(items, key=lambda x: x[0])

    width = max(len(k) for k, _ in items)

    print("\ndgcv settings")
    print("-" * (width + 20))
    for k, v in items:
        print(f"{k:<{width}} : {v!r}")
    print("-" * (width + 20))


def reset_dgcv_settings():
    """
    Reset dgcv settings to their default values.
    """
    from ._config import _vscodepatch, default_engine_inference

    dgcvSR = get_dgcv_settings_registry()

    dgcvSR.clear()
    dgcvSR.update(
        {
            "use_latex": True,
            "theme": "paper_graphite",
            "format_displays": True,
            "version_specific_defaults": f"v{__version__}",
            "ask_before_overwriting_objects_in_vmf": True,
            "forgo_warnings": False,
            "default_symbolic_engine": default_engine_inference(),
            "secondary_symbolic_engine": "auto",
            "secondary_time_budget": None,
            "generic_structure_constants": None,
            "verbose_label_printing": False,
            "VLP": vlp,
            "conjugation_prefix": "BAR",
            "fallback_conjugate_prefix": "anti_",
            "preferred_variable_format": "complex",
            "pass_solve_requests_to_symbolic_engine": True,
            "extra_support_for_math_in_tables": _vscodepatch,
            "use_numeric_methods": False,
            "use_rank_basis_extraction": True,
            "default_numeric_engine": "numpy",
            "numeric_error_thresholds": {
                "abs_tol": 1e-9,
                "rel_tol": 1e-9,
                "policy": "balanced",  # balanced, reckless,
            },
            "DEBUG": False,
            "print_style": "readable",
            "force_rich_display": False,
            "simplify_singularity_ideals_by_default": True,
            "__": dict(),
        }
    )

    invalidate_engine_cache()
    invalidate_notebook_cache()
