"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv._aux._config


Description: This module provides utility functions for dgcv's Variable Management Framework, maintaining
a registry of instantiated mathematical object for interaction with dgcv functions.

Functions
---------
- get_variable_registry: Returns the current state of the `variable_registry`,
  which holds information about objects tracked in the VMF.
- clear_variable_registry: Resets the `variable_registry` to its initial state.
- get_dgcv_settings_registry: Returns the current state of the dictionary storing setting default affecting dgcv functions.

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

import base64
import collections.abc
import mimetypes
import re
import sys
import warnings
from functools import lru_cache
from importlib import resources
from typing import List, Literal, Optional

from ... import __version__

_working_namespace_ = None

dgcv_categories = {
    "vector_field",
    "tensor_field",
    "differential_form",
    "algebra",
    "algebra_element",
    "algebra_subspace",
    "subalgebra",
    "subalgebra_element",
}

greek_letters = {
    "alpha": "\\alpha",
    "beta": "\\beta",
    "gamma": "\\gamma",
    "delta": "\\delta",
    "epsilon": "\\epsilon",
    "varepsilon": "\\varepsilon",
    "zeta": "\\zeta",
    "eta": "\\eta",
    "theta": "\\theta",
    "vartheta": "\\vartheta",
    "iota": "\\iota",
    "kappa": "\\kappa",
    "lam": "\\lambda",
    "lambda": "\\lambda",
    "mu": "\\mu",
    "nu": "\\nu",
    "xi": "\\xi",
    "pi": "\\pi",
    "varpi": "\\varpi",
    "rho": "\\rho",
    "varrho": "\\varrho",
    "sigma": "\\sigma",
    "varsigma": "\\varsigma",
    "tau": "\\tau",
    "upsilon": "\\upsilon",
    "phi": "\\phi",
    "varphi": "\\varphi",
    "chi": "\\chi",
    "psi": "\\psi",
    "omega": "\\omega",
    "Gamma": "\\Gamma",
    "Delta": "\\Delta",
    "Theta": "\\Theta",
    "Lam": "\\Lambda",
    "Lambda": "\\Lambda",
    "Xi": "\\Xi",
    "Pi": "\\Pi",
    "Sigma": "\\Sigma",
    "Upsilon": "\\Upsilon",
    "Phi": "\\Phi",
    "Psi": "\\Psi",
    "Omega": "\\Omega",
    "ell": "\\ell",
    "hbar": "\\hbar",
}

__all__ = [
    "get_variable_registry",
    "get_dgcv_settings_registry",
    "configure_convenient_labels",
]


# -----------------------------------------------------------------------------
# VMF
# -----------------------------------------------------------------------------
def working_namespace():
    """
    Fetch namespace dict of the active session
    """
    global _working_namespace_
    if _working_namespace_ is None:
        _working_namespace_ = sys.modules["__main__"].__dict__
    return _working_namespace_


def update_working_namespace(update_dict):
    namespace = working_namespace()
    namespace.update(update_dict)


def update_working_namespace_k_v(key, value):
    namespace = working_namespace()
    namespace[key] = value


class StringifiedSymbolsDict(collections.abc.MutableMapping):
    """
    A lightweight dictionary that stores keys as their string representations.
    When setting or getting an item with a key, it is converted to its string form.
    """

    def __init__(self, initial_data=None):
        self._data = {}
        if initial_data:
            self.update(initial_data)

    def _convert_key(self, key):
        return key if isinstance(key, str) else str(key)

    def __getitem__(self, key):
        return self._data[self._convert_key(key)]

    def __setitem__(self, key, value):
        self._data[self._convert_key(key)] = value

    def __delitem__(self, key):
        del self._data[self._convert_key(key)]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def copy(self):
        new_copy = StringifiedSymbolsDict()
        new_copy._data = self._data.copy()
        return new_copy

    def __repr__(self):
        return f"StringifiedSymbolsDict({self._data})"


def _fresh_variable_registry():
    return {
        "standard_variable_systems": {},
        "complex_variable_systems": {},
        "finite_algebra_systems": {},
        "misc": {},
        "eds": {"atoms": {}, "coframes": {}},
        "protected_variables": set(),
        "temporary_variables": set(),
        "obscure_variables": set(),
        "conversion_dictionaries": {
            "holToReal": StringifiedSymbolsDict(),
            "realToSym": StringifiedSymbolsDict(),
            "symToHol": StringifiedSymbolsDict(),
            "symToReal": StringifiedSymbolsDict(),
            "realToHol": StringifiedSymbolsDict(),
            "conjugation": StringifiedSymbolsDict(),
            "find_parents": StringifiedSymbolsDict(),
            "real_part": StringifiedSymbolsDict(),
            "im_part": StringifiedSymbolsDict(),
        },
        "dgcv_enforced_real_atoms": dict(),
        "_labels": {},
        "paths": {},
    }


variable_registry = _fresh_variable_registry()
vlp = re.compile(
    r"""
    ^(?:\\left\((?P<content>.*)\\right\))?   
    _\{\\operatorname\{v\.\}(?P<j>\d+)\}$    
    """,
    re.VERBOSE,
)
_vscodepatch = None


def environment_inference():
    global _vscodepatch
    if _vscodepatch is None:
        try:
            import os

            try:
                from .._backends._notebooks import in_kernel

                in_jupyter = in_kernel()
            except Exception:
                from IPython import get_ipython  # type: ignore

                ip = get_ipython()
                in_jupyter = (
                    ip is not None and "ZMQInteractiveShell" in ip.__class__.__name__
                )

            in_vscode = (
                "VSCODE_PID" in os.environ
                or "VSCODE_IPC_HOOK" in os.environ
                or os.environ.get("TERM_PROGRAM") == "vscode"
            )

            _vscodepatch = bool(in_jupyter and in_vscode)
        except Exception:
            _vscodepatch = False

    return _vscodepatch


_sage_kernel_default = None


def on_sage_kernel_inference():
    global _sage_kernel_default
    if _sage_kernel_default is None:
        try:
            from .._backends._notebooks import on_sage_kernel

            _sage_kernel_default = bool(on_sage_kernel())
        except Exception:
            _sage_kernel_default = False
    return _sage_kernel_default


def legacy_engine_inference():
    return "sage" if on_sage_kernel_inference() else "sympy"


def default_engine_inference():
    return "builtin"


dgcv_settings_registry = {
    "use_latex": True,
    "theme": "paper_graphite",
    "format_displays": True,
    "version_specific_defaults": f"v{__version__}",
    "ask_before_overwriting_objects_in_vmf": True,
    "forgo_warnings": False,
    "default_symbolic_engine": default_engine_inference(),
    "secondary_symbolic_engine": "auto",
    "verbose_label_printing": False,
    "print_style": "readable",
    "VLP": vlp,
    "conjugation_prefix": "BAR",
    "compile_latex_conjugation": True,
    "preferred_variable_format": "complex",
    "pass_solve_requests_to_symbolic_engine": True,
    "use_rank_basis_extraction": True,
    "extra_support_for_math_in_tables": environment_inference(),
    "use_numeric_methods": False,
    "default_numeric_engine": "numpy",
    "numeric_error_thresholds": {
        "abs_tolerance": 1e-9,
        "rel_tolerance": 1e-9,
        "policy": "balanced",  # balanced, reckless,
    },
    "DEBUG": False,
    "force_rich_display": False,
    "simplify_singularity_ideals_by_default": True,
    "forgo_CAS_provenance_pruning": False,
    "forgo_builtin_probabilistic_shortcuts": False,
    "secondary_time_budget": None,
    "generic_structure_constants": None,
    "_solve_default": "auto",
    "__": dict(),
}


def get_variable_registry():
    return variable_registry


def get_dgcv_settings_registry():
    return dgcv_settings_registry


def clear_variable_registry():
    global variable_registry
    variable_registry = _fresh_variable_registry()


def canonicalize(obj, with_simplify=False, depth=1000):
    if hasattr(obj, "_eval_canonicalize"):
        obj = obj._eval_canonicalize(depth=depth)
    if with_simplify is True:
        return obj._eval_simplify() if hasattr(obj, "_eval_simplify") else obj
    else:
        return obj


class dgcv_exception_note(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


def _try_wrap_html(s: str):
    """
    Return an IPython.display.HTML object if IPython is available; otherwise
    return the string unchanged.
    """
    try:
        from .._backends._notebooks import is_ipython_available
    except Exception:
        is_ipython_available = None

    if callable(is_ipython_available):
        try:
            if not is_ipython_available():
                return s
        except Exception:
            return s

    try:
        from IPython.display import HTML  # type: ignore # local import by design

        return HTML(s)
    except Exception:
        return s


def latex_in_html(
    html_string,
    extra_support_for_math_in_tables=None,
    container_id: str | None = None,
    katex_selector: str | None = None,
):
    if extra_support_for_math_in_tables is None:
        extra_support_for_math_in_tables = (
            dgcv_settings_registry.get("extra_support_for_math_in_tables", False)
            is True
        )

    if hasattr(html_string, "to_html"):
        try:
            body = html_string.to_html(escape=False)
        except Exception:
            body = str(html_string)
    else:
        body = str(html_string)

    if extra_support_for_math_in_tables is True:
        if katex_selector:
            if container_id:
                selector = f"#{container_id} {katex_selector}"
            else:
                selector = katex_selector
            render_target = f"document.querySelector({selector!r})"
        elif container_id:
            render_target = f'document.getElementById("{container_id}")'
        else:
            render_target = "document.body"
        # This is the display workaround given in the github pages for the Jupyter VSCode extension.
        katex_inject_string = f"""<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.45/dist/katex.min.css" integrity="sha384-UA8juhPf75SzzAMA/4fo3yOU7sBJ0om7SCD2GHq0fZqZco6tr1UCV7nUbk9J90JM" crossorigin="anonymous">
<script type="module">
    import renderMathInElement from "https://cdn.jsdelivr.net/npm/katex@0.16.45/dist/contrib/auto-render.mjs";
    const renderRoot = {render_target};
    if (renderRoot) {{
        renderMathInElement(renderRoot, {{
            delimiters: [
                {{left: "$$", right: "$$", display: true}},
                {{left: "$", right: "$", display: false}}
            ]
        }});
    }}
</script>"""
        return _try_wrap_html(katex_inject_string + body)

    if isinstance(html_string, str):
        return _try_wrap_html(html_string)
    return html_string


def latex_in_html_offline(
    html_string,
    extra_support_for_math_in_tables=None,
    container_id: str | None = None,
    katex_selector: str | None = None,
):
    if extra_support_for_math_in_tables is None:
        extra_support_for_math_in_tables = (
            dgcv_settings_registry.get("extra_support_for_math_in_tables", False)
            is True
        )

    body = _latex_in_html_body(html_string)

    if not extra_support_for_math_in_tables:
        if isinstance(html_string, str):
            return _try_wrap_html(html_string)
        return html_string

    render_target = _latex_in_html_render_target(
        container_id=container_id,
        katex_selector=katex_selector,
    )
    katex_css = _latex_in_html_katex_css()
    katex_module, auto_render_module = _latex_in_html_katex_modules()

    injected = f"""<style>{katex_css}</style>
<script type="module">
    const katexModuleSource = {katex_module!r};
    const autoRenderModuleSource = {auto_render_module!r};

    const katexUrl = URL.createObjectURL(
        new Blob([katexModuleSource], {{ type: "text/javascript" }})
    );
    const autoRenderUrl = URL.createObjectURL(
        new Blob(
            [autoRenderModuleSource.replace("../katex.mjs", katexUrl)],
            {{ type: "text/javascript" }}
        )
    );

    try {{
        const {{ default: renderMathInElement }} = await import(autoRenderUrl);
        const renderRoot = {render_target};
        if (renderRoot) {{
            renderMathInElement(renderRoot, {{
                delimiters: [
                    {{ left: "$$", right: "$$", display: true }},
                    {{ left: "$", right: "$", display: false }},
                ],
            }});
        }}
    }} finally {{
        URL.revokeObjectURL(autoRenderUrl);
        URL.revokeObjectURL(katexUrl);
    }}
</script>"""

    return _try_wrap_html(injected + body)


def _latex_in_html_body(html_string) -> str:
    if hasattr(html_string, "to_html"):
        try:
            return html_string.to_html(escape=False)
        except Exception:
            return str(html_string)
    return str(html_string)


def _latex_in_html_render_target(
    *,
    container_id: str | None,
    katex_selector: str | None,
) -> str:
    if katex_selector:
        selector = (
            f"#{container_id} {katex_selector}" if container_id else katex_selector
        )
        return f"document.querySelector({selector!r})"
    if container_id:
        return f'document.getElementById("{container_id}")'
    return "document.body"


@lru_cache(maxsize=1)
def _latex_in_html_katex_modules() -> tuple[str, str]:
    dist = resources.files("dgcv").joinpath("assets", "katex", "dist")
    katex_module = dist.joinpath("katex.mjs").read_text(encoding="utf-8")
    auto_render_module = dist.joinpath("contrib", "auto-render.mjs").read_text(
        encoding="utf-8"
    )
    return katex_module, auto_render_module


@lru_cache(maxsize=1)
def _latex_in_html_katex_css() -> str:
    dist = resources.files("dgcv").joinpath("assets", "katex", "dist")
    css = dist.joinpath("katex.min.css").read_text(encoding="utf-8")

    def replace_font_url(match: re.Match[str]) -> str:
        font_rel_path = match.group("path")
        font_path = dist.joinpath(font_rel_path)
        font_bytes = font_path.read_bytes()
        mime_type = (
            mimetypes.guess_type(font_path.name)[0] or "application/octet-stream"
        )
        encoded = base64.b64encode(font_bytes).decode("ascii")
        return f'url("data:{mime_type};base64,{encoded}")'

    return re.sub(
        r'url\((?P<quote>["\']?)(?P<path>fonts/[^)\'"]+)(?P=quote)\)',
        replace_font_url,
        css,
    )


_ELEMENTARY_LABELS = (
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "sinh",
    "cosh",
    "tanh",
    "coth",
    "sech",
    "csch",
    "asin",
    "acos",
    "atan",
    "acot",
    "asec",
    "acsc",
    "asinh",
    "acosh",
    "atanh",
    "acoth",
    "asech",
    "acsch",
)


def configure_convenient_labels(
    libraries: Optional[
        List[
            Literal[
                "all",
                "most",
                "complex variables",
                "symbolic expressions",
                "elementary functions",
                "abbreviations",
            ]
        ]
    ] = None,
    verbose: bool = False,
    safe_labeling: bool = False,
    custom_labels: Optional[dict] = None,
):
    if libraries is None:
        libraries = ["most"]
    include_all = "all" in libraries
    include_most = include_all or "most" in libraries
    relabeling_map = (
        {
            "simplify": "simplify_dgcv",
            "expand": "expand_dgcv",
            "factor": "factor_dgcv",
            "cancel": "cancel_dgcv",
            "conjugate": "conjugate_dgcv",
            "im": "im_dgcv",
            "re": "re_dgcv",
        }
        if safe_labeling
        else dict()
    )
    if isinstance(custom_labels, dict) and all(
        isinstance(k, str) for k in custom_labels
    ):
        relabeling_map |= custom_labels
    configured_by_library: dict[str, list[str]] = {}

    if include_most or "complex variables" in libraries:
        from .._backends import im, re
        from .._backends._symbolic_router import conjugate
        from .._backends._types_and_constants import imag_unit

        new_functions = {
            "I": imag_unit(),
            "conjugate": conjugate,
            "im": im,
            "re": re,
        }
        new_functions = {relabeling_map.get(k, k): v for k, v in new_functions.items()}
        working_namespace().update(new_functions)
        configured_by_library["complex variables"] = sorted(
            new_functions, key=str.lower
        )

    if include_most or "symbolic expressions" in libraries:
        from .._backends import (
            cancel_dgcv,
            expand_dgcv,
            factor_dgcv,
            ratio,
            simplify_dgcv,
            sqrt_dgcv,
            subs_dgcv,
            symbol,
        )

        new_functions = {
            "subs": subs_dgcv,
            "take_quotient": ratio,
            "simplify": simplify_dgcv,
            "expand": expand_dgcv,
            "factor": factor_dgcv,
            "cancel": cancel_dgcv,
            "symbol": symbol,
            "sqrt": sqrt_dgcv,
        }
        new_functions = {relabeling_map.get(k, k): v for k, v in new_functions.items()}
        working_namespace().update(new_functions)
        configured_by_library["symbolic expressions"] = sorted(
            new_functions, key=str.lower
        )
    if include_most or "elementary functions" in libraries:
        from .._backends import _symbolic_router

        new_functions = {
            name: getattr(_symbolic_router, name) for name in _ELEMENTARY_LABELS
        }
        new_functions = {relabeling_map.get(k, k): v for k, v in new_functions.items()}
        working_namespace().update(new_functions)
        configured_by_library["elementary functions"] = list(new_functions)

    if include_all or "abbreviations" in libraries:
        from ...core.vector_fields_and_differential_forms import (
            coordinate_differential_form,
            coordinate_vector_field,
        )

        new_functions = {
            "coor_DF": coordinate_differential_form,
            "coor_VF": coordinate_vector_field,
        }
        new_functions = {relabeling_map.get(k, k): v for k, v in new_functions.items()}
        working_namespace().update(new_functions)
        configured_by_library["miscellaneous abbreviations"] = sorted(
            new_functions, key=str.lower
        )

    if not configured_by_library:
        return

    to_print = (
        True if not dgcv_settings_registry.get("forgo_warnings", False) else verbose
    )

    if not to_print:
        return

    engine = dgcv_settings_registry.get("default_symbolic_engine", "builtin")

    bullets = "\n".join(
        f"  • For {lib}: {', '.join(funcs)}"
        for lib, funcs in configured_by_library.items()
    )

    message = f"The following labels were configured in the namespace:\n\n{bullets}"
    if not dgcv_settings_registry.get("forgo_warnings", False):
        message += (
            "\n\n"
            f"User warning: The current dgcv settings have {engine} registered as the default symbolic engine.\n"
            "The above labels supersede equivalently-named objects from the default engine, "
            "extending functionality where applicable and falling back to the "
            f"{engine} implementations otherwise."
        )

    print(message)


# -----------------------------------------------------------------------------
# warning system
# -----------------------------------------------------------------------------


def configure_warnings():
    warnings.simplefilter("once", category=dgcvWarning)
    warnings.simplefilter("once", category=dgcvDeprecationWarning)
    warnings.simplefilter("always", category=dgcvOperationsNote)
    warnings.showwarning = _dgcv_showwarning


_original_showwarning = warnings.showwarning


class dgcvWarning(UserWarning):
    pass


class dgcvOperationsNote(UserWarning):
    pass


class dgcvDeprecationWarning(DeprecationWarning):
    def __init__(
        self,
        message,
        *,
        old_kw=None,
        new_kw=None,
        since=None,
        sunset=None,
    ):
        super().__init__(message)
        self.old_kw = old_kw
        self.new_kw = new_kw
        self.since = since
        self.sunset = sunset


class debug_log(UserWarning):
    pass


def _dgcv_showwarning(message, category, filename, lineno, file=None, line=None):
    stream = file if file is not None else sys.stderr

    if issubclass(category, dgcvDeprecationWarning):
        parts = []

        if getattr(message, "old_kw", None):
            parts.append(f"deprecated keyword={message.old_kw!r}")

        if getattr(message, "new_kw", None):
            parts.append(f"use {message.new_kw!r} instead")

        if getattr(message, "since", None):
            parts.append(f"since {message.since}")

        if getattr(message, "sunset", None):
            parts.append(f"scheduled for removal: {message.sunset}")

        suffix = f" ({'; '.join(parts)})" if parts else ""

        print(f"dgcv deprecation: {message}{suffix}", file=stream)
        return
    if issubclass(category, dgcvWarning):
        print(f"dgcv warning: {message}", file=stream)
        return
    if issubclass(category, dgcvOperationsNote):
        print(f"dgcv operations note: {message}", file=stream)
        return
    if issubclass(category, debug_log):
        if get_dgcv_settings_registry().get("DEBUG", False):
            print(f"DEBUG note: {message}", file=stream)
        return
    _original_showwarning(message, category, filename, lineno, file=file, line=line)


def dgcv_warning(
    message, warning_class=None, stacklevel=2, wc_label=None, **warning_kwargs
):
    if warning_class is None:
        if isinstance(wc_label, str):
            if wc_label == "dgcvOperationsNote":
                warning_class = dgcvOperationsNote
            elif wc_label == "debug_log":
                warning_class = debug_log
            else:
                warning_class = dgcvWarning
        else:
            warning_class = dgcvWarning

    if isinstance(warning_class, type) and issubclass(warning_class, Warning):
        warning = (
            warning_class(message, **warning_kwargs)
            if warning_kwargs
            else warning_class(message)
        )
    else:
        warning = warning_class

    warnings.warn(warning, stacklevel=stacklevel)
