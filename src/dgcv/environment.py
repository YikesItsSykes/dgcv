"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv.environment


---
Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from ._aux._backends._engine import available_engine_kinds, engine_kind
from ._aux._backends._notebooks import in_kernel, on_sage_kernel
from ._aux._backends._numeric_engine import available_numeric_kinds, numeric_kind
from ._aux._utilities._config import (
    environment_inference,
    get_dgcv_settings_registry,
)

__all__ = ["environment_summary", "reproducibility_note"]

_UNKNOWN = "unknown"
_ABSENT = "(not installed)"

_DISTRIBUTIONS = (
    ("dgcv", "dgcv", None),
    ("sympy", "sympy", "sympy"),
    ("sage", "sagemath-standard", "sage.version"),
    ("numpy", "numpy", "numpy"),
    ("IPython", "ipython", "IPython"),
    ("ipykernel", "ipykernel", "ipykernel"),
)

_PROSE_NAMES = {
    "sympy": "SymPy",
    "sage": "SageMath",
    "numpy": "NumPy",
    "python": "Python",
}

_RESULT_SETTINGS = (
    "default_symbolic_engine",
    "pass_solve_requests_to_symbolic_engine",
    "_solve_default",
    "_exceptional_cartan_basis",
    "use_rank_basis_extraction",
    "preferred_variable_format",
    "simplify_singularity_ideals_by_default",
    "use_numeric_methods",
    "version_specific_defaults",
)


def _guard(f, default=_UNKNOWN):
    try:
        return f()
    except Exception:
        return default


def _loaded_version(module_name):
    if not module_name:
        return None
    try:
        mod = sys.modules.get(module_name)
        if mod is None:
            return None
        for attr in ("__version__", "version"):
            value = getattr(mod, attr, None)
            if isinstance(value, str):
                return value
    except Exception:
        return None
    return None


def _distribution_version(dist, module_name):
    try:
        return version(dist)
    except PackageNotFoundError:
        loaded = _loaded_version(module_name)
        return loaded if loaded else _ABSENT
    except Exception:
        loaded = _loaded_version(module_name)
        return loaded if loaded else _UNKNOWN


def _symbolic_engine():
    settings = _guard(get_dgcv_settings_registry, {})
    requested = settings.get("default_symbolic_engine", _UNKNOWN)
    active = _guard(engine_kind, None)
    if active is None:
        active = requested
    available = _guard(available_engine_kinds, ())
    listed = ", ".join(str(k) for k in available) if available else "none"
    return f"{active} (available: {listed})"


def _numeric_engine(settings):
    available = _guard(available_numeric_kinds, ())
    listed = ", ".join(str(k) for k in available) if available else "none"
    if settings.get("use_numeric_methods", False) is not True:
        return f"off (available: {listed})"
    active = _guard(numeric_kind, _UNKNOWN)
    return f"{active} (available: {listed})"


def _frontend():
    kernel = _guard(in_kernel, None)
    if kernel is None:
        return _UNKNOWN
    where = "jupyter kernel" if kernel else "no kernel attached"
    shim = _guard(environment_inference, None)
    if shim is None:
        return where
    return f"{where}, katex layer {'on' if shim else 'off'}"


def _process_fields():
    seed = os.environ.get("PYTHONHASHSEED")
    return {"PYTHONHASHSEED": seed} if seed else {}


def environment_summary(
    verbose: bool = False, plain_text: bool = True, return_data: bool = False
):
    """
    Report the environment details.
    """
    data = _collect_verbose() if verbose else _collect_concise()

    if return_data:
        return data

    if plain_text:
        print(_as_text(data) if verbose else _as_sentence(data))
        return None

    _show_table(data)
    return None


def reproducibility_note(return_text: bool = False):
    """
    Prints a note sumarizing the present computing environment. Intended as a note to include with any published script for reproducibility.
    """
    data = _collect_concise()
    core = {k: v for k, v in data.get("core", {}).items() if v != _ABSENT}
    python = core.pop("python", None)
    engine = data.get("engine", {}).get("symbolic engine")

    host = core.pop("sage") if ("sage" in core and engine != "sage") else None
    listed = [f"{_PROSE_NAMES.get(k, k)} {v}" for k, v in core.items()]
    under = f" running under {_PROSE_NAMES['sage']} {host}" if host else ""

    text = "Reproducibility note: "
    if listed:
        text += f"running with {_join(listed)}{under}"
        if python:
            text += f"{',' if under else ''} on {_PROSE_NAMES['python']} {python}."
        else:
            text += "."
    elif python:
        text += f"running {_PROSE_NAMES['python']} {python}."
    else:
        text += "the computing environment could not be determined."

    notebook = data.get("notebook")
    if notebook:
        pairs = _join([f"{k} {v}" for k, v in notebook.items()])
        text += f" The notebook was rendered with {pairs}."

    if return_text:
        return text
    print(text)
    return None


def _join(items):
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _collect_concise():
    settings = _guard(get_dgcv_settings_registry, {})
    engine = _guard(engine_kind, None) or settings.get(
        "default_symbolic_engine", _UNKNOWN
    )
    sage_kernel = _guard(on_sage_kernel, False)

    core = {
        "dgcv": _distribution_version("dgcv", None),
        "python": _guard(platform.python_version),
    }
    if engine == "sage" or sage_kernel:
        core["sage"] = _distribution_version("sagemath-standard", "sage.version")
    if engine != "sage":
        core["sympy"] = _distribution_version("sympy", "sympy")
    if settings.get("use_numeric_methods", False) is True:
        core["numpy"] = _distribution_version("numpy", "numpy")

    data = {"core": core, "engine": {"symbolic engine": engine}}
    if _guard(in_kernel, False):
        data["notebook"] = {
            "IPython": _distribution_version("ipython", "IPython"),
            "ipykernel": _distribution_version("ipykernel", "ipykernel"),
        }
    return data


def _as_sentence(data):
    core = data.get("core", {})
    engine = data.get("engine", {}).get("symbolic engine")
    named_engines = [k for k in core if k in ("sympy", "sage")]
    parts = []
    for key, value in core.items():
        mark = " (symbolic engine)" if key == engine and len(named_engines) > 1 else ""
        parts.append(f"{key} {value}{mark}")
    out = ", ".join(parts)
    if engine not in core:
        out += f", symbolic engine: {engine}"
    notebook = data.get("notebook")
    if notebook:
        listed = ", ".join(f"{k} {v}" for k, v in notebook.items())
        out += f"; notebook: {listed}"
    return out


def _collect_verbose():
    settings = _guard(get_dgcv_settings_registry, {})

    versions = {}
    for label, dist, module_name in _DISTRIBUTIONS:
        versions[label] = _distribution_version(dist, module_name)
    versions["python"] = _guard(
        lambda: f"{platform.python_version()} ({platform.python_implementation()})"
    )
    versions["platform"] = _guard(platform.platform)

    data = {
        "versions": {
            "dgcv": versions["dgcv"],
            "python": versions["python"],
            "platform": versions["platform"],
            "sympy": versions["sympy"],
            "sage": versions["sage"],
            "numpy": versions["numpy"],
            "IPython": versions["IPython"],
            "ipykernel": versions["ipykernel"],
        },
        "engines": {
            "symbolic engine": _symbolic_engine(),
            "numeric engine": _numeric_engine(settings),
        },
        "display": {"frontend": _frontend()},
        "settings affecting results": {
            key: settings.get(key, _UNKNOWN)
            for key in _RESULT_SETTINGS
            if key in settings
        },
        "process": _process_fields(),
    }
    if settings.get("use_numeric_methods", False) is True:
        data["settings affecting results"]["numeric_error_thresholds"] = settings.get(
            "numeric_error_thresholds", _UNKNOWN
        )

    return data


def _sections(data):
    for section, fields in data.items():
        if isinstance(fields, dict):
            if fields:
                yield section, fields
        else:
            yield section, {section: fields}


def _as_text(data):
    lines = ["dgcv environment"]
    for section, fields in _sections(data):
        width = max(len(k) for k in fields)
        lines.append(f"  [{section}]")
        for key, value in fields.items():
            lines.append(f"    {key:<{width}}  {value}")
    return "\n".join(lines)


def _show_table(data):
    from ._aux._utilities._styles import get_style
    from ._aux.printing._tables import build_plain_table
    from ._aux.printing.printing._dgcv_display import show

    rows = [
        [section, key, str(value)]
        for section, fields in _sections(data)
        for key, value in fields.items()
    ]
    theme = _guard(lambda: get_dgcv_settings_registry().get("theme", "dark"), "dark")
    show(
        build_plain_table(
            columns=["section", "field", "value"],
            rows=rows,
            theme_css_vars=_guard(lambda: get_style(theme, legacy=False), ""),
            caption="dgcv environment",
            container_id="dgcv-environment-summary",
        )
    )
