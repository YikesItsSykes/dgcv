"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv._aux._backends

module: dgcv._aux._backends._engine.py


Description: manages dgcv's interfacing with available CAS libraries.

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
import importlib
import importlib.util
import sys

from .._utilities._config import dgcv_warning, get_dgcv_settings_registry

__all__ = [
    "is_sage_available",
    "is_sympy_available",
    "sage_module_if_available",
    "sympy_module_if_available",
    "available_engine_kinds",
    "engine_module",
    "engine_kind",
    "engine_capability",
    "invalidate_engine_cache",
]


# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------
_engine_kind = None
_engine_module = None
_sympy_module = None
_sage_module = None
_sage_available = None
_builtin_module = None
_builtin_fallback_warned = False
_capabilities = None


def is_sage_available():
    global _sage_available
    if _sage_available is not None:
        return _sage_available
    try:
        spec = importlib.util.find_spec("sage.all")
        _sage_available = spec is not None
    except (ImportError, ModuleNotFoundError):
        _sage_available = False
    return _sage_available


def _get_sage_module():
    global _sage_module
    if _sage_module is not None:
        return _sage_module
    if not is_sage_available():
        raise RuntimeError("Sage is not available in the current environment.")
    _sage_module = importlib.import_module("sage.all")
    return _sage_module


def sage_module_if_available():
    if not is_sage_available():
        return None
    try:
        return _get_sage_module()
    except Exception:
        return None


def is_sympy_available():
    try:
        return importlib.util.find_spec("sympy") is not None
    except (ImportError, ModuleNotFoundError):
        return False


def _get_sympy_module():
    global _sympy_module
    if _sympy_module is not None:
        return _sympy_module
    if not is_sympy_available():
        raise RuntimeError("SymPy is not available in the current environment.")
    _sympy_module = importlib.import_module("sympy")
    return _sympy_module


def sympy_module_if_available():
    if not is_sympy_available():
        return None
    try:
        return _get_sympy_module()
    except Exception:
        return None


def available_engine_kinds():
    out = []
    if is_sage_available():
        out.append("sage")
    if is_sympy_available():
        out.append("sympy")
    out.append("builtin")
    return tuple(out)


def _get_builtin_module():
    global _builtin_module
    if _builtin_module is None:
        from ..._symbolic_scalars import _engine_api

        _builtin_module = _engine_api
    return _builtin_module


def _builtin_fallback(requested):
    global _builtin_fallback_warned
    if not _builtin_fallback_warned:
        _builtin_fallback_warned = True
        dgcv_warning(
            f"dgcv: default symbolic engine setting is {requested!r} but neither Sage nor SymPy is available; "
            "using dgcv's builtin symbolic engine (rational expressions only).",
            stacklevel=2,
        )
    return "builtin"


def notify_vmf_cleared():
    mod = sys.modules.get("dgcv._symbolic_scalars._secondary")
    if mod is not None:
        mod.invalidate_secondary_cache()
    mod = sys.modules.get("dgcv._symbolic_scalars._poly")
    if mod is not None:
        mod._kinds_stale[0] = True


def invalidate_engine_cache():
    global _engine_kind, _engine_module, _capabilities
    _engine_kind = None
    _engine_module = None
    _capabilities = None
    notify_vmf_cleared()

    try:
        from ._types_and_constants import invalidate_types_and_constants_cache

        invalidate_types_and_constants_cache()
    except Exception:
        pass

    try:
        from ._cls_coercion import invalidate_cls_coercion_cache

        invalidate_cls_coercion_cache()
    except Exception:
        pass


def _resolve_engine_kind():
    settings = get_dgcv_settings_registry()
    requested = str(settings.get("default_symbolic_engine", "builtin")).lower()

    if requested in ("sagemath",):
        requested = "sage"

    if requested == "builtin":
        return "builtin"

    if requested == "sage":
        if is_sage_available():
            return "sage"
        if is_sympy_available():
            dgcv_warning(
                "dgcv: default symbolic engine setting is 'sage' but Sage is not available; "
                "falling back to 'sympy'.",
                stacklevel=2,
            )
            return "sympy"
        return _builtin_fallback(requested)

    if requested == "sympy":
        if is_sympy_available():
            return "sympy"
        if is_sage_available():
            dgcv_warning(
                "dgcv: default symbolic engine setting is 'sympy' but SymPy is not available; "
                "falling back to 'sage'.",
                stacklevel=2,
            )
            return "sage"
        return _builtin_fallback(requested)

    dgcv_warning(
        f"dgcv: unrecognized symbolic engine {requested!r}; falling back to 'builtin'.",
        stacklevel=2,
    )
    return "builtin"


def engine_kind():
    global _engine_kind
    if _engine_kind is None:
        _engine_kind = _resolve_engine_kind()
        if _engine_kind == "builtin":
            _get_builtin_module()
    return _engine_kind


def engine_capability(name):
    global _capabilities
    caps = _capabilities
    if caps is None:
        caps = getattr(engine_module(), "dgcv_capabilities", None)
        if caps is None:
            caps = {}
        _capabilities = caps
    return caps.get(name)


def engine_module():
    global _engine_module
    kind = engine_kind()
    if kind is None:
        return None
    if _engine_module is not None:
        return _engine_module
    if kind == "sage":
        _engine_module = _get_sage_module()
    elif kind == "sympy":
        _engine_module = _get_sympy_module()
    elif kind == "builtin":
        _engine_module = _get_builtin_module()
    else:
        raise RuntimeError(f"dgcv: unknown symbolic engine kind {kind!r}")
    return _engine_module
