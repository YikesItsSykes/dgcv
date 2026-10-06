"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv._aux._backends

module: dgcv._aux._backends._updates


Description: manages compatibility with older versions of dgcv scripts

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True, order=True)
class version:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, s: str) -> "version":
        s = s.strip()
        if s.startswith(("v", "V")):
            s = s[1:]
        parts = (s.split(".") + ["0", "0"])[:3]
        return cls(int(parts[0]), int(parts[1]), int(parts[2]))


_infer_engine = "__infer__"


baseline_defaults: Dict[str, Any] = {
    "use_latex": False,
    "theme": "blue",
    "format_displays": False,
    "version_specific_defaults": None,
    "ask_before_overwriting_objects_in_vmf": True,
    "forgo_warnings": False,
    "default_symbolic_engine": "sympy",
    "verbose_label_printing": True,
    "VLP": None,
    "extra_support_for_math_in_tables": False,
    "conjugation_prefix": "BAR",
    "_solve_default": "solve",
    "forgo_CAS_provenance_pruning": False,
    "forgo_builtin_probabilistic_shortcuts": False,
}


changes: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("0.2.6", {"theme": "appalachian"}),
    ("0.3.0", {"theme": "graph_paper", "verbose_label_printing": False}),
    ("0.3.3", {"_solve_default": "auto"}),
    (
        "0.3.13",
        {
            "use_latex": True,
            "format_displays": True,
            "extra_support_for_math_in_tables": "infer",
        },
    ),
    ("0.4.20", {"theme": "paper_graphite"}),
    ("0.4.36", {"default_symbolic_engine": _infer_engine}),
    ("0.5.0", {"default_symbolic_engine": "builtin"}),
)


_parsed_changes = tuple((version.parse(v), patch) for v, patch in changes)


def needs_sympy_hook(target_version: str) -> bool:
    v = version.parse(target_version)
    return v <= version.parse("0.3.10")


def defaults_for_version(
    target_version: str,
    *,
    current_version: str,
    vlp: Any,
) -> Dict[str, Any]:
    target = version.parse(target_version)
    out = deepcopy(baseline_defaults)

    for v, patch in _parsed_changes:
        if v <= target:
            out.update(patch)

    if out.get("default_symbolic_engine") == _infer_engine:
        from .._utilities._config import legacy_engine_inference

        out["default_symbolic_engine"] = legacy_engine_inference()

    out["VLP"] = vlp
    out["version_specific_defaults"] = f"v{target.major}.{target.minor}.{target.patch}"
    return out
