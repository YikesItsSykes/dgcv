"""
package: dgcv - Differential Geometry with Complex Variables
module: dgcv._aux._vmf._function_arguments

Author (of this module): David Gamble Sykes
Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes
Licensed under the Apache License, Version 2.0
SPDX-License-Identifier: Apache-2.0
"""

from .._backends._engine import engine_kind
from .._backends._types_and_constants import is_atomic, to_active_engine
from .._utilities._config import dgcv_warning
from .vmf import vmf_lookup

DEPENDENCE_PROPERTIES = ("holomorphic", "antiholomorphic", "real_analytic")
FUNCTION_PROPERTIES = DEPENDENCE_PROPERTIES + ("real",)

_COORDINATE_FORMAT = {
    "holo": "complex",
    "anti": "complex",
    "real": "real",
    "imag": "real",
}


def function_properties(name, properties, real=False):
    if properties is None:
        names = frozenset()
    elif isinstance(properties, str):
        names = frozenset((properties,))
    else:
        names = frozenset(properties)
    for p in sorted(names, key=str):
        if p not in FUNCTION_PROPERTIES:
            raise ValueError(
                f"function_dgcv: unknown property {p!r} for `{name}`; the properties are 'holomorphic', 'antiholomorphic', 'real_analytic' and 'real'"
            )
    given = [p for p in DEPENDENCE_PROPERTIES if p in names]
    if len(given) > 1:
        dgcv_warning(
            f"function_dgcv: {' and '.join(repr(p) for p in given)} were given together for `{name}`; they are ignored"
        )
        names = names.difference(given)
        given = []
    real = bool(real) or "real" in names
    if real:
        names = names | {"real"}
    return real, given[0] if given else None, names


def _coordinate(name, arg):
    a = to_active_engine(arg)
    if engine_kind() == "builtin":
        from ...eds._zero_forms import _as_leaf

        leaf = _as_leaf(a)
        if leaf is not None:
            a = leaf
    if not is_atomic(a):
        raise ValueError(
            f"function_dgcv: with properties, every argument of `{name}` must be a variable"
        )
    info = vmf_lookup(a, flattened_relatives=True)
    fmt = _COORDINATE_FORMAT.get(info.get("sub_type"))
    rel = info.get("flattened_relatives")
    if fmt is None or not (isinstance(rel, tuple) and len(rel) == 4):
        return a, None, None
    return a, fmt, rel


def complete_function_arguments(name, args, dependence):
    out = []
    block = []
    seen = set()
    insert_at = 0
    fmt = None
    converted = False
    for arg in args:
        a, a_fmt, rel = _coordinate(name, arg)
        if a_fmt is None:
            out.append(a)
            continue
        if dependence == "real_analytic":
            if fmt is None:
                fmt = a_fmt
            elif fmt != a_fmt:
                raise ValueError(
                    f"function_dgcv: `{name}` with 'real_analytic' received both holomorphic and real coordinates; drop the property or use one coordinate type"
                )
            primary, secondary = rel[:2] if fmt == "complex" else rel[2:]
        else:
            primary = rel[0] if dependence == "holomorphic" else rel[1]
            secondary = None
            if a_fmt == "real":
                converted = True
        key = str(primary)
        if key not in seen:
            seen.add(key)
            out.append(primary)
            if secondary is not None:
                block.append(secondary)
        insert_at = len(out)
    if block:
        out[insert_at:insert_at] = block
    if converted:
        dgcv_warning(
            f"function_dgcv: the arguments of `{name}` were switched to {dependence} coordinates: ({', '.join(str(a) for a in out)})"
        )
    return tuple(out)
