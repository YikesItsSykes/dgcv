"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

module: dgcv.eds._zf_ops

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers
from math import prod

from .._aux._backends._symbolic_router import ratio


class OpSpec:
    __slots__ = ("tag", "arity", "normalize", "lower", "text", "latex", "cfd")

    def __init__(self, tag, arity, normalize, lower, text, latex, cfd=None):
        self.tag = tag
        self.arity = arity
        self.normalize = normalize
        self.lower = lower
        self.text = text
        self.latex = latex
        self.cfd = cfd


OP_REGISTRY = {}


def register_op(tag, arity, normalize, lower, text, latex):
    OP_REGISTRY[tag] = OpSpec(tag, arity, normalize, lower, text, latex)
    return OP_REGISTRY[tag]


def op_spec(tag):
    spec = OP_REGISTRY.get(tag)
    if spec is None:
        raise ValueError(f"`abstract_ZF` does not support the operation tag {tag!r}")
    return spec


def check_arity(tag, args):
    spec = op_spec(tag)
    if spec.arity is not None and len(args) != spec.arity:
        raise ValueError(
            f"`abstract_ZF` operation {tag!r} expects {spec.arity} operands, received {len(args)}"
        )
    return spec


def _lower_add(args):
    return sum(args)


def _lower_mul(args):
    return prod(args)


def _lower_pow(args):
    return args[0] ** args[1]


def _lower_sub(args):
    return args[0] - args[1]


def _lower_div(args):
    if all(isinstance(arg, numbers.Number) for arg in args):
        return ratio(args[0], args[1])
    return args[0] / args[1]


def _text_add(parts):
    out = parts[0]
    for part in parts[1:]:
        out += f" - {part[1:]}" if part.startswith("-") else f" + {part}"
    return out


def _text_sub(parts):
    return " - ".join(parts)


def _text_mul(parts):
    negative = False
    kept = []
    for part in parts:
        if part == "-1":
            negative = not negative
        elif part != "1":
            kept.append(part)
    if not kept:
        return "-1" if negative else "1"
    out = "*".join(kept)
    if negative:
        out = out[1:] if out.startswith("-") else f"-{out}"
    return out


def _text_div(parts):
    return f"{parts[0]}/{parts[1]}"


def _text_pow(parts):
    return f"{parts[0]}**{parts[1]}"


def _latex_add(parts):
    return " + ".join(parts).replace("+ -", "-")


def _latex_sub(parts):
    return " - ".join(parts)


def _latex_mul(parts):
    return " ".join(parts)


def _latex_div(parts):
    return f"\\frac{{{parts[0]}}}{{{parts[1]}}}"


def _latex_pow(parts):
    return f"{parts[0]}^{{{parts[1]}}}"


LOWER = {
    "add": _lower_add,
    "mul": _lower_mul,
    "pow": _lower_pow,
    "sub": _lower_sub,
    "div": _lower_div,
}
TEXT = {
    "add": _text_add,
    "sub": _text_sub,
    "mul": _text_mul,
    "div": _text_div,
    "pow": _text_pow,
}
LATEX = {
    "add": _latex_add,
    "sub": _latex_sub,
    "mul": _latex_mul,
    "div": _latex_div,
    "pow": _latex_pow,
}
ARITY = {"add": None, "mul": None, "pow": 2, "sub": 2, "div": 2}
