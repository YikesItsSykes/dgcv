"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv._aux.printing.printing._class_printers

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

# -----------------------------------------------------------------------------
# imports
# -----------------------------------------------------------------------------
from __future__ import annotations

import re
from typing import Any

from ..._backends._display import latex as _backend_latex
from ..._backends._symbolic_router import _scalar_is_zero
from ..._utilities._config import get_dgcv_settings_registry
from ._scalars import (
    _coeff_latex,
    _dual_marker,
    _print_style,
    _scalar_is_minus_one,
    _scalar_is_one,
    _scalar_mul,
    _shape_joiner,
)
from ._string_processing import (
    _format_label_with_hi_low,
    _process_var_label,
    coeff_needs_parens_latex,
    coeff_needs_parens_plain,
    convert_to_greek,
    latex_superscript,
)


def tensor_VS_printer(tp) -> str:
    terms = tp.coeff_dict
    smul = _scalar_mul("plain")
    joiner = _shape_joiner(getattr(tp, "shape", "general"), "plain")
    dual_plain = _dual_marker("plain")

    def coeff_prefix(scalar) -> str:
        if _scalar_is_one(scalar):
            return ""
        if _scalar_is_minus_one(scalar):
            return "-"
        s = str(scalar)
        if coeff_needs_parens_plain(s):
            return f"({s}){smul}"
        return f"{s}{smul}"

    BL: dict[Any, list[str]] = {}

    def labler(idx, card) -> str:
        if card not in BL:
            vsl = card.space
            BL[card] = vsl.basis_labels or [
                f"VS{card.uid}_{j + 1}" for j in range(vsl.dimension)
            ]
        return BL[card][idx]

    def dual_labler(idx, card) -> str:
        labels = getattr(card.space, "dual_basis_labels", None)
        if labels is not None:
            return labels[idx]
        return f"{labler(idx, card)}{dual_plain}"

    formatted_terms: list[str] = []
    for vec, scalar in terms.items():
        if _scalar_is_zero(scalar):
            continue
        if not vec:
            formatted_terms.append(str(scalar))
            continue
        basis_elements = [
            labler(index, card) if valence == 1 else dual_labler(index, card)
            for index, valence, card in vec
        ]
        basis = joiner.join(basis_elements)
        formatted_terms.append(f"{coeff_prefix(scalar)}{basis}")

    if not formatted_terms:
        vec = next(iter(terms.keys()))
        basis = joiner.join(labler(index, card) for index, _, card in vec)
        return f"0{basis}"

    out = formatted_terms[0]
    for t in formatted_terms[1:]:
        out += t if t.startswith("-") else f"+{t}"
    return out


def _pwrap_for_dual(base: str) -> str:
    # only for things like x^2, so x^2^* becomes (x^2)^*
    # intentionally minimal: basis labels are close to atomic
    if "^" in base and not base.rstrip().endswith(")"):
        return rf"\left({base}\right)"
    return base


def _alias_factor_latex(factor) -> str:
    if isinstance(factor, str):
        return _process_var_label(factor)
    return factor._repr_latex_(raw=True)


def tensor_latex_alias(alias) -> str:
    terms, primary = alias
    joiner = _shape_joiner("general", "latex")
    dual_latex = _dual_marker("latex")
    formatted_terms: list[str] = []
    for vec, scalar in terms.items():
        basis_elements = [
            _alias_factor_latex(vec[1]),
            rf"{_pwrap_for_dual(_alias_factor_latex(vec[0]))}^{dual_latex}",
        ]
        basis = joiner.join(basis_elements)

        c = _coeff_latex(scalar, basis)
        if c == "":
            formatted_terms.append(basis)
        elif c == "-":
            formatted_terms.append(rf"- {basis}")
        else:
            formatted_terms.append(rf"{c} {basis}")
    out = formatted_terms[0] if formatted_terms else "0"
    for t in formatted_terms[1:]:
        out += t if t.startswith("-") else f" + {t}"
    out = f"{_process_var_label(primary)} = {out}"
    return out


def tensor_latex_helper(tp) -> str:
    terms = tp.coeff_dict
    joiner = _shape_joiner(getattr(tp, "shape", "general"), "latex")
    dual_latex = _dual_marker("latex")

    def labler(idx, card) -> str:
        return card.space.basis[idx]._repr_latex_(raw=True)

    def dual_labler(idx, card) -> str:
        labels = getattr(card.space, "_tex_dual_basis_labels", None)
        if labels is not None:
            return labels[idx]
        return rf"{_pwrap_for_dual(labler(idx, card))}^{dual_latex}"

    formatted_terms: list[str] = []
    for vec, scalar in terms.items():
        if _scalar_is_zero(scalar):
            continue
        basis_elements = [
            labler(index, card) if valence == 1 else dual_labler(index, card)
            for index, valence, card in vec
        ]
        basis = joiner.join(basis_elements)

        c = _coeff_latex(scalar, basis)
        if c == "":
            formatted_terms.append(basis)
        elif c == "-":
            formatted_terms.append(rf"- {basis}")
        else:
            formatted_terms.append(rf"{c} {basis}")

    if not formatted_terms:
        vec = next(iter(terms.keys()))
        basis = joiner.join(labler(index, card) for index, _, card in vec)
        return "0" + basis

    out = formatted_terms[0]
    for t in formatted_terms[1:]:
        out += t if t.startswith("-") else f" + {t}"
    return out


def lincomb_plain(
    coeff_dict,
    labels,
    *,
    valence: int = 1,
    label_transform=None,
    fallback_label: str = "e_1",
    include_zero_term: bool = True,
) -> str:
    style = _print_style()
    smul = _scalar_mul("plain")
    dual_plain = _dual_marker("plain")

    def _bl(lbl: str) -> str:
        s = label_transform(lbl) if callable(label_transform) else lbl
        return s if valence == 1 else f"{s}{dual_plain}"

    def _coeff_prefix(c) -> str:
        if _scalar_is_one(c):
            return ""
        if _scalar_is_minus_one(c):
            return "-"
        s = str(c)
        if coeff_needs_parens_plain(s):
            return f"({s}){smul}"
        return f"{s}{smul}"

    terms: list[str] = []
    for idx, c in coeff_dict.items():
        lbl = labels[idx]
        if _scalar_is_zero(c):
            continue

        bl = _bl(str(lbl))

        if _scalar_is_one(c):
            terms.append(bl)
        elif _scalar_is_minus_one(c):
            terms.append(f"-{bl}")
        else:
            terms.append(f"{_coeff_prefix(c)}{bl}")

    if not terms:
        if not include_zero_term:
            return "0"
        return f"0{smul}{_bl(str(fallback_label))}"

    if style == "literal":
        out = terms[0]
        for t in terms[1:]:
            out += t if t.startswith("-") else f"+{t}"
        return out

    return " + ".join(terms).replace("+ -", "- ")


def lincomb_latex(
    coeff_dict,
    labels=None,
    *,
    vectorSpace=None,
    valence: int = 1,
    fallback_label: str | None = None,
    verbose: bool = False,
    raw: bool = False,
    apply_vlp_trim: bool = False,
):
    if labels is None:
        if vectorSpace is None:
            raise ValueError("lincomb_latex expects either labels or vectorSpace")
        if getattr(vectorSpace, "_tex_basis_labels", None) is not None:
            labels = vectorSpace._tex_basis_labels
            proDone = True
        elif getattr(vectorSpace, "basis_labels", None) is not None:
            labels = vectorSpace.basis_labels
            proDone = False
        else:
            labels = [
                f"_e_{{{i + 1}}}" for i in range(getattr(vectorSpace, "dimension", 0))
            ]
            proDone = False
    else:
        proDone = False

    if fallback_label is None:
        bl = (
            getattr(vectorSpace, "basis_labels", None)
            if vectorSpace is not None
            else None
        )
        fallback_label = str(bl[0]) if isinstance(bl, (list, tuple)) and bl else "e_1"

    def _process_label(lbl: str) -> str:
        if proDone:
            return lbl

        if vectorSpace is not None and getattr(
            vectorSpace, "_basis_labels_parent", False
        ):
            m = re.search(r"(_\d+)$", lbl)
            if m:
                tail = f"_{{{m.group(1)[1:]}}}"
                return (
                    f"\\left({vectorSpace._repr_latex_(abbrev=True, raw=True)}\\right)"
                    + tail
                )

        m = re.search(r"(\d+)$", lbl)
        if m and "_" not in lbl:
            lbl = lbl[: m.start(1)] + "_" + m.group(1)
        return _format_label_with_hi_low(lbl)

    def _coeff_tex(c) -> str:
        if _scalar_is_one(c):
            return ""
        if _scalar_is_minus_one(c):
            return "-"
        s = _backend_latex(c)
        if coeff_needs_parens_latex(s):
            return rf"\left({s}\right)"
        return s

    def _star(lbl: str) -> str:
        return latex_superscript(lbl, "*")

    terms: list[str] = []
    for idx, c in coeff_dict.items():
        lbl = labels[idx]
        if _scalar_is_zero(c):
            continue

        bl = _process_label(str(lbl))
        bl = bl if valence == 1 else _star(bl)

        if _scalar_is_one(c):
            terms.append(bl)
            continue
        if _scalar_is_minus_one(c):
            terms.append(rf"-{bl}")
            continue

        ct = _coeff_tex(c)
        terms.append(rf"{ct} {bl}")

    if not terms:
        if verbose:
            out0 = rf"0 {fallback_label}"
            return out0 if raw else rf"${out0}$"
        return "0" if raw else "$0$"

    result = " + ".join(terms).replace("+ -", "- ")

    if apply_vlp_trim:
        reg = get_dgcv_settings_registry()
        if reg.get("verbose_label_printing") is False:
            m = reg["VLP"].match(result)
            if m and m.group("content") is not None:
                result = m.group("content")
            elif m:
                result = result[: result.rfind("_{\\operatorname{v.}")]

    return result if raw else rf"$\displaystyle {result}$"


def space_display(
    *,
    fmt: str,
    basis_words,
    dim: int | None = None,
    label: str | None = None,
    label_tex: str | None = None,
    mathfrak_label: bool = False,
    unlabeled_plain: str = "Unnamed",
    unlabeled_tex: str = r"\text{Unnamed}",
    max_dim: int = 20,
    raw: bool = False,
    abbrev: bool = False,
    use_displaystyle: bool = True,
    plain_wrapper: str = "<{}>",
    latex_wrapper: str = r"\langle {}\rangle",
    label_only_if_not_vlp: bool = False,
):
    reg = get_dgcv_settings_registry()
    vlp = bool(reg.get("verbose_label_printing", False))

    if fmt not in ("plain", "latex"):
        raise ValueError("fmt must be 'plain' or 'latex'")

    def _truncate(words: list[str]) -> list[str]:
        d = len(words) if dim is None else dim
        if d <= max_dim or len(words) <= max_dim:
            return words
        k = max_dim // 2
        return words[:k] + ["..."] + words[-k:]

    def format_suffix(sub: str) -> str:
        if sub == "negative":
            return "-"
        if sub == "positive":
            return "+"
        if re.fullmatch(r"m\d+", sub):
            return f"-{sub[1:]}"
        if re.fullmatch(r"[a-zA-Z]+", sub) and len(sub) > 1:
            return rf"\text{{{sub}}}"
        return sub

    def _label_tex(label0: str | None, *, mathfrak: bool) -> str:
        if not label0:
            return unlabeled_tex
        s = str(label0)

        idx_double = s.find("__")
        idx_single = -1
        i = 0
        while i < len(s):
            if s[i] == "_":
                if i + 1 < len(s) and s[i + 1] == "_":
                    i += 2
                    continue
                idx_single = i
                break
            i += 1

        if idx_single != -1 or idx_double != -1:
            if idx_single != -1 and idx_double != -1:
                if idx_single < idx_double:
                    main = s[:idx_single]
                    sub = s[idx_single + 1 : idx_double]
                    sup = s[idx_double + 2 :]
                else:
                    main = s[:idx_double]
                    sup = s[idx_double + 2 : idx_single]
                    sub = s[idx_single + 1 :]
                main_conv = convert_to_greek(main)
                if mathfrak and main.islower() and main_conv == main:
                    main_conv = rf"\mathfrak{{{main}}}"
                return rf"{main_conv}_{{{format_suffix(sub)}}}^{{{format_suffix(sup)}}}"

            elif idx_double != -1:
                main, sup = s[:idx_double], s[idx_double + 2 :]
                main_conv = convert_to_greek(main)
                if mathfrak and main.islower() and main_conv == main:
                    main_conv = rf"\mathfrak{{{main}}}"
                return rf"{main_conv}^{{{format_suffix(sup)}}}"

            else:
                main, sub = s[:idx_single], s[idx_single + 1 :]
                main_conv = convert_to_greek(main)
                if mathfrak and main.islower() and main_conv == main:
                    main_conv = rf"\mathfrak{{{main}}}"
                return rf"{main_conv}_{{{format_suffix(sub)}}}"

        if s[-1].isdigit():
            head = "".join(ch for ch in s if ch.isalpha())
            tail = "".join(ch for ch in s if ch.isdigit())
            head_conv = convert_to_greek(head)
            if mathfrak and head.islower() and head_conv == head:
                head_conv = rf"\mathfrak{{{head}}}"
            return rf"{head_conv}_{{{tail}}}"

        head_conv = convert_to_greek(s)
        if mathfrak and s.islower() and head_conv == s:
            return rf"\mathfrak{{{s}}}"
        return head_conv

    if fmt == "latex":
        lab = (
            label_tex
            if label_tex is not None
            else _label_tex(label, mathfrak=mathfrak_label)
        )
    else:
        lab = None

    if abbrev:
        if fmt == "plain":
            return str(label) if label else unlabeled_plain
        return lab if raw else f"${lab}$"

    if (not vlp) and label_only_if_not_vlp:
        if fmt == "plain":
            return str(label) if label else unlabeled_plain
        return (
            lab
            if raw
            else (rf"$\displaystyle {lab}$" if use_displaystyle else f"${lab}$")
        )

    toks = list(basis_words() if callable(basis_words) else (basis_words or []))
    toks = [str(t) for t in toks]
    toks = _truncate(toks)

    if fmt == "plain":
        core = plain_wrapper.format(", ".join(toks))
        if vlp and label:
            return f"{label}={core}"
        return core

    inner = ", ".join(toks).replace("...", r"\dots")
    core = latex_wrapper.format(inner)

    if vlp and (label or label_tex):
        core = f"{lab}={core}"

    if raw:
        return core
    return rf"$\displaystyle {core}$" if use_displaystyle else f"${core}$"
