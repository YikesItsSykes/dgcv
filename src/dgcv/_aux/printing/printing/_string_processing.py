"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv._aux.printing.printing._string_processing

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

import re

from ..._utilities._config import get_dgcv_settings_registry, greek_letters


# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------
def _format_label_with_hi_low(
    label: str,
    infer_suffix: bool = False,
    prime_processing: bool = True,
    decorations_support: bool = True,
) -> str:
    if not label:
        return label

    text = str(label)

    pref = get_dgcv_settings_registry().get("conjugation_prefix", "BAR")
    is_conjugate = False
    if pref and text.startswith(pref):
        text = text[len(pref) :]
        is_conjugate = True

    # Entry order matters in this dict. If one key string is contained in another, the longer one must come first.
    decoration_prefixes = {
        "widehat_": r"\widehat",
        "widetilde_": r"\widetilde",
        "overline_": r"\overline",
        "underline_": r"\underline",
        "tilde_": r"\widetilde",
        "hat_": r"\hat",
        "bar_": r"\bar",
        "vec_": r"\vec",
        "ddot_": r"\ddot",
        "dot_": r"\dot",
        "check_": r"\check",
        "breve_": r"\breve",
        "overline": r"\overline",
        "bar": r"\bar",
    }

    if decorations_support:
        decoration_prefixes = {
            **decoration_prefixes,
            "overleftrightarrow_": r"\overleftrightarrow",
            "overrightarrow_": r"\overrightarrow",
            "overleftarrow_": r"\overleftarrow",
            "underbrace_": r"\underbrace",
            "overbrace_": r"\overbrace",
            "underbar_": r"\underbar",
            "mathcal_": r"\mathcal",
            "mathbb_": r"\mathbb",
            "mathbf_": r"\mathbf",
            "mathrm_": r"\mathrm",
            "mathit_": r"\mathit",
            "mathsf_": r"\mathsf",
        }

    decorations = []
    while text:
        matched = False
        for prefix in decoration_prefixes:
            if text.startswith(prefix):
                decorations.append(decoration_prefixes[prefix])
                text = text[len(prefix) :]
                matched = True
                break
        if not matched:
            break

    def split_explicit_indices(raw: str) -> tuple[str, list[str], list[str]]:
        first_hi = raw.find("_hi_")
        first_low = raw.find("_low_")
        starts = [i for i in (first_hi, first_low) if i != -1]
        if not starts:
            return raw, [], []

        split_at = min(starts)
        remaining = raw[:split_at]
        suffix = raw[split_at:]

        superscripts = []
        subscripts = []
        active = None

        for piece in suffix.split("_"):
            if not piece:
                continue
            if piece == "hi":
                active = superscripts
            elif piece == "low":
                active = subscripts
            elif active is not None:
                active.append(piece)

        return remaining, superscripts, subscripts

    def split_implicit_indices(raw: str) -> tuple[str, list[str], list[str]]:
        first_underscore = raw.find("_")
        if first_underscore == -1:
            return raw, [], []

        base = raw[:first_underscore]
        tail = raw[first_underscore:]

        superscripts = []
        subscripts = []
        pos = 0
        tail_len = len(tail)

        while pos < tail_len:
            if tail[pos] != "_":
                break

            run_start = pos
            while pos < tail_len and tail[pos] == "_":
                pos += 1
            underscore_count = pos - run_start

            word_start = pos
            while pos < tail_len and tail[pos] != "_":
                pos += 1
            word = tail[word_start:pos]

            if not word:
                continue

            if underscore_count >= 2:
                superscripts.append(word)
            else:
                subscripts.append(word)

        return base, superscripts, subscripts

    def format_superscripts(items: list[str]) -> str:
        if not items:
            return ""

        prime_count = 0
        if prime_processing:
            idx = 0
            item_count = len(items)
            while idx < item_count and re.fullmatch(r"p+", items[idx]):
                prime_count += len(items[idx])
                idx += 1
            items = items[idx:]

        prime_tex = r"\prime" * prime_count
        if items:
            return prime_tex + ("," if prime_tex else "") + ",".join(items)
        return prime_tex

    text, explicit_superscripts, explicit_subscripts = split_explicit_indices(text)
    base_part, inline_superscripts, inline_subscripts = split_implicit_indices(text)

    def format_suffix(word: str) -> str:
        if word == "negative":
            return "-"
        if word == "positive":
            return "+"
        if re.fullmatch(r"m\d+", word):
            return f"-{word[1:]}"
        if re.fullmatch(r"[a-zA-Z]+", word) and len(word) > 1:
            return rf"\text{{{word}}}"
        return word

    superscripts = [
        format_suffix(word) for word in inline_superscripts + explicit_superscripts
    ]
    subscripts = [
        format_suffix(word) for word in inline_subscripts + explicit_subscripts
    ]

    if not superscripts and not subscripts and "_" not in text and infer_suffix:
        m = re.match(r"^(.*?)(\d+)$", base_part)
        if m:
            base_part = m.group(1)
            subscripts = [m.group(2)]

    base_tex = latexify_base(base_part)
    for decoration_cmd in reversed(decorations):
        base_tex = f"{decoration_cmd}{{{base_tex}}}"

    superscript_tex = format_superscripts(superscripts)

    out = base_tex
    if superscript_tex:
        out += f"^{{{superscript_tex}}}"
    if subscripts:
        out += f"_{{{','.join(subscripts)}}}"

    if is_conjugate:
        out = rf"\overline{{{out}}}"

    return out


def latexify_base(base: str) -> str:
    if not base:
        return base
    if "_" not in base and base and base[-1].isdigit():
        m = re.match(r"^(.+?)(\d+)$", base)
        if m:
            name, digits = m.groups()
            return f"{name}_{{{digits}}}"
    if base.startswith("_"):
        return r"\_" + format_latex_subscripts(base[1:])
    return format_latex_subscripts(base)


def _latex_escape_text(s):
    return (
        s.replace("\\", r"\textbackslash ")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("$", r"\$")
        .replace("^", r"\^{}")
        .replace("~", r"\~{}")
    )


def convert_to_greek(var_name):
    stem = var_name.rstrip("0123456789")
    if stem in greek_letters:
        return greek_letters[stem] + var_name[len(stem) :]
    return var_name


def format_latex_subscripts(var_name, nest_braces=False):
    """for use_latex branches"""
    if var_name[-1] == "_":
        var_name = var_name[:-1]
    if var_name == "":
        return r"\_"
    if var_name[0] == "_":
        return format_latex_subscripts(var_name=var_name[1:], nest_braces=nest_braces)
    parts = var_name.split("_")
    if len(parts) == 1:
        return convert_to_greek(var_name)
    base = convert_to_greek(parts[0])
    subscript = ", ".join(parts[1:])
    if nest_braces is True:
        return f"{{{base}_{{{subscript}}}}}"
    else:
        return f"{base}_{{{subscript}}}"


def latex_superscript(base: str, power: str) -> str:
    if power == "":
        return base
    if len(power) > 1:
        power = f"{{{power}}}"
    if "^" in base:
        return rf"\left({base}\right)^{{{power}}}"
    return f"{base}^{power}"


_dollars = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)


def clean_LaTeX(word: str, replacements: dict[str, str] | None = None) -> str:
    word = re.sub(r"\\displaystyle\s*", "", word)
    word = _dollars.sub(r"\\[\1\\]", word)
    word = _collapse_double_braces(word)
    word = _format_display_math(word)
    if replacements:
        for old, new in replacements.items():
            word = word.replace(old, new)
    return word


_strip_display_pat = re.compile(r"|".join(map(re.escape, ["$", r"\displaystyle"])))


def _strip_display_dollars(s: str | None) -> str:
    if not isinstance(s, str):
        return ""
    return _strip_display_pat.sub("", s).lstrip()


def _strip_displaystyles(s: str | None) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace(r"\displaystyle", "").lstrip()


def _strip_dollars(s: str | None) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace("$", "")


def _unwrap_math_delims(s: str) -> str:
    t = s.strip()
    if t.startswith("$$") and t.endswith("$$") and len(t) >= 4:
        return t[2:-2].strip()
    if t.startswith("$") and t.endswith("$") and len(t) >= 2:
        return t[1:-1].strip()
    return t


def _collapse_double_braces(s: str) -> str:
    out = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "{" and i + 1 < n and s[i + 1] == "{":
            depth = 0
            j = i
            last = None
            while j < n:
                c = s[j]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        last = j
                        break
                j += 1
            if last is not None and last > i + 1 and s[last - 1] == "}":
                inner = s[i + 2 : last - 1]
                inner = _collapse_double_braces(inner)
                out.append("{")
                out.append(inner)
                out.append("}")
                i = last + 1
                continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _format_display_math(s: str) -> str:
    out = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "\\" and i + 1 < n and s[i + 1] in ("[", "]"):
            if out and out[-1] != "\n":
                out.append("\n")
            out.append("\\")
            out.append(s[i + 1])
            if i + 2 < n and s[i + 2] != "\n":
                out.append("\n")
            i += 2
            continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _strip_outer_parens_plain(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        depth = 0
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    return s
        return s[1:-1].strip()
    return s


def _strip_outer_parens_latex(s: str) -> str:
    s = s.strip()
    if s.startswith(r"\left(") and s.endswith(r"\right)"):
        body = s[len(r"\left(") : -len(r"\right)")].strip()
        return body
    return _strip_outer_parens_plain(s)


def _needs_parens_plain(s: str) -> bool:
    s = s.strip()
    if not s:
        return False

    dp = db = dc = 0
    for i, ch in enumerate(s):
        if ch == "(":
            dp += 1
            continue
        if ch == ")":
            dp = max(0, dp - 1)
            continue
        if ch == "[":
            db += 1
            continue
        if ch == "]":
            db = max(0, db - 1)
            continue
        if ch == "{":
            dc += 1
            continue
        if ch == "}":
            dc = max(0, dc - 1)
            continue

        if dp or db or dc:
            continue

        if ch == "+":
            return True
        if ch == "-" and i > 0:
            return True

    return False


def _needs_parens_latex(s: str) -> bool:
    s = s.strip()
    if not s:
        return False

    i = 0
    n = len(s)
    ctx = []
    frac_wait = 0

    while i < n:
        if s.startswith(r"\frac", i):
            i += 5
            frac_wait = 2
            continue

        ch = s[i]

        if ch == "^" and i + 1 < n and s[i + 1] == "{":
            ctx.append("sup")
            i += 2
            continue
        if ch == "_" and i + 1 < n and s[i + 1] == "{":
            ctx.append("sub")
            i += 2
            continue

        if ch == "{":
            if frac_wait > 0:
                ctx.append("frac")
                frac_wait -= 1
            else:
                ctx.append("brace")
            i += 1
            continue
        if ch == "}":
            if ctx:
                ctx.pop()
            i += 1
            continue

        if (ch == "+" or ch == "-") and not ctx:
            return True

        i += 1

    return False


def _split_leading_minus(s: str):
    s = s.strip()
    if s.startswith("-"):
        return True, s[1:].strip()
    return False, s


def coeff_needs_parens_plain(s: str) -> bool:
    return _needs_parens_plain(s)


def coeff_needs_parens_latex(s: str) -> bool:
    s = s.strip()
    if not s:
        return False

    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    seen_top_word = False

    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        if ch.isspace():
            i += 1
            continue

        if ch == "\\":
            j = i + 1
            while j < n and s[j].isalpha():
                j += 1
            if paren_depth == bracket_depth == brace_depth == 0:
                seen_top_word = True
            i = j
            continue

        if ch == "(":
            paren_depth += 1
            i += 1
            continue
        if ch == ")":
            paren_depth = max(0, paren_depth - 1)
            i += 1
            continue
        if ch == "[":
            bracket_depth += 1
            i += 1
            continue
        if ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
            i += 1
            continue
        if ch == "{":
            brace_depth += 1
            i += 1
            continue
        if ch == "}":
            brace_depth = max(0, brace_depth - 1)
            i += 1
            continue

        top_level = paren_depth == bracket_depth == brace_depth == 0

        if top_level:
            if ch == "+":
                return True
            if ch == "-":
                if seen_top_word:
                    return True
                i += 1
                continue
            seen_top_word = True

        i += 1

    return False


def _process_var_label(var, bypass=False) -> str:
    if bypass:
        return var
    return _format_label_with_hi_low(str(var), infer_suffix=True)


def _verbose_labels():
    dgcvSR = get_dgcv_settings_registry()
    verbosity = dgcvSR.get("__", dict())
    return dgcvSR.get("verbose_label_printing", False) or "verbose" in verbosity


def _process_label(lbl):
    m = re.search(r"(\d+)$", lbl)
    if m and "_" not in lbl:
        lbl = lbl[: m.start(1)] + "_" + m.group(1)
    return _format_label_with_hi_low(lbl)
