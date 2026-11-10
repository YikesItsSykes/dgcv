import numbers
import re

from ._string_processing import _format_label_with_hi_low, label_tex_parts
from ..._utilities._config import get_dgcv_settings_registry


def derivative_index_tex(head, own, derivatives, *, conjugated=False, sep=";", sup=""):
    tail = ",".join(derivatives)
    top = f"^{{{sup}}}" if sup else ""
    if not own:
        body = rf"\overline{{{head}{top}}}" if conjugated else f"{head}{top}"
        return f"{body}_{{{tail}}}" if tail else body
    if not tail:
        body = f"{head}{top}_{{{own}}}"
        return rf"\overline{{{body}}}" if conjugated else body
    if not conjugated:
        return f"{head}{top}_{{{own}{sep}{tail}}}"
    outer_top = f"^{{\\vphantom{{{sup}}}}}" if sup else ""
    return (
        rf"\overline{{{head}{top}_{{{own}\vphantom{{{sep}{tail}}}}}}}"
        rf"{{}}{outer_top}_{{\vphantom{{{own}}}{sep}{tail}}}"
    )


def stem_derivative_tex(stem, derivatives, conjugated=False, prefix=""):
    head, sup, own, sep = stem
    return derivative_index_tex(
        prefix + head, own, derivatives, conjugated=conjugated, sep=sep, sup=sup
    )


def label_derivative_parts(label):
    head, sup, digits, subscripts = label_tex_parts(label)
    if digits:
        own = digits + (";" + ",".join(subscripts) if subscripts else "")
        return head, sup, own, "," if subscripts else ";"
    return head, sup, ",".join(subscripts), ","


def conjugation_prefix():
    return get_dgcv_settings_registry().get("conjugation_prefix", "_c_")


def ext_der_latex(string, order):
    if order == 1:
        return f"D\\left({string}\\right)"
    if order is not None and order > 1:
        return f"D^{order}\\left({string}\\right)"
    return string


def split_conjugation_prefix(label):
    pref = conjugation_prefix()
    if label and label[0 : len(pref)] == pref:
        return label[len(pref) :], True
    return label, False


def ext_der_str(string, order):
    if isinstance(order, numbers.Integral) and order != 0:
        return f"{string}_extD_{order}"
    return string


def bar_label_latex(label):
    base, conjugated = split_conjugation_prefix(label)
    match = re.search(r"(\d+)$", base)
    if match and "_" not in base:
        base = base[: match.start(1)] + "__" + match.group(1)
    tex = _format_label_with_hi_low(base)
    if not conjugated:
        return tex
    head, sup, digits, subscripts = label_tex_parts(base)
    if digits and subscripts:
        return derivative_index_tex(head, digits, subscripts, conjugated=True, sup=sup)
    cuts = [i for i in (tex.find("^"), tex.find("_")) if i != -1]
    if cuts:
        idx = min(cuts)
        return f"\\overline{{{tex[:idx]}}}{tex[idx:]}"
    return f"\\overline{{{tex}}}"


def ordinal_latex(count):
    if count == 0:
        return r"0^\text{th}"
    if count == 1:
        return r"1^\text{st}"
    if count == 2:
        return r"2^\text{nd}"
    if count == 3:
        return r"3^\text{rd}"
    return str(count) + r"^\text{th}"


def ext_der_repr(string, order):
    if isinstance(order, numbers.Integral) and order > 0:
        return f"extDer({string},order = {order})"
    return string
