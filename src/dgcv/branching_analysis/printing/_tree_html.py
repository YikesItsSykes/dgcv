from __future__ import annotations

import html
import uuid

from .._traversal import count_leaves
from ._conditions_latex import condition_strings, represents_strings
from ._styles import html_style


def grouped_branch_conditions(node, latex, filter_conditions=None):
    params = node.system_parameters
    vari = node.system_variables
    represents = represents_strings(node, latex, filter_conditions)
    extra = {"represents": represents} if represents else {}
    return {
        "parameters": condition_strings(
            node, {"d"}, latex, filter_conditions, key_filter=lambda e: e in params
        ),
        "variables": condition_strings(
            node, {"d"}, latex, filter_conditions, key_filter=lambda e: e in vari
        ),
        "corollaries": condition_strings(
            node,
            {"c"},
            latex,
            filter_conditions,
            key_filter=lambda e: e in params or e in vari,
        ),
        **extra,
    }


def build_verbose_tree(node, latex, filter_conditions=None, evaluate=True, sep=", "):
    from .._tasks import leaf_annotation

    branch_conditions = grouped_branch_conditions(
        node, latex=latex, filter_conditions=filter_conditions
    )
    out = {node.label: {"branch_conditions": branch_conditions, "descendants": {}}}
    if getattr(node, "note", None) is not None:
        out[node.label]["note"] = str(node.note)
    if node._subcases:
        for _, subtree in node._subcases.items():
            out[node.label]["descendants"] |= build_verbose_tree(
                subtree,
                latex,
                filter_conditions=filter_conditions,
                evaluate=evaluate,
                sep=sep,
            )
    else:
        out[node.label]["descendants"] = leaf_annotation(node, evaluate, sep=sep)
    return out


def verbose_tree(node, filter_conditions=None, evaluate=True, sep=", "):
    return build_verbose_tree(
        node, False, filter_conditions=filter_conditions, evaluate=evaluate, sep=sep
    )


def latex_verbose_tree(node, filter_conditions=None, evaluate=True, sep=", "):
    return build_verbose_tree(
        node, True, filter_conditions=filter_conditions, evaluate=evaluate, sep=sep
    )


def fold_markup(
    container_id, counter, depth, collapse_depth, count=None, collapse_threshold=None
):
    fold_id = f"{container_id}-f{counter[0]}"
    counter[0] += 1
    collapsed = isinstance(collapse_depth, int) and depth >= collapse_depth
    if not collapsed and isinstance(collapse_threshold, int) and count is not None:
        collapsed = count >= collapse_threshold
    checked = " checked" if collapsed else ""
    return (
        fold_id,
        f'<input type="checkbox" class="tree-toggle" id="{fold_id}"{checked}>',
    )


def fold_count_html(count):
    plural = "branch" if count == 1 else "branches"
    return f'<div class="fold-count">{count} {plural}</div>'


def to_html_tree(
    data,
    path="",
    is_root=True,
    root_label=None,
    container_id=None,
    slim=False,
    foldable=True,
    collapse_depth=None,
    collapse_threshold=6,
    depth=0,
    counter=None,
):
    if not isinstance(data, dict):
        return ""

    if is_root and container_id is None:
        container_id = f"tree-{uuid.uuid4().hex[:8]}"

    if counter is None:
        counter = [0]

    folds = bool(foldable) and not slim

    res = (
        f'<div id="{container_id}" class="tree-container"><ul>'
        if is_root
        else '<ul class="children-ul">'
    )

    if not isinstance(root_label, str):
        root_label = "root"
    if is_root and root_label not in data and len(data) == 1:
        root_label = next(iter(data))

    if is_root and root_label in data:
        root_node = data[root_label] if isinstance(data[root_label], dict) else {}
        root_note = root_node.get("note")
        note_html = (
            f'<div class="note-msg"><div class="cell-content">'
            f'{html.escape("Note: " + str(root_note))}</div></div>'
            if root_note is not None
            else ""
        )
        root_has_children = isinstance(root_node.get("descendants"), dict)
        chip_html = '<span class="cond-chip node-chip">root</span>' if not slim else ""
        root_text = (
            '<span class="node-phantom">root</span>'
            if root_label == "root"
            else html.escape(root_label)
        )
        res += "<li>"
        if folds and root_has_children:
            count = count_leaves(root_node)
            fold_id, toggle = fold_markup(
                container_id, counter, depth, collapse_depth
            )
            res += toggle
            label_html = (
                f'<label class="node-label" style="min-width: 10px;" '
                f'for="{fold_id}">{chip_html}{root_text}</label>'
            )
            count_html = fold_count_html(count)
        else:
            label_html = (
                f'<div class="node-label" style="min-width: 10px;">'
                f"{chip_html}{root_text}</div>"
            )
            count_html = ""
        res += (
            f'<div class="compound-node root-wrapper">'
            f"{label_html}{count_html}{note_html}</div>"
        )
        if root_has_children:
            res += to_html_tree(
                data[root_label],
                "",
                False,
                container_id=container_id,
                slim=slim,
                foldable=foldable,
                collapse_depth=collapse_depth,
                collapse_threshold=collapse_threshold,
                depth=depth + 1,
                counter=counter,
            )
        res += "</li>"
    else:
        items = list(data.get("descendants", {}).items())
        for key, value in items:
            clean_key = str(key).strip("_")
            current_path = f"{path}.{clean_key}" if path else clean_key

            bc = value.get("branch_conditions", {}) if isinstance(value, dict) else {}
            if isinstance(bc, dict):
                groups = [
                    ("parameters", bc.get("parameters", [])),
                    ("variables", bc.get("variables", [])),
                    ("corollaries", bc.get("corollaries", [])),
                    ("conditions", bc.get("represents", [])),
                ]
            elif isinstance(bc, str):
                groups = [(None, [bc.strip("[]'")] if bc else [])]
            else:
                groups = [(None, list(bc))]

            descendants = value.get("descendants") if isinstance(value, dict) else value
            is_dict = isinstance(descendants, dict)

            res += "<li>"
            chip_html = (
                '<span class="cond-chip node-chip">case</span>' if not slim else ""
            )
            if folds and is_dict:
                count = count_leaves(value)
                fold_id, toggle = fold_markup(
                    container_id,
                    counter,
                    depth,
                    collapse_depth,
                    count=count,
                    collapse_threshold=collapse_threshold,
                )
                res += toggle
                label_html = (
                    f'<label class="node-label" for="{fold_id}">'
                    f"{chip_html}{html.escape(current_path)}</label>"
                )
            else:
                label_html = (
                    f'<div class="node-label">'
                    f"{chip_html}{html.escape(current_path)}</div>"
                )
            res += '<div class="compound-node">'
            res += label_html

            inner = ""
            for chip_label, group_items in groups:
                if not group_items:
                    continue
                body = ",\u00a0  ".join(html.escape(str(c)) for c in group_items)
                chip = (
                    f'<span class="cond-chip">{html.escape(chip_label)}</span>'
                    if chip_label
                    else ""
                )
                inner += (
                    f'<div class="cond-compartment">{chip}'
                    f'<div class="cond-content">{body}</div></div>'
                )
            if not inner:
                inner = (
                    '<div class="cond-compartment">'
                    '<span class="cond-chip">conditions</span>'
                    '<div class="cond-content">None</div></div>'
                )
            res += f'<div class="cond-box">{inner}</div>'

            if not is_dict and str(descendants):
                sampled = isinstance(bc, dict) and bool(bc.get("represents"))
                sample_chip = (
                    '<span class="cond-chip sample-chip">via numeric<br>sample</span>'
                    if sampled and not slim
                    else ""
                )
                res += (
                    f'<div class="complete-msg">{sample_chip}'
                    f'<div class="cell-content">'
                    f"{html.escape(str(descendants))}</div></div>"
                )

            if folds and is_dict:
                res += fold_count_html(count)

            note = value.get("note") if isinstance(value, dict) else None
            if note is not None:
                res += (
                    f'<div class="note-msg"><div class="cell-content">'
                    f'{html.escape("Note: " + str(note))}</div></div>'
                )

            res += "</div>"
            if is_dict:
                res += to_html_tree(
                    value,
                    current_path,
                    False,
                    container_id=container_id,
                    slim=slim,
                    foldable=foldable,
                    collapse_depth=collapse_depth,
                    collapse_threshold=collapse_threshold,
                    depth=depth + 1,
                    counter=counter,
                )
            res += "</li>"

    res += "</ul>"
    return (res + "</div>") if is_root else res


def full_tree_html(
    data,
    theme=None,
    root_label=None,
    slim=False,
    foldable=True,
    collapse_depth=None,
    collapse_threshold=6,
):

    cid = f"tree-{uuid.uuid4().hex[:8]}"
    styles = html_style(theme=theme, container_id=cid, slim=slim)
    tree_html = to_html_tree(
        data,
        is_root=True,
        root_label=root_label,
        container_id=cid,
        slim=slim,
        foldable=foldable,
        collapse_depth=collapse_depth,
        collapse_threshold=collapse_threshold,
    )

    return styles + tree_html
