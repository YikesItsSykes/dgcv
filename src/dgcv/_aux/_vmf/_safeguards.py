"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv._aux._safeguards


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
import random
import string

from .._utilities._config import (
    dgcv_warning,
    get_dgcv_settings_registry,
    get_variable_registry,
    working_namespace,
)
from .vmf import clearVar, vmf_lookup

_passkey = "".join(random.choices(string.ascii_letters + string.digits, k=16))
public_key = "".join(random.choices(string.ascii_letters + string.digits, k=8))


# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------
def get_dgcv_category(obj):
    if getattr(obj, "_dgcv_class_check", None) == _passkey:
        return getattr(obj, "_dgcv_category", None)
    return None


def query_dgcv_categories(obj, cat):
    if getattr(obj, "_dgcv_class_check", None) == _passkey:
        if isinstance(cat, (list, tuple, set, dict)):
            return any(
                c in getattr(obj, "_dgcv_categories", {get_dgcv_category(obj)})
                for c in cat
            )
        return cat in getattr(obj, "_dgcv_categories", {get_dgcv_category(obj)})


def check_dgcv_category(obj):
    return getattr(obj, "_dgcv_class_check", None) == _passkey


def create_key(prefix=None, avoid_caller_globals=False, key_length=8):
    """
    Generates a unique alphanumeric key with an optional prefix.

    Parameters
    ----------
    prefix : str, optional
        A string to prepend to the generated key. Defaults to an empty string if not provided.
    avoid_caller_globals : bool, optional
        If True, makes sure that the key does not conflict with existing keys in the caller's global namespace.

    Returns
    -------
    str
        An alphanumeric key.
    """
    if not isinstance(prefix, str):
        prefix = ""

    caller_globals = {}
    if avoid_caller_globals:
        caller_globals = working_namespace()
    while True:
        key = prefix + "".join(
            random.choices(string.ascii_letters + string.digits, k=key_length)
        )
        if not avoid_caller_globals or key not in caller_globals:
            return key


def retrieve_passkey():
    """
    Returns the internal passkey for use within dgcv functions.
    """
    return _passkey


def retrieve_public_key():
    """
    Returns the public key for use in function and variable names.
    """
    return public_key


def protected_caller_globals():
    """
    Returns a set of globally protected variable labels that should not be overwritten.
    These include standard Python built-ins, special variables, and common modules.

    Returns
    -------
    set
        A set of protected global variable names.
    """
    return {
        # Built-in functions
        "print",
        "len",
        "sum",
        "max",
        "min",
        "str",
        "int",
        "float",
        "list",
        "dict",
        "set",
        "tuple",
        "open",
        "range",
        "enumerate",
        "map",
        "filter",
        # Common modules and objects
        "math",
        "numpy",
        "sympy",
        "os",
        "sys",
        "config",
        "inspect",
        "re",
        # Special variables
        "__name__",
        "__file__",
        "__doc__",
        "__builtins__",
        "__package__",
    }


_BRANCH_NAMES = {
    "standard_variable_systems": "standard coordinate",
    "complex_variable_systems": "complex coordinate",
    "finite_algebra_systems": "finite-dimensional algebra",
    "eds_atoms": "differential form",
    "eds_coframes": "coframe",
    "eds_frames": "frame",
    "eds_jet_spaces": "jet space",
}


def _branch_of(path):
    if not isinstance(path, (tuple, list)) or not path:
        return None
    if path[0] == "eds":
        return f"eds_{path[2]}" if len(path) > 2 else "eds"
    return path[0]


def _path_of(entry):
    p = entry.get("path") if isinstance(entry, dict) else None
    return p if isinstance(p, tuple) and len(p) > 1 else None


def _owners(label, vr):
    owners = set()
    p = _path_of(vr["paths"].get(label))
    if p is not None and p[1] != label:
        owners.add(p[1])
    for system_label, meta in vr["_labels"].items():
        if system_label != label and label in meta.get("children", ()):
            owners.add(system_label)
    for system_label, info in vr.get("eds", {}).get("coframes", {}).items():
        if info.get("cousins_parent") == label:
            owners.add(system_label)
    return owners


def _cleared_by(system_label, vr, seen):
    if system_label in seen:
        return seen
    seen.add(system_label)
    meta = vr["_labels"].get(system_label, {})
    seen.update(meta.get("children", ()))
    for name, entry in vr["paths"].items():
        p = _path_of(entry)
        if p is not None and p[1] == system_label:
            seen.add(name)
    if _branch_of(meta.get("path")) == "eds_coframes":
        info = vr["eds"]["coframes"].get(system_label, {})
        for child in list(meta.get("children", ())) + [
            info.get("cousins_parent"),
            info.get("frame_label"),
        ]:
            if child is not None and child in vr["_labels"]:
                _cleared_by(child, vr, seen)
    if _branch_of(meta.get("path")) == "eds_jet_spaces":
        for child in meta.get("children", ()):
            if child in vr["_labels"]:
                _cleared_by(child, vr, seen)
    return seen


def _kind_name(system_label, vr):
    meta = vr["_labels"].get(system_label, {})
    return _BRANCH_NAMES.get(_branch_of(meta.get("path")), "registered")


def validate_label_list(basis_labels, *, kind=None, system_labels=()):
    """
    Checks a list of labels against the dgcv Variable Management Framework (its `paths`, `_labels` and
    system dictionaries; the working namespace is not consulted). A label in `system_labels` that is
    currently a system label of a branch in `kind` and belongs to no other system is left to the caller,
    which replaces it; every other registered label consults `ask_before_overwriting_objects_in_vmf`.

    Parameters
    ----------
    basis_labels : list of str
        The labels to validate.
    kind : str or iterable of str, optional
        The registry branches the caller creates (`standard_variable_systems`, `complex_variable_systems`,
        `finite_algebra_systems`, `eds_atoms`, `eds_coframes`).
    system_labels : iterable of str, optional
        The labels the caller registers as system labels.
    """
    vr = get_variable_registry()
    kinds = {kind} if isinstance(kind, str) else set(kind or ())
    replaced = set()
    for label in system_labels:
        meta = vr["_labels"].get(label)
        if (
            meta is not None
            and _branch_of(meta.get("path")) in kinds
            and not _owners(label, vr)
        ):
            replaced.add(label)
    freed = set()
    for label in replaced:
        _cleared_by(label, vr, freed)

    permitted = (
        get_dgcv_settings_registry()["ask_before_overwriting_objects_in_vmf"] is False
    )
    existing_labels = []
    to_clear = []
    detailed_message = ""
    for label in basis_labels:
        if label is None or label in freed:
            continue
        is_system = label in vr["_labels"]
        owners = _owners(label, vr)
        if not is_system and not owners:
            continue
        existing_labels.append(label)
        if not owners:
            system_type_name = _kind_name(label, vr)
            if permitted:
                detailed_message += (
                    f"\n • Label '{label}' was already assigned as the label for a {system_type_name} system.\n"
                    "    The old object (and all of its dependant relatives) was deleted from the VMF (and "
                    "the global namespace) to resolve the environment for re-assignments."
                )
                to_clear.append(label)
            else:
                detailed_message += (
                    f"\n`validate_label_list` detected '{label}' within the dgcv Variable Management Framework "
                    f"assigned as the label for a {system_type_name} system.\n"
                    f"Apply the dgcv function `clearVar('{label}')` to clear the obstructing objects."
                )
            continue
        roots = sorted(o for o in owners if not (_owners(o, vr) & owners))
        for parent_label in roots:
            system_type_name = _kind_name(parent_label, vr)
            if permitted:
                detailed_message += (
                    f"\n • Label '{label}' was already assigned as the label for an object in a {system_type_name} system.\n"
                    "    The old object (and all of its dependant relatives) was deleted from the "
                    "VMF (and the global namespace) to resolve the environment for re-assignments. "
                )
                to_clear.append(parent_label)
            else:
                detailed_message += (
                    f"\n`validate_label_list` detected '{label}' within the dgcv Variable Management Framework "
                    f"associated with the {system_type_name} system '{parent_label}'.\n Apply "
                    f"the dgcv function `clearVar('{parent_label}')` to first clear the obstructing objects."
                )

    if len(existing_labels) > 0:
        if permitted:
            warning_message = (
                f"The following labels were already registered in the dgcv Variable Management Framework: {existing_labels}. "
                "The setting from `set_dgcv_settings(ask_before_overwriting_objects_in_vmf=False)` is active in the current session, so "
                "these previously used labels have been re-assigned to new objects."
            )
        else:
            warning_message = (
                f"Warning: The following labels are already registered in the dgcv Variable Management Framework: {existing_labels}. "
                "By default, `dgcv` creator functions will not overwrite such objects. Either clear them with `clearVar` "
                "before attempting label re-assignment or set the over-riding setting "
                "`set_dgcv_settings(ask_before_overwriting_objects_in_vmf=False)`."
            )

        if detailed_message:
            warning_message += detailed_message

        if permitted:
            to_clear = list(dict.fromkeys(to_clear))
            if get_dgcv_settings_registry()["forgo_warnings"] is not True:
                dgcv_warning(
                    warning_message
                    + "\n To suppress warnings such as this, set `set_dgcv_settings(forgo_warnings=True)`."
                )
                clearVar(*to_clear)
            else:
                clearVar(*to_clear, report=False)
        else:
            raise ValueError(warning_message)


def creator_label_check_enabled():
    return (
        get_dgcv_settings_registry().get("_creator_label_collision_check", True)
        is not False
    )


def validate_creator_labels(kind, system_labels, member_labels=()):
    """
    The creator contract (own label replaces, foreign label consults) behind the versioned setting
    `_creator_label_collision_check`; see `validate_label_list`.
    """
    if not creator_label_check_enabled():
        return
    system_labels = tuple(label for label in system_labels if label is not None)
    labels = list(system_labels) + [
        label for label in member_labels if label is not None
    ]
    validate_label_list(
        list(dict.fromkeys(labels)), kind=kind, system_labels=system_labels
    )


def validate_label(label, remove_guardrails=False):
    """
    Checks if the provided variable label starts with dgcv's designated prefix for complex variables, and reformats it to a safe fallback if necessary.
    Also checks if the label is a protected global or in 'protected_variables', unless remove_guardrails is True.

    Parameters
    ----------
    label : str
        A string representing the variable label to be validated.
    remove_guardrails : bool, optional
        If True, skips the check for protected global names and 'protected_variables' (default is False).

    Returns
    -------
    str
        The reformatted label.
    """
    if not remove_guardrails:
        if label in protected_caller_globals():
            raise ValueError(
                f"dgcv recognizes label '{label}' as a protected global name and recommends not using it as a variable name. Set remove_guardrails=True to force it."
            )
        protected_vr = get_variable_registry().get("protected_variables", set())
        if label in protected_vr:
            info = vmf_lookup(label, path=True)
            system_type = info.get("type", "unregistered")
            path = info.get("path", ["", "unknown"])
            system_label = (
                path[1]
                if isinstance(path, (list, tuple)) and len(path) > 1
                else "unknown"
            )
            if system_type == "unregistered":
                raise ValueError(
                    f"The label {label} is currently protected from reassignment in the VMF"
                    "but it is not associated with any system of objects in the VMF. This is"
                    "likely a bug in `dgcv`. Please report it to dgcv maintainers."
                )
            raise ValueError(
                f"The label {label} is currently protected from reassignment in the VMF"
                f"It's associated system is: {system_label}"
                f"It's associated system type is: {system_type}"
            )

    dgcvSR = get_dgcv_settings_registry()
    pref, fallback = (
        dgcvSR.get("conjugation_prefix", "BAR"),
        dgcvSR.get("fallback_conjugate_prefix", "anti_"),
    )
    if label.startswith(pref):
        reformatted_label = fallback + label[len(pref) :]
        dgcv_warning(
            f"Label '{label}' starts with {pref}, which is reserved by the `conjugation_prefix` "
            "value currently set in dgcv settings. It has been automatically reformatted to "
            f"'{reformatted_label}'."
        )
    else:
        reformatted_label = label

    return reformatted_label


def unique_label(
    label,
    remove_guardrails=False,
    version_prefix="v",
    tex_v_prefix="v.",
    tex_label=None,
    protected=set(),
):
    """
    Validate a label and, if it already exists in the VMF, append
    a minimal version suffix to make it unique.


    Parameters
    ----------
    label : str
    remove_guardrails : bool, optional
        If True, skip guardrail checks inside validate_label (still picks a unique name).
    version_prefix : str, optional
        Suffix to prepend to the numeric version (default "v").
    tex_v_prefix : str, optional
        Suffix to prepend to the numeric version in LaTeX (default "v.").
    tex_label : str | None, optional
        LaTeX companion string. When a version is added to the plain label,
        the same version suffix is appended here.

    Returns
    -------
    str | tuple[str, str]
        If tex_label is None, returns the processed label string.
        Otherwise returns a pair (label, latex_label).
    """
    base = validate_label(label, remove_guardrails=remove_guardrails)

    vr = get_variable_registry()
    taken = set()

    taken.update(working_namespace().keys())

    labels_index = vr.get("_labels", {})
    taken.update(labels_index.keys())
    for meta in labels_index.values():
        children = meta.get("children", set())
        if isinstance(children, (set, list, tuple)):
            taken.update(children)

    protected_vr = vr.get("protected_variables", set())
    if isinstance(protected_vr, (set, list, tuple)):
        taken.update(protected_vr)
    if protected:
        if isinstance(protected, str):
            taken.add(protected)
        else:
            for name in protected:
                if isinstance(name, str):
                    taken.add(name)

    def _augment_tex_label(tl: str, version_number: int) -> str:
        suffix = f"_{{\\operatorname{{{tex_v_prefix}}}{version_number}}}"
        if tl is None:
            return None
        needs_wrap = False
        if tl.startswith("\\left(") and tl.endswith(")"):
            needs_wrap = False
        elif tl.startswith("(") and tl.endswith(")"):
            needs_wrap = False
        else:
            if any(ch in tl for ch in (" ", "\\", "{", "}", "+", "-", "*", "/")):
                needs_wrap = True
        core = f"\\left({tl}\\right)" if needs_wrap else tl
        return f"{core}{suffix}"

    if base not in taken:
        if tex_label is None:
            return base
        return base, tex_label

    n = 1
    while True:
        candidate = f"{base}_{version_prefix}{n}"
        if candidate not in taken:
            if tex_label is None:
                return candidate
            return candidate, _augment_tex_label(tex_label, n)
        n += 1
