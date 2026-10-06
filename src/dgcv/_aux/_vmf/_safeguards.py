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


def validate_label_list(basis_labels):
    """
    Validates a list of basis labels by checking if they are already present in active namespace.

    Parameters
    ----------
    basis_labels : list of str
        The list of basis labels to validate.
    """
    existing_labels = []
    to_clear = []
    detailed_message = ""
    if get_dgcv_settings_registry()["ask_before_overwriting_objects_in_vmf"] is False:
        overwritePermissionGranted = True
    else:
        overwritePermissionGranted = False

    namespace = working_namespace()
    variable_registry = get_variable_registry()
    for label in basis_labels:
        if label in namespace:
            existing_labels.append(label)
            for system_type, sub_type, system_type_name, family_names_address in [
                (
                    "standard_variable_systems",
                    "",
                    "standard coordinate",
                    "variable_relatives",
                ),
                (
                    "complex_variable_systems",
                    "",
                    "complex coordinate",
                    "variable_relatives",
                ),
                (
                    "finite_algebra_systems",
                    "",
                    "finite-dimensional algebra",
                    "family_names",
                ),
                ("eds", "atoms", "atomic differential forms", "family_relatives"),
                ("eds", "coframes", "exterior differential", "children"),
            ]:
                if system_type in variable_registry:
                    if sub_type != "" and sub_type in variable_registry[system_type]:
                        innerVR = variable_registry[system_type][sub_type]
                    else:
                        innerVR = variable_registry[system_type]
                    if label in innerVR:
                        if overwritePermissionGranted is True:
                            detailed_message += (
                                f"\n •Label'{label}' was already assigned as the label for a {system_type_name} system.\n"
                                rf"    The old object (and all of its dependant relatives) was deleted from the VMF (and "
                                f"the global namespace) to resolve the environment for re-assignments."
                            )
                            to_clear.append(label)
                        else:
                            detailed_message += (
                                f"\n`validate_basis_labels` detected '{label}' within the dgcv Variable Management Framework "
                                f"assigned as the label for a {system_type_name} system.\n"
                                f"Apply the dgcv function `clearVar('{label}')` to clear the obstructing objects."
                            )
                    else:
                        for parent_label, parent_data in innerVR.items():
                            if (
                                family_names_address in parent_data
                                and label in parent_data[family_names_address]
                            ):
                                if overwritePermissionGranted is True:
                                    detailed_message += (
                                        f"\n • Label '{label}' was already assigned as the label for an object in a {system_type_name} system.\n"
                                        r"    The old object (and all of its dependant relatives) was deleted from the "
                                        f"VMF (and the global namespace) to resolve the environment for re-assignments. "
                                    )
                                    to_clear.append(parent_label)
                                else:
                                    detailed_message += (
                                        f"\n`validate_basis_labels` detected '{label}' within the dgcv Variable Management Framework "
                                        f"associated with the {system_type_name} system '{parent_label}'.\n Apply"
                                        f"the dgcv function `clearVar('{parent_label}')` to first clear the obstructing objects."
                                    )

    if len(existing_labels) > 0:
        if overwritePermissionGranted is True:
            warning_message = (
                f"The following basis labels were already defined in the current namespace: {existing_labels}. "
                "The setting from `set_dgcv_settings(ask_before_overwriting_objects_in_vmf=False)` is active in the current session, so "
                "these previously used labels have been re-assigned to new objects."
            )
        else:
            warning_message = (
                f"Warning: The following basis labels are already defined in the current namespace: {existing_labels}. "
                "By default, `dgcv` creator functions will not overwrite such objects.  Either clear them from the "
                " global namespace before attempting label re-assignment or set the over-riding setting "
                "`set_dgcv_settings(ask_before_overwriting_objects_in_vmf=True)`."
            )

        if detailed_message:
            warning_message += detailed_message

        if overwritePermissionGranted is True:
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
