"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv._aux.vmf

This module defines defines functions for interacting with the Variable Management Framework (VMF),
which is dgcv's system for managing object creation and labeling. It additionaly contains functions
for internal use by the VMF. The intended public functions include the following

Functions for listing and clearing objects in the VMF:
    - clear_vmf(): primary multipurpose tool for removing objects from the VMF.
    - listVar(): Lists the "parent names" of objects currently tracked by the dgcv VMF.
    - clearVar(): Clears the variables from the dgcv registry and deletes them from caller's globals().

Functions for summarizing the state of the VMF:
    - vmf_summary(): Takes a snapshot of the current dgcv VMF and reports a summary in an html table or plain text report.


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
from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence

from .._backends._display_engine import is_rich_displaying_available
from .._backends._engine import notify_vmf_cleared
from .._backends._types_and_constants import is_atomic
from .._utilities._config import (
    dgcv_warning,
    dgcvDeprecationWarning,
    get_dgcv_settings_registry,
    get_variable_registry,
    greek_letters,
    latex_in_html,
    working_namespace,
)
from .._utilities._styles import get_style
from ..printing._tables import build_plain_table

__all__ = ["clear_vmf", "listVar", "clearVar", "vmf_lookup", "vmf_summary"]


# -----------------------------------------------------------------------------
# clearing and listing
# -----------------------------------------------------------------------------
def listVar(
    standard_only=False,
    complex_only=False,
    algebras_only=False,
    zeroForms_only=False,
    coframes_only=False,
    temporary_only=False,
    obscure_only=False,
    protected_only=False,
):
    """
    This function lists all parent labels for objects tracked within the dgcv Variable Management Framework (VMF). In particular strings that are keys in dgcv's internal `standard_variable_systems`, `complex_variable_systems`, 'finite_algebra_systems', 'eps' dictionaries, etc. It also accepts optional keywords to filter the results, showing only temporary, protected, or "obscure" object system labels.

    Parameters
    ----------
    standard_only : bool, optional
        If True, only standard variable system labels will be listed.
    complex_only : bool, optional
        If True, only complex variable system labels will be listed.
    algebras_only : bool, optional
        If True, only finite algebra system labels will be listed.
    zeroForms_only : bool, optional
        If True, only zeroFormAtom system labels will be listed.
    coframes_only : bool, optional
        If True, only coframe system labels will be listed.
    temporary_only : bool, optional
        If True, only variable system labels marked as temporary will be listed.
    protected_only : bool, optional
        If True, only variable system labels marked as protected will be listed.

    Returns
    -------
    list
        A list of object system labels matching the provided filters.

    Notes
    -----
    - If no filters are specified, the function returns all labels tracked in the VMF.
    - If multiple filters are specified, the function combines them, displaying
      labels that meet any of the selected criteria.
    """
    variable_registry = get_variable_registry()

    # Collect all labels
    standard_labels = set(variable_registry["standard_variable_systems"].keys())
    complex_labels = set(variable_registry["complex_variable_systems"].keys())
    algebra_labels = set(variable_registry["finite_algebra_systems"].keys())
    zeroForm_labels = set(
        variable_registry["eds"]["atoms"].keys()
    )  # New zeroFormAtom labels
    coframe_labels = set(variable_registry["eds"]["coframes"].keys())

    selected_labels = set()
    if standard_only:
        selected_labels |= standard_labels
    if complex_only:
        selected_labels |= complex_labels
    if algebras_only:
        selected_labels |= algebra_labels
    if zeroForms_only:
        selected_labels |= zeroForm_labels
    if coframes_only:
        selected_labels |= coframe_labels

    all_labels = (
        selected_labels
        if selected_labels
        else standard_labels
        | complex_labels
        | algebra_labels
        | zeroForm_labels
        | coframe_labels
    )

    # Apply additional property filters
    if temporary_only:
        all_labels = all_labels & variable_registry["temporary_variables"]
    if obscure_only:
        all_labels = all_labels & variable_registry["obscure_variables"]
    if protected_only:
        all_labels = all_labels & variable_registry["protected_variables"]

    # Return the filtered labels list
    return list(all_labels)


def _clearVar_single(label):
    """
    Helper function that clears a single variable system
    from the dgcv variable management framework. It returns
    a tuple (system_type, label) indicating what was cleared, intended
    for compilation into a printout.
    """
    parent_path = vmf_lookup(label, path=True).get("path", None)
    if parent_path:
        label = parent_path[1]
    else:
        return
    registry = get_variable_registry()
    global_vars = working_namespace()
    cleared_info = None

    paths = registry.get("paths", {})
    std = registry.get("standard_variable_systems", {})

    system_label = None
    if label in std:
        system_label = label
    else:
        entry = paths.get(label)
        p = entry.get("path") if isinstance(entry, dict) else None
        if isinstance(p, tuple) and len(p) >= 2 and p[0] == "standard_variable_systems":
            system_label = p[1]

    if system_label is not None and system_label in std:
        system_dict = std[system_label]
        family_names = system_dict.get("family_names", ())
        if isinstance(family_names, str):
            family_names = (family_names,)
        else:
            family_names = tuple(family_names) if family_names is not None else ()

        to_pop = set(family_names)
        to_pop.add(system_label)

        if system_dict.get("differential_system"):
            for v in family_names:
                to_pop.add(f"D_{v}")
                to_pop.add(f"d_{v}")
            to_pop.add(f"D_{system_label}")
            to_pop.add(f"d_{system_label}")

        for name in to_pop:
            global_vars.pop(name, None)
            paths.pop(name, None)

        if system_dict.get("tempVar"):
            registry["temporary_variables"].discard(system_label)
        if system_dict.get("obsVar"):
            registry["obscure_variables"].discard(system_label)

        std.pop(system_label, None)
        registry.get("_labels", {}).pop(system_label, None)

        cleared_info = ("standard", system_label)
        return cleared_info

    if label not in registry.get("_labels", {}):
        return None

    path = registry["_labels"][label]["path"]
    branch = path[0]

    if branch == "standard_variable_systems":
        system_dict = registry[branch][label]
        family_names = system_dict["family_names"]
        if isinstance(family_names, str):
            family_names = (family_names,)
        for var in family_names:
            global_vars.pop(var, None)
            paths.pop(var, None)
        global_vars.pop(label, None)
        paths.pop(label, None)
        if system_dict.get("differential_system"):
            for var in family_names:
                global_vars.pop(f"D_{var}", None)
                global_vars.pop(f"d_{var}", None)
                paths.pop(f"D_{var}", None)
                paths.pop(f"d_{var}", None)
            global_vars.pop(f"D_{label}", None)
            global_vars.pop(f"d_{label}", None)
            paths.pop(f"D_{label}", None)
            paths.pop(f"d_{label}", None)
        if system_dict.get("tempVar"):
            registry["temporary_variables"].discard(label)
        if system_dict.get("obsVar"):
            registry["obscure_variables"].discard(label)
        del registry[branch][label]
        cleared_info = ("standard", label)

    elif branch == "complex_variable_systems":
        system_dict = registry[branch][label]
        family_houses = system_dict["family_houses"]
        real_parent, imag_parent = family_houses[-2], family_houses[-1]
        registry["protected_variables"].discard(real_parent)
        registry["protected_variables"].discard(imag_parent)
        for house in family_houses:
            global_vars.pop(house, None)
            paths.pop(house, None)
        variable_relatives = system_dict["variable_relatives"]
        for var_label, var_data in variable_relatives.items():
            global_vars.pop(var_label, None)
            paths.pop(var_label, None)
            if var_data.get("DFClass"):
                global_vars.pop(f"D_{var_label}", None)
                paths.pop(f"D_{var_label}", None)
            if var_data.get("VFClass"):
                global_vars.pop(f"d_{var_label}", None)
                paths.pop(f"d_{var_label}", None)
        conv = registry["conversion_dictionaries"]

        for var_label, var_data in variable_relatives.items():
            pos = var_data.get("complex_positioning")

            if pos == "holomorphic":
                conv["conjugation"].pop(var_label, None)
                conv["holToReal"].pop(var_label, None)
                conv["symToReal"].pop(var_label, None)
                conv["real_part"].pop(var_label, None)
                conv["im_part"].pop(var_label, None)

            elif pos == "antiholomorphic":
                conv["conjugation"].pop(var_label, None)
                conv["symToHol"].pop(var_label, None)
                conv["symToReal"].pop(var_label, None)
                conv["real_part"].pop(var_label, None)
                conv["im_part"].pop(var_label, None)

            elif pos in ("real", "imaginary"):
                conv["realToHol"].pop(var_label, None)
                conv["realToSym"].pop(var_label, None)
                conv["find_parents"].pop(var_label, None)
        registry["temporary_variables"].discard(label)
        del registry[branch][label]
        paths.pop(label, None)
        cleared_info = ("complex", label)

    elif branch == "finite_algebra_systems":
        system_dict = registry[branch][label]
        family_names = system_dict.get("family_names", ())
        for member in family_names:
            global_vars.pop(member, None)
            paths.pop(member, None)
        global_vars.pop(label, None)
        paths.pop(label, None)
        del registry[branch][label]
        cleared_info = ("algebra", label)

    elif branch == "eds" and path[2] == "atoms":
        system_dict = registry["eds"]["atoms"][label]
        family_names = system_dict["family_names"]
        if isinstance(family_names, str):
            family_names = (family_names,)
        for var in family_names:
            global_vars.pop(var, None)
            paths.pop(var, None)
        for var in system_dict.get("family_relatives", {}):
            global_vars.pop(var, None)
            paths.pop(var, None)
        global_vars.pop(label, None)
        from ..printing.printing._eds import conjugation_prefix

        global_vars.pop(f"{conjugation_prefix()}{label}", None)
        paths.pop(label, None)
        if system_dict.get("tempVar"):
            registry["temporary_variables"].discard(label)
        del registry["eds"]["atoms"][label]
        cleared_info = ("DFAtom", label)

    elif branch == "eds" and path[2] == "coframes":
        coframe_info = registry["eds"]["coframes"][label]
        cousins_parent = coframe_info.get("cousins_parent")
        children = tuple(coframe_info.get("children", ()) or ())
        global_vars.pop(label, None)
        paths.pop(label, None)
        del registry["eds"]["coframes"][label]
        cleared_info = ("coframe", (label, cousins_parent, children))

    registry["_labels"].pop(label, None)

    return cleared_info


def clearVar(*labels, report=True):
    cleared_standard = []
    cleared_complex = []
    cleared_algebras = []
    cleared_diffFormAtoms = []
    cleared_coframes = []

    seen = set()

    for label in labels:
        info = _clearVar_single(label)
        if not info:
            continue

        system_type, cleared_label = info

        if system_type == "standard":
            if cleared_label not in seen:
                seen.add(cleared_label)
                cleared_standard.append(cleared_label)

        elif system_type == "complex":
            if cleared_label not in seen:
                seen.add(cleared_label)
                cleared_complex.append(cleared_label)

        elif system_type == "algebra":
            if cleared_label not in seen:
                seen.add(cleared_label)
                cleared_algebras.append(cleared_label)

        elif system_type == "DFAtom":
            if cleared_label not in seen:
                seen.add(cleared_label)
                cleared_diffFormAtoms.append(cleared_label)

        elif system_type == "coframe":
            coframe_label, cousins_system_label, children = cleared_label
            if ("coframe", coframe_label) not in seen:
                seen.add(("coframe", coframe_label))
                cleared_coframes.append((coframe_label, cousins_system_label))
                clearVar(*children, report=False)
                if cousins_system_label is not None:
                    clearVar(cousins_system_label, report=False)

    if report:
        if cleared_standard:
            print(
                f"Cleared standard variable systems from the dgcv variable management framework: {', '.join(cleared_standard)}"
            )
        if cleared_complex:
            print(
                f"Cleared complex variable systems from the dgcv variable management framework: {', '.join(cleared_complex)}"
            )
        if cleared_algebras:
            print(
                f"Cleared finite algebra systems from the dgcv variable management framework: {', '.join(cleared_algebras)}"
            )
        if cleared_diffFormAtoms:
            print(
                f"Cleared differential form systems from the dgcv variable management framework: {', '.join(cleared_diffFormAtoms)}"
            )
        if cleared_coframes:
            for cf_label, cp_label in cleared_coframes:
                if cp_label is None:
                    print(f"Cleared coframe '{cf_label}' along with its 1-forms")
                else:
                    print(
                        f"Cleared coframe '{cf_label}' along with its 1-forms and the zero form atom system '{cp_label}'"
                    )
    if seen:
        notify_vmf_cleared()


def clear_vmf(
    objects_to_clear=None,
    categories_to_clear: Sequence | None = None,
    report: bool = True,
):
    """
    Clear objects registered in dgcv's Variable Management Framework (VMF).
    Parameters specify what to clear, and if no parameters are set it will
    default to clear everything.


    Parameters
    ----------
    objects_to_clear : object or iterable of objects, optional
        Specific VMF-tracked objects to remove. May be a single object
        or an iterable of objects. Any object currently registered in
        the VMF is valid. If an object is linked to a system of of
        VMF-tracked object, then the full system will be cleared.

    categories_to_clear : sequence of str, optional
        One or more category labels specifying which classes of VMF
        objects to clear. If None, defaults to ("all",).

        Valid category strings are:
            - "all"
            - "standard coor"
            - "complex coor"
            - "algebras"
            - "coframes"
            - "zeroForms"
            - "temporary"
            - "obscure"
            - "protected"

    report : bool, default True
        If True, prints a summary of cleared objects.

    Returns
    -------
    None
        Removes the selected objects from the VMF and active namespace.
    """
    if categories_to_clear is None and objects_to_clear is None:
        categories_to_clear = ("all",)

    cats = set(categories_to_clear)

    if "all" in cats:
        objects = listVar()
    else:
        objects = listVar(
            standard_only=("standard coor" in cats),
            complex_only=("complex coor" in cats),
            algebras_only=("algebras" in cats),
            zeroForms_only=("zeroForms" in cats),
            coframes_only=("coframes" in cats),
            temporary_only=("temporary" in cats),
            obscure_only=("obscure" in cats),
            protected_only=("protected" in cats),
        )

    if objects_to_clear is not None:
        if isinstance(objects_to_clear, Iterable) and not isinstance(
            objects_to_clear, (str, bytes)
        ):
            objects += list(objects_to_clear)
        else:
            objects.append(objects_to_clear)

    return clearVar(*objects, report=report)


# -----------------------------------------------------------------------------
# look up
# -----------------------------------------------------------------------------


def _eds_lookup(
    registry,
    p,
    kind,
    obj,
    path,
    relatives,
    flattened_relatives,
    system_index,
    differential_system,
):
    system_label = p[1]
    section = p[2]
    name = p[3] if len(p) > 3 else None
    eds = registry.get("eds", {})

    if section == "coframes":
        info = eds.get("coframes", {}).get(system_label, {})
        coframe = info.get("coframe")
        out = {"type": "coframe", "sub_type": "coframe"}
        if path:
            out["path"] = ("eds", system_label, "coframes")
        if relatives:
            forms = tuple(getattr(coframe, "forms", ()) or ())
            out["relatives"] = {
                "system_label": system_label,
                "forms": forms,
                "structure_coefficients": tuple(info.get("cousins_vals") or ()),
                "structure_coefficient_system": info.get("cousins_parent"),
                "children": tuple(info.get("children") or ()),
            }
            if flattened_relatives:
                out["flattened_relatives"] = forms
        if system_index:
            out["system_index"] = None
        if differential_system:
            out["differential_system"] = None
        return out

    info = eds.get("atoms", {}).get(system_label, {})
    base_kind = (
        "differential_form"
        if str(kind).startswith("differential_form")
        else "zero_form"
    )
    out = {"type": base_kind}
    if name is None:
        out["sub_type"] = "system"
    elif info.get("real"):
        out["sub_type"] = "real"
    elif str(kind).endswith("_conjugate"):
        out["sub_type"] = "conjugate"
    else:
        out["sub_type"] = "primary"
    if path:
        out["path"] = tuple(p)
    if relatives:
        family_values = tuple(info.get("family_values") or ())
        pair = info.get("family_relatives", {}).get(name) if name else None
        conjugate = None
        if pair is not None:
            conjugate = pair[1] if out["sub_type"] == "primary" else pair[0]
        coframes = eds.get("coframes", {})
        out["relatives"] = {
            "system_label": system_label,
            "family": family_values,
            "conjugates": tuple(info.get("conjugates", {}).values()),
            "conjugate": conjugate,
            "coframe": info.get("primary_coframe"),
            "is_structure_coefficient": any(
                cf.get("cousins_parent") == system_label for cf in coframes.values()
            ),
            "degree": info.get("degree"),
        }
        if flattened_relatives:
            out["flattened_relatives"] = family_values + tuple(
                info.get("conjugates", {}).values()
            )
    if system_index:
        family_names = tuple(info.get("family_names") or ())
        out["system_index"] = family_names.index(name) if name in family_names else None
    if differential_system:
        order = getattr(obj, "differential_order", 0) if obj is not None else 0
        if order:
            out["differential_system"] = {
                "coframe": getattr(obj, "coframe", None),
                "order": order,
            }
        else:
            out["differential_system"] = None
    return out


def vmf_lookup(
    obj: any,
    *,
    path: bool = False,
    relatives: bool = False,
    flattened_relatives: bool = False,
    system_index: bool = False,
    differential_system: bool = False,
) -> dict:
    """
    Query the Variable Management Framework (VMF) for metadata associated
    with an object.

    Parameters
    ----------
    obj : any
        Typically, an object tracked by the VMF, but untracked objects
        are accepted too.

    path : bool, default False
        If True, include the registry path identifying where the object
        is stored within the VMF hierarchy.

    relatives : bool, default False
        If True, include structured information about related objects
        (e.g., coordinate families, associated systems, etc.).

    flattened_relatives : bool, default False
        If True, return related objects as a flattened tuple rather than
        a structured container. Implies `relatives=True`.

    system_index : bool, default False
        If True, include index information for objects belonging to
        indexed systems.

    differential_system : bool, default False
        If True, return metadata specific to differential system
        registrations.

    Returns
    -------
    dict
        A dictionary containing registration metadata for the object.
        If the object is not registered, returns an indicator reflecting
        its status.
    """
    registry = get_variable_registry()

    if flattened_relatives:
        relatives = True

    def _rel_empty():
        return {
            "standard": None,
            "holo": None,
            "anti": None,
            "real": None,
            "imag": None,
            "system_label": None,
        }

    def _flatten(rel: dict) -> tuple:
        st = rel.get("standard")
        if st:
            return st

        fam = []
        for k in ("holo", "anti", "real", "imag"):
            v = rel.get(k)
            if v is None:
                continue
            if isinstance(v, tuple):
                fam.extend(v)
            else:
                fam.append(v)
        return tuple(fam)

    def _out_unregistered():
        out = {"type": "unregistered", "sub_type": None}
        if path:
            out["path"] = None
        if relatives:
            rel = _rel_empty()
            out["relatives"] = rel
            if flattened_relatives:
                out["flattened_relatives"] = tuple()
        if system_index:
            out["system_index"] = None
        if differential_system:
            out["differential_system"] = None
        return out

    paths = registry.get("paths")
    if not isinstance(paths, dict):
        return _out_unregistered()

    def _is_path_tuple(p):
        return isinstance(p, tuple) and len(p) >= 2

    def _sys_std(system_label: str):
        return registry.get("standard_variable_systems", {}).get(system_label, {})

    def _sys_cplx(system_label: str):
        return registry.get("complex_variable_systems", {}).get(system_label, {})

    if isinstance(obj, str):
        entry = paths.get(obj)
        p = entry.get("path") if isinstance(entry, dict) else None
        if not _is_path_tuple(p) or not len(p) >= 2:
            return _out_unregistered()

        branch, system_label = p[0], p[1]

        if branch == "standard_variable_systems":
            out = {"type": "coordinate", "sub_type": "standard"}
            if path:
                out["path"] = ("standard_variable_systems", system_label)
            if relatives:
                rel = _rel_empty()
                rel["system_label"] = system_label
                sys = _sys_std(system_label)
                fam = sys.get("family_values")
                rel["standard"] = sys.get("family_values")
                out["relatives"] = rel
                if flattened_relatives:
                    out["flattened_relatives"] = _flatten(rel)
            if system_index:
                out["system_index"] = None
            if differential_system:
                out["differential_system"] = None
            return out

        if branch == "complex_variable_systems":
            out = {"type": "coordinate", "sub_type": "complex"}
            if path:
                out["path"] = ("complex_variable_systems", system_label)
            if relatives:
                rel = _rel_empty()
                rel["system_label"] = system_label
                sys = _sys_cplx(system_label)

                fam = (
                    sys.get("family_values")
                    if sys.get("family_type") != "single"
                    else tuple((var,) for var in sys.get("family_values"))
                )
                if fam is not None:
                    rel["holo"], rel["anti"], rel["real"], rel["imag"] = fam
                out["relatives"] = rel
                if flattened_relatives:
                    out["flattened_relatives"] = _flatten(rel)
            if system_index:
                out["system_index"] = None
            if differential_system:
                out["differential_system"] = None
            return out

        if branch == "eds":
            return _eds_lookup(
                registry,
                p,
                entry.get("kind"),
                None,
                path,
                relatives,
                flattened_relatives,
                system_index,
                differential_system,
            )

        out = {"type": branch, "sub_type": branch}
        if path:
            out["path"] = (branch, system_label)
        if relatives:
            out["relatives"] = _rel_empty()
            if flattened_relatives:
                out["flattened_relatives"] = tuple()
        if system_index:
            out["system_index"] = None
        if differential_system:
            out["differential_system"] = None
        return out

    from ._safeguards import get_dgcv_category

    category = get_dgcv_category(obj)
    if category in ("zeroFormAtom", "abstDFAtom"):
        entry = paths.get(getattr(obj, "label", None))
        p = entry.get("path") if isinstance(entry, dict) else None
        if _is_path_tuple(p) and p[0] == "eds":
            return _eds_lookup(
                registry,
                p,
                entry.get("kind"),
                obj,
                path,
                relatives,
                flattened_relatives,
                system_index,
                differential_system,
            )
        if not is_atomic(obj):
            return _out_unregistered()
    elif category == "abst_coframe":
        for coframe_label, info in registry.get("eds", {}).get("coframes", {}).items():
            if info.get("coframe") is obj:
                return _eds_lookup(
                    registry,
                    ("eds", coframe_label, "coframes"),
                    "coframe",
                    obj,
                    path,
                    relatives,
                    flattened_relatives,
                    system_index,
                    differential_system,
                )
        return _out_unregistered()

    if not is_atomic(obj):
        return _out_unregistered()

    label = str(obj)
    entry = paths.get(label)
    p = entry.get("path") if isinstance(entry, dict) else None

    out = {"type": "unregistered", "sub_type": None}
    if path:
        out["path"] = p if _is_path_tuple(p) else None
    if system_index:
        out["system_index"] = None
    if differential_system:
        out["differential_system"] = None

    if not _is_path_tuple(p):
        if relatives:
            rel = _rel_empty()
            out["relatives"] = rel
            if flattened_relatives:
                out["flattened_relatives"] = tuple()
        return out

    branch, system_label = p[0], p[1]

    if branch == "standard_variable_systems":
        out["type"] = "coordinate"
        out["sub_type"] = "standard"

        sys = _sys_std(system_label)

        if system_index:
            data = sys.get("variable_relatives", {}).get(label, {})
            out["system_index"] = data.get("system_index")

        if differential_system and sys.get("differential_system", False):
            data = sys.get("variable_relatives", {}).get(label, {})
            out["differential_system"] = {
                "vf": data.get("VFClass", None),
                "df": data.get("DFClass", None),
            }

        if relatives:
            rel = _rel_empty()
            rel["system_label"] = system_label
            fam = sys.get("family_values")
            rel["standard"] = (
                fam[0]
                if sys.get("family_type") == "single"
                else tuple(fam)
                if fam is not None
                else (obj,)
            )
            out["relatives"] = rel
            if flattened_relatives:
                out["flattened_relatives"] = _flatten(rel)

        return out

    if branch != "complex_variable_systems":
        if relatives:
            rel = _rel_empty()
            out["relatives"] = rel
            if flattened_relatives:
                out["flattened_relatives"] = tuple()
        return out

    out["type"] = "coordinate"

    sys = _sys_cplx(system_label)
    var_rel = sys.get("variable_relatives", {})
    data = var_rel.get(label, {})

    if system_index:
        out["system_index"] = data.get("system_index")

    pos = data.get("complex_positioning")
    if pos == "holomorphic":
        out["sub_type"] = "holo"
    elif pos == "antiholomorphic":
        out["sub_type"] = "anti"
    elif pos == "real":
        out["sub_type"] = "real"
    elif pos == "imaginary":
        out["sub_type"] = "imag"
    else:
        out["sub_type"] = None

    if relatives:
        rel = _rel_empty()
        rel["system_label"] = system_label
        fam = data.get("complex_family")
        if fam is not None:
            rel["holo"], rel["anti"], rel["real"], rel["imag"] = fam
        out["relatives"] = rel
        if flattened_relatives:
            out["flattened_relatives"] = _flatten(rel)

    if differential_system:
        if sys.get("differential_system", False):
            out["differential_system"] = {
                "vf": data.get("VFClass", None),
                "df": data.get("DFClass", None),
            }

    return out


# -----------------------------------------------------------------------------
# general utilities
# -----------------------------------------------------------------------------
def order_coordinates(coordinates):
    def sort_key(x):
        info = vmf_lookup(x, system_index=True, relatives=True)
        hierarchy = {"standard": 0, "holo": 1, "anti": 2, "real": 3, "imag": 4}
        none_guard, info1, info2 = (
            {None: "zzzzz"},
            info["relatives"]["system_label"],
            info["system_index"],
        )
        return (
            hierarchy.get(info["sub_type"], 9),
            none_guard.get(info1, info1),
            none_guard.get(info2, info2),
        )

    return tuple(sorted(coordinates, key=sort_key))


# -----------------------------------------------------------------------------
# displaying summaries
# -----------------------------------------------------------------------------
def DGCV_snapshot(style=None, use_latex=None, complete_report=None):
    dgcv_warning(
        "`DGCV_snapshot` has been deprecated as part of the shift toward standardized naming conventions in the `dgcv` library.",
        dgcvDeprecationWarning,
        stacklevel=2,
        old_kw="DGCV_snapshot",
        new_kw="vmf_summary",
        sunset="2027",
    )
    return vmf_summary(
        style=style, use_latex=use_latex, complete_report=complete_report
    )


def variableSummary(*args, **kwargs):
    dgcv_warning(
        "variableSummary() is deprecated and will be removed in a future version. "
        "Please use vmf_summary() instead.",
        dgcvDeprecationWarning,
        stacklevel=2,
        old_kw="variableSummary",
        new_kw="vmf_summary",
        sunset="2027",
    )
    return vmf_summary(*args, **kwargs)


def vmf_summary(
    theme=None,
    use_latex=None,
    complete_report=None,
    *,
    plain_text: bool = False,
    **kwargs,  # supporting the old style kwarg
):
    if not is_rich_displaying_available():
        plain_text = True
    if plain_text:
        vr = get_variable_registry()
        force_report = True if complete_report is True else False
        if complete_report is None:
            complete_report = True

        sections = _vmf_plain_build_sections(vr, force_report=force_report)
        sections = _vmf_plain_pad_headers(sections)
        if not sections:
            print("VMF empty")
            return

        out = "\n\n".join(s for s in sections if s).strip()
        print(out if out else "VMF empty")
        return

    if theme is None:
        theme = kwargs.get("style", None) or get_dgcv_settings_registry().get(
            "theme", "dark"
        )
    if use_latex is None:
        use_latex = get_dgcv_settings_registry()["use_latex"]

    force_report = True if complete_report is True else False
    if complete_report is None:
        complete_report = True

    vr = get_variable_registry()
    builders = []
    if (
        vr["standard_variable_systems"]
        or vr["complex_variable_systems"]
        or force_report
    ):
        builders.append(_snapshot_coor_)
    if vr["finite_algebra_systems"] or force_report:
        builders.append(_snapshot_algebras_)
    if vr["eds"]["atoms"] or force_report:
        builders.append(_snapshot_eds_atoms_)
    if vr["eds"]["coframes"] or force_report:
        builders.append(_snapshot_coframes_)

    if not builders and not force_report:
        print("There are no objects currently registered in the dgcv VMF.")
        return

    container_id = f"dgcv-vmf-summary-{uuid.uuid4().hex[:8]}"
    theme_vars, theme_data = get_style(theme, return_theme_data=True, minimal=True)
    scoped_theme = theme_vars.replace(":root", f"#{container_id}")
    border_radius = int(
        theme_data.custom_css_vars.get("--dgcv-border-radius", "12px").replace("px", "")
    )
    scoped_theme = theme_vars.replace(":root", f"#{container_id}")

    html_parts = []
    total = len(builders)

    for i, builder in enumerate(builders):
        is_first = i == 0
        is_last = i == total - 1
        r = border_radius
        if total == 1:
            corner_kwargs = {"ul": r, "ur": r, "ll": r, "lr": r}
        elif is_first:
            corner_kwargs = {"ul": r, "ur": r, "ll": 0, "lr": 0}
        elif is_last:
            corner_kwargs = {"ul": 0, "ur": 0, "ll": r, "lr": r}
        else:
            corner_kwargs = {"ul": 0, "ur": 0, "ll": 0, "lr": 0}
        view = builder(style=theme, use_latex=use_latex, slim=True, **corner_kwargs)
        html_parts.append(f'<div class="section">{view.to_html()}</div>')

    combined_html = f"""
<div id="{container_id}">
  <style>
    {scoped_theme}

    #{container_id} .stack {{ display: flex; flex-direction: column; gap: 16px; align-items: stretch; width: 100%; margin: 0; }}
    #{container_id} .section {{ width: 100%; }}

    #{container_id} .dgcv-table-wrap {{ overflow-x: auto; max-width: 100%;}}
    #{container_id} .dgcv-caption-wrapper {{ display: flex; justify-content: flex-start; width: 100%; margin: 0; padding: 0; }}

    #{container_id} .section .dgcv-table-caption {{ 
        background: var(--dgcv-special-background, var(--dgcv-bg-surface));
        color: var(--dgcv-special-text, var(--dgcv-text-heading)); 
        font-weight: bold; 
        padding: 8px 20px; 
        border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main); 
        box-sizing: border-box; 
        text-align: left; 
        width: fit-content; 
        margin: 0;
        text-shadow: var(--dgcv-text-shadow, none);
        border-top-left-radius: 12px; 
        border-top-right-radius: 12px;
    }}
    #{container_id} .section .dgcv-table-caption .dgcv-plaque {{
        display: inline-block;
        background: var(--plaque-fill);
        padding: 4px 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: var(--dgcv-border-width, 1px) solid var(--plaque-border, var(--dgcv-border-main));
        border-top-left-radius: calc(12px - 4px);
        border-top-right-radius: calc(12px - 4px);
    }}
    #{container_id} .section:not(:first-child) .dgcv-table-caption {{
        border-top-left-radius: 0;
        border-top-right-radius: 0;
    }}
    #{container_id} .section:not(:first-child) .dgcv-table-caption .dgcv-plaque {{
        border-top-left-radius: 0;
        border-top-right-radius: 0;
    }}


    #{container_id} .dgcv-data-table {{ 
        width: 100%;
        border-collapse: separate; 
        border-spacing: 0;
        background: var(--dgcv-special-background, var(--dgcv-bg-surface));
        border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main); 
        border-image: var(--dgcv-border-image, none);
        box-shadow: var(--dgcv-table-shadow, none); 
        overflow: hidden; 
        border-top-left-radius: 0;
        border-top-right-radius: var(--m-r-tr);
        border-bottom-right-radius: var(--m-r-br);
        border-bottom-left-radius: var(--m-r-bl);
    }}

    #{container_id} .dgcv-data-table thead th, #{container_id} .dgcv-data-table th.col_heading {{ 
        background-color: var(--dgcv-bg-surface); 
        color: var(--dgcv-text-heading); 
        border-bottom: 2px solid var(--dgcv-border-main); 
        padding: 8px 12px; 
        text-shadow: var(--dgcv-text-shadow, none);
    }}

    #{container_id} .dgcv-data-table tbody tr {{ background-color: var(--dgcv-bg-primary); color: var(--dgcv-text-main); }}
    #{container_id} .dgcv-data-table tbody tr:nth-child(even) {{ background-color: var(--dgcv-bg-alt); color: var(--dgcv-text-alt); }}
    #{container_id} .dgcv-data-table tbody td {{ padding: 8px 12px; border-bottom: 1px solid var(--dgcv-border-alt); }}
    #{container_id} .dgcv-data-table td .table-cell {{ overflow-x: visible; white-space: normal; }}

    #{container_id} .dgcv-data-table tr:first-child th:first-child {{ border-top-left-radius: inherit; }}
    #{container_id} .dgcv-data-table tr:first-child th:last-child {{ border-top-right-radius: var(--m-r-tr); }}
    #{container_id} .dgcv-data-table tr:last-child td:first-child {{ border-bottom-left-radius: var(--m-r-bl); }}
    #{container_id} .dgcv-data-table tr:last-child td:last-child {{ border-bottom-right-radius: var(--m-r-br); }}

    #{container_id} .dgcv-data-table tbody tr {{ transition: var(--dgcv-hover-transition, transform 0.2s, box-shadow 0.2s, background-color 0.2s); }}
    #{container_id} .dgcv-data-table tbody tr:hover {{ background-color: var(--dgcv-bg-hover) !important; color: var(--dgcv-text-hover) !important; transform: var(--dgcv-hover-transform, none); }}

    #{container_id} .dgcv-data-table td, #{container_id} .dgcv-data-table th {{ text-align: left; border-right: 1px solid var(--dgcv-border-main); }}
    #{container_id} .dgcv-data-table td:last-child, #{container_id} .dgcv-data-table th:last-child {{ border-right: none; }}
    #{container_id} .dgcv-flex {{ 
        display: grid; 
        grid-template-columns: 1fr 340px; 
        grid-template-rows: auto 1fr; 
        column-gap: 10px; 
        align-items: start; 
    }}
    #{container_id} .dgcv-main {{ display: grid; grid-row: 1 / -1; grid-column: 1; grid-template-rows: subgrid; }}
    #{container_id} .dgcv-side {{ grid-row: 2; grid-column: 2; box-sizing: border-box; display: flex; flex-direction: column; align-items: stretch; max-width: 100%; }}
    #{container_id} .dgcv-side-panel {{
        border: var(--dgcv-border-width, 1px) solid var(--dgcv-border-main);
        background: var(--dgcv-special-background, var(--dgcv-bg-surface));
        color: var(--dgcv-special-text, var(--dgcv-text-heading));
        border-radius: var(--s-r-tl) var(--s-r-tr) var(--s-r-br) var(--s-r-bl);
        padding: 12px;
        height: 100%;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
    }}
    #{container_id} .dgcv-side-plaque {{
        border-radius: calc(var(--s-r-tl) - 12px) calc(var(--s-r-tr) - 12px) calc(var(--s-r-br) - 12px) calc(var(--s-r-bl) - 12px);
    }}
    @media (max-width: 900px) {{
        #{container_id} .dgcv-flex {{ display: flex; flex-direction: column; align-items: flex-start; row-gap: 10px; }}
        #{container_id} .dgcv-main, #{container_id} .dgcv-side {{ max-width: 100%; width: 100%; }}
        #{container_id} .dgcv-main {{ display: flex; flex-direction: column; grid-row: unset; grid-column: unset; }}
        #{container_id} .dgcv-side {{ grid-row: unset; grid-column: unset; }}
    }}
  </style>
  <div class="stack">
    {"".join(html_parts)}
  </div>
</div>
""".strip()

    class _HTMLWrapper:
        def __init__(self, html):
            self._html = html

        def to_html(self, *args, **kwargs):
            return self._html

        def _repr_html_(self):
            return self._html

    return latex_in_html(_HTMLWrapper(combined_html))


# -----------------------------------------------------------------------------
# plain text summary helpers
# -----------------------------------------------------------------------------
def _vmf_plain_fmt_dim(n) -> str:
    return f"{n} dimensional" if n is not None else "? dimensional"


def _vmf_plain_fmt_span(xs, *, max_items: int = 12) -> str:
    inner = _vmf_plain_fmt_list(xs, max_items=max_items)
    return f"<{inner}>" if inner else "<>"


def _vmf_plain_fmt_grading(grading, *, max_items: int = 12) -> str:
    if grading is None:
        return "None"
    if not isinstance(grading, (list, tuple)) or not grading:
        return "None"

    vecs = []
    for g in grading:
        if isinstance(g, (list, tuple)):
            inner = _vmf_plain_fmt_list(g, max_items=max_items)
            vecs.append(f"({inner})")
        else:
            vecs.append(str(g))
    return "[" + ", ".join(vecs) + "]"


def _vmf_plain_pad_headers(sections: list[str]) -> list[str]:
    if not sections:
        return sections

    headers = []
    split_sections = []

    for block in sections:
        lines = block.splitlines()
        if not lines:
            split_sections.append(lines)
            headers.append("")
            continue
        headers.append(lines[0])
        split_sections.append(lines)

    max_len = max(len(h) for h in headers if h)

    for lines in split_sections:
        if not lines:
            continue
        header = lines[0]
        pad = max_len - len(header)
        if pad > 0:
            lines[0] = header + "=" * pad

    return ["\n".join(lines) for lines in split_sections]


def _vmf_plain_build_sections(vr: dict, *, force_report: bool) -> list[str]:
    out: list[str] = []
    s = _vmf_plain_snapshot_coordinate_systems(vr)
    if s or force_report:
        if s:
            out.append(s)
    s = _vmf_plain_snapshot_algebras(vr)
    if s or force_report:
        if s:
            out.append(s)
    s = _vmf_plain_snapshot_eds_atoms(vr)
    if s or force_report:
        if s:
            out.append(s)
    s = _vmf_plain_snapshot_coframes(vr)
    if s or force_report:
        if s:
            out.append(s)
    return out


def _vmf_plain_snapshot_coordinate_systems(vr: dict) -> str:

    c = vr.get("complex_variable_systems", {}) or {}
    s = vr.get("standard_variable_systems", {}) or {}

    complex_keys = sorted(c.keys())
    standard_keys = sorted(s.keys())
    total = len(complex_keys) + len(standard_keys)

    if total == 0:
        return ""

    lines: list[str] = [f"=== Coordinate Systems ({total}) ==="]

    if complex_keys:
        lines.append("Complex Systems:")
        for syslabel in complex_keys:
            system = c.get(syslabel, {}) or {}

            fn = system.get("family_names", ())
            diff = system.get("differential_system", False)

            if isinstance(fn, (list, tuple)) and len(fn) == 4:
                hol_names = fn[0]
                bar_names = fn[1]
                real_names = fn[2]
                imag_names = fn[3]
            else:
                hol_names = bar_names = real_names = imag_names = ()

            dim = len(hol_names) if isinstance(hol_names, (list, tuple)) else None

            lines.append(
                f"  - {syslabel} ({_vmf_plain_fmt_dim(dim)}, with frame/coframe = {diff})"
            )

            if hol_names:
                lines.append(
                    f"      - holomorphic: {_vmf_plain_fmt_list(hol_names, max_items=3)}"
                )

            if bar_names:
                lines.append(
                    f"      - antiholomorphic: {_vmf_plain_fmt_list(bar_names, max_items=3)}"
                )

            if real_names:
                lines.append(
                    f"      - real: {_vmf_plain_fmt_list(real_names, max_items=3)}"
                )

            if imag_names:
                lines.append(
                    f"      - imaginary: {_vmf_plain_fmt_list(imag_names, max_items=3)}"
                )

    if standard_keys:
        lines.append("Standard Systems:")
        for syslabel in standard_keys:
            system = s.get(syslabel, {}) or {}

            fn = system.get("family_names", ())
            diff = system.get("differential_system", False)

            dim = len(fn) if isinstance(fn, (list, tuple)) else None

            lines.append(
                f"  - {syslabel} ({_vmf_plain_fmt_dim(dim)}, with frame/coframe = {diff})"
            )

            if fn:
                lines.append(f"      - vars: {_vmf_plain_fmt_list(fn, max_items=3)}")

    return "\n".join(lines).rstrip()


def _vmf_plain_snapshot_algebras(vr: dict) -> str:
    fa = vr.get("finite_algebra_systems", {}) or {}
    keys = sorted(fa.keys())
    if not keys:
        return ""

    global_dict = working_namespace()

    lines: list[str] = [f"=== Finite-dimensional Algebras ({len(keys)}) ==="]
    for label in keys:
        system = fa.get(label, {}) or {}
        family_values = system.get("family_values", ())
        obj = global_dict.get(label, None)

        if obj is not None:
            try:
                nm = obj.__str__(VLP=False)
            except TypeError:
                nm = str(getattr(obj, "label", None) or label or "Unnamed Algebra")
        else:
            nm = str(label) if label else "Unnamed Algebra"

        dim = getattr(obj, "dimension", None)
        if dim is None:
            dim = (
                len(family_values)
                if isinstance(family_values, (list, tuple))
                else (1 if family_values else None)
            )

        lines.append(f"  - {nm} ({_vmf_plain_fmt_dim(dim)})")

        basis = getattr(obj, "basis", None)
        if not isinstance(basis, (list, tuple)) or not basis:
            basis = family_values if isinstance(family_values, (list, tuple)) else ()
        lines.append(f"      - basis: {_vmf_plain_fmt_span(basis, max_items=12)}")

        grading = getattr(obj, "grading", None)
        if grading is None:
            grading = system.get("grading", None)
        lines.append(
            f"      - grading: {_vmf_plain_fmt_grading(grading, max_items=12)}"
        )

    return "\n".join(lines).rstrip()


def _vmf_plain_snapshot_eds_atoms(vr: dict) -> str:
    atoms = (vr.get("eds", {}) or {}).get("atoms", {}) or {}
    keys = sorted(atoms.keys())
    if not keys:
        return ""

    lines: list[str] = [f"=== EDS Atoms ({len(keys)}) ==="]
    for label in keys:
        system = atoms.get(label, {}) or {}
        deg = system.get("degree", "----")
        lines.append(f"  - {label} (degree = {deg})")

        family_values = system.get("family_values", ())
        forms = (
            family_values
            if isinstance(family_values, (list, tuple))
            else ([family_values] if family_values else [])
        )
        lines.append(
            f"      - forms: {_vmf_plain_fmt_list(forms, max_items=12) if forms else '----'}"
        )

        if system.get("real", False):
            lines.append("      - conjugates: ----")
        else:
            conjugates = system.get("conjugates", {}) or {}
            conj_vals = list(conjugates.values())
            lines.append(
                f"      - conjugates: {_vmf_plain_fmt_list(conj_vals, max_items=12) if conj_vals else '----'}"
            )

        pc = system.get("primary_coframe", None)
        lines.append(
            f"      - primary coframe: {str(pc) if pc is not None else '----'}"
        )

    return "\n".join(lines).rstrip()


def _vmf_plain_snapshot_coframes(vr: dict) -> str:
    coframes = (vr.get("eds", {}) or {}).get("coframes", {}) or {}
    keys = sorted(coframes.keys())
    if not keys:
        return ""

    global_dict = working_namespace()
    lines: list[str] = [f"=== Coframes ({len(keys)}) ==="]
    for label in keys:
        system = coframes.get(label, {}) or {}
        coframe_obj = global_dict.get(label, None)

        forms = getattr(coframe_obj, "forms", None) if coframe_obj is not None else None
        if not forms:
            children = list(system.get("children", []) or [])
            if children:
                try:
                    forms = [global_dict.get(ch, ch) for ch in children]
                except Exception:
                    forms = children
            else:
                forms = []

        lines.append(f"  - {label} = {_vmf_plain_fmt_span(forms, max_items=12)}")

    return "\n".join(lines).rstrip()


def _vmf_plain_fmt_dim(n) -> str:
    return f"{n} dimensional" if n is not None else "? dimensional"


def _vmf_plain_truncate(xs, *, max_items: int):
    xs = list(xs or [])
    if len(xs) <= max_items:
        return xs
    k = max_items // 2
    return xs[:k] + ["..."] + xs[-k:]


def _vmf_plain_fmt_list(xs, *, max_items: int = 10) -> str:
    toks = [str(x) for x in _vmf_plain_truncate(xs, max_items=max_items)]
    return ", ".join(toks)


# -----------------------------------------------------------------------------
# rich display summary helpers
# -----------------------------------------------------------------------------
def _snapshot_coor_(style=None, use_latex=None, slim=False, **kwargs):
    if style is None:
        style = get_dgcv_settings_registry()["theme"]
    if use_latex is None:
        use_latex = get_dgcv_settings_registry()["use_latex"]

    def convert_to_greek(var_name):
        for name, greek in greek_letters.items():
            if var_name.lower().startswith(name):
                return var_name.replace(name, greek, 1)
        return var_name

    def format_latex_subscripts(var_name, nest_braces=False):
        if var_name and var_name[-1] == "_":
            var_name = var_name[:-1]
        parts = var_name.split("_")
        if len(parts) == 1:
            return convert_to_greek(var_name)
        base = convert_to_greek(parts[0])
        subscript = ", ".join(parts[1:])
        return (
            f"{{{base}_{{{subscript}}}}}" if nest_braces else f"{base}_{{{subscript}}}"
        )

    def format_variable_name(var_name, system_type, use_latex=False):
        vr = variable_registry
        if system_type == "standard":
            info = vr["standard_variable_systems"].get(var_name, {})
            family_names = info.get("family_names", var_name)
            initial_index = info.get("initial_index", 1)
        elif system_type == "complex":
            info = vr["complex_variable_systems"].get(var_name, {})
            fn = info.get("family_names", ())
            family_names = (
                fn[0]
                if (fn and isinstance(fn, (list, tuple)) and len(fn) > 0)
                else var_name
            )
            initial_index = info.get("initial_index", 1)
        elif system_type == "algebra":
            info = vr["finite_algebra_systems"].get(var_name, {})
            family_names = info.get("family_names", var_name)
            initial_index = None
        else:
            family_names, initial_index = var_name, None

        if isinstance(family_names, (list, tuple)) and len(family_names) > 1:
            if use_latex:
                if initial_index is not None:
                    content = (
                        f"{format_latex_subscripts(var_name)} = "
                        f"\\left( {format_latex_subscripts(var_name, nest_braces=True)}_{{{initial_index}}}, "
                        f"\\ldots, {format_latex_subscripts(var_name, nest_braces=True)}_{{{initial_index + len(family_names) - 1}}} \\right)"
                    )
                else:
                    content = (
                        f"{convert_to_greek(var_name)} = {convert_to_greek(var_name)}"
                    )
            else:
                content = f"{var_name} = ({family_names[0]}, ..., {family_names[-1]})"
        else:
            content = f"{convert_to_greek(var_name)}" if use_latex else f"{var_name}"
        return f"${content}$" if use_latex else content

    def build_object_string(
        obj_type, var_name, start_index, tuple_len, system_type, use_latex=False
    ):
        from ..printing.printing._string_processing import _process_var_label

        if tuple_len == 1:
            if use_latex:
                sym = _process_var_label(var_name)
                return (
                    f"$\\frac{{\\partial}}{{\\partial {sym}}}$"
                    if obj_type == "D"
                    else f"$\\operatorname{{d}} {sym}$"
                )
            return f"{obj_type}_{var_name}"
        else:
            if use_latex:
                left = _process_var_label(f"{var_name}_{start_index}")
                right = _process_var_label(f"{var_name}_{start_index + tuple_len - 1}")
                s = (
                    f"\\frac{{\\partial}}{{\\partial {left}}}, \\ldots, \\frac{{\\partial}}{{\\partial {right}}}"
                    if obj_type == "D"
                    else f"\\operatorname{{d}} {left}, \\ldots, \\operatorname{{d}} {right}"
                )
                return f"${s}$"
            return f"{obj_type}_{var_name}{start_index},...,{obj_type}_{var_name}{start_index + tuple_len - 1}"

    def build_object_string_for_complex(
        obj_type, family_houses, family_names, start_index, use_latex=False
    ):
        from ..printing.printing._string_processing import _process_var_label

        parts = []
        if isinstance(family_names, (list, tuple)) and len(family_names) == 4:
            for i, part in enumerate(family_houses):
                part_names = family_names[i]
                if isinstance(part_names, (list, tuple)) and len(part_names) > 1:
                    if use_latex:
                        left = _process_var_label(f"{part}_{start_index}")
                        right = _process_var_label(
                            f"{part}_{start_index + len(part_names) - 1}"
                        )
                        core = (
                            f"\\frac{{\\partial}}{{\\partial {left}}}, \\ldots, \\frac{{\\partial}}{{\\partial {right}}}"
                            if obj_type == "D"
                            else f"\\operatorname{{d}} {left}, \\ldots, \\operatorname{{d}} {right}"
                        )
                        parts.append(f"${core}$")
                    else:
                        parts.append(
                            f"{obj_type}_{part}{start_index},...,{obj_type}_{part}{start_index + len(part_names) - 1}"
                        )
                else:
                    if use_latex:
                        sym = _process_var_label(part)
                        parts.append(f"${obj_type}_{sym}$")
                    else:
                        parts.append(f"{obj_type}_{part}")
        else:
            if isinstance(family_names, (list, tuple)) and len(family_names) > 1:
                if use_latex:
                    left = _process_var_label(f"{family_houses[0]}_{start_index}")
                    right = _process_var_label(
                        f"{family_houses[0]}_{start_index + len(family_names) - 1}"
                    )
                    parts.append(
                        f"$\\frac{{\\partial}}{{\\partial {left}}}$, $\\ldots$, $\\frac{{\\partial}}{{\\partial {right}}}$"
                        if obj_type == "D"
                        else f"$\\operatorname{{d}} {left}$, $\\ldots$, $\\operatorname{{d}} {right}$"
                    )
                else:
                    parts.append(
                        f"{obj_type}_{family_houses[0]}{start_index},...,{obj_type}_{family_houses[0]}{start_index + len(family_names) - 1}"
                    )
            else:
                if use_latex:
                    sym = _process_var_label(family_houses[0])
                    parts.append(f"${obj_type}_{sym}$")
                else:
                    parts.append(f"{obj_type}_{family_houses[0]}")
        return ", ".join(parts)

    variable_registry = get_variable_registry()
    data = []
    var_system_labels = []

    for var_name in sorted(
        variable_registry.get("complex_variable_systems", {}).keys()
    ):
        system = variable_registry["complex_variable_systems"][var_name]
        fn = system.get("family_names", ())
        hol_names = (
            fn[0] if (fn and isinstance(fn, (list, tuple)) and len(fn) == 4) else []
        )
        tuple_len = len(hol_names) if isinstance(hol_names, (list, tuple)) else 1
        start_index = system.get("initial_index", 1)
        var_system_labels.append(
            format_variable_name(var_name, "complex", use_latex=use_latex)
        )
        family_houses = system.get("family_houses", ("N/A", "N/A", "N/A", "N/A"))
        real_names, imag_names = (
            (fn[2], fn[3])
            if (isinstance(fn, (list, tuple)) and len(fn) == 4)
            else ("N/A", "N/A")
        )

        if use_latex:
            real_part = (
                f"${format_latex_subscripts(family_houses[2])} = \\left( {format_latex_subscripts(family_houses[2], nest_braces=True)}_{{{start_index}}}, \\ldots, {format_latex_subscripts(family_houses[2], nest_braces=True)}_{{{start_index + len(real_names) - 1}}} \\right)$"
                if isinstance(real_names, (list, tuple)) and len(real_names) > 1
                else f"${format_latex_subscripts(family_houses[2])}$"
            )
            imag_part = (
                f"${format_latex_subscripts(family_houses[3])} = \\left( {format_latex_subscripts(family_houses[3], nest_braces=True)}_{{{start_index}}}, \\ldots, {format_latex_subscripts(family_houses[3], nest_braces=True)}_{{{start_index + len(imag_names) - 1}}} \\right)$"
                if isinstance(imag_names, (list, tuple)) and len(imag_names) > 1
                else f"${format_latex_subscripts(family_houses[3])}$"
            )
        else:
            real_part = (
                f"{family_houses[2]} = ({real_names[0]}, ..., {real_names[-1]})"
                if isinstance(real_names, (list, tuple)) and len(real_names) > 1
                else f"{family_houses[2]}"
            )
            imag_part = (
                f"{family_houses[3]} = ({imag_names[0]}, ..., {imag_names[-1]})"
                if isinstance(imag_names, (list, tuple)) and len(imag_names) > 1
                else f"{family_houses[3]}"
            )

        if system.get("differential_system", False):
            vf_str = build_object_string_for_complex(
                "D", family_houses, fn, start_index, use_latex
            )
            df_str = build_object_string_for_complex(
                "d", family_houses, fn, start_index, use_latex
            )
        else:
            vf_str = df_str = "----"
        data.append([tuple_len, real_part, imag_part, vf_str, df_str])

    for var_name in sorted(
        variable_registry.get("standard_variable_systems", {}).keys()
    ):
        system = variable_registry["standard_variable_systems"][var_name]
        family_names = system.get("family_names", ())
        tuple_len = len(family_names) if isinstance(family_names, (list, tuple)) else 1
        start_index = system.get("initial_index", 1)
        var_system_labels.append(
            format_variable_name(var_name, "standard", use_latex=use_latex)
        )

        if system.get("differential_system", False):
            vf_str = build_object_string(
                "D", var_name, start_index, tuple_len, "standard", use_latex
            )
            df_str = build_object_string(
                "d", var_name, start_index, tuple_len, "standard", use_latex
            )
        else:
            vf_str = df_str = "----"
        data.append([tuple_len, "----", "----", vf_str, df_str])

    combined_data = [[label] + row for label, row in zip(var_system_labels, data)]
    columns = [
        "Coordinate System",
        "# of Variables",
        "Real Part",
        "Imaginary Part",
        "Vector Fields",
        "Differential Forms",
    ]

    theme_vars = get_style(style, legacy=False, minimal=True)

    extra_css = """
.dgcv-data-table td, .dgcv-data-table th {
    text-align: left;
}
    """

    view = build_plain_table(
        columns=columns,
        rows=combined_data,
        caption="Initialized Coordinate Systems",
        theme_css_vars=theme_vars,
        extra_css=extra_css,
        table_attrs='style="table-layout:auto;"',
        cell_align=None,
        escape_cells=False,
        escape_headers=True,
        nowrap=False,
        slim=slim,
        truncate_chars=None,
        **kwargs,
    )
    return view


def _snapshot_algebras_(style=None, use_latex=None, slim=False, **kwargs):
    if style is None:
        style = get_dgcv_settings_registry()["theme"]
    if use_latex is None:
        use_latex = get_dgcv_settings_registry()["use_latex"]

    registry = get_variable_registry()
    finite_algebras = registry.get("finite_algebra_systems", {}) or {}
    global_dict = working_namespace()

    def _basis_label(x):
        if use_latex:
            try:
                return f"${x._repr_latex_(raw=True)}$"
            except Exception:
                return f"${str(x)}$"
        return repr(x)

    def _format_basis(values):
        if isinstance(values, (list, tuple)):
            n = len(values)
            if n == 0:
                return "-" if not use_latex else "$\\text{-}$"
            if n > 5:
                return f"{_basis_label(values[0])}, ..., {_basis_label(values[-1])}"
            return ", ".join(_basis_label(v) for v in values)
        return _basis_label(values)

    def _format_alg_label(label):
        try:
            if use_latex:
                return global_dict[label]._repr_latex_(abbrev=True)
        except Exception:
            pass
        return label

    def _format_grading(label):
        try:
            alg = global_dict[label]
            if alg._gradingNumber * alg.dimension > 30:
                return f"{alg._gradingNumber} gradings assigned"
            grading = getattr(alg, "grading", None)
            if isinstance(grading, (list, tuple)):
                if all(isinstance(g, (list, tuple)) for g in grading):
                    return ", ".join(f"({', '.join(map(str, g))})" for g in grading)
                else:
                    return str(grading)
        except Exception:
            pass
        return "None"

    rows = []
    for label in sorted(finite_algebras.keys()):
        system = finite_algebras[label] or {}
        family_values = system.get("family_values", ())
        basis_str = _format_basis(family_values)

        if isinstance(family_values, (list, tuple)):
            dim = len(family_values)
        else:
            dim = 1

        alg_label = _format_alg_label(label)
        if use_latex and isinstance(alg_label, str) and not alg_label.startswith("$"):
            alg_label = f"${alg_label}$"

        grading_str = _format_grading(label)

        rows.append([alg_label, basis_str, dim, grading_str])

    columns = ["Algebra Label", "Basis", "Dimension", "Grading"]

    theme_vars = get_style(style, legacy=False, minimal=True)

    extra_css = """
.dgcv-data-table td, .dgcv-data-table th {
    text-align: left;
}
    """

    view = build_plain_table(
        columns=columns,
        rows=rows,
        caption="Initialized Finite-dimensional Algebras",
        theme_css_vars=theme_vars,
        extra_css=extra_css,
        table_attrs='style="table-layout:auto; overflow-x:auto;"',
        cell_align=None,
        escape_cells=False,
        escape_headers=True,
        nowrap=False,
        truncate_chars=None,
        slim=slim,
        **kwargs,
    )
    return view


def _snapshot_eds_atoms_(style=None, use_latex=None, slim=False, **kwargs):
    if style is None:
        style = get_dgcv_settings_registry()["theme"]
    if use_latex is None:
        use_latex = get_dgcv_settings_registry()["use_latex"]

    vr = get_variable_registry()
    eds_atoms_registry = (vr.get("eds", {}) or {}).get("atoms", {}) or {}

    columns = [
        "DF System",
        "Degree",
        "# Elements",
        "Differential Forms",
        "Conjugate Forms",
        "Primary Coframe",
    ]
    rows = []

    if eds_atoms_registry:
        for label, system in sorted(eds_atoms_registry.items()):
            df_system = label
            degree = system.get("degree", "----")
            family_values = system.get("family_values", ())
            num_elements = len(family_values) if isinstance(family_values, tuple) else 1

            if isinstance(family_values, tuple):
                if len(family_values) > 3:
                    diff_forms = f"{family_values[0]}, ..., {family_values[-1]}"
                else:
                    diff_forms = ", ".join(str(x) for x in family_values)
            else:
                diff_forms = str(family_values)

            if use_latex and family_values:
                if isinstance(family_values, tuple) and len(family_values) > 3:
                    left = (
                        family_values[0]._latex()
                        if hasattr(family_values[0], "_latex")
                        else str(family_values[0])
                    )
                    right = (
                        family_values[-1]._latex()
                        if hasattr(family_values[-1], "_latex")
                        else str(family_values[-1])
                    )
                    diff_forms = f"$ {left}, ..., {right} $"
                elif isinstance(family_values, tuple):
                    inner = ", ".join(
                        (x._latex() if hasattr(x, "_latex") else str(x))
                        for x in family_values
                    )
                    diff_forms = f"$ {inner} $"

            real_status = system.get("real", False)
            if real_status:
                conjugate_forms = "----"
            else:
                conjugates = system.get("conjugates", {})
                if conjugates:
                    conj_list = list(conjugates.values())
                    if len(conj_list) > 3:
                        conjugate_forms = f"{conj_list[0]}, ..., {conj_list[-1]}"
                    else:
                        conjugate_forms = ", ".join(str(x) for x in conj_list)
                    if use_latex and conj_list and hasattr(conj_list[0], "_latex"):
                        if len(conj_list) > 3:
                            left = conj_list[0]._latex()
                            right = conj_list[-1]._latex()
                            conjugate_forms = f"$ {left}, ..., {right} $"
                        else:
                            inner = ", ".join(x._latex() for x in conj_list)
                            conjugate_forms = f"$ {inner} $"
                else:
                    conjugate_forms = "----"

            primary_coframe = system.get("primary_coframe", None)
            if primary_coframe is None:
                primary_coframe_str = "----"
            else:
                primary_coframe_str = (
                    primary_coframe._latex()
                    if use_latex and hasattr(primary_coframe, "_latex")
                    else repr(primary_coframe)
                )

            rows.append(
                [
                    df_system,
                    degree,
                    num_elements,
                    diff_forms,
                    conjugate_forms,
                    primary_coframe_str,
                ]
            )

    theme_vars = get_style(style, legacy=False, minimal=True)

    extra_css = """
    .dgcv-data-table td, .dgcv-data-table th {
        text-align: left;
    }
    """

    return build_plain_table(
        columns=columns,
        rows=rows,
        caption="Initialized abstract differential forms in the VMF scope",
        theme_css_vars=theme_vars,
        extra_css=extra_css,
        table_attrs='style="table-layout:auto; overflow-x:auto;"',
        cell_align=None,
        escape_cells=False,
        escape_headers=True,
        nowrap=False,
        slim=slim,
        truncate_chars=None,
        **kwargs,
    )


def _snapshot_coframes_(style=None, use_latex=None, slim=False, **kwargs):
    """
    Returns a summary table listing coframes in the VMF scope
    """
    if style is None:
        style = get_dgcv_settings_registry()["theme"]
    if use_latex is None:
        use_latex = get_dgcv_settings_registry()["use_latex"]

    vr = get_variable_registry()
    coframes_registry = (vr.get("eds", {}) or {}).get("coframes", {}) or {}
    global_dict = working_namespace()

    def _latex_of(obj):
        try:
            if hasattr(obj, "_repr_latex_"):
                s = obj._repr_latex_(raw=True)
                if s is not None:
                    return s.strip()
            if hasattr(obj, "_latex"):
                s = obj._latex()
                if s is not None:
                    return s.strip()
        except Exception:
            pass
        return None

    def _fmt_one(x):
        if use_latex:
            s = _latex_of(x)
            if s is not None:
                return f"${s}$"
        return repr(x)

    def _fmt_list(xs):
        xs = list(xs or [])
        if not xs:
            return "$\\varnothing$" if use_latex else "∅"
        if len(xs) > 3:
            return f"{_fmt_one(xs[0])}, ..., {_fmt_one(xs[-1])}"
        return ", ".join(_fmt_one(x) for x in xs)

    rows = []
    for label, system in sorted(coframes_registry.items()):
        coframe_obj = global_dict.get(label, label)
        coframe_label = label

        if isinstance(coframe_obj, str):
            children = list(system.get("children", []) or [])
            if children and all(ch in global_dict for ch in children):
                forms = [global_dict[ch] for ch in children]
            else:
                forms = []
        else:
            forms = list(getattr(coframe_obj, "forms", []) or [])
        forms_cell = _fmt_list(forms)

        cousins = list(system.get("cousins_vals", []) or [])
        cousins_cell = _fmt_list(cousins)

        rows.append([coframe_label, forms_cell, cousins_cell])

    columns = ["Coframe Label", "Coframe 1-Forms", "Structure Coefficients"]

    theme_vars = get_style(style, legacy=False, minimal=True)

    extra_css = """
    .dgcv-data-table td, .dgcv-data-table th {
        text-align: left;
    }
    """

    from dgcv._aux.printing._tables import build_plain_table

    view = build_plain_table(
        columns=columns,
        rows=rows,
        caption="Initialized Abstract Coframes",
        theme_css_vars=theme_vars,
        extra_css=extra_css,
        table_attrs='style="table-layout:auto; overflow-x:auto;"',
        cell_align=None,
        escape_cells=False,
        escape_headers=True,
        nowrap=False,
        slim=slim,
        truncate_chars=None,
        **kwargs,
    )
    return view


# -----------------------------------------------------------------------------
# labeling
# -----------------------------------------------------------------------------


A_ORD = ord("a")


def label_to_value_fixed(s):
    v = 0
    for ch in s:
        v = v * 26 + (ord(ch) - A_ORD)  # 26=alphabet length
    return v


def value_to_label_fixed(value, length):
    chars = []
    for _ in range(length):
        chars.append(chr(A_ORD + (value % 26)))
        value //= 26
    return "".join(reversed(chars))


def first_available_label():
    keys = get_variable_registry().get("_labels", {}).keys()
    existing_by_len = {}
    n = 0
    for k in keys:
        if not k:
            continue
        L = len(k)
        existing_by_len.setdefault(L, []).append(k)
        n += 1

    L = 1
    while True:
        total = 26**L
        present_list = existing_by_len.get(L, [])
        if len(present_list) < total:
            seen = bytearray(
                total
                if total <= (len(present_list) + 1) * 4
                else (len(present_list) + 1)
            )
            if len(seen) == total:
                for s in present_list:
                    seen[label_to_value_fixed(s)] = 1
                for i, b in enumerate(seen):
                    if not b:
                        return value_to_label_fixed(i, L)
            else:
                present_vals = {label_to_value_fixed(s) for s in present_list}
                for i in range(total):
                    if i not in present_vals:
                        return value_to_label_fixed(i, L)
        L += 1
