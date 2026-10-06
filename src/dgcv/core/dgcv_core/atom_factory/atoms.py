from numbers import Integral

from ...._aux._backends._engine import engine_kind
from ...._aux._backends._types_and_constants import symbol
from ...._aux._utilities._config import (
    get_dgcv_settings_registry,
    get_variable_registry,
    update_working_namespace,
    update_working_namespace_k_v,
)
from ...._aux._vmf._safeguards import retrieve_passkey, validate_label
from ...._aux._vmf.vmf import clearVar
from ...combinatorics.combinatorics import build_nd_array, carProd
from ..fields import differential_form_class, vector_field_class

indexing_places = {
    "hi": "__",
    "low": "_",
    "up": "__",
    "down": "_",
    "u": "__",
    "d": "_",
    "h": "__",
    "l": "_",
    "__": "__",
    "_": "_",
    "": "",
}

_atom_properties = ("real", "nonnegative", "positive")


def effective_assumptions(name, assumptions, targeted_assumptions):
    """Per-symbol assumptions, matching how `symbol(...)` is constructed."""
    base = assumptions if assumptions else {}
    if targeted_assumptions:
        return base | targeted_assumptions.get(name, base)
    return base


def register_enforced_real(
    vars, var_names, assumptions, targeted_assumptions, enforced_real_dict
):
    if enforced_real_dict is None:
        return
    for name, v in zip(var_names, vars):
        eff = effective_assumptions(name, assumptions, targeted_assumptions)
        if any(eff.get(k, False) for k in _atom_properties):
            enforced_real_dict[v.conjugate()] = v


def variableProcedure(
    variables_label,
    number_of_variables=None,
    initialIndex=1,
    multiindex_shape=None,
    index_placement=None,
    return_created_object=None,
    withVF=False,
    assumptions: dict = None,
    targeted_assumptions: dict = None,
    _tempVar=None,
    _doNotUpdateVar=None,
    _calledFromCVP=None,
    _calledFromFactory=None,
    remove_guardrails=None,
    _obscure=None,
    _return_labels=False,
    _return_flattened=False,
    **kwargs,
):
    """
    Initializes one or more standard variable systems (single or tuples) and integrates them into dgcv's Variable Management Framework.
    """
    passkey = retrieve_passkey()
    variable_registry = get_variable_registry()
    settings = get_dgcv_settings_registry()

    if kwargs.get("assumeReal", False) is True:  # deprecated kwarg support
        if not isinstance(assumptions, dict):
            assumptions = {"real": True}
        else:
            assumptions["real"] = True

    index_placement = (
        []
        if index_placement is None
        else [indexing_places.get(place, "_") for place in index_placement]
    )

    if _return_flattened is True:
        _return_labels = True
    if _return_labels is True:
        return_created_object = True

    if settings["ask_before_overwriting_objects_in_vmf"] and not (
        _calledFromCVP == passkey or _calledFromFactory == passkey
    ):
        labels_iter = (
            tuple(variables_label)
            if isinstance(variables_label, (list, tuple))
            else (variables_label,)
        )
        protected = variable_registry.get("protected_variables", set())
        for j in labels_iter:
            if j in protected:
                raise Exception(
                    f"{variables_label} is already assigned to the real or imaginary part of a complex variable system, "
                    "so dgcv variable creation functions will not reassign it as a standard variable. Instead, use the "
                    "clearVar function to remove the conflicting CV system first before implementing such reassignments."
                )

    kind = engine_kind()
    enforce_real = kind == "sage"
    enforced_real_dict = (
        variable_registry.get("dgcv_enforced_real_atoms", None)
        if enforce_real
        else None
    )

    labels = (
        tuple(variables_label)
        if isinstance(variables_label, (list, tuple))
        else (variables_label,)
    )

    rco = [] if return_created_object is True else None
    sl_out = [] if _return_labels is True else None
    vvf = [] if _return_flattened is True else None

    paths = variable_registry.get("paths", None)

    for j in labels:
        labelLoc = (
            validate_label(j, remove_guardrails=remove_guardrails)
            if (not _calledFromCVP == passkey)
            else j
        )

        if _doNotUpdateVar != passkey:
            clearVar(labelLoc, report=False)

        temp_flag = True if _tempVar == passkey else None
        obscure_flag = True if _obscure == passkey else None

        if temp_flag is True:
            variable_registry["temporary_variables"].add(labelLoc)
        if obscure_flag is True:
            variable_registry["obscure_variables"].add(labelLoc)
        if isinstance(multiindex_shape, (list, tuple)) and all(
            isinstance(n, Integral) and n > 0 for n in multiindex_shape
        ):
            if len(multiindex_shape) > len(index_placement):
                index_placement += ["_"] * (
                    len(multiindex_shape) - len(index_placement)
                )
            indices = list(
                carProd(
                    *[range(initialIndex, initialIndex + n) for n in multiindex_shape]
                )
            )
            var_names = [
                f"{labelLoc}{''.join([a + str(b) for a, b in zip(index_placement, idx)])}"
                for idx in indices
            ]
            if _doNotUpdateVar != passkey or (
                _calledFromCVP == passkey or _calledFromFactory == passkey
            ):  # CVP doesn't manage sub-parant names
                clearVar(*var_names, report=False)
            if targeted_assumptions:
                merger = assumptions if assumptions else dict()
                vars = tuple(
                    symbol(
                        name,
                        assumptions=merger | targeted_assumptions.get(name, merger),
                    )
                    for name in var_names
                )
            elif assumptions:
                vars = tuple(
                    symbol(name, assumptions=assumptions) for name in var_names
                )
            else:
                vars = tuple(symbol(name) for name in var_names)

            var_values_flattened = vars

            new_globals = dict(zip(var_names, vars))
            var_values = build_nd_array(vars, multiindex_shape)
            new_globals[labelLoc] = var_values
            update_working_namespace(new_globals)

            if enforce_real:
                register_enforced_real(
                    vars,
                    var_names,
                    assumptions,
                    targeted_assumptions,
                    enforced_real_dict,
                )

            if _doNotUpdateVar != passkey:
                variable_registry["standard_variable_systems"][labelLoc] = {
                    "family_type": "multi_index",
                    "family_shape": multiindex_shape,
                    "family_names": tuple(var_names),
                    "family_values": var_values,
                    "differential_system": None,
                    "tempVar": temp_flag,
                    "obsVar": obscure_flag,
                    "initial_index": initialIndex,
                    "variable_relatives": {
                        var_name: {
                            "VFClass": None,
                            "DFClass": None,
                            "assumptions": effective_assumptions(
                                var_name, assumptions, targeted_assumptions
                            )
                            or None,
                            "system_index": idx,
                        }
                        for idx, var_name in enumerate(var_names)
                    },
                }
                variable_registry["_labels"][labelLoc] = {
                    "path": ("standard_variable_systems", labelLoc),
                    "children": set(var_names),
                }

                if paths is not None:
                    paths[labelLoc] = {
                        "kind": "standard_variable_system",
                        "path": ("standard_variable_systems", labelLoc),
                    }

                    base = ("standard_variable_systems", labelLoc, "family_values")
                    for idx, var_name in zip(indices, var_names):
                        offs = tuple(int(k - initialIndex) for k in idx)
                        paths[var_name] = {
                            "kind": "standard_variable",
                            "path": base + offs,
                        }

        elif number_of_variables is None:
            if targeted_assumptions:
                merger = assumptions if assumptions else dict()
                sym = symbol(
                    labelLoc,
                    assumptions=merger | targeted_assumptions.get(labelLoc, merger),
                )
            elif assumptions:
                sym = symbol(labelLoc, assumptions=assumptions)
            else:
                sym = symbol(labelLoc)
            var_values = (sym,)
            var_values_flattened = var_values
            var_names = [labelLoc]
            update_working_namespace_k_v(labelLoc, sym)

            if enforce_real:
                register_enforced_real(
                    (sym,),
                    (labelLoc,),
                    assumptions,
                    targeted_assumptions,
                    enforced_real_dict,
                )

            if _doNotUpdateVar != passkey:
                v_a = (
                    effective_assumptions(labelLoc, assumptions, targeted_assumptions)
                    or None
                )
                variable_registry["standard_variable_systems"][labelLoc] = {
                    "family_type": "single",
                    "family_names": (labelLoc,),
                    "family_values": var_values,
                    "differential_system": None,
                    "tempVar": temp_flag,
                    "obsVar": obscure_flag,
                    "initial_index": None,
                    "variable_relatives": {
                        labelLoc: {
                            "VFClass": None,
                            "DFClass": None,
                            "assumptions": v_a,
                            "system_index": 0,
                        }
                    },
                }
                variable_registry["_labels"][labelLoc] = {
                    "path": ("standard_variable_systems", labelLoc),
                    "children": set(),
                }

                if paths is not None:
                    paths[labelLoc] = {
                        "kind": "standard_variable",
                        "path": (
                            "standard_variable_systems",
                            labelLoc,
                            "family_values",
                            0,
                        ),
                    }

        elif isinstance(number_of_variables, Integral) and number_of_variables >= 0:
            lengthLoc = number_of_variables
            if isinstance(index_placement, (list, tuple)):
                if len(index_placement) > 0:
                    index_placement = index_placement[0]
                else:
                    index_placement = ""
            if index_placement is None:
                index_placement = ""
            else:
                index_placement = indexing_places.get(index_placement, "_")
            var_names = [
                f"{labelLoc}{index_placement}{i}"
                for i in range(initialIndex, lengthLoc + initialIndex)
            ]
            if targeted_assumptions:
                merger = assumptions if assumptions else dict()
                vars = tuple(
                    symbol(
                        name,
                        assumptions=merger | targeted_assumptions.get(name, merger),
                    )
                    for name in var_names
                )
            elif assumptions:
                vars = tuple(
                    symbol(name, assumptions=assumptions) for name in var_names
                )
            else:
                vars = tuple(symbol(name) for name in var_names)
            var_values = vars
            var_values_flattened = vars
            if _doNotUpdateVar != passkey or (
                _calledFromCVP == passkey or _calledFromFactory == passkey
            ):  # CVP doesn't manage sub-parant names
                clearVar(*vars, report=False)
                new_globals = dict(zip(var_names, vars))
                new_globals[labelLoc] = vars
                update_working_namespace(new_globals)

            if enforce_real:
                register_enforced_real(
                    vars,
                    var_names,
                    assumptions,
                    targeted_assumptions,
                    enforced_real_dict,
                )

            vtuple = tuple(vars)

            def var_v_a(elem):
                return (
                    effective_assumptions(elem, assumptions, targeted_assumptions)
                    or None
                )

            if withVF:
                vf_instances = [
                    vector_field_class(
                        coeff_dict={(j, 1, labelLoc): 1},
                        dgcvType="standard",
                        variable_spaces={labelLoc: vtuple},
                    )
                    for j in range(len(vtuple))
                ]
                update_working_namespace(
                    dict(zip([f"D_{vn}" for vn in var_names], vf_instances))
                )

                df_instances = [
                    differential_form_class(
                        coeff_dict={(j, 0, labelLoc): 1},
                        dgcvType="standard",
                        variable_spaces={labelLoc: vtuple},
                    )
                    for j in range(len(vtuple))
                ]
                update_working_namespace(
                    dict(zip([f"d_{vn}" for vn in var_names], df_instances))
                )

                if _doNotUpdateVar != passkey:
                    variable_registry["standard_variable_systems"][labelLoc] = {
                        "family_type": "tuple",
                        "family_values": vtuple,
                        "family_names": tuple(var_names),
                        "differential_system": True,
                        "tempVar": temp_flag,
                        "initial_index": initialIndex,
                        "obsVar": obscure_flag,
                        "variable_relatives": {
                            var_name: {
                                "VFClass": vf_instances[i],
                                "DFClass": df_instances[i],
                                "assumptions": var_v_a(var_name),
                                "system_index": i,
                            }
                            for i, var_name in enumerate(var_names)
                        },
                    }
            else:
                if _doNotUpdateVar != passkey:
                    variable_registry["standard_variable_systems"][labelLoc] = {
                        "family_type": "tuple",
                        "family_values": vtuple,
                        "family_names": tuple(var_names),
                        "differential_system": False,
                        "tempVar": temp_flag,
                        "initial_index": initialIndex,
                        "obsVar": obscure_flag,
                        "variable_relatives": {
                            var_name: {
                                "VFClass": None,
                                "DFClass": None,
                                "assumptions": var_v_a(var_name),
                                "system_index": i,
                            }
                            for i, var_name in enumerate(var_names)
                        },
                    }
            variable_registry["_labels"][labelLoc] = {
                "path": ("standard_variable_systems", labelLoc),
                "children": set(var_names),
            }

            if paths is not None:
                paths[labelLoc] = {
                    "kind": "standard_variable_system",
                    "path": ("standard_variable_systems", labelLoc),
                }

                base_vals = ("standard_variable_systems", labelLoc, "family_values")
                base_rel = (
                    "standard_variable_systems",
                    labelLoc,
                    "variable_relatives",
                )

                for i, var_name in enumerate(var_names):
                    paths[var_name] = {
                        "kind": "standard_variable",
                        "path": base_vals + (i,),
                    }
                    if withVF:
                        paths[f"D_{var_name}"] = {
                            "kind": "vector_field",
                            "path": base_rel + (var_name, "VFClass"),
                        }
                        paths[f"d_{var_name}"] = {
                            "kind": "differential_form",
                            "path": base_rel + (var_name, "DFClass"),
                        }
        else:
            raise ValueError(
                "variableProcedure expected its second argument number_of_variables (optional) to be a positive integer, if provided."
            )

        if rco is not None:
            rco.append(var_values)
        if sl_out is not None:
            sl_out.append(var_names)
        if vvf is not None:
            vvf.append(var_values_flattened)
    if _return_flattened is True:
        return rco, sl_out, vvf
    if _return_labels:
        return rco, sl_out
    return rco
