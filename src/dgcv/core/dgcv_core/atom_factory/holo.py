from __future__ import annotations

from numbers import Integral, Number
from typing import Literal

from ...._aux._backends._symbolic_router import _engine_conjugate
from ...._aux._backends._types_and_constants import half, imag_unit
from ...._aux._utilities._config import (
    get_dgcv_settings_registry,
    get_variable_registry,
    update_working_namespace,
    update_working_namespace_k_v,
)
from ...._aux._vmf._safeguards import retrieve_passkey, validate_label
from ...._aux._vmf.vmf import clearVar
from ...combinatorics.combinatorics import carProd
from ..fields import differential_form_class, vector_field_class
from .atoms import variableProcedure


def complexVarProc(
    holom_label,
    real_label,
    im_label,
    number_of_variables=None,
    initialIndex=1,
    multiindex_shape=None,
    index_placement=None,
    default_var_format: Literal["complex", "real", "mixed"] = None,
    remove_guardrails=None,
    return_created_object=True,
    assumptions: dict = None,
    targeted_assumptions: dict = None,
):
    """
    Initializes a complex variable system, linking a holomorphic variable with its real and imaginary parts and
    a symbolic representative of its complex conjugate.
    """
    default_var_format = {None: "mixed", "real": "real", "complex": "complex"}.get(
        default_var_format, "mixed"
    )
    if not assumptions:
        assumptions = {"real": True}
    variable_registry = get_variable_registry()
    conv = variable_registry["conversion_dictionaries"]
    find_parents = conv["find_parents"]
    # protected_vars = variable_registry["protected_variables"]

    conj_updates = {}
    holToReal_updates = {}
    realToSym_updates = {}
    symToHol_updates = {}
    symToReal_updates = {}
    realToHol_updates = {}
    real_part_updates = {}
    im_part_updates = {}
    complex_system_updates = {}

    # entry format: (labelLoc1, var_names1, var_namesBAR, var_names2, var_names3, lengthLoc)
    tuple_system_data = []

    def _register_complex_paths(system_label: str) -> None:
        paths = variable_registry.get("paths", None)
        if paths is None:
            return

        sys = variable_registry.get("complex_variable_systems", {}).get(system_label)
        if not isinstance(sys, dict):
            return

        paths[system_label] = {
            "kind": "complex_variable_system",
            "path": ("complex_variable_systems", system_label),
        }

        houses = sys.get("family_houses")
        if isinstance(houses, (tuple, list)):
            base = ("complex_variable_systems", system_label, "family_houses")
            for i, house in enumerate(houses):
                if isinstance(house, str):
                    paths[house] = {
                        "kind": "complex_variable_house",
                        "path": base + (i,),
                    }

        rel = sys.get("variable_relatives")
        if isinstance(rel, dict):
            base = ("complex_variable_systems", system_label, "variable_relatives")
            for member_label in rel.keys():
                if isinstance(member_label, str):
                    paths[member_label] = {
                        "kind": "coordinate",
                        "path": base + (member_label, "variable_value"),
                    }

    def validate_variable_labels(*labels, remove_guardrails=False):
        reformatted_labels = []
        seen_labels = set()
        for label in labels:
            reformatted_label = validate_label(
                label, remove_guardrails=remove_guardrails
            )
            if reformatted_label in seen_labels:
                raise ValueError(
                    f"Duplicate label found while formatting '{labels}'. Each label must be unique."
                )
            seen_labels.add(reformatted_label)
            reformatted_labels.append(reformatted_label)
        return tuple(reformatted_labels)

    if isinstance(holom_label, str):
        holom_label = [holom_label]
        real_label = [real_label]
        im_label = [im_label]

    rco = [] if return_created_object is True else None

    pref = get_dgcv_settings_registry().get("conjugation_prefix", "BAR")
    for j in range(len(holom_label)):
        if remove_guardrails:
            labelLoc1 = holom_label[j]
            labelLoc2 = real_label[j]
            labelLoc3 = im_label[j]
        else:
            labelLoc1, labelLoc2, labelLoc3 = validate_variable_labels(
                holom_label[j], real_label[j], im_label[j]
            )
        labelLocBAR = f"{pref}{labelLoc1}"

        clearVar(labelLoc1, report=False)
        clearVar(labelLoc2, report=False)
        clearVar(labelLoc3, report=False)
        clearVar(labelLocBAR, report=False)

        def _valid_multiindex_shape(ms):
            return isinstance(ms, (list, tuple)) and all(
                isinstance(n, Integral) and n > 0 for n in ms
            )

        def _multiindex_indices(ms):
            return list(carProd(*[range(initialIndex, initialIndex + n) for n in ms]))

        def _multiindex_names(base, idxs):
            return [f"{base}_{'_'.join(map(str, idx))}" for idx in idxs]

        # protected_vars.update({labelLoc2, labelLoc3})

        # Multi-index System Case
        if number_of_variables is None and _valid_multiindex_shape(multiindex_shape):
            var_arr1_t, var_str1_t, flat1_t = variableProcedure(
                labelLoc1,
                initialIndex=initialIndex,
                multiindex_shape=multiindex_shape,
                index_placement=index_placement,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
                _return_flattened=True,
            )
            var_arrBAR_t, var_strBAR_t, flatBAR_t = variableProcedure(
                labelLocBAR,
                initialIndex=initialIndex,
                multiindex_shape=multiindex_shape,
                index_placement=index_placement,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
                return_created_object=True,
                _return_flattened=True,
            )
            var_arr2_t, var_str2_t, flat2_t = variableProcedure(
                labelLoc2,
                initialIndex=initialIndex,
                multiindex_shape=multiindex_shape,
                index_placement=index_placement,
                assumptions=assumptions,
                targeted_assumptions=targeted_assumptions,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
                return_created_object=True,
                _return_flattened=True,
            )
            var_arr3_t, var_str3_t, flat3_t = variableProcedure(
                labelLoc3,
                initialIndex=initialIndex,
                multiindex_shape=multiindex_shape,
                index_placement=index_placement,
                assumptions=assumptions,
                targeted_assumptions=targeted_assumptions,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
                return_created_object=True,
                _return_flattened=True,
            )
            var_arr1, var_arrBAR, var_arr2, var_arr3 = (
                var_arr1_t[0],
                var_arrBAR_t[0],
                var_arr2_t[0],
                var_arr3_t[0],
            )
            var_str1, var_strBAR, var_str2, var_str3 = (
                var_str1_t[0],
                var_strBAR_t[0],
                var_str2_t[0],
                var_str3_t[0],
            )
            flat1, flatBAR, flat2, flat3 = (
                flat1_t[0],
                flatBAR_t[0],
                flat2_t[0],
                flat3_t[0],
            )
            if rco is not None:
                rco += [var_arr1, var_arrBAR, var_arr2, var_arr3]

            complex_system_updates[labelLoc1] = {
                "family_type": "multi_index",
                "family_shape": tuple(multiindex_shape),
                "family_names": (var_str1, var_strBAR, var_str2, var_str3),
                "family_values": (var_arr1, var_arrBAR, var_arr2, var_arr3),
                "family_houses": (labelLoc1, labelLocBAR, labelLoc2, labelLoc3),
                "differential_system": True,
                "initial_index": initialIndex,
                "variable_relatives": {},
            }

            all_var_strs = list(var_str1 + var_strBAR + var_str2 + var_str3)
            variable_registry["_labels"][labelLoc1] = {
                "path": ("complex_variable_systems", labelLoc1),
                "children": set(
                    all_var_strs
                    + [f"D_{v}" for v in all_var_strs]
                    + [f"d_{v}" for v in all_var_strs]
                ),
            }

            totalVarListLoc = list(zip(flat1, flatBAR, flat2, flat3))
            for comp_var, bar_comp_var, real_var, imag_var in totalVarListLoc:
                find_parents[real_var] = (comp_var, bar_comp_var)
                find_parents[imag_var] = (comp_var, bar_comp_var)

                conj_updates[comp_var] = bar_comp_var
                conj_updates[bar_comp_var] = comp_var
                holToReal_updates[comp_var] = real_var + imag_unit() * imag_var
                realToSym_updates[real_var] = half() * (comp_var + bar_comp_var)
                realToSym_updates[imag_var] = (
                    -imag_unit() * half() * (comp_var - bar_comp_var)
                )
                symToHol_updates[bar_comp_var] = _engine_conjugate(comp_var)
                symToReal_updates[comp_var] = real_var + imag_unit() * imag_var
                symToReal_updates[bar_comp_var] = real_var - imag_unit() * imag_var
                realToHol_updates[real_var] = half() * (comp_var + _engine_conjugate(comp_var))
                realToHol_updates[imag_var] = (
                    imag_unit() * half() * (_engine_conjugate(comp_var) - comp_var)
                )
                real_part_updates[comp_var] = real_var
                real_part_updates[bar_comp_var] = real_var
                im_part_updates[comp_var] = imag_var
                im_part_updates[bar_comp_var] = -imag_var

            tuple_system_data.append(
                (
                    labelLoc1,
                    flat1,
                    flatBAR,
                    flat2,
                    flat3,
                    len(flat1),
                )
            )

            conv["conjugation"].update(conj_updates)
            conv["holToReal"].update(holToReal_updates)
            conv["realToSym"].update(realToSym_updates)
            conv["symToHol"].update(symToHol_updates)
            conv["symToReal"].update(symToReal_updates)
            conv["realToHol"].update(realToHol_updates)
            conv["real_part"].update(real_part_updates)
            conv["im_part"].update(im_part_updates)

        # Single Variable System
        elif number_of_variables is None:
            var_hol_tuple = variableProcedure(
                labelLoc1,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
                return_created_object=True,
            )[0]
            var_bar_tuple = variableProcedure(
                labelLocBAR,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
                return_created_object=True,
            )[0]
            var_real_tuple = variableProcedure(
                labelLoc2,
                _doNotUpdateVar=retrieve_passkey(),
                assumptions=assumptions,
                targeted_assumptions=targeted_assumptions,
                _calledFromCVP=retrieve_passkey(),
                return_created_object=True,
            )[0]
            var_im_tuple = variableProcedure(
                labelLoc3,
                _doNotUpdateVar=retrieve_passkey(),
                assumptions=assumptions,
                targeted_assumptions=targeted_assumptions,
                _calledFromCVP=retrieve_passkey(),
                return_created_object=True,
            )[0]

            if rco is not None:
                rco += [var_hol_tuple, var_bar_tuple, var_real_tuple, var_im_tuple]
            var_hol = var_hol_tuple[0]
            var_bar = var_bar_tuple[0]
            var_real = var_real_tuple[0]
            var_im = var_im_tuple[0]

            # conversion updates
            conj_updates[var_hol] = var_bar
            conj_updates[var_bar] = var_hol
            holToReal_updates[var_hol] = var_real + imag_unit() * var_im
            realToSym_updates[var_real] = half() * (var_hol + var_bar)
            realToSym_updates[var_im] = -imag_unit() * half() * (var_hol - var_bar)
            symToHol_updates[var_bar] = _engine_conjugate(var_hol)
            symToReal_updates[var_hol] = var_real + imag_unit() * var_im
            symToReal_updates[var_bar] = var_real - imag_unit() * var_im
            realToHol_updates[var_real] = half() * (var_hol + _engine_conjugate(var_hol))
            realToHol_updates[var_im] = (
                imag_unit() * half() * (_engine_conjugate(var_hol) - var_hol)
            )
            real_part_updates[var_hol] = var_real
            real_part_updates[var_bar] = var_real
            im_part_updates[var_hol] = var_im
            im_part_updates[var_bar] = -var_im

            conv["conjugation"].update(conj_updates)
            conv["holToReal"].update(holToReal_updates)
            conv["realToSym"].update(realToSym_updates)
            conv["symToHol"].update(symToHol_updates)
            conv["symToReal"].update(symToReal_updates)
            conv["realToHol"].update(realToHol_updates)
            conv["real_part"].update(real_part_updates)
            conv["im_part"].update(im_part_updates)

            # update VMF
            def var_v_a(elem):  # for registering assumptions
                if targeted_assumptions:
                    t_a = targeted_assumptions.get(elem, None)
                    return {"real": True} | t_a if t_a else assumptions
                return assumptions

            variable_registry["complex_variable_systems"][labelLoc1] = {
                "family_type": "single",
                "family_names": (
                    (labelLoc1,),
                    (labelLocBAR,),
                    (labelLoc2,),
                    (labelLoc3,),
                ),
                "family_values": (var_hol, var_bar, var_real, var_im),
                "family_houses": (labelLoc1, labelLocBAR, labelLoc2, labelLoc3),
                "differential_system": True,
                "initial_index": None,
                "variable_relatives": {
                    labelLoc1: {
                        "complex_positioning": "holomorphic",
                        "complex_family": (var_hol, var_bar, var_real, var_im),
                        "variable_value": var_hol,
                        "VFClass": None,
                        "DFClass": None,
                        "assumptions": None,
                        "system_index": 0,
                    },
                    labelLocBAR: {
                        "complex_positioning": "antiholomorphic",
                        "complex_family": (var_hol, var_bar, var_real, var_im),
                        "variable_value": var_bar,
                        "VFClass": None,
                        "DFClass": None,
                        "assumptions": None,
                        "system_index": 1,
                    },
                    labelLoc2: {
                        "complex_positioning": "real",
                        "complex_family": (var_hol, var_bar, var_real, var_im),
                        "variable_value": var_real,
                        "VFClass": None,
                        "DFClass": None,
                        "assumptions": var_v_a(labelLoc2),
                        "system_index": 2,
                    },
                    labelLoc3: {
                        "complex_positioning": "imaginary",
                        "complex_family": (var_hol, var_bar, var_real, var_im),
                        "variable_value": var_im,
                        "VFClass": None,
                        "DFClass": None,
                        "assumptions": var_v_a(labelLoc3),
                        "system_index": 3,
                    },
                },
            }
            _register_complex_paths(labelLoc1)

            variable_registry["_labels"][labelLoc1] = {
                "path": ("complex_variable_systems", labelLoc1),
                "children": {
                    labelLocBAR,
                    labelLoc2,
                    labelLoc3,
                    f"D_{labelLoc1}",
                    f"d_{labelLoc1}",
                    f"D_{labelLocBAR}",
                    f"d_{labelLocBAR}",
                    f"D_{labelLoc2}",
                    f"d_{labelLoc2}",
                    f"D_{labelLoc3}",
                    f"d_{labelLoc3}",
                },
            }

            def create_differential_objects_single(
                var_hol, var_bar, var_real, var_im, default_var_format
            ):
                vs = (var_hol, var_bar, var_real, var_im)
                sys = labelLoc1  # the complex system label
                if default_var_format == "real":
                    # Differential objects using the real/imaginary parts.
                    inh_dict = {"_validated_format": "real"}
                    vf_instance_hol = vector_field_class(
                        coeff_dict={(2, 1, sys): half(), (3, 1, sys): -imag_unit() / 2},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    vf_instance_aHol = vector_field_class(
                        coeff_dict={(2, 1, sys): half(), (3, 1, sys): imag_unit() / 2},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    vf_instance_real = vector_field_class(
                        coeff_dict={(2, 1, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    vf_instance_im = vector_field_class(
                        coeff_dict={(3, 1, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_hol = differential_form_class(
                        coeff_dict={(2, 0, sys): 1, (3, 0, sys): imag_unit()},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_aHol = differential_form_class(
                        coeff_dict={(2, 0, sys): 1, (3, 0, sys): -imag_unit()},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_real = differential_form_class(
                        coeff_dict={(2, 0, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_im = differential_form_class(
                        coeff_dict={(3, 0, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                elif default_var_format == "complex":
                    inh_dict = {"_validated_format": "complex"}
                    vf_instance_hol = vector_field_class(
                        coeff_dict={(0, 1, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    vf_instance_aHol = vector_field_class(
                        coeff_dict={(1, 1, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    vf_instance_real = vector_field_class(
                        coeff_dict={(0, 1, sys): 1, (1, 1, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    vf_instance_im = vector_field_class(
                        coeff_dict={
                            (0, 1, sys): imag_unit(),
                            (1, 1, sys): -imag_unit(),
                        },
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_hol = differential_form_class(
                        coeff_dict={(0, 0, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_aHol = differential_form_class(
                        coeff_dict={(1, 0, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_real = differential_form_class(
                        coeff_dict={(0, 0, sys): half(), (1, 0, sys): half()},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_im = differential_form_class(
                        coeff_dict={
                            (0, 0, sys): -imag_unit() / 2,
                            (1, 0, sys): imag_unit() / 2,
                        },
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                else:  # default_var_format == "mixed"
                    inh_dict = {"_validated_format": "complex"}
                    inh_dict_real = {"_validated_format": "real"}
                    vf_instance_hol = vector_field_class(
                        coeff_dict={(0, 1, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    vf_instance_aHol = vector_field_class(
                        coeff_dict={(1, 1, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    vf_instance_real = vector_field_class(
                        coeff_dict={(2, 1, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict_real,
                    )
                    vf_instance_im = vector_field_class(
                        coeff_dict={(3, 1, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict_real,
                    )
                    df_instance_hol = differential_form_class(
                        coeff_dict={(0, 0, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_aHol = differential_form_class(
                        coeff_dict={(1, 0, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict,
                    )
                    df_instance_real = differential_form_class(
                        coeff_dict={(2, 0, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict_real,
                    )
                    df_instance_im = differential_form_class(
                        coeff_dict={(3, 0, sys): 1},
                        dgcvType="complex",
                        variable_spaces={sys: vs},
                        _inheritance=inh_dict_real,
                    )
                return (
                    vf_instance_hol,
                    vf_instance_aHol,
                    vf_instance_real,
                    vf_instance_im,
                    df_instance_hol,
                    df_instance_aHol,
                    df_instance_real,
                    df_instance_im,
                )

            # differential objects for singleton systems
            (
                vf_instance_hol,
                vf_instance_aHol,
                vf_instance_real,
                vf_instance_im,
                df_instance_hol,
                df_instance_aHol,
                df_instance_real,
                df_instance_im,
            ) = create_differential_objects_single(
                var_hol, var_bar, var_real, var_im, default_var_format
            )

            update_working_namespace(
                {
                    f"D_{labelLoc1}": vf_instance_hol,
                    f"D_{labelLocBAR}": vf_instance_aHol,
                    f"d_{labelLoc1}": df_instance_hol,
                    f"d_{labelLocBAR}": df_instance_aHol,
                    f"D_{labelLoc2}": vf_instance_real,
                    f"D_{labelLoc3}": vf_instance_im,
                    f"d_{labelLoc2}": df_instance_real,
                    f"d_{labelLoc3}": df_instance_im,
                }
            )

            find_parents[var_real] = (var_hol, var_bar)
            find_parents[var_im] = (var_hol, var_bar)

            address = variable_registry["complex_variable_systems"][labelLoc1][
                "variable_relatives"
            ]
            address[labelLoc1] |= {
                "VFClass": vf_instance_hol,
                "DFClass": df_instance_hol,
            }
            address[labelLocBAR] |= {
                "VFClass": vf_instance_aHol,
                "DFClass": df_instance_aHol,
            }
            address[labelLoc2] |= {
                "VFClass": vf_instance_real,
                "DFClass": df_instance_real,
            }
            address[labelLoc3] |= {
                "VFClass": vf_instance_im,
                "DFClass": df_instance_im,
            }

        # tuple system case
        elif isinstance(number_of_variables, Number) and number_of_variables > 0:
            lengthLoc = number_of_variables
            var_names1, var_names1_str = variableProcedure(
                labelLoc1,
                lengthLoc,
                initialIndex=initialIndex,
                index_placement=index_placement,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
                return_created_object=True,
                _return_labels=True,
            )
            var_namesBAR, var_namesBAR_str = variableProcedure(
                labelLocBAR,
                lengthLoc,
                initialIndex=initialIndex,
                index_placement=index_placement,
                _doNotUpdateVar=retrieve_passkey(),
                _calledFromCVP=retrieve_passkey(),
                return_created_object=True,
                _return_labels=True,
            )
            var_names2, var_names2_str = variableProcedure(
                labelLoc2,
                lengthLoc,
                initialIndex=initialIndex,
                index_placement=index_placement,
                _doNotUpdateVar=retrieve_passkey(),
                assumptions=assumptions,
                targeted_assumptions=targeted_assumptions,
                _calledFromCVP=retrieve_passkey(),
                return_created_object=True,
                _return_labels=True,
            )
            var_names3, var_names3_str = variableProcedure(
                labelLoc3,
                lengthLoc,
                initialIndex=initialIndex,
                index_placement=index_placement,
                _doNotUpdateVar=retrieve_passkey(),
                assumptions=assumptions,
                targeted_assumptions=targeted_assumptions,
                _calledFromCVP=retrieve_passkey(),
                return_created_object=True,
                _return_labels=True,
            )
            var_names1, var_namesBAR, var_names2, var_names3 = (
                var_names1[0],
                var_namesBAR[0],
                var_names2[0],
                var_names3[0],
            )
            if rco is not None:
                rco += [var_names1, var_namesBAR, var_names2, var_names3]

            # Build string labels for registry.
            var_str1, var_strBAR, var_str2, var_str3 = (
                var_names1_str[0],
                var_namesBAR_str[0],
                var_names2_str[0],
                var_names3_str[0],
            )

            # VMF update
            complex_system_updates[labelLoc1] = {
                "family_type": "tuple",
                "family_names": (var_str1, var_strBAR, var_str2, var_str3),
                "family_values": (var_names1, var_namesBAR, var_names2, var_names3),
                "family_houses": (labelLoc1, labelLocBAR, labelLoc2, labelLoc3),
                "differential_system": True,
                "initial_index": initialIndex,
                "variable_relatives": {},
            }

            all_var_strs = list(var_str1 + var_strBAR + var_str2 + var_str3)
            variable_registry["_labels"][labelLoc1] = {
                "path": ("complex_variable_systems", labelLoc1),
                "children": set(
                    all_var_strs
                    + [f"D_{v}" for v in all_var_strs]
                    + [f"d_{v}" for v in all_var_strs]
                ),
            }

            # conversion dict updates
            totalVarListLoc = list(
                zip(var_names1, var_namesBAR, var_names2, var_names3)
            )
            for idx, (comp_var, bar_comp_var, real_var, imag_var) in enumerate(
                totalVarListLoc
            ):
                find_parents[real_var] = (comp_var, bar_comp_var)
                find_parents[imag_var] = (comp_var, bar_comp_var)

                conj_updates[comp_var] = bar_comp_var
                conj_updates[bar_comp_var] = comp_var
                holToReal_updates[comp_var] = real_var + imag_unit() * imag_var
                realToSym_updates[real_var] = half() * (comp_var + bar_comp_var)
                realToSym_updates[imag_var] = (
                    -imag_unit() * half() * (comp_var - bar_comp_var)
                )
                symToHol_updates[bar_comp_var] = _engine_conjugate(comp_var)
                symToReal_updates[comp_var] = real_var + imag_unit() * imag_var
                symToReal_updates[bar_comp_var] = real_var - imag_unit() * imag_var
                realToHol_updates[real_var] = half() * (comp_var + _engine_conjugate(comp_var))
                realToHol_updates[imag_var] = (
                    imag_unit() * half() * (_engine_conjugate(comp_var) - comp_var)
                )
                real_part_updates[comp_var] = real_var
                real_part_updates[bar_comp_var] = real_var
                im_part_updates[comp_var] = imag_var
                im_part_updates[bar_comp_var] = -imag_var

            tuple_system_data.append(
                (labelLoc1, var_names1, var_namesBAR, var_names2, var_names3, lengthLoc)
            )

            conv["conjugation"].update(conj_updates)
            conv["holToReal"].update(holToReal_updates)
            conv["realToSym"].update(realToSym_updates)
            conv["symToHol"].update(symToHol_updates)
            conv["symToReal"].update(symToReal_updates)
            conv["realToHol"].update(realToHol_updates)
            conv["real_part"].update(real_part_updates)
            conv["im_part"].update(im_part_updates)
        else:
            raise ValueError(
                "variableProcedure expected its second argument number_of_variables to be a positive integer, if provided."
            )

    variable_registry["complex_variable_systems"].update(complex_system_updates)

    # differential objects
    for (
        labelLoc1,
        var_names1,
        var_namesBAR,
        var_names2,
        var_names3,
        lengthLoc,
    ) in tuple_system_data:
        relatives = variable_registry["complex_variable_systems"][labelLoc1][
            "variable_relatives"
        ]
        totalVarListLoc = list(zip(var_names1, var_namesBAR, var_names2, var_names3))

        # conversion dictionary updates
        conj_updates_batch = {comp: anti for comp, anti, _, _ in totalVarListLoc}
        conj_updates_batch.update({anti: comp for comp, anti, _, _ in totalVarListLoc})

        holToReal_updates_batch = {
            comp: real + imag_unit() * imag for comp, _, real, imag in totalVarListLoc
        }

        realToSym_updates_batch = {}
        for comp, anti, real, imag in totalVarListLoc:
            realToSym_updates_batch[real] = half() * (comp + anti)
            realToSym_updates_batch[imag] = -imag_unit() * half() * (comp - anti)

        symToHol_updates_batch = {
            anti: _engine_conjugate(comp) for comp, anti, _, _ in totalVarListLoc
        }

        symToReal_updates_batch = {}
        for comp, anti, real, imag in totalVarListLoc:
            symToReal_updates_batch[comp] = real + imag_unit() * imag
            symToReal_updates_batch[anti] = real - imag_unit() * imag

        realToHol_updates_batch = {}
        for comp, _, real, _ in totalVarListLoc:
            realToHol_updates_batch[real] = half() * (comp + _engine_conjugate(comp))
        for comp, _, _, imag in totalVarListLoc:
            realToHol_updates_batch[imag] = (
                imag_unit() * half() * (_engine_conjugate(comp) - comp)
            )

        real_part_updates_batch = {}
        im_part_updates_batch = {}
        for comp, anti, real, imag in totalVarListLoc:
            real_part_updates_batch[comp] = real
            real_part_updates_batch[anti] = real
            im_part_updates_batch[comp] = imag
            im_part_updates_batch[anti] = -imag

        # differential objects
        for idx, (comp_var, bar_comp_var, real_var, imag_var) in enumerate(
            totalVarListLoc
        ):
            sys = labelLoc1
            N = lengthLoc

            vs = (
                tuple(var_names1)
                + tuple(var_namesBAR)
                + tuple(var_names2)
                + tuple(var_names3)
            )
            if default_var_format == "real":
                i_real = idx + 2 * N
                i_im = idx + 3 * N

                inh_dict = {"_validated_format": "real"}
                D_comp = vector_field_class(
                    coeff_dict={
                        (i_real, 1, sys): half(),
                        (i_im, 1, sys): -imag_unit() / 2,
                    },
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                D_bar_comp = vector_field_class(
                    coeff_dict={
                        (i_real, 1, sys): half(),
                        (i_im, 1, sys): imag_unit() / 2,
                    },
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                D_real = vector_field_class(
                    coeff_dict={(i_real, 1, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                D_im = vector_field_class(
                    coeff_dict={(i_im, 1, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_comp = differential_form_class(
                    coeff_dict={(i_real, 0, sys): 1, (i_im, 0, sys): imag_unit()},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_bar_comp = differential_form_class(
                    coeff_dict={(i_real, 0, sys): 1, (i_im, 0, sys): -imag_unit()},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_real = differential_form_class(
                    coeff_dict={(i_real, 0, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_im = differential_form_class(
                    coeff_dict={(i_im, 0, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
            elif default_var_format == "complex":
                sys = labelLoc1
                N = lengthLoc

                vs = (
                    tuple(var_names1)
                    + tuple(var_namesBAR)
                    + tuple(var_names2)
                    + tuple(var_names3)
                )

                i_hol = idx
                i_anti = idx + N
                inh_dict = {"_validated_format": "complex"}

                D_comp = vector_field_class(
                    coeff_dict={(i_hol, 1, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                D_bar_comp = vector_field_class(
                    coeff_dict={(i_anti, 1, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                D_real = vector_field_class(
                    coeff_dict={
                        (i_hol, 1, sys): 1,
                        (i_anti, 1, sys): 1,
                    },
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                D_im = vector_field_class(
                    coeff_dict={
                        (i_hol, 1, sys): imag_unit(),
                        (i_anti, 1, sys): -imag_unit(),
                    },
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_comp = differential_form_class(
                    coeff_dict={(i_hol, 0, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_bar_comp = differential_form_class(
                    coeff_dict={(i_anti, 0, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_real = differential_form_class(
                    coeff_dict={(i_hol, 0, sys): half(), (i_anti, 0, sys): half()},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_im = differential_form_class(
                    coeff_dict={
                        (i_hol, 0, sys): -imag_unit() / 2,
                        (i_anti, 0, sys): imag_unit() / 2,
                    },
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
            else:  # default_var_format == "mixed"
                sys = labelLoc1
                N = lengthLoc

                vs = (
                    tuple(var_names1)
                    + tuple(var_namesBAR)
                    + tuple(var_names2)
                    + tuple(var_names3)
                )

                i_hol = idx
                i_anti = idx + N
                i_real = idx + 2 * N
                i_im = idx + 3 * N
                inh_dict = {"_validated_format": "complex"}
                inh_dict_real = {"_validated_format": "real"}

                D_comp = vector_field_class(
                    coeff_dict={(i_hol, 1, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                D_bar_comp = vector_field_class(
                    coeff_dict={(i_anti, 1, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                D_real = vector_field_class(
                    coeff_dict={(i_real, 1, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict_real,
                )
                D_im = vector_field_class(
                    coeff_dict={(i_im, 1, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict_real,
                )
                d_comp = differential_form_class(
                    coeff_dict={(i_hol, 0, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_bar_comp = differential_form_class(
                    coeff_dict={(i_anti, 0, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict,
                )
                d_real = differential_form_class(
                    coeff_dict={(i_real, 0, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict_real,
                )
                d_im = differential_form_class(
                    coeff_dict={(i_im, 0, sys): 1},
                    dgcvType="complex",
                    variable_spaces={sys: vs},
                    _inheritance=inh_dict_real,
                )

            # Register the differential objects in VMF
            update_working_namespace_k_v(f"D_{comp_var}", D_comp)
            update_working_namespace_k_v(f"D_{bar_comp_var}", D_bar_comp)
            update_working_namespace_k_v(f"d_{comp_var}", d_comp)
            update_working_namespace_k_v(f"d_{bar_comp_var}", d_bar_comp)
            update_working_namespace_k_v(f"D_{real_var}", D_real)
            update_working_namespace_k_v(f"D_{imag_var}", D_im)
            update_working_namespace_k_v(f"d_{real_var}", d_real)
            update_working_namespace_k_v(f"d_{imag_var}", d_im)

            # update VMF
            def var_v_a(elem):  # for registering assumptions
                if targeted_assumptions:
                    t_a = targeted_assumptions.get(elem, None)
                    return {"real": True} | t_a if t_a else assumptions
                return assumptions

            relatives[str(comp_var)] = {
                "complex_positioning": "holomorphic",
                "complex_family": (comp_var, bar_comp_var, real_var, imag_var),
                "variable_value": comp_var,
                "VFClass": D_comp,
                "DFClass": d_comp,
                "assumptions": None,
                "system_index": idx,
            }
            relatives[str(bar_comp_var)] = {
                "complex_positioning": "antiholomorphic",
                "complex_family": (comp_var, bar_comp_var, real_var, imag_var),
                "variable_value": bar_comp_var,
                "VFClass": D_bar_comp,
                "DFClass": d_bar_comp,
                "assumptions": None,
                "system_index": idx + N,
            }
            srv = str(real_var)
            relatives[srv] = {
                "complex_positioning": "real",
                "complex_family": (comp_var, bar_comp_var, real_var, imag_var),
                "variable_value": real_var,
                "VFClass": D_real,
                "DFClass": d_real,
                "assumptions": var_v_a(srv),
                "system_index": idx + 2 * N,
            }
            siv = str(imag_var)
            relatives[siv] = {
                "complex_positioning": "imaginary",
                "complex_family": (comp_var, bar_comp_var, real_var, imag_var),
                "variable_value": imag_var,
                "VFClass": D_im,
                "DFClass": d_im,
                "assumptions": var_v_a(siv),
                "system_index": idx + 3 * N,
            }
            _register_complex_paths(labelLoc1)
    if tuple_system_data:
        conv["conjugation"].update(conj_updates_batch)
        conv["holToReal"].update(holToReal_updates_batch)
        conv["realToSym"].update(realToSym_updates_batch)
        conv["symToHol"].update(symToHol_updates_batch)
        conv["symToReal"].update(symToReal_updates_batch)
        conv["realToHol"].update(realToHol_updates_batch)
        conv["real_part"].update(real_part_updates_batch)
        conv["im_part"].update(im_part_updates_batch)

    rv = rco if return_created_object is True else None
    return rv
