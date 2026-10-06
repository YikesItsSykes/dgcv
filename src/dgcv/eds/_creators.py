"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.eds - Exterior Differential Systems

module: dgcv.eds._creators

---
Author (of this module): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/


Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

import numbers

from .._aux._utilities._config import (
    dgcv_warning,
    get_variable_registry,
    update_working_namespace,
    update_working_namespace_k_v,
    working_namespace,
)
from .._aux._vmf._safeguards import retrieve_passkey, validate_label
from .._aux._vmf.vmf import clearVar
from .._aux.printing.printing._eds import conjugation_prefix
from ..core.combinatorics.combinatorics import carProd
from ._atoms import zero_form_atom
from ._coframes import abstract_coframe
from ._forms import abstract_differential_form, abstract_differential_form_atom


def _register_atom_paths(variable_registry, label, family_names, conjugates, kind):
    paths = variable_registry["paths"]
    paths[label] = {"kind": "eds_atoms", "path": ("eds", label, "atoms")}
    for name in family_names:
        paths[name] = {"kind": kind, "path": ("eds", label, "atoms", name)}
    for name in conjugates:
        paths[name] = {
            "kind": f"{kind}_conjugate",
            "path": ("eds", label, "atoms", name),
        }


def _register_coframe(label, coframe, children, cousins, cousins_parent):
    vr = get_variable_registry()
    vr["eds"]["coframes"][label] = {
        "dimension": coframe.dimension,
        "coframe": coframe,
        "children": list(children),
        "cousins": list(cousins),
        "cousins_vals": (
            working_namespace()[cousins_parent] if cousins_parent is not None else None
        ),
        "cousins_parent": cousins_parent,
    }
    vr["_labels"][label] = {
        "path": ("eds", label, "coframes"),
        "children": set(children) | set(cousins),
    }
    vr["paths"][label] = {"kind": "coframe", "path": ("eds", label, "coframes")}
    update_working_namespace_k_v(label, coframe)


def createZeroForm(
    labels,
    index_set={},
    initial_index=1,
    assumeReal=False,
    coframe=None,
    coframe_independants=dict(),
    verbose_labeling=False,
    remove_guardrails=None,
):
    if not isinstance(labels, str):
        raise TypeError(
            "`createZeroForm` requires its first argument to be a string, which will be used in lables for the created zero forms."
        )

    def reformat_string(input_string: str):
        # Replace commas with spaces, then split on spaces
        substrings = input_string.replace(",", " ").split()
        # Return the list of non-empty substrings
        return [s for s in substrings if len(s) > 0]

    if isinstance(index_set, numbers.Integral) and index_set > 0:
        if not isinstance(initial_index, numbers.Integral):
            initial_index = 1
        index_set = {"lower": list(range(initial_index, index_set + initial_index))}
    elif not isinstance(index_set, dict):
        index_set = {}
    if isinstance(labels, str):
        labels = reformat_string(labels)

    for label in labels:
        _zeroFormFactory(
            label,
            index_set=index_set,
            assumeReal=assumeReal,
            coframe=coframe,
            coframe_independants=coframe_independants,
            verbose_labeling=verbose_labeling,
            _tempVar=None,
            _doNotUpdateVar=None,
            remove_guardrails=remove_guardrails,
        )


def _zeroFormFactory(
    label,
    index_set={},
    assumeReal=False,
    coframe=None,
    coframe_independants=dict(),
    verbose_labeling=False,
    _tempVar=None,
    _doNotUpdateVar=None,
    remove_guardrails=None,
):
    """
    Initializes zeroFormAtom systems, registers them in the VMF,
    and updates caller globals().

    Parameters:
    - label (str): Base label for the zeroFormAtom instances.
    - index_set (dict, optional): Determines tuple structure with 'upper' and 'lower' keys.
    - assumeReal (bool, optional): If True, marks instances as real and skips conjugate handling.
    - coframe (abstCoframe, optional): sets the primary coframe associated with the form.
    - _tempVar (None, optional): If set, marks the variable system as temporary.
    - _doNotUpdateVar (None, optional): If set, prevents clearing/replacing existing instances.
    - remove_guardrails (None, optional): If set, bypasses safeguards for label validation.
    """
    variable_registry = get_variable_registry()
    eds_atoms = variable_registry["eds"]["atoms"]
    passkey = retrieve_passkey()

    label = validate_label(label) if not remove_guardrails else label

    if _doNotUpdateVar is None:
        clearVar(label, report=False)

    if _tempVar == passkey:
        variable_registry["temporary_variables"].add(label)
        _tempVar = True
    else:
        _tempVar = None

    family_type = "single"
    if index_set:
        family_type = "tuple"

        def expand_indices(index):
            if index is None:
                return []
            if isinstance(index, numbers.Integral):
                return [[index]]
            if isinstance(index, list):
                return [[i] if isinstance(i, numbers.Integral) else i for i in index]
            raise ValueError("Indices must be an integer or a list of integers/lists.")

        upper_indices = expand_indices(index_set.get("upper", None))
        lower_indices = expand_indices(index_set.get("lower", None))
        lhPairs = index_set.get("low_hi_pairs", None)
        if (
            isinstance(lhPairs, (list, tuple))
            and len(lhPairs) == 1
            and len(lhPairs[0][0]) == 0
            and len(lhPairs[0][1]) == 0
        ):
            lhPairs = None

        if upper_indices and lower_indices:
            index_combinations = list(carProd(lower_indices, upper_indices))
            if lhPairs:
                index_combinations += list(lhPairs)
        elif lower_indices:
            index_combinations = [
                (lo, []) for lo in lower_indices
            ]  # Treat upper as empty
            if lhPairs:
                index_combinations += list(lhPairs)
        elif upper_indices:
            index_combinations = [
                ([], hi) for hi in upper_indices
            ]  # Treat lower as empty
            if lhPairs:
                index_combinations += list(lhPairs)
        elif lhPairs:
            index_combinations = list(lhPairs)
        else:
            index_combinations = [((), ())]  # No indices

        def labeler(index_pair, verbose):
            lower, upper = index_pair
            if verbose is True or len(upper) != 0 or len(lower) > 1:
                return (
                    f"{label}"
                    + (f"_low_{'_'.join(map(str, lower))}" if lower else "")
                    + (f"_hi_{'_'.join(map(str, upper))}" if upper else "")
                )
            elif len(lower) == 1:
                return f"{label}" + str(lower[0])
            else:
                return f"{label}"

        family_names = [
            labeler(index_pair, verbose_labeling) for index_pair in index_combinations
        ]
    else:
        family_names = [label]

    family_values = tuple(
        zero_form_atom(
            name,
            coframe=coframe,
            _markers={"real"} if assumeReal else frozenset(),
            coframe_independants=coframe_independants,
        )
        for name in family_names
    )

    # handle conjugates
    family_relatives = {}
    conjugates = {}
    for atom in family_values:
        if assumeReal:
            family_relatives[atom.label] = (atom, atom)  # Self-conjugate if real
        else:
            conj_atom = atom._eval_conjugate()
            family_relatives[atom.label] = (atom, conj_atom)
            family_relatives[conj_atom.label] = (atom, conj_atom)
            conjugates[conj_atom.label] = conj_atom

    # store in VMF
    eds_atoms[label] = {
        "family_type": family_type,
        "primary_coframe": coframe,
        "degree": 0,
        "family_values": family_values,
        "family_names": tuple(family_names),
        "tempVar": _tempVar,
        "real": assumeReal if assumeReal else None,
        "conjugates": conjugates,
        "family_relatives": family_relatives,
    }
    variable_registry["_labels"][label] = {
        "path": ("eds", label, "atoms"),
        "children": set(family_names + list(conjugates.keys())),
    }
    _register_atom_paths(
        variable_registry, label, family_names, conjugates, "zero_form"
    )

    # update active namespace
    update_working_namespace_k_v(
        label, family_values if family_type == "tuple" else family_values[0]
    )
    update_working_namespace(zip(family_names, family_values))
    if assumeReal is not True:
        pref = conjugation_prefix()
        update_working_namespace_k_v(f"{pref}{label}", tuple(conjugates.values()))
    update_working_namespace(conjugates)


def createDiffForm(
    labels,
    degree,
    number_of_variables=None,
    initialIndex=1,
    assumeReal=False,
    remove_guardrails=None,
    return_obj=False,
):
    """
    Initializes abstract_DF_atom systems, registers them in the VMF,
    and updates caller globals().

    Parameters:
    - label (str): Base label for the abstract_DF_atom instances.
    - degree (int): Degree of DF to be created.
    - number_of_variables (int, optional): Determines the number of DF created.
    - initialIndex (int, optional): Determines the starting index for DF enumeration
    - assumeReal (bool, optional): If True, marks instances as real and skips conjugate handling.
    - remove_guardrails (None, optional): If set, bypasses safeguards for label validation.
    """
    if isinstance(labels, (list, tuple)):
        if len(set(labels)) < len(labels):
            raise NameError(
                "`createDiffForm` was given the same label repeatedly. Labels need to be distinct."
            )
        if isinstance(degree, (list, tuple)):
            if len(degree) < len(labels):
                degree = list(degree) + ([degree[-1]] * (len(labels) - len(degree)))
        else:
            degree = [degree] * len(labels)
        if isinstance(number_of_variables, (list, tuple)):
            if len(number_of_variables) < len(labels):
                number_of_variables = list(number_of_variables) + (
                    [number_of_variables[-1]] * (len(labels) - len(number_of_variables))
                )
        else:
            number_of_variables = [number_of_variables] * len(labels)
        if isinstance(initialIndex, (list, tuple)):
            if len(initialIndex) < len(labels):
                initialIndex = list(initialIndex) + (
                    [initialIndex[-1]] * (len(labels) - len(initialIndex))
                )
        else:
            initialIndex = [initialIndex] * len(labels)
        if isinstance(assumeReal, (list, tuple)):
            if len(assumeReal) < len(labels):
                assumeReal = list(assumeReal) + (
                    [assumeReal[-1]] * (len(labels) - len(assumeReal))
                )
        else:
            assumeReal = [assumeReal] * len(labels)
        if isinstance(remove_guardrails, (list, tuple)):
            if len(remove_guardrails) < len(labels):
                remove_guardrails = list(remove_guardrails) + (
                    [remove_guardrails[-1]] * (len(labels) - len(remove_guardrails))
                )
        else:
            remove_guardrails = [remove_guardrails] * len(labels)
        if isinstance(return_obj, (list, tuple)):
            if len(return_obj) < len(labels):
                return_obj = list(return_obj) + (
                    [return_obj[-1]] * (len(labels) - len(return_obj))
                )
        else:
            return_obj = [return_obj] * len(labels)
        for idx in range(len(labels)):
            createDiffForm(
                labels[idx],
                degree[idx],
                number_of_variables[idx],
                initialIndex[idx],
                assumeReal[idx],
                remove_guardrails[idx],
                return_obj[idx],
            )
    else:
        if not isinstance(labels, str):
            raise TypeError(
                "`createDiffForm` requires its first argument to be a string, which will be used in lables for the created DF."
            )

        def reformat_string(input_string: str):
            # Replace commas with spaces, then split on spaces
            substrings = input_string.replace(",", " ").split()
            # Return the list of non-empty substrings
            return [s for s in substrings if len(s) > 0]

        if isinstance(labels, str):
            labels = reformat_string(labels)
        if return_obj is True:
            returnList = []
        for label in labels:
            if return_obj is True:
                returnList.append(
                    _DFFactory(
                        label,
                        degree,
                        number_of_variables=number_of_variables,
                        initialIndex=initialIndex,
                        assumeReal=assumeReal,
                        remove_guardrails=remove_guardrails,
                        return_obj=True,
                    )
                )
            else:
                _DFFactory(
                    label,
                    degree,
                    number_of_variables=number_of_variables,
                    initialIndex=initialIndex,
                    assumeReal=assumeReal,
                    remove_guardrails=remove_guardrails,
                )
        if return_obj is True:
            if len(returnList) == 1:
                return returnList[0]
            return returnList


def _DFFactory(
    label,
    degree,
    number_of_variables=None,
    initialIndex=1,
    assumeReal=False,
    _tempVar=None,
    _doNotUpdateVar=None,
    remove_guardrails=None,
    return_obj=False,
):
    """
    Initializes abstract_DF_atom systems, registers them in the VMF,
    and updates caller globals().

    Parameters:
    - label (str): Base label for the zeroFormAtom instances.
    - degree (int): Degree of DF to be created.
    - number_of_variables (int, optional): Determines the number of DF created.
    - initialIndex (int, optional): Determines the starting index for DF enumeration
    - assumeReal (bool, optional): If True, marks instances as real and skips conjugate handling.
    - _tempVar (None, optional): If set, marks the variable system as temporary.
    - _doNotUpdateVar (None, optional): If set, prevents clearing/replacing existing instances.
    - remove_guardrails (None, optional): If set, bypasses safeguards for label validation.
    """
    pref = conjugation_prefix()
    variable_registry = get_variable_registry()
    eds_atoms = variable_registry["eds"]["atoms"]
    passkey = retrieve_passkey()

    label = validate_label(label) if not remove_guardrails else label

    if _doNotUpdateVar is None:
        clearVar(label, report=False)

    if _tempVar == passkey:
        variable_registry["temporary_variables"].add(label)
        _tempVar = True
    else:
        _tempVar = None

    family_type = "single"
    if number_of_variables:
        family_type = "tuple"

        family_names = [
            f"{label}{index}"
            for index in range(initialIndex, number_of_variables + initialIndex)
        ]
    else:
        family_names = [label]

    family_values = tuple(
        abstract_differential_form_atom(
            1,
            degree,
            label=name,
            ext_deriv_order=0,
            _markers={"real"} if assumeReal else frozenset(),
        )
        for name in family_names
    )

    # handle conjugates
    family_relatives = {}
    conjugates = {}
    for atom in family_values:
        if assumeReal:
            family_relatives[atom.label] = (atom, atom)  # Self-conjugate if real
        else:
            conj_atom = atom._eval_conjugate()
            family_relatives[atom.label] = (atom, conj_atom)
            family_relatives[conj_atom.label] = (atom, conj_atom)
            conjugates[conj_atom.label] = conj_atom

    # store in VMF
    eds_atoms[label] = {
        "family_type": family_type,
        "primary_coframe": None,
        "degree": degree,
        "family_values": family_values,
        "family_names": tuple(family_names),
        "tempVar": _tempVar,
        "real": assumeReal if assumeReal else None,
        "conjugates": conjugates,
        "family_relatives": family_relatives,
    }

    children_set = set(family_names)
    if not assumeReal:
        children_set.update(conjugates.keys())
    variable_registry["_labels"][label] = {
        "path": ("eds", label, "atoms"),
        "children": children_set,
    }
    _register_atom_paths(
        variable_registry, label, family_names, conjugates, "differential_form"
    )

    # update namespaces
    label_value = family_values if family_type == "tuple" else family_values[0]
    out = {"original": label_value}
    update_working_namespace_k_v(label, label_value)
    update_working_namespace(zip(family_names, family_values))
    if assumeReal is not True:
        cl_value = tuple(conjugates.values())
        out["conjugated"] = cl_value if family_type == "tuple" else cl_value[0]
        update_working_namespace_k_v(f"{pref}{label}", cl_value)
    update_working_namespace(conjugates)
    if return_obj:
        return out


def createCoframe(
    label,
    coframe_labels,
    str_eqns=None,
    str_eqns_labels=None,
    complete_to_complex_cf=None,
    integrable_complex_struct=False,
    markers=dict(),
    remove_guardrails=False,
):
    """
    Create a coframe with specified 1-forms and structure equations.

    Parameters:
    - label (str): The name of the coframe.
    - coframe_labels (list of str): Labels for 1-forms in the coframe.
    - str_eqns (dict, optional): Pre-populated structure equations, keyed by (i, j, k) tuples.
    - str_eqns_labels (str, optional): Prefix for generating labels for missing terms in str_eqns.
    - markers (dict, optional): a dict whose key are strings from `coframe_labels`, and values of sets of properties associated with each coframe element
    - complete_to_complex_cf (any, optional): builds a larger coframe by include complex conjugate duals of coframe elements marked as holomorphic/antiholomorphic
    - remove_guardrails (bool, optional): Pass to validate_label for customization.
    """

    if str_eqns is None:
        str_eqns = {}
    min_conj_rules = {}

    coframe_labels = list(coframe_labels)

    if complete_to_complex_cf is None:
        complete_to_complex_cf = ["standard"] * len(coframe_labels)
    elif complete_to_complex_cf is True:
        complete_to_complex_cf = []
        for coVec in coframe_labels:
            if "holomorphic" in markers.get(coVec, {}):
                complete_to_complex_cf += ["holomorphic"]
                if "antiholomorphic" in markers.get(coVec, {}):
                    dgcv_warning(
                        'A coframe label was given to `createCoframe` with conflicting property markers "holomorphic" and "antiholomorphic". The coframe was created assuming "holomorphic" is correct'
                    )
                if "real" in markers.get(coVec, {}):
                    dgcv_warning(
                        'A coframe label was given to `createCoframe` with conflicting property markers "holomorphic" and "real". The coframe was created assuming "holomorphic" is correct'
                    )
            elif "antiholomorphic" in markers.get(coVec, {}):
                complete_to_complex_cf += ["antiholomorphic"]
                if "real" in markers.get(coVec, {}):
                    dgcv_warning(
                        'A coframe label was given to `createCoframe` with conflicting property markers "antiholomorphic" and "real". The coframe was created assuming "antiholomorphic" is correct'
                    )
            elif "real" in markers.get(coVec, {}):
                complete_to_complex_cf += ["real"]
            else:
                complete_to_complex_cf += ["standard"]
    elif complete_to_complex_cf == "fromHol":
        complete_to_complex_cf = ["holomorphic"] * len(coframe_labels)
    elif complete_to_complex_cf == "fromAntihol":
        complete_to_complex_cf = ["antiholomorphic"] * len(coframe_labels)
    elif not isinstance(complete_to_complex_cf, (list, tuple)) or len(
        complete_to_complex_cf
    ) != len(coframe_labels):
        dgcv_warning(
            "`createCoframe` was given an unexpected value for `complete_to_complex_cf`. Proceeding as if `complete_to_complex_cf=None`."
        )
        complete_to_complex_cf = ["standard"] * len(coframe_labels)

    closed_assumptions = [
        next(iter(markers[j].intersection({"closed"})), None) if j in markers else None
        for j in coframe_labels
    ]

    # Create abstract_DF_atom instances for coframe labels
    elem_list = []
    conjugates_list = []
    conjugates_labels = []
    augments_counter = 0
    for count, coframe_label in enumerate(coframe_labels):
        if complete_to_complex_cf[count] == "real":
            coframe_label = validate_label(
                coframe_label, remove_guardrails=remove_guardrails
            )
            elem = _DFFactory(coframe_label, 1, assumeReal=True, return_obj=True)[
                "original"
            ]
            elem_list.append(elem)
        elif (
            complete_to_complex_cf[count] == "holomorphic"
            or complete_to_complex_cf[count] == "antiholomorphic"
        ):
            pref = conjugation_prefix()
            prefL = len(pref)
            if coframe_label[0:prefL] == pref:
                paired_label = validate_label(
                    coframe_label[prefL:], remove_guardrails=remove_guardrails
                )
                elems = _DFFactory(paired_label, 1, return_obj=True)
                elem = elems["conjugated"]
                cElem = elems["original"]
            else:
                coframe_label = validate_label(
                    coframe_label, remove_guardrails=remove_guardrails
                )
                elems = _DFFactory(coframe_label, 1, return_obj=True)
                elem = elems["original"]
                cElem = elems["conjugated"]
            elem_list.append(elem)
            conjugates_list.append(cElem)
            conjugates_labels.append(cElem.label)
            min_conj_rules = min_conj_rules | {
                count: len(coframe_labels) + augments_counter
            }
            augments_counter += 1
        else:
            coframe_label = validate_label(
                coframe_label, remove_guardrails=remove_guardrails
            )
            elem = _DFFactory(coframe_label, 1, return_obj=True)["original"]
            elem_list.append(elem)

    init_dimension = len(coframe_labels)
    coframe_labels += conjugates_labels
    elem_list += conjugates_list

    coframe_dict = {elem: abstract_differential_form([]) for elem in elem_list}

    # register the coframe in VMF
    coframe = abstract_coframe(elem_list, coframe_dict, min_conj_rules)
    label = validate_label(label, remove_guardrails=remove_guardrails)
    update_working_namespace_k_v(label, coframe)

    # populate missing terms in str_eqns
    coeff_labels = []
    low_hi_pairs = []
    if str_eqns_labels is not None:
        if integrable_complex_struct:
            first_index_bound = init_dimension
        else:
            first_index_bound = len(coframe_labels)
        for i in range(first_index_bound):
            for j in range(i + 1, len(coframe_labels)):
                for k in range(init_dimension):
                    if (i, j, k) not in str_eqns or str_eqns[(i, j, k)] is None:
                        low_hi_pairs.append([[i + 1, j + 1], [k + 1]])
                        scale = 0 if closed_assumptions[k] == "closed" else 1
                        coeff_label = (
                            f"{str_eqns_labels}_low_{i + 1}_{j + 1}_hi_{k + 1}"
                        )
                        coeff_label = validate_label(
                            coeff_label, remove_guardrails=remove_guardrails
                        )

                        coeff_labels.append(coeff_label)

                        str_eqns[coeff_label] = [(i, j, k), scale]
        _zeroFormFactory(
            str_eqns_labels,
            {"low_hi_pairs": low_hi_pairs},
            coframe=coframe,
            remove_guardrails=remove_guardrails,
        )
        str_eqns = {
            (k if isinstance(k, tuple) else v[0]): (
                v if isinstance(k, tuple) else v[1] * working_namespace()[k]
            )
            for k, v in str_eqns.items()
        }
    else:
        # Fill missing terms with None
        for i in range(len(coframe_labels)):
            for j in range(i + 1, len(coframe_labels)):
                for k in range(len(coframe_labels)):
                    if (i, j, k) not in str_eqns:
                        str_eqns[(i, j, k)] = None

    # update coframe!!!!
    update_dict = {}
    for k in range(init_dimension):
        kth_term_list = []
        for i in range(len(coframe_labels)):
            for j in range(i + 1, len(coframe_labels)):
                if (i, j, k) in str_eqns and str_eqns[(i, j, k)] is not None:
                    kth_term_list.append(
                        str_eqns[(i, j, k)] * elem_list[i] * elem_list[j]
                    )
        if kth_term_list:
            update_dict[elem_list[k]] = abstract_differential_form(kth_term_list)
    inv_dict = {v: k for k, v in min_conj_rules.items()}
    for v in range(init_dimension, len(coframe_labels)):
        conj_eqn = update_dict.get(elem_list[inv_dict[v]])
        if conj_eqn is not None and not conj_eqn.is_zero:
            update_dict[elem_list[v]] = conj_eqn._eval_conjugate()

    coframe.update_structure_equations(replace_eqns=update_dict)

    _register_coframe(label, coframe, coframe_labels, coeff_labels, str_eqns_labels)
