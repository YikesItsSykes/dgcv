from __future__ import annotations

from numbers import Integral
from typing import List, Literal

from ...._aux._utilities._config import dgcv_warning, get_dgcv_settings_registry
from ...._aux._vmf._safeguards import (
    retrieve_passkey,
    retrieve_public_key,
    validate_creator_labels,
)
from ...combinatorics.combinatorics import carProd
from .atoms import indexing_places, variableProcedure
from .holo import complexVarProc
from .systems import varWithVF


def createVariables(
    variable_label: str,
    real_label: str | int | None = None,
    imaginary_label: str | None = None,
    number_of_variables: int | None = None,
    initialIndex: int | None = 1,
    withVF: bool | None = None,
    complex: bool | None = None,
    multiindex_shape: List[int] = None,
    index_placement: List[
        Literal["up", "down", "hi", "low", "h", "l", "u", "d", "__", "_"]
    ] = None,
    return_created_object: bool | None = None,
    remove_guardrails: bool | None = None,
    default_var_format: Literal["real", "complex", "mixed"] = None,
    temporary_variables: bool | None = False,
    assumptions: dict = None,
    targeted_assumptions: dict = None,
    **kwargs,
):
    """
    This function serves as the default interface for creating variables within the dgcv package. It supports creating
    both standard coordinate systems and complex coordinate systems, with options for initializing coordinate vector fields
    and differential forms. Variables created through `createVariables` are automatically tracked within dgcv's Variable
    Management Framework (VMF) and are assigned labels validated through a safeguards routine that prevents overwriting important labels (e.g., standard Python built-ins).

    Parameters
    ----------
    variable_label : str
        The label for the primary variable or system of variables to be created. If creating a complex variable system,
        this will correspond to the holomorphic variable(s), whilst antiholomorphic variable(s) recieve this label
        pre-pended with "BAR".

    real_label : str, optional
        The label for the real part of the complex variable system. Required only when creating complex variable systems.

    imaginary_label : str, optional
        The label for the imaginary part of the complex variable system. Required only when creating complex variable
        systems.

    number_of_variables : int, optional
        The number of variables to be created, used to initialize a tuple of variables rather than a single variable
        (e.g., x=(x1, x2, x3) rather than just x).

    initialIndex : int, optional, default=1
        The starting index for tuple variables, allowing for flexible indexing when initializing variable systems
        as tuples (e.g., x=(x0, x1, x2) with `initialIndex=0`).

    withVF : bool, optional (default True)
        If set to True, creates associated coordinate vector fields and differential forms for the variable(s) in the
        system.

    complex : bool, optional
        Specifies whether to create a complex variable system. If not provided, the function will infer whether to
        create a complex system based on whether `real_label` or `imaginary_label` is provided. If provided a complex
        variable system will be created regardless of `real_label` and `imaginary_label` settings, and string labels
        are automatically created for `real_label` and `imaginary_label` if they are not provided.

    multiindex_shape : tuple[int, ...], optional
        If provided, creates a multi-index variable system with shape given by `multiindex_shape`, using index values
        starting at `initialIndex`. For standard systems this creates an N-dimensional array handle stored in the global
        namespace under `variable_label` (or each label if a list is provided). For complex systems, this creates a
        multi-index complex variable system with holomorphic/antiholomorphic and real/imaginary parts.

    assumptions : dict, optional
        This can be a dictionary carrying optional assumptions. Currently supported assumptions are
        {'real':bool, 'nonnegative':bool, 'positive':bool}. Include any of those keys with a boolean value to apply the assumption.

    remove_guardrails : bool, optional
        If set to True, bypasses dgcv's safeguard system for variable labeling, allowing one to overwrite certain
        reserved labels. Use with caution, as it may overwrite important variables in the global namespace.

    default_var_format : {'complex', 'real'}, optional
        Relevant only for complex variable systems. Specifies whether the system's vector fields and differential forms
        default to real coordinate expressions (`real`) or holomorphic coordinate expressions (`complex`). If not
        provided, the default is holomorphic coordinates.

    temporary_variables : bool, optional
        If True, variables are created as temporary variables under dgcv's VMF conventions.

    Returns
    -------
    None or list
        If `return_created_object=True`, returns a list of created handles (for standard systems) or created labels (for
        complex systems, consistent with dgcv conventions). Otherwise returns None.

    Functionality
    -------------
    - Creates standard or complex variable systems.
    - Automatically registers all created variables, vector fields, and differential forms in dgcv's VMF.
    - Safeguards are applied to prevent overwriting critical Python or dgcv internal functions by default.

    Notes
    -----
    - For complex variable systems, dgcv initializes associated differential objects by default.
    - For multi-index standard systems, `withVF=True` is supported and will create VF/DF objects for each atomic entry.
    - For multi-index complex systems, VF/DF objects are created for each atomic entry in the system.
    - Use `vmf_summary()` for a clear summary of variables created and tracked within the dgcv VMF.
    """
    supported_assumptions = {"real", "nonnegative", "positive"}
    if assumptions is None:
        assumptions = dict()
    elif isinstance(assumptions, (list, tuple, set)):
        assumptions = {k: True for k in supported_assumptions if k in assumptions}
    elif isinstance(assumptions, str) and assumptions in supported_assumptions:
        assumptions = {assumptions: True}
    elif not isinstance(assumptions, dict):
        assumptions = dict()
        dgcv_warning(
            "The `assumptions` parameter given to `createVariables` was in an unsupported format."
        )
    if targeted_assumptions is None:
        targeted_assumptions = dict()
    elif not isinstance(targeted_assumptions, dict):
        targeted_assumptions = dict()
        dgcv_warning(
            "The `targeted_assumptions` parameter given to `createVariables` was in an unsupported format."
        )
    else:
        valid = True
        new_k_v = dict()
        for k, v in targeted_assumptions.items():
            if not isinstance(k, str):
                valid = False
                break
            if isinstance(v, (list, tuple, set)):
                for key in v:
                    if isinstance(key, "str") and key in supported_assumptions:
                        new_k_v[k] = new_k_v.get(k, {}) | {key: True}
            elif isinstance(v, dict):
                new_k_v[k] = {
                    key: v.get(key) for key in supported_assumptions if key in v
                }
            elif isinstance(v, str) and v in supported_assumptions:
                new_k_v[k] = {v: True}
            else:
                valid = False
                break
        if valid:
            targeted_assumptions = new_k_v
        else:
            targeted_assumptions = dict()
            dgcv_warning(
                "The `targeted_assumptions` parameter given to `createVariables` was in an unsupported format."
            )

    if kwargs.get("assumeReal", False) is True:  # deprecated kwarg support
        assumptions["real"] = True

    def _valid_multiindex_shape(ms):
        return isinstance(ms, (list, tuple)) and all(
            isinstance(n, Integral) and n > 0 for n in ms
        )

    def reformat_string(input_string: str):
        substrings = input_string.replace(",", " ").split()
        return [s for s in substrings if len(s) > 0]

    # validation checks
    if not isinstance(variable_label, str):
        raise TypeError(
            "`createVariables` requires its first argument to be a string, which will be used in labels for the created variables."
        )

    if (
        isinstance(real_label, Integral)
        and imaginary_label is None
        and number_of_variables is None
    ):
        number_of_variables = real_label
        real_label = None

    if real_label is not None and not isinstance(real_label, str):
        raise TypeError(
            "A non-string value cannot be assigned to the `real_label` keyword of `createVariables`."
        )
    if imaginary_label is not None and not isinstance(imaginary_label, str):
        raise TypeError(
            "A non-string value cannot be assigned to the `imaginary_label` keyword of `createVariables`."
        )

    if multiindex_shape is not None and not _valid_multiindex_shape(multiindex_shape):
        raise TypeError("`multiindex_shape` must be a tuple/list of positive integers.")

    if multiindex_shape is not None and number_of_variables is not None:
        raise ValueError(
            "Provide at most one of `number_of_variables` and `multiindex_shape`."
        )

    complex_requested = (
        complex is True or real_label is not None or imaginary_label is not None
    )

    if default_var_format is not None and not complex_requested:
        dgcv_warning(
            "`default_var_format` is only relevant for complex variable systems; it was disregarded."
        )
        default_var_format = None

    if complex and not withVF and not kwargs.get("skip_warnings", False):
        dgcv_warning(
            "`createVariables` was called with `complex=True` and `withVF=False`. The latter keyword was disregarded because "
            "dgcv initializes associated differential objects whenever complex variable systems are created."
        )

    if complex and assumptions.get("real", False) is True:
        dgcv_warning(
            "`createVariables` was called with `complex=True` and another keyword setting requested `assume real`. The latter setting was disregarded because it is incompatible with "
            "dgcv's variable assumptions for elements in its complex variable systems."
        )
        assumptions.pop("real", None)

    if complex is False and (real_label is not None or imaginary_label is not None):
        dgcv_warning(
            "`createVariables` received `complex=False` and values for `real_label` and/or `imaginary_label`. Honoring "
            "`complex=False`, only a standard variable system was created and the latter labels were disregarded."
        )
        real_label = None
        imaginary_label = None
        complex_requested = False

    if complex is True or complex_requested:
        if (
            complex is not True
            and complex is not None
            and (real_label or imaginary_label)
        ):
            dgcv_warning(
                "The keyword `complex` was set to a non-bool value. Since a string value was also assigned to either "
                "`real_label` or `imaginary_label`, `createVariables` proceeded under the assumption that it should "
                "create a complex variable system. Set `complex=False` to force a standard variable system."
            )
        complex = True

        if real_label is None and imaginary_label is None:
            key_string = retrieve_public_key()
            real_label = variable_label + "REAL" + key_string
            imaginary_label = variable_label + "IM" + key_string
            dgcv_warning(
                "`createVariables` received `complex=True` and did not receive assignments for `real_label` or "
                "`imaginary_label`, so intentionally obscure labels were created for both."
            )
        else:
            if real_label is None:
                real_label = variable_label + "REAL" + retrieve_public_key()
                dgcv_warning(
                    "`createVariables` received a value for `imaginary_label` but not `real_label`, so an intentionally "
                    "obscure label was created for the real variables."
                )
            if imaginary_label is None:
                imaginary_label = variable_label + "IM" + retrieve_public_key()
                dgcv_warning(
                    "`createVariables` received a value for `real_label` but not `imaginary_label`, so an intentionally "
                    "obscure label was created for the imaginary variables."
                )

    if withVF is None:
        withVF = True
    variable_label = reformat_string(variable_label)
    if isinstance(real_label, str):
        real_label = reformat_string(real_label)
    if isinstance(imaginary_label, str):
        imaginary_label = reformat_string(imaginary_label)

    def _planned_member_labels(house):
        if multiindex_shape is not None:
            places = [indexing_places.get(place, "_") for place in index_placement or []]
            places += ["_"] * (len(multiindex_shape) - len(places))
            return [
                f"{house}{''.join(a + str(b) for a, b in zip(places, idx))}"
                for idx in carProd(
                    *[range(initialIndex, initialIndex + n) for n in multiindex_shape]
                )
            ]
        if number_of_variables is not None:
            place = indexing_places.get(index_placement[0], "_") if index_placement else ""
            return [
                f"{house}{place}{i}"
                for i in range(initialIndex, number_of_variables + initialIndex)
            ]
        return [house]

    houses = []
    members = []
    pref = get_dgcv_settings_registry().get("conjugation_prefix", "BAR")
    for j, house in enumerate(variable_label):
        others = (
            [f"{pref}{house}", real_label[j], imaginary_label[j]]
            if complex and j < len(real_label) and j < len(imaginary_label)
            else []
        )
        houses += others
        for name in [house] + others:
            members += _planned_member_labels(name)
    validate_creator_labels(
        ("standard_variable_systems", "complex_variable_systems"),
        variable_label,
        houses + members,
    )

    if complex:
        rv = complexVarProc(
            variable_label,
            real_label,
            imaginary_label,
            number_of_variables=number_of_variables,
            initialIndex=initialIndex,
            multiindex_shape=multiindex_shape,
            index_placement=index_placement,
            default_var_format=default_var_format,
            remove_guardrails=remove_guardrails,
            return_created_object=return_created_object,
            assumptions=assumptions,
            targeted_assumptions=targeted_assumptions,
        )
        return rv if rv is not None else None

    if withVF:
        rv = varWithVF(
            variable_label,
            number_of_variables=number_of_variables,
            initialIndex=initialIndex,
            multiindex_shape=multiindex_shape,
            index_placement=index_placement,
            _doNotUpdateVar=False,
            _calledFromCVP=None,
            remove_guardrails=remove_guardrails,
            return_created_object=return_created_object,
            assumptions=assumptions,
            targeted_assumptions=targeted_assumptions,
        )
        return rv if rv is not None else None

    pk = retrieve_passkey()
    tv = pk if temporary_variables else None
    rv = variableProcedure(
        variable_label,
        number_of_variables=number_of_variables,
        initialIndex=initialIndex,
        multiindex_shape=multiindex_shape,
        index_placement=index_placement,
        _tempVar=tv,
        _doNotUpdateVar=None,
        _calledFromCVP=None,
        _calledFromFactory=pk,
        remove_guardrails=remove_guardrails,
        return_created_object=return_created_object,
        assumptions=assumptions,
        targeted_assumptions=targeted_assumptions,
    )
    return rv if rv is not None else None
