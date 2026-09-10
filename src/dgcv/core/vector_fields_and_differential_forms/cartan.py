from __future__ import annotations

from typing import Any, Dict, Sequence

from ..._aux._backends._symbolic_router import get_free_symbols
from ..._aux._backends._types_and_constants import expr_numeric_types, is_atomic
from ..._aux._vmf._safeguards import get_dgcv_category, query_dgcv_categories
from ..._aux._vmf.vmf import order_coordinates, vmf_lookup
from ..dgcv_core import VF_bracket, differential_form_class
from .retrieval import coordinate_differential_form, coordinate_vector_field


def makeZeroForm(
    expr: Any,
    varSpace: Sequence[Any] | None = None,
    *,
    variable_spaces: dict | None = None,
    data_shape: str = "all",
    dgcvType: str = "standard",
    _simplifyKW=None,
    parameters=set(),
) -> "differential_form_class":
    if variable_spaces is None:
        exprVars = get_free_symbols(expr)
        if exprVars:
            params = getattr(expr, "parameters", None)
            if params:
                exprVars = set(exprVars) - set(params)
        else:
            exprVars = set()

        variable_spaces = _prep_symb_set_for_ext_der(
            exprVars, use_for_zero_form=True, full_dict=True
        )

    return differential_form_class(
        coeff_dict={tuple(): expr},
        valence=tuple(),
        data_shape=data_shape,
        dgcvType=dgcvType,
        _simplifyKW=_simplifyKW,
        variable_spaces=variable_spaces,
        parameters=parameters,
    )


def _prep_symb_set_for_ext_der(
    symbols, use_for_zero_form=False, full_dict=False, prefer=None
):
    registry: Dict[Any, Any] = {}
    seen = set()

    for atom in order_coordinates(symbols):
        atom_data = vmf_lookup(
            atom,
            path=True,
            flattened_relatives=True,
            differential_system=True,
        )
        if atom_data.get("type") != "coordinate":
            if atom_data.get("type") == "unregistered" and is_atomic(atom):
                if None in registry:
                    registry[None][atom] = atom
                else:
                    registry[None] = {atom: atom}
            continue

        path = atom_data.get("path")
        if not (isinstance(path, tuple) and len(path) >= 2):
            continue

        syslbl = path[1]

        if full_dict or use_for_zero_form:
            if syslbl in registry:
                continue

            sys_key = atom_data.get("differential_system") or syslbl
            sys_data = vmf_lookup(sys_key, flattened_relatives=True)

            flattened = sys_data.get("flattened_relatives")
            if isinstance(flattened, tuple) and flattened:
                registry[syslbl] = flattened
            continue

        sysreg = registry.setdefault(syslbl, {})

        subtype = atom_data.get("sub_type")
        rel = atom_data.get("relatives") or {}

        if subtype in {"holo", "anti", "real", "imag"}:
            pair = (rel.get("holo"), rel.get("anti"))
            if pair in seen:
                continue
            seen.add(pair)
            if prefer == "real" or (prefer is None and subtype in {"real", "imag"}):
                directions = (rel.get("real"), rel.get("imag"))
            else:
                directions = pair
            for direction in directions:
                if direction is not None:
                    sysreg[direction] = pair
        elif atom not in sysreg:
            ds = atom_data.get("differential_system")
            if ds is not None:
                sysreg[atom] = (atom,)

    return registry


def exteriorDerivative(form_or_scalar: Any, **kwargs) -> "differential_form_class":
    obj = form_or_scalar

    if get_dgcv_category(obj) != "tensor_field":
        if isinstance(obj, expr_numeric_types()):
            obj = makeZeroForm(obj)
        else:
            raise TypeError(
                "exteriorDerivative expects a differential form or scalar expression."
            )

    if not query_dgcv_categories(obj, {"differential_form"}):
        raise TypeError(
            "exteriorDerivative expects a differential form or scalar expression."
        )

    form = obj

    exprVars = get_free_symbols(form)
    if exprVars:
        params = getattr(form, "parameters", None)
        if params:
            exprVars = set(exprVars) - set(params)
    else:
        exprVars = set()

    fmt = getattr(form, "_validated_format", None)
    prefer = fmt if fmt in {"real", "complex"} else None

    relevant_vars = _prep_symb_set_for_ext_der(exprVars, prefer=prefer)
    accumulation = 0
    for key, system in relevant_vars.items():
        if key is None:
            for atom in system:
                vf, df = (
                    coordinate_vector_field(atom),
                    coordinate_differential_form(atom),
                )
                accumulation += df * (form.apply(vf))
            continue
        for atom in system:
            ds = vmf_lookup(atom, differential_system=True).get("differential_system")
            if ds is None:
                continue
            vf = ds.get("vf")
            df = ds.get("df")
            if vf is None or df is None:
                continue
            accumulation += df * (form.apply(vf))

    return accumulation


def interiorProduct(
    vector_field: "vector_field_class",
    differential_form: "differential_form_class",
) -> "differential_form_class":
    if not query_dgcv_categories(vector_field, {"vector_field"}):
        raise TypeError("interiorProduct expects a vector_field_class instance.")
    if not query_dgcv_categories(differential_form, {"differential_form"}):
        raise TypeError(
            f"interiorProduct expects a differential_form_class instance. Recieved {differential_form}"
        )

    return differential_form(vector_field)


def LieDerivative(
    vector_field: "vector_field_class",
    obj: Any,
    **kwargs,
):
    if not query_dgcv_categories(vector_field, {"vector_field"}):
        raise TypeError(
            "LieDerivative expects the first argument to be a vector field."
        )

    if isinstance(obj, expr_numeric_types()):
        return vector_field(obj)

    if get_dgcv_category(obj) != "tensor_field":
        raise TypeError("LieDerivative expects a tensor field or scalar expression.")

    if query_dgcv_categories(obj, {"differential_form"}):
        return exteriorDerivative(interiorProduct(vector_field, obj)) + interiorProduct(
            vector_field, exteriorDerivative(obj)
        )

    if query_dgcv_categories(obj, {"vector_field"}):
        return VF_bracket(vector_field, obj)

    raise TypeError(
        "LieDerivative currently only supports differential forms and vector fields."
    )
