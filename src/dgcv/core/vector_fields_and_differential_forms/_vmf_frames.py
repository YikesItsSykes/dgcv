from __future__ import annotations

from typing import Any, List, Sequence

from ..._aux._backends._symbolic_router import get_free_symbols
from .cartan import _prep_symb_set_for_ext_der
from .retrieval import coordinate_differential_form, coordinate_vector_field


def _coordinate_basis_from_vmf(
    coordinate_space: Sequence[Any],
    *,
    coorType: str,  # "df" or "vf"
) -> List[Any]:
    getter = coordinate_vector_field if coorType == "vf" else coordinate_differential_form
    out: List[Any] = []
    for coordinate in coordinate_space:
        obj = getter(coordinate)
        if obj is None:
            raise TypeError(
                "`coordinate_space` must consist of coordinates registered in the dgcv VMF with a differential system, or of unregistered symbols. "
                "Suggestion: initialize coordinates with `createVariables(..., withVF=True)` (or `createVariables(..., complex=True)` as appropriate)."
            )
        out.append(obj)
    return out


def _format_preference(objs: Sequence[Any]):
    formats = {getattr(obj, "_validated_format", None) for obj in objs}
    if formats <= {"complex"}:
        return "complex"
    if formats <= {"real"}:
        return "real"
    return None


def _infer_coordinate_space_from_objs(objs: Sequence[Any], parameters=()) -> List[Any]:
    symbols: set[Any] = set()
    for obj in objs:
        free = get_free_symbols(obj)
        if free:
            params = getattr(obj, "parameters", None)
            if params:
                free = set(free) - set(params)
            symbols |= set(free)
    symbols -= set(parameters)
    registry = _prep_symb_set_for_ext_der(symbols, prefer=_format_preference(objs))
    coordinates = [atom for system in registry.values() for atom in system]
    return sorted(coordinates, key=str)
