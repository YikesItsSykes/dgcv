from __future__ import annotations

from numbers import Integral
from typing import List, Literal

from ...._aux._backends._types_and_constants import imag_unit
from ...arrays import matrix_dgcv
from .creator import createVariables


def createMatrixCoordinates(
    variable_label: str,
    real_label: str | int | None = None,
    imaginary_label: str | None = None,
    initialIndex: int | None = 1,
    withVF: bool | None = None,
    complex: bool | None = None,
    multiindex_shape: List[int] = None,
    matrix_shape: List[
        Literal["sym", "symmetric", "skew", "hermitian", "skew_hermitian", "general"]
    ] = None,
    index_placement: List[
        Literal["up", "down", "hi", "low", "h", "l", "u", "d", "__", "_"]
    ] = None,
    remove_guardrails: bool | None = None,
    default_var_format: Literal["real", "complex", "mixed"] = None,
    temporary_variables: bool | None = False,
    assumptions: dict = None,
    targeted_assumptions: dict = None,
    **kwargs,
):
    if multiindex_shape is None:
        multiindex_shape = (1, 1)
    elif isinstance(multiindex_shape, Integral):
        multiindex_shape = (multiindex_shape,)
    if not isinstance(multiindex_shape, (tuple, list)) or not all(
        isinstance(idx, Integral) and idx >= 0 for idx in multiindex_shape
    ):
        raise ValueError(
            "multiindex_shape in createMatrixCoordinates should be a length 2 tuple of nonnegative integers."
        )
    if len(multiindex_shape) == 0:
        multiindex_shape = (1, 1)
    elif len(multiindex_shape) == 1:
        multiindex_shape = (multiindex_shape[0], multiindex_shape[0])
    elif len(multiindex_shape) > 2:
        multiindex_shape = multiindex_shape[:2]
    passthrough = {
        "variable_label": variable_label,
        "real_label": real_label,
        "imaginary_label": imaginary_label,
        "initialIndex": initialIndex,
        "withVF": withVF,
        "index_placement": index_placement,
        "return_created_object": True,
        "default_var_format": default_var_format,
        "remove_guardrails": remove_guardrails,
        "temporary_variables": temporary_variables,
        "assumptions": assumptions,
        "targeted_assumptions": targeted_assumptions,
    }
    if matrix_shape in {"sym", "symmetric", "skew"}:
        skew = matrix_shape == "skew"
        if multiindex_shape[0] != multiindex_shape[1]:
            raise ValueError(
                'If setting `matrix_shape="symmetric" then multiindex_shape should be square.`'
            )
        n = multiindex_shape[0]
        dim = n * (n - 1) // 2 if skew else n * (n + 1) // 2
        variables = createVariables(
            **passthrough, number_of_variables=dim, complex=complex
        )
        coordinate_dict = {}
        idx = 0
        start = 1 if skew else 0
        for idx1 in range(start, n):
            for idx2 in range(n - idx1):
                vari = variables[0][idx]
                idx += 1
                coordinate_dict[(idx1 + idx2, idx2)] = -vari if skew else vari
                coordinate_dict[(idx2, idx1 + idx2)] = vari
        return matrix_dgcv(coordinate_dict, shape=multiindex_shape)
    if matrix_shape in {"hermitian", "skew_hermitian"}:
        if multiindex_shape[0] != multiindex_shape[1]:
            raise ValueError(
                'If setting `matrix_shape="symmetric" then multiindex_shape should be square.`'
            )
        n = multiindex_shape[0]
        dim = n * (n + 1) // 2
        variables = createVariables(
            **passthrough, number_of_variables=dim, complex=True, skip_warnings=True
        )
        coordinate_dict = {}
        idx = 0
        herm = matrix_shape == "hermitian"
        if herm:
            secondary_idx, secondary_factor = 2, 1
        else:
            secondary_idx, secondary_factor = 3, imag_unit()
        for idx1 in range(n):
            for idx2 in range(n - idx1):
                if idx1 == 0:
                    vari = secondary_factor * variables[secondary_idx][idx]
                    idx += 1
                    coordinate_dict[(idx2, idx1 + idx2)] = vari
                else:
                    vari = variables[0][idx]
                    barvari = variables[1][idx]
                    idx += 1
                    coordinate_dict[(idx2, idx1 + idx2)] = vari
                    coordinate_dict[(idx1 + idx2, idx2)] = barvari if herm else -barvari
        return matrix_dgcv(coordinate_dict, shape=multiindex_shape)
    n, m = multiindex_shape
    dim = n * m
    variables = createVariables(**passthrough, number_of_variables=dim, complex=complex)
    coordinate_dict = {}
    idx = 0
    for idx1 in range(n):
        for idx2 in range(m):
            vari = variables[0][idx]
            idx += 1
            coordinate_dict[(idx1, idx2)] = vari
    return matrix_dgcv(coordinate_dict, shape=multiindex_shape)
