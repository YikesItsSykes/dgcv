from __future__ import annotations

from typing import List, Literal, Optional

from .._aux._backends._symbolic_router import (
    _scalar_is_zero,
)
from .._aux._backends._types_and_constants import (
    symbol,
)
from ..core.base import annotated_container


def structure_equations(
    target_alg,
    formatting: Optional[Literal["dict", "list"]] = "dict",
    new_basis_labels: Optional[str | List[str]] = None,
    abbreviate_for_skew_struct: bool = None,
    initial_index: int = 1,
    list_symbols_as_strings: bool = False,
):
    if new_basis_labels is not None:
        if (
            isinstance(new_basis_labels, (list, tuple))
            and len(new_basis_labels) >= target_alg.dimension
            and len(set(new_basis_labels)) == len(new_basis_labels)
        ):
            atoms = [symbol(lab) for lab in new_basis_labels]
        elif isinstance(new_basis_labels, str):
            atoms = [
                symbol(f"{new_basis_labels}{i + initial_index}")
                for i in range(target_alg.dimension)
            ]
        else:
            atoms = [symbol(str(lab)) for lab in target_alg.basis]
    else:
        atoms = [symbol(str(lab)) for lab in target_alg.basis]
    str_eqns = dict()
    labels = [str(atom) for atom in atoms] if list_symbols_as_strings else atoms

    if abbreviate_for_skew_struct is None:
        abbreviate_for_skew_struct = True if target_alg.is_Lie_algebra() else False
    for i in range(target_alg.dimension):
        start = (
            i + 1 if abbreviate_for_skew_struct and target_alg.is_skew_symmetric else 0
        )
        for j in range(start, target_alg.dimension):
            val = sum(
                c * atoms[idx]
                for idx, c in target_alg.structureData[i, j]._data.items()
            )
            if not _scalar_is_zero(val):
                str_eqns[(atoms[i], atoms[j])] = val
    if formatting == "list":
        str_eqns = [[[k[0], k[1]], v] for k, v in str_eqns.items()]
    return annotated_container(
        [str_eqns, labels],
        _dgcv_notes={
            "signature": "algebra_str_eqns",
            "skew_aware_sparse": abbreviate_for_skew_struct,
        },
    )
