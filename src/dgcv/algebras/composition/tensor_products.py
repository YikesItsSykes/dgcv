from __future__ import annotations

import numbers

from ..._aux._backends._types_and_constants import expr_numeric_types
from ..._aux._utilities._config import dgcv_warning
from ..._aux._vmf._safeguards import create_key, get_dgcv_category, retrieve_passkey


def tensor_product(
    target_alg,
    other,
    grading=None,
    label=None,
    basis_labels=None,
    register_in_vmf=False,
    initial_basis_index=None,
    simplify_products_by_default=None,
    build_all_gradings=False,
):
    if get_dgcv_category(other) in {
        "algebra",
        "vectorspace",
        "subalgebra",
        "algebra_subspace",
        "vector_subspace",
    }:
        if simplify_products_by_default is None:
            simplify_products_by_default = getattr(
                target_alg, "simplify_products_by_default", False
            )
        if build_all_gradings is not True:
            grad1 = target_alg.grading[:1] or [[0] * target_alg.dimension]
            grad2 = other.grading[:1] or [[0] * other.dimension]
        else:
            grad1 = target_alg.grading or [[0] * target_alg.dimension]
            grad2 = other.grading or [[0] * other.dimension]
        builtG = []
        for gl1 in grad1:
            for gl2 in grad2:
                builtG.append([w1 + w2 for w1 in gl1 for w2 in gl2])
        if not isinstance(grading, (list, tuple)):
            grading = []
        if isinstance(grading, (list, tuple)):
            if all(isinstance(elem, (list, tuple)) for elem in grading):
                grading = [list(elem) for elem in grading] + builtG
            elif all(isinstance(elem, expr_numeric_types()) for elem in grading):
                grading = [list(grading)] + builtG
            elif grading is not None:
                dgcv_warning(
                    "The supplied grading data format is incompatible, and was ignored."
                )
                grading = builtG
            else:
                grading = builtG

        if isinstance(basis_labels, (tuple, list)):
            if (
                not all(isinstance(elem, str) for elem in basis_labels)
                or len(basis_labels) != target_alg.dimension * other.dimension
            ):
                dgcv_warning(
                    f"`basis_labels` is in an unsupported format and was ignored. Received {basis_labels}, types: {[type(lab) for lab in basis_labels]}, target length {target_alg.dimension}*{other.dimension}"
                )
                basis_labels = None
        _markers = {
            "prod": True,
            "lockKey": retrieve_passkey(),
            "tensor_decomposition": (target_alg, other),
        }
        if label is None:
            label = f"{target_alg.label}_tensor_{other.label}"
            _markers["_tex_label"] = (
                f"{target_alg._repr_latex_(raw=True, abbrev=True)}\\otimes {other._repr_latex_(raw=True, abbrev=True)}"
            )
        if basis_labels is None:
            basis_labels = [
                f"{elem1.__repr__()}_tensor_{elem2.__repr__()}"
                for elem1 in target_alg.basis
                for elem2 in other.basis
            ]
            _markers["_tex_basis_labels"] = [
                f"{elem1._repr_latex_(raw=True)}\\otimes {elem2._repr_latex_(raw=True)}"
                for elem1 in target_alg.basis
                for elem2 in other.basis
            ]
        if isinstance(basis_labels, str):
            pref = basis_labels
            IIdx = (
                initial_basis_index
                if isinstance(initial_basis_index, numbers.Integral)
                else 1
            )
            basis_labels = [
                f"{pref}{i + IIdx}"
                for i in range(target_alg.dimension * other.dimension)
            ]
        if not isinstance(label, str) or label == "":
            label = "Alg_" + create_key()

        if register_in_vmf is True:
            from ..creators import createAlgebra

            return createAlgebra(
                target_alg.dimension * other.dimension,
                label,
                basis_labels=basis_labels,
                grading=grading,
                return_created_object=True,
                simplify_products_by_default=simplify_products_by_default,
                _markers=_markers,
            )
        else:
            _markers["registered"] = False
            return target_alg.__class__(
                target_alg.dimension * other.dimension,
                grading=grading,
                simplify_products_by_default=simplify_products_by_default,
                _label=label,
                _basis_labels=basis_labels,
                _calledFromCreator=retrieve_passkey(),
                _markers=_markers,
            )
    else:
        return NotImplemented
