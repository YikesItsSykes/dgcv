from __future__ import annotations

import numbers

from .._aux._backends._types_and_constants import (
    expr_numeric_types,
)
from .._aux._utilities._config import (
    dgcv_warning,
    get_dgcv_settings_registry,
)
from .._aux._vmf._safeguards import (
    create_key,
    get_dgcv_category,
    retrieve_passkey,
)
from ..core.base import dgcv_class
from ..core.morphisms.morphisms import homomorphism
from .linear_algebra import (
    linear_representation,
)


class algebra_dual(dgcv_class):
    def __init__(self, alg, invert_grad_weights=True):
        object.__setattr__(self, "dual_algebra", alg)
        basis = alg.basis
        object.__setattr__(
            self,
            "basis",
            basis.dual_view()
            if hasattr(basis, "dual_view")
            else tuple([elem.dual() for elem in basis]),
        )
        object.__setattr__(self, "label", alg.label + "_dual")
        if invert_grad_weights is not False:
            object.__setattr__(
                self,
                "grading",
                [
                    elem.negated()
                    if hasattr(elem, "negated")
                    else tuple(-j for j in elem)
                    for elem in alg.grading
                ],
            )
        object.__setattr__(self, "_dgcv_categories", {"algebra_dual"})

    def __getattr__(self, name):
        return getattr(self.dual_algebra, name)

    def __setattr__(self, name, value):
        if name == "dual_algebra":
            object.__setattr__(self, name, value)
        else:
            setattr(self.dual_algebra, name, value)

    def __delattr__(self, name):
        if name == "dual_algebra":
            raise AttributeError("Cannot delete 'dual_algebra'")
        delattr(self.dual_algebra, name)

    def __dir__(self):
        # Merge proxy attributes with algebra_class attributes
        return sorted(
            set(dir(type(self)))
            | set(self.__dict__.keys())
            | set(dir(self.dual_algebra))
        )

    def dual(self):
        return self.dual_algebra

    def __str__(self):
        reg = get_dgcv_settings_registry()
        vlp = bool(reg.get("verbose_label_printing", False))

        alg = getattr(self, "dual_algebra", None)
        if alg is None:
            return "<>"

        if vlp is False:
            lab = getattr(alg, "label", None)
            return f"{lab}^*" if lab else "Unnamed^*"

        nm = str(getattr(alg, "label", None) or "Unnamed")
        b = getattr(self, "basis", None) or []
        core = "<" + ", ".join(str(e) for e in b) + ">"
        return f"{nm}^*={core}"

    def _repr_latex_(self, raw: bool = False, abbrev: bool = False, **kwargs):
        reg = get_dgcv_settings_registry()
        vlp = bool(reg.get("verbose_label_printing", False))

        alg = getattr(self, "dual_algebra", None)
        texS = (
            alg._repr_latex_(raw=True, abbrev=True) if alg is not None else r"\text{?}"
        )
        texS = str(texS).replace("$", "").replace(r"\displaystyle", "")
        if "^" in texS:
            texS = f"\\left({texS}\\right)"
        texS = f"{texS}^{{*}}"

        if abbrev or (vlp is False):
            out = texS
            return out if raw else f"$\\displaystyle {out}$"

        b = getattr(self, "basis", None) or []
        inner = ", ".join(e._repr_latex_(raw=True) for e in b)
        inner = str(inner).replace("$", "").replace(r"\displaystyle", "")

        out = texS if not inner else texS + rf"=\langle {inner}\rangle"
        return out if raw else f"$\\displaystyle {out}$"

    def _latex(self, printer=None, raw: bool = True, **kwargs):
        return self._repr_latex_(raw=raw)

    def direct_sum(
        self,
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
            _markers = {
                "sum": True,
                "lockKey": retrieve_passkey(),
                "base_field": (
                    "real"
                    if self.base_field == "real"
                    and getattr(other, "base_field", "complex") == "real"
                    else "complex"
                ),
            }
            if build_all_gradings is not True:
                grad1 = self.grading[:1] or [[0] * self.dimension]
                grad2 = other.grading[:1] or [[0] * other.dimension]
            else:
                grad1 = self.grading or [[0] * self.dimension]
                grad2 = other.grading or [[0] * other.dimension]
            builtG = []
            for gl1 in grad1:
                for gl2 in grad2:
                    builtG.append(list(gl1) + list(gl2))
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

            if label is None:
                label = f"{self.label}_plus_{other.label}"
                _markers["_tex_label"] = (
                    f"{self._repr_latex_(raw=True, abbrev=True)}\\oplus {other._repr_latex_(raw=True, abbrev=True)}"
                )
            if basis_labels is None:
                basis_labels = [elem.__repr__() for elem in self.basis] + [
                    elem.__repr__() for elem in other.basis
                ]
                _markers["_tex_basis_labels"] = [
                    elem._repr_latex_(raw=True) for elem in self.basis
                ] + [elem._repr_latex_(raw=True) for elem in other.basis]

            return linear_representation(
                homomorphism(self, other.endomorphism_algebra)
            ).semidirect_sum(
                grading=grading,
                label=label,
                basis_labels=basis_labels,
                register_in_vmf=register_in_vmf,
                initial_basis_index=initial_basis_index,
                simplify_products_by_default=simplify_products_by_default,
                _markers=_markers,
            )
        else:
            return NotImplemented

    def __add__(self, other):
        return self.direct_sum(other)

    def tensor_product(
        self,
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
                    self, "simplify_products_by_default", False
                )
            if build_all_gradings is not True:
                grad1 = self.grading[:1] or [[0] * self.dimension]
                grad2 = other.grading[:1] or [[0] * other.dimension]
            else:
                grad1 = self.grading or [[0] * self.dimension]
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
                    or len(basis_labels) != self.dimension * other.dimension
                ):
                    dgcv_warning(
                        f"`basis_labels` is in an unsupported format and was ignored. Recieved {basis_labels}, types: {[type(lab) for lab in basis_labels]}, target length {self.dimension}*{other.dimension}"
                    )
                    basis_labels = None
            _markers = {
                "prod": True,
                "lockKey": retrieve_passkey(),
                "tensor_decomposition": (self, other),
                "base_field": (
                    "real"
                    if self.base_field == "real"
                    and getattr(other, "base_field", "complex") == "real"
                    else "complex"
                ),
            }
            if label is None:
                label = f"{self.label}_tensor_{other.label}"
                _markers["_tex_label"] = (
                    f"{self._repr_latex_(raw=True, abbrev=True)}\\otimes {other._repr_latex_(raw=True, abbrev=True)}"
                )
            if basis_labels is None or not isinstance(basis_labels, str):
                basis_labels = [
                    f"{elem1.__repr__()}_tensor_{elem2.__repr__()}"
                    for elem1 in self.basis
                    for elem2 in other.basis
                ]
                _markers["_tex_basis_labels"] = [
                    f"{elem1._repr_latex_(raw=True)}\\otimes {elem2._repr_latex_(raw=True)}"
                    for elem1 in self.basis
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
                    f"{pref}{i + IIdx}" for i in range(self.dimension * other.dimension)
                ]
            if not isinstance(label, str) or label == "":
                label = "Alg_" + create_key()

            if register_in_vmf is True:
                from .creators import createAlgebra

                return createAlgebra(
                    self.dimension * other.dimension,
                    label,
                    basis_labels=basis_labels,
                    grading=grading,
                    return_created_object=True,
                    simplify_products_by_default=simplify_products_by_default,
                    _markers=_markers,
                )
            else:
                _markers["registered"] = False
                return self.dual_algebra.__class__(
                    self.dimension * other.dimension,
                    grading=grading,
                    simplify_products_by_default=simplify_products_by_default,
                    _label=label,
                    _basis_labels=basis_labels,
                    _calledFromCreator=retrieve_passkey(),
                    _markers=_markers,
                )

        else:
            return NotImplemented

    def __matmul__(self, other):
        return self.tensor_product(other)
