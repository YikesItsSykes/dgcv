"""
package: dgcv - Differential Geometry with Complex Variables

sub-package: dgcv.core

module: dgcv.core.coordinate_maps


---
Author (of this sub-package): David Gamble Sykes

Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0

SPDX-License-Identifier: Apache-2.0
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from ..._aux._backends._symbolic_router import im, re, simplify, subs
from ..._aux._utilities._config import get_variable_registry
from ..._aux._vmf._safeguards import get_dgcv_category, query_dgcv_categories
from ..._aux._vmf.vmf import vmf_lookup
from ...special_fields.complex_structures import KahlerStructure
from ...special_fields.Riemannian_geometry import metricClass
from ..base import dgcv_class
from ..conversions.conversions import allToReal, allToSym
from ..vector_fields_and_differential_forms import get_DF, get_VF

__all__ = ["coordinate_map"]


# -----------------------------------------------------------------------------
# classes
# -----------------------------------------------------------------------------
class coordinate_map(dgcv_class):
    def __init__(
        self, coordinates1, coordinates2, coordinate_formulas, holomorphic=None
    ):
        if not all(
            isinstance(j, (list, tuple))
            for j in [coordinates1, coordinates2, coordinate_formulas]
        ):
            raise TypeError(
                "`coordinate_map` needs all arguments given in the form of lists or tuples."
            )
        if len(coordinates2) != len(coordinate_formulas):
            raise TypeError(
                "`coordinate_map` recieved incompatible initialization data. The number of coordinates provided for the range (i.e., in the second argument position) must match the number of components in the provided formulas for the mapping's components (i.e., in the third argument position)."
            )

        vr = get_variable_registry()

        self.domain = tuple(coordinates1)
        self.range = tuple(coordinates2)

        if holomorphic:

            def get_real_parts(varList):
                reals = []
                ims = []
                for var in varList:
                    varStr = str(var)
                    for parent in vr["complex_variable_systems"]:
                        if (
                            varStr
                            in vr["complex_variable_systems"][parent][
                                "variable_relatives"
                            ]
                        ):
                            realParts = vr["complex_variable_systems"][parent][
                                "variable_relatives"
                            ][varStr]["complex_family"][2:]
                            reals = reals + [realParts[0]]
                            ims = ims + [realParts[1]]
                return reals + ims

            if all(
                j in vr["conversion_dictionaries"]["symToReal"]
                for j in self.domain + self.range
            ):
                self.domain = tuple(get_real_parts(self.domain))
                self.range = tuple(get_real_parts(self.range))
                coordinate_formulas = [
                    simplify(re(allToReal(j))) for j in coordinate_formulas
                ] + [simplify(im(allToReal(j))) for j in coordinate_formulas]
            else:
                raise TypeError(
                    "When setting `holomorphic=True`, `coordinate_map` expects the variables given for the domain and range to be all holomorphic parts of dgcv complex variable systems. It will infer the appropriate action of antiholomophic/real/imaginary parts using holomorphicity."
                )

        self.domain_varSpace_type, self.domain_frame, self.domain_coframe = (
            coordinate_map.validate_coordinates(self.domain)
        )
        self.range_varSpace_type, self.range_frame, self.range_coframe = (
            coordinate_map.validate_coordinates(self.range)
        )
        self._varSpace_type = self.domain_varSpace_type
        self.coordinate_formulas = list(coordinate_formulas)
        idx_transforms = dict()
        for idx, var in enumerate(self.domain):
            data = vmf_lookup(var, path=True, system_index=True)
            path = data.get("path")
            sysl = path[1] if path else None
            idx_transforms[idx] = {"label": sysl, "idx": data.get("system_index")}
        self._idx_transforms = idx_transforms
        self.domain_dimension = len(self.domain)
        self.range_dimension = len(self.range)

        self._JacobianMatrix = None
        self._domain_frame_image = None

    def __dgcv_simplify__(self, method=None, **kwargs):
        return self._eval_simplify(**kwargs)

    def _eval_simplify(self, **kwargs):
        self.coordinate_formulas = [simplify(f, **kwargs) for f in self.coordinate_formulas]
        self._JacobianMatrix = None
        return self

    def _repr_latex_(self, raw: bool = False, **kwargs):
        s = self._latex(raw=True, **kwargs)
        return s if raw else f"$\\displaystyle {s}$"

    def _latex(self, printer=None, raw: bool = True, **kwargs):
        from ..._aux.printing.printing._dgcv_display import LaTeX

        def _trunc_indices(n: int):
            if n <= 4:
                return list(range(n)), None, []
            return [0, 1], 2, [n - 2, n - 1]

        def _join_trunc(seq, n: int):
            lead, mid, tail = _trunc_indices(n)
            parts = [LaTeX(seq[i]) for i in lead]
            if mid is not None:
                parts.append(r"\ldots")
            parts.extend(LaTeX(seq[i]) for i in tail)
            return ", ".join(parts)

        dom = _join_trunc(self.domain, self.domain_dimension)
        ran = _join_trunc(self.range, self.range_dimension)

        n = self.range_dimension
        lead, mid, tail = _trunc_indices(n)
        map_parts = [
            f"{LaTeX(self.range[i])} = {LaTeX(self.coordinate_formulas[i])}"
            for i in lead
        ]
        if mid is not None:
            map_parts.append(r"\ldots")
        map_parts.extend(
            f"{LaTeX(self.range[i])} = {LaTeX(self.coordinate_formulas[i])}"
            for i in tail
        )
        maps = ", ".join(map_parts)

        s = rf"\left\langle {dom}\right\rangle\to\left\langle {ran}\right\rangle\;:\;\left({maps}\right)"
        return s if raw else rf"$\displaystyle {s}$"

    def __str__(self):
        return f"coordinate_map({self.domain} -> {self.range})"

    @property
    def domain_frame_image(self):
        if self._domain_frame_image is None:
            self._domain_frame_image = [
                self.differential(vf) for vf in self.domain_frame
            ]

        return self._domain_frame_image

    @property
    def JacobianMatrix(self):
        if self._JacobianMatrix is None:
            self._JacobianMatrix = [
                [simplify(j(k)) for j in self.domain_frame]
                for k in self.coordinate_formulas
            ]
        return self._JacobianMatrix

    @staticmethod
    def validate_coordinates(varSpace):
        if len(varSpace) != len(set(varSpace)):
            raise TypeError(
                "`coordinate_map` was a list of variables for coordinates (either for the domain or range) that have repeated values.)"
            )
        formatting = set()
        for var in varSpace:
            vtype = vmf_lookup(var)["sub_type"]
            if vtype:
                formatting.add(vtype)
            else:
                raise TypeError(
                    "`coordinate_map` was given coordinates containing variables that were not initialized in the dgcv variable management framework (VMF). Use variable creation functions like `createVariables` to initialize variables while automatically registering them in the VMF."
                )
        reals, holos = (
            ("real" in formatting or "imag" in formatting),
            ("holo" in formatting or "anti" in formatting),
        )
        if reals and holos:
            raise TypeError(
                "`coordinate_map` was given coordinates containing variables that were not initialized in the dgcv variable management framework (VMF), which is not supported. Use variable creation functions like `createVariables` to initialize variables while automatically registering them in the VMF."
            )
        elif reals:
            return (
                "real",
                [allToReal(j) for j in get_VF(*varSpace)],
                [allToReal(j) for j in get_DF(*varSpace)],
            )
        elif holos:
            return (
                "complex",
                [allToSym(j) for j in get_VF(*varSpace)],
                [allToSym(j) for j in get_DF(*varSpace)],
            )
        else:
            return "standard", get_VF(*varSpace), get_DF(*varSpace)

    def differential(self, vf):
        if not query_dgcv_categories(vf, {"vector_field"}):
            raise ValueError(
                "coordinate_map.differential only operates on dgcv vector fields"
            )
        inputCoeffs = [vf(var) for var in self.domain]
        vf_list = []
        for j in range(len(self.range)):
            vf_list += [
                sum(
                    [
                        inputCoeffs[k] * self.JacobianMatrix[j][k]
                        for k in range(len(self.domain))
                    ]
                )
                * self.range_frame[j]
            ]
        return sum(vf_list)

    def pull_back(self, tf):
        if get_dgcv_category(tf) == "tensor_field":
            hp = tf.homogeneous_parts
            if len(hp) > 1:
                return sum(self.pull_back(tf_arg) for tf_arg in hp)

            vf_basis = self.domain_frame_image
            valence = tuple(tf.valence)
            if not all(val == 0 for val in valence):
                raise ValueError(
                    "coordinate_map.pull_back only operates on covariant tensor fields, e.g., differential forms."
                )

            def entry_rule(iT):
                argumentList = [vf_basis[iT[j]] for j in range(len(iT))]
                return tf(*argumentList)

            deg = tf.total_degree
            dimension = len(self.domain)

            def generate_indices(degree, dimension, shape, min_index=0):
                def incr(idx):
                    return (
                        idx
                        if shape == "symmetric"
                        else idx + 1
                        if shape == "skew"
                        else 0
                    )

                if degree == 1:
                    return [(i,) for i in range(min_index, dimension)]
                else:
                    return [
                        (i,) + t
                        for i in range(min_index, dimension)
                        for t in generate_indices(degree - 1, dimension, shape, incr(i))
                    ]

            def new_key(idxs):
                t1, t2 = [], []
                for idx in idxs:
                    id_data = self._idx_transforms[idx]
                    t1.append(id_data["idx"])
                    t2.append(id_data["label"])
                return tuple(t1) + valence + tuple(t2)

            sparse_data = {
                new_key(indices): entry_rule(indices)
                for indices in generate_indices(
                    deg, dimension, getattr(tf, "data_shape", "general")
                )
            }
            return tf.__class__(
                coeff_dict=sparse_data,
                data_shape=tf.data_shape,
                _simplifyKW=tf._simplifyKW,
                parameters=tf.parameters,
            )

        if isinstance(tf, metricClass):
            return metricClass(self.pull_back(tf.SymTensorField))

        if isinstance(tf, KahlerStructure):
            return KahlerStructure(
                self.pull_back(tf.kahlerForm), formatting=tf.formatting
            )

        raise TypeError(
            "`coordinate_map.pull_back` received an unsupported object type."
        )

    @property
    def substitution_dict(self):
        return {var: value for var, value in zip(self.range, self.coordinate_formulas)}

    def __call__(self, other):
        return subs(other, self.substitution_dict)
