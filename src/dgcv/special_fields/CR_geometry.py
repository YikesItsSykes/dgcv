"""
package: dgcv - Differential Geometry with Complex Variables

module: dgcv.special_fields.CR_geometry

Description: This module provides tools specific to CR (Cauchy-Riemann) geometry within the dgcv package.
It includes functions for constructing CR hypersurfaces and  computing symmetries.

Key classes:
    - CR_structure
    - abstract_CR_structure

Key Functions:
    - tangencyObstruction(): Computes the tangency obstruction for a holomorphic vector field's
      real part to be tangent to a CR hypersurface.
    - weightedHomogeneousVF(): Creates a general weighted homogeneous vector field in a
      specified coordinate space.
    - findWeightedCRSymmetries(): Attempts to find infinitesimal symmetries of a weighted CR
      hypersurface.
    - model2Nondegenerate(): Builds the defining equation for a 2-nondegenerate model
      hypersurface.

      ---
Author (of this module): David Gamble Sykes
Project page: https://realandimaginary.com/dgcv/

Copyright (c) 2024-present David Gamble Sykes

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# -----------------------------------------------------------------------------
# imports and broadcasting
# -----------------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable, List, Literal, Sequence, Tuple

from .._aux._backends import simplify_dgcv
from .._aux._backends._calculus import diff
from .._aux._backends._engine import engine_capability, engine_kind, engine_module
from .._aux._backends._numeric_router import rational_sample
from .._aux._backends._polynomials import PolyBackendError, poly_terms
from .._aux._backends._symbolic_router import (
    _scalar_is_zero,
    as_numer_denom,
    cancel,
    conjugate,
    expand,
    get_free_symbols,
    im,
    re,
    subs,
)
from .._aux._backends._types_and_constants import (
    _disposable_symbols,
    exact_fraction,
    expr_numeric_types,
    imag_unit,
    rational,
    symbol,
)
from .._aux._utilities._config import dgcv_warning
from .._aux._vmf._safeguards import (
    create_key,
    query_dgcv_categories,
    retrieve_passkey,
)
from .._aux._backends._parallel import parallel_map, thread_count
from .._aux._vmf.vmf import order_coordinates, vmf_lookup
from ..core.arrays import matrix_dgcv
from ..eds import zero_form_class
from ..core.base import dgcv_class
from ..core.conversions.conversions import (
    allToReal,
    allToSym,
    realToHol,
    realToSym,
    symToHol,
)
from ..core.dgcv_core import (
    complex_struct_op,
    holVF_coeffs,
    polynomial_dgcv,
    realPartOfVF,
    wedge,
)
from ..core.polynomials.polynomials import (
    _weighted_exponent_tuples,
    createMultigradedPolynomial,
)
from ..core.solvers import solve_dgcv
from ..core.vector_fields_and_differential_forms import (
    LieDerivative,
    annihilator,
    exteriorDerivative,
    get_VF,
)
from .complex_structures import Del, DelBar
from .filtered_structures import distribution, filtration_class
from .filtered_structures._distribution import _flag_span

__all__ = [
    "CR_structure",
    "abstract_CR_structure",
    "tangencyObstruction",
    "weightedHomogeneousVF",
    "findWeightedCRSymmetries",
    "model2Nondegenerate",
]


# -----------------------------------------------------------------------------
# classes
# -----------------------------------------------------------------------------
def _graph_coordinate_order(graph_coordinates, eqns_data):
    import graphlib

    tree = {
        gc: {eq["graph_coor"] for eq in eqns_data if gc in eq["fun_symbols"]}
        for gc in graph_coordinates
    }
    try:
        order = list(graphlib.TopologicalSorter(tree).static_order())
    except graphlib.CycleError:
        return None
    return {coor: c for c, coor in enumerate(order)}


def _complex_fraction(value):
    a, b = exact_fraction(re(value)), exact_fraction(im(value))
    if a is None or b is None:
        return None
    return a, b

class CR_structure(dgcv_class):
    """
    Represents local CR structure on a real submanifold in complex space given locally by defining equations.
    """

    def __init__(
        self,
        holomorphic_coordinates: list | tuple,
        defining_equations: list | tuple | dict,
        default_coordinate_format: Literal["real", "complex"] = None,
        parameters: set | list | tuple = None,
        enforce_finding_real_tangent_bundle_basis: bool = False,
        weights: list | tuple = None,
    ):
        self._weight_map = None
        if isinstance(weights, (list, tuple)):
            if len(weights) != len(holomorphic_coordinates):
                dgcv_warning(
                    "The provided `weights` list is not the same length as the provided holomorphic coordinates list, and was ignored."
                )
            else:
                self._weight_map = {
                    v: w for v, w in zip(holomorphic_coordinates, weights)
                }
        formatted_coordinates = []
        parameters = set() if parameters is None else set(parameters)
        seen_coordinates = set()
        omitted = set()
        for var in holomorphic_coordinates:
            holoVar = (
                vmf_lookup(var, relatives=True)
                .get("relatives", dict())
                .get("holo", None)
            )
            if holoVar is None:
                omitted.add(holoVar)
            elif holoVar not in seen_coordinates:
                formatted_coordinates.append(holoVar)
                seen_coordinates.add(holoVar)
        if len(omitted) > 0:
            dgcv_warning(
                "Some of the elements provided in `holomorphic_coordinates` are not part of a complex coordinate system registered in the dgcv VMF. They have been omitted from the CR structure's coordinate system, and regarded as structure parameters instead."
            )
            parameters |= omitted
        self.default_coordinate_format = {None: "complex", "real": "real"}.get(
            default_coordinate_format, "complex"
        )
        self.graph_format = False

        if isinstance(defining_equations, dict):
            self.graph_format = True
            eqns_data = []
            ri_types = {"real", "imag"}
            graph_coordinates = set()
            added_vars = set()
            for k, v in defining_equations.items():
                real_k = allToReal(k)
                key_info = vmf_lookup(real_k, relatives=True)
                if key_info.get("type", None) != "coordinate":
                    self.graph_format = False
                    dgcv_warning(
                        "Initializing a `CR_structure` with `defining_equations` in the dictionary format is intended for inputing equations in a graph format (e.g., graph coordinates equal functions of other coordinates). To process it as `graph form`, the dictionary's keys must belong to a coordinate system registered in the dgcv VMF. The given dictionary fails this condition, so preserving `graph form` has been abandoned. Tip: use `createVariables` to build such coordinate systems."
                    )
                    break
                if key_info.get("sub_type", None) not in ri_types:
                    self.graph_format = False
                    dgcv_warning(
                        "Initializing a `CR_structure` with `defining_equations` in the dictionary format is intended for inputing equations in a graph format (e.g., graph coordinates equal functions of other coordinates). To process it as `graph form`, the dictionary's keys must be real or imaginary parts of holomorphic coordinates belonging to a coordinate system registered in the dgcv VMF. The given dictionary fails this condition, so preserving `graph form` has been abandoned."
                    )
                    break
                real_k = key_info["relatives"][key_info["sub_type"]]
                holoVar = key_info.get("relatives", dict()).get("holo", None)
                if holoVar not in seen_coordinates:
                    added_vars.add(holoVar)
                    seen_coordinates.add(holoVar)
                    formatted_coordinates.append(holoVar)

                real_v = allToReal(v)
                v_fv = get_free_symbols(real_v)
                graph_coordinates.add(real_k)
                conv_v = (
                    real_v
                    if self.default_coordinate_format == "real"
                    else realToSym(real_v)
                )
                conv_k = (
                    real_k
                    if self.default_coordinate_format == "real"
                    else realToSym(real_k)
                )
                eqns_data.append(
                    {
                        "graph_coor": real_k,
                        "graph_function": conv_v,
                        "conv_graph_coor": conv_k,
                        "real_graph_fun": real_v,
                        "fun_symbols": v_fv,
                    }
                )
            graph_coordinates_ordering = _graph_coordinate_order(
                graph_coordinates, eqns_data
            )
            if graph_coordinates_ordering is None:
                self.graph_format = False
                dgcv_warning(
                    "Initializing a `CR_structure` with `defining_equations` in the dictionary format is intended for inputing equations in a graph format (e.g., graph coordinates equal functions of other coordinates like {x_1=f_1,...}, such that [x_j<x_k iff f_j doesn't depend on x_j] determines a partial order). The given dictionary however has an equation of the form x=f(x,y,...), so preserving `graph form` has been abandoned."
                )

            if self.graph_format is True:
                if len(added_vars) > 0:
                    dgcv_warning(
                        f"Some of the given graph coordinates come from holomorphic coordinates not present in the given `holomorphic_coordinates` list, so new coordinates {added_vars} have been added to it."
                    )

                def sort_key(elem):
                    return graph_coordinates_ordering.get(elem["graph_coor"], 0)

                eqns_data = sorted(eqns_data, key=sort_key)
                self.flattened_defining_equations = tuple(
                    [
                        eqn_data["graph_function"] - eqn_data["conv_graph_coor"]
                        for eqn_data in eqns_data
                    ]
                )
                self.graph_equations = {
                    eqn_data["graph_coor"]: eqn_data["graph_function"]
                    for eqn_data in eqns_data
                }
                param_extraction = set()
                for eqn in eqns_data:
                    for var in eqn["fun_symbols"]:
                        holoVar = (
                            vmf_lookup(var, relatives=True)
                            .get("relatives", dict())
                            .get("holo", None)
                        )
                        if holoVar not in seen_coordinates:
                            if var not in parameters:
                                param_extraction.add(var)
                                parameters.add(var)
                if len(param_extraction) > 0:
                    dgcv_warning(
                        f"The given equations include the variables {param_extraction}, which are not present in the given coordinate or parameter sets. They have been added to the structure parameters."
                    )
                self.parameters = parameters
                self.holomorphic_coordinates = tuple(
                    order_coordinates(formatted_coordinates)
                )
            else:
                try:
                    defining_equations = [v - k for k, v in defining_equations.items()]
                except Exception:
                    raise TypeError(
                        "Given `defining_equations` parameter is in an unsupported format."
                    )
        if self.graph_format is False:
            if isinstance(defining_equations, expr_numeric_types()):
                defining_equations = [defining_equations]
            if not isinstance(defining_equations, (list, tuple)):
                raise TypeError(
                    "Given `defining_equations` parameter is in an unsupported format."
                )
            self.flattened_defining_equations = (
                [allToReal(eqn) for eqn in defining_equations]
                if self.default_coordinate_format == "real"
                else [allToSym(eqn) for eqn in defining_equations]
            )
            param_extraction = set()
            for eqn in self.flattened_defining_equations:
                for var in get_free_symbols(eqn):
                    holoVar = (
                        vmf_lookup(var, relatives=True)
                        .get("relatives", dict())
                        .get("holo", None)
                    )
                    if holoVar not in seen_coordinates:
                        if var not in parameters:
                            param_extraction.add(var)
                            parameters.add(var)
            if len(param_extraction) > 0:
                dgcv_warning(
                    f"The given equations include the variables {param_extraction}, which are not present in the given coordinate or parameter sets. They have been added to the structure parameters."
                )
            self.parameters = parameters
            self.holomorphic_coordinates = tuple(
                order_coordinates(formatted_coordinates)
            )

            self.graph_equations = None
        self._efrtbb = enforce_finding_real_tangent_bundle_basis
        self.CR_codimension = len(self.flattened_defining_equations)
        self._CR_distribution = None
        self._holomorphic_CR_distribution = None
        self._tangent_bundle = None
        self._real_tangent_bundle = None
        self._antiholomorphic_CR_dist = None
        self._antiholomorphic_coordinates = None
        self._real_part_coordinates = None
        self._imag_part_coordinates = None
        self._real_coordinates = None
        self._complex_coordinates = None
        self._levi_2form = None
        self._graph_format_cache = None
        self._recovered_graph_cache = None
        self._LFM = None
        self._Freeman_filtration = None
        self._Freeman_filtration_list = None
        self._Freeman_divisors = None
        self._real_def_eqns_cache = None
        self._symmetries = dict()

        super().__init__()

    @property
    def _graph_format_restriction(self):
        if self._graph_format_cache is None:
            if not self.graph_format:
                self._graph_format_cache = dict()
            else:
                if self.default_coordinate_format == "real":
                    self._graph_format_cache = self.graph_equations
                else:
                    self._graph_format_cache = {
                        k: allToReal(
                            v,
                            convert_everything=False,
                            variables_scope=list(self.graph_equations.keys()),
                        )
                        for k, v in self.graph_equations.items()
                    }
        return self._graph_format_cache

    @property
    def _real_defining_equations(self):
        if self._real_def_eqns_cache is None:
            self._real_def_eqns_cache = (
                self.flattened_defining_equations
                if self.default_coordinate_format == "real"
                else [allToReal(eqn) for eqn in self.flattened_defining_equations]
            )
        return self._real_def_eqns_cache

    @property
    def CR_dimension(self):
        if self.default_coordinate_format == "complex":
            return len(self.holomorphic_CR_distribution.vf_basis)
        return self.CR_distribution.rank // 2

    @property
    def CR_distribution(self):
        if self._CR_distribution is None:
            if self.default_coordinate_format == "complex":
                Imag = imag_unit()
                hol = self.holomorphic_CR_distribution.vf_basis
                conjs = [conjugate(X, symbolic=True) for X in hol]
                self._CR_distribution = distribution(
                    [X + Xc for X, Xc in zip(hol, conjs)]
                    + [Imag * (Xc - X) for X, Xc in zip(hol, conjs)],
                    formatting="complex",
                    assume_spanning_sections_linearly_indep=True,
                )
                return self._CR_distribution
            t_basis = self.tangent_bundle.vf_basis
            canonf = self.tangent_bundle.anticanonical_bundle
            Jt_basis = [complex_struct_op(vf) for vf in t_basis]
            vl = create_key("var")
            variables = _disposable_symbols(vl, len(Jt_basis))
            gen_elem = sum(v * elem for v, elem in zip(variables, Jt_basis))
            obst = wedge(gen_elem, canonf)
            sol = solve_dgcv(
                expand(obst),
                variables,
                method="linear_parametric",
                simplify_result=False,
            )
            gen_sol = subs(gen_elem, sol[0])
            varnames = {str(var) for var in variables}
            free_vars = set()
            for expr in sol[0].values():
                free_vars |= {
                    var for var in get_free_symbols(expr) if str(var) in varnames
                }
            if not free_vars:
                dgcv_warning(
                    "The CR distribution was computed as trivial (rank 0) because the solved "
                    "obstruction system retained no free parameters. If a positive-dimensional CR "
                    "distribution was expected, the symbolic backend likely failed to produce a "
                    "parametric solution for the obstruction system."
                )
            zeroing = {var: 0 for var in free_vars}
            self._CR_distribution = distribution(
                [subs(gen_sol, zeroing | {var: 1}) for var in free_vars],
                assume_spanning_sections_linearly_indep=True,
            )
        return self._CR_distribution

    @property
    def holomorphic_CR_distribution(self):
        if self._holomorphic_CR_distribution is None:
            if (
                self._antiholomorphic_CR_dist is None
                and self.default_coordinate_format == "complex"
            ):
                basis = annihilator(
                    [Del(rho) for rho in self.flattened_defining_equations],
                    self.holomorphic_coordinates,
                    polynomial_bases=True,
                    coherent_coordinates_checked=True,
                )
                self._holomorphic_CR_distribution = distribution(
                    basis,
                    formatting="complex",
                    assume_spanning_sections_linearly_indep=True,
                )
            elif self._antiholomorphic_CR_dist is None:
                Imag = imag_unit()
                self._holomorphic_CR_distribution = distribution(
                    [
                        X - Imag * complex_struct_op(X)
                        for X in self.CR_distribution.vf_basis
                    ],
                    find_basis=True,
                    find_polynomial_spanners=True,
                    dimension_hint=self.CR_dimension,
                )
            else:
                self._holomorphic_CR_distribution = conjugate(
                    self._antiholomorphic_CR_dist, symbolic=True
                )
        return self._holomorphic_CR_distribution

    @property
    def antiholomorphic_CR_distribution(self):
        if self._antiholomorphic_CR_dist is None:
            if self.default_coordinate_format == "complex":
                self._antiholomorphic_CR_dist = distribution(
                    [
                        conjugate(X, symbolic=True)
                        for X in self.holomorphic_CR_distribution.vf_basis
                    ],
                    formatting="complex",
                    assume_spanning_sections_linearly_indep=True,
                )
            elif self._holomorphic_CR_distribution is None:
                Imag = imag_unit()
                self._antiholomorphic_CR_dist = distribution(
                    [
                        X + Imag * complex_struct_op(X)
                        for X in self.CR_distribution.vf_basis
                    ],
                    find_basis=True,
                    find_polynomial_spanners=True,
                    dimension_hint=self.CR_dimension,
                )
            else:
                self._antiholomorphic_CR_dist = conjugate(
                    self._holomorphic_CR_distribution, symbolic=True
                )
        return self._antiholomorphic_CR_dist

    @property
    def real_coordinates(self):
        if self._real_coordinates is None:
            if self._real_part_coordinates is None:
                self._real_part_coordinates = tuple(
                    [
                        vmf_lookup(var, relatives=True)["relatives"]["real"]
                        for var in self.holomorphic_coordinates
                    ]
                )
            if self._imag_part_coordinates is None:
                self._imag_part_coordinates = tuple(
                    [
                        vmf_lookup(var, relatives=True)["relatives"]["imag"]
                        for var in self.holomorphic_coordinates
                    ]
                )
            self._real_coordinates = (
                self._real_part_coordinates + self._imag_part_coordinates
            )
        return self._real_coordinates

    @property
    def tangent_bundle(self):
        if self._tangent_bundle is None:
            if self._efrtbb is True:
                defeqns = self._real_defining_equations
                coor = self.real_coordinates
            else:
                defeqns = self.flattened_defining_equations
                coor = self.default_coordinates
            d_rho_list = [exteriorDerivative(eqn) for eqn in defeqns]
            basis = annihilator(
                d_rho_list,
                coor,
                polynomial_bases=True,
                coherent_coordinates_checked=True,
            )
            if self._efrtbb is True and self.default_coordinate_format != "real":
                basis = [realToSym(arg) for arg in basis]
            self._tangent_bundle = distribution(
                basis,
                formatting=self.default_coordinate_format,
                assume_spanning_sections_linearly_indep=True,
            )
        return self._tangent_bundle

    @property
    def real_tangent_bundle(self):
        if self._real_tangent_bundle is None:
            Imag, tangent = imag_unit(), self.tangent_bundle.vf_basis
            self._real_tangent_bundle = (
                self.tangent_bundle
                if self.default_coordinate_format == "real"
                else distribution(
                    [X + conjugate(X, symbolic=True) for X in tangent]
                    + [Imag * (conjugate(X, symbolic=True) - X) for X in tangent],
                    find_basis=True,
                )
            )
        return self._real_tangent_bundle

    @property
    def complex_coordinates(self):
        if self._complex_coordinates is None:
            if self._antiholomorphic_coordinates is None:
                self._antiholomorphic_coordinates = tuple(
                    [
                        vmf_lookup(var, relatives=True)["relatives"]["anti"]
                        for var in self.holomorphic_coordinates
                    ]
                )
            self._complex_coordinates = (
                self.holomorphic_coordinates + self._antiholomorphic_coordinates
            )
        return self._complex_coordinates

    @property
    def default_coordinates(self):
        if self.default_coordinate_format == "real":
            return self.real_coordinates
        return self.complex_coordinates

    def Levi_form_skew_unrestricted(
        self, vf1, vf2, format_scalar_as_vector=False, skip_fast_simplify=False
    ):
        """
        Evaluates a representative of the Levi form on a pair of vector fields.
        Intended for applying to pairs of vector fields from the real CR
        distribution or its complexification (i.e., from the holomorphic +
        antiholomorphic CR distributions).

        By default the returned value is either a 1-by-n array when the CR
        codimension is n>2, or a scalar when the CR codimension is 1. To make
        the codimension 1 case also return an array for consistent formatting
        set the keyword `format_scalar_as_vector=True`.

        Caution/note: Intended only for applying to pairs of vector fields from
        the complexified CR distribution. It will not raise errors or warnings
        if applied to other vector fields however.
        """
        if self._levi_2form is None:
            self._levi_2form = (2 / imag_unit()) * matrix_dgcv(
                [Del(DelBar(rho)) for rho in self.flattened_defining_equations]
            )
        out = self._levi_2form.apply(lambda x: x.__call__(vf1, vf2))
        if skip_fast_simplify is False:
            out = expand(out)
        if self.CR_codimension == 1 and not format_scalar_as_vector:
            out = out[0]
        return out

    def Levi_form_skew(self, vf1, vf2, format_scalar_as_vector=False):
        """
        Evaluates a representative of the Levi form on a pair of vector fields.
        Intended for applying to pairs of vector fields from the real CR
        distribution or its complexification (i.e., from the holomorphic +
        antiholomorphic CR distributions). This Levi form is that of the
        distribution, in the usual sense, rather than the sesquilinear Levi form
        of CR geometry defined with conjugation hooked into an argument.

        By default the returned value is either a 1-by-n array when the CR
        codimension is n>2, or a scalar when the CR codimension is 1. To make
        the codimension 1 case also return an array for consistent formatting
        set the keyword `format_scalar_as_vector=True`.

        Caution/note: Intended only for applying to pairs of vector fields from
        the complexified CR distribution. It will not raise errors or warnings
        if applied to other vector fields however.
        """
        if self.graph_format is False:
            dgcv_warning(
                "This CR structure was not set up in `graph format`, so the computed Levi form value has not been restricted to the submanifold. Suggestion: provide structure data in `graph format` or use `Levi_form_skew_unrestricted` instead."
            )
        return subs(
            self.Levi_form_skew_unrestricted(
                vf1, vf2, format_scalar_as_vector=format_scalar_as_vector
            ),
            self._graph_format_restriction,
        )

    def Levi_form_unrestricted(self, vf1, vf2, format_scalar_as_vector=False):
        """
        Evaluates a representative of the Levi form on a pair of vector fields.
        Intended for applying to pairs of vector fields from the holomorphic CR
        distribution.

        By default the returned value is either a 1-by-n array when the CR
        codimension is n>2, or a scalar when the CR codimension is 1. To make
        the codimension 1 case also return an array for consistent formatting
        set the keyword `format_scalar_as_vector=True`.

        Caution/note: Intended only for applying to pairs of vector fields from
        the holomorphic CR distribution. It will not raise errors or warnings
        if applied to other vector fields however.
        """
        return (
            imag_unit()
            / 2
            * self.Levi_form_skew_unrestricted(
                vf1,
                conjugate(vf2, symbolic=True),
                format_scalar_as_vector=format_scalar_as_vector,
            )
        )

    def Levi_form(self, vf1, vf2, format_scalar_as_vector=False):
        """
        Evaluates a representative of the Levi form on a pair of vector fields.
        This Levi form is the usual CR geometry one, i.e.,
        sesquilinear with complex conjugation hooked into the ordinary skew
        symmetric Levi form of the complexified CR distribution.

        By default the returned value is either a 1-by-n array when the CR
        codimension is n>2, or a scalar when the CR codimension is 1. To make
        the codimension 1 case also return an array for consistent formatting
        set the keyword `format_scalar_as_vector=True`.

        Caution/note: Intended only for applying to pairs of vector fields from
        the holomorphic CR distribution. It will not raise errors or warnings
        if applied to other vector fields however.
        """
        if self.graph_format is False:
            dgcv_warning(
                "This CR structure was not set up in `graph format`, so the computed Levi form value has not been restricted to the submanifold. Suggestion: provide structure data in `graph format` or use `Levi_form_unrestricted` instead."
            )
        return subs(
            self.Levi_form_unrestricted(vf1, vf2), self._graph_format_restriction
        )

    @property
    def Levi_form_matrix(self):
        if self._LFM is None:
            vf_basis = self.holomorphic_CR_distribution.vf_basis

            def entry(i, j):
                return self.Levi_form(vf_basis[i], vf_basis[j])

            self._LFM = matrix_dgcv(
                shape=(self.CR_dimension, self.CR_dimension), entry_rule=entry
            )
        return self._LFM

    @property
    def Freeman_filtration(self):
        if self._Freeman_filtration is None:
            self._compute_Freeman_filtration()
        return self._Freeman_filtration

    def _compute_Freeman_filtration(self, track_singularities=False):
        levels = [list(self.holomorphic_CR_distribution.vf_basis)]
        local_vars = _disposable_symbols(create_key("var"), len(levels[0]))
        if self.default_coordinate_format == "complex":
            filtration_bases, divisors = self._Freeman_descent(
                levels, local_vars, track_singularities
            )
        else:
            filtration_bases = self._Freeman_descent_legacy(levels, local_vars)
            divisors = None
        self._Freeman_divisors = divisors
        alligned_bases = filtration_bases[:1]
        shortened_alligneds = list(alligned_bases)
        if len(filtration_bases) > 1:
            span = _flag_span(
                list(filtration_bases[0]),
                tuple(self.default_coordinates) + tuple(self.parameters),
                False,
            )
            dim = len(filtration_bases[0])
            for basis in filtration_bases[1:]:
                current_basis, new_elems = list(alligned_bases[0]), []
                discrep = len(basis) - dim
                dim = len(basis)
                for elem in basis:
                    if discrep == 0:
                        break
                    if not span.independent(elem):
                        continue
                    discrep -= 1
                    current_basis.append(elem)
                    new_elems.append(elem)
                alligned_bases = [current_basis] + alligned_bases
                shortened_alligneds.append(new_elems)
        self._Freeman_filtration = filtration_class(
            shortened_alligneds, assume_spanning_sections_linearly_indep=True
        )
        self._Freeman_filtration_list = [
            distribution(
                base,
                assume_compatibility=True,
                formatting=self.default_coordinate_format,
                assume_spanning_sections_linearly_indep=True,
            )
            for base in alligned_bases
        ]

    def _Freeman_level_equations(self, general_elem, level, first):
        antihol = self.antiholomorphic_CR_distribution.vf_basis
        if first:
            eqns = []
            for vf in antihol:
                eqns.extend(
                    self.Levi_form_skew_unrestricted(
                        general_elem, vf, format_scalar_as_vector=True
                    )
                )
            return eqns
        phis = annihilator(
            list(level),
            self.holomorphic_coordinates,
            polynomial_bases=True,
            coherent_coordinates_checked=True,
        )
        return [
            expand(phi(LieDerivative(general_elem, vf)))
            for vf in antihol
            for phi in phis
        ]

    def _Freeman_descent(self, levels, local_vars, track_singularities):
        divisors = []
        first = True
        while len(levels[0]) > 0:
            level = levels[0]
            if (
                not first
                and not track_singularities
                and self._Freeman_level_is_final(level)
            ):
                return [[]] + levels, None
            v_trunc = local_vars[: len(level)]
            general_elem = sum(v * elem for v, elem in zip(v_trunc, level))
            eqns = self._Freeman_level_equations(general_elem, level, first)
            first = False
            if not track_singularities and self._Freeman_kernel_is_trivial(
                eqns, v_trunc
            ):
                return [[]] + levels, None
            if track_singularities:
                solution, divs = solve_dgcv(
                    eqns,
                    v_trunc,
                    method="linear_parametric",
                    simplify_result=False,
                    return_divisors=True,
                )
                divisors.append(divs)
            else:
                solution = solve_dgcv(
                    eqns, v_trunc, method="linear_parametric", simplify_result=False
                )
            gen_solution = subs(general_elem, solution[0])
            var_set, found = set(v_trunc), set()
            for expr in solution[0].values():
                found |= {var for var in get_free_symbols(expr) if var in var_set}
            free_vars = [var for var in v_trunc if var in found]
            zeroing = {var: 0 for var in free_vars}
            new_level = [
                subs(gen_solution, zeroing | {var: 1}).scale_to_polynomial_attempt()
                for var in free_vars
            ]
            if len(new_level) == len(level):
                break
            levels = [new_level] + levels
            if not new_level:
                break
        return levels, (divisors[::-1] if track_singularities else None)

    def _Freeman_level_is_final(self, level):
        dfs = []
        for coord in self.holomorphic_coordinates:
            ds = vmf_lookup(coord, differential_system=True).get("differential_system")
            df = ds.get("df") if ds else None
            if df is None:
                return False
            dfs.append(df)
        antihol = self.antiholomorphic_CR_distribution.vf_basis

        def components(vf):
            out = []
            for df in dfs:
                value = df(vf)
                cd = getattr(value, "coeff_dict", None)
                if cd is not None:
                    value = next(iter(cd.values()), 0)
                out.append(value)
            return out

        level_rows = [components(elem) for elem in level]
        bracket_rows = [
            [components(LieDerivative(elem, vf)) for elem in level] for vf in antihol
        ]
        symbols = set()
        for row in level_rows + [row for rows in bracket_rows for row in rows]:
            for entry in row:
                symbols |= set(get_free_symbols(entry))
        m, n = len(level), len(dfs)
        unknowns = m * (1 + len(antihol))
        for _ in range(8):
            point = {s: rational_sample() for s in symbols}
            try:
                A = [[_complex_fraction(subs(e, point)) for e in row] for row in level_rows]
                B = [
                    [[_complex_fraction(subs(e, point)) for e in row] for row in rows]
                    for rows in bracket_rows
                ]
            except ZeroDivisionError:
                continue
            except ValueError as err:
                if "division by zero" not in str(err):
                    raise
                continue
            if any(x is None for row in A for x in row) or any(
                x is None for rows in B for row in rows for x in row
            ):
                return False
            real_rows = []
            for k, Bk in enumerate(B):
                for i in range(n):
                    c = [(Fraction(0), Fraction(0))] * unknowns
                    for a in range(m):
                        c[a] = Bk[a][i]
                    for b in range(m):
                        c[m + k * m + b] = (-A[b][i][0], -A[b][i][1])
                    re_part = [x[0] for x in c]
                    im_part = [x[1] for x in c]
                    real_rows.append(re_part + [-x for x in im_part])
                    real_rows.append(im_part + re_part)
            return self._fraction_rank(real_rows) == 2 * unknowns
        return False

    def _Freeman_kernel_is_trivial(self, eqns, variables):
        var_set = set(variables)
        symbols = set()
        for eqn in eqns:
            symbols |= set(get_free_symbols(eqn))
        symbols -= var_set
        zeros = {var: 0 for var in variables}
        for _ in range(8):
            point = {s: rational_sample() for s in symbols}
            try:
                evaluated = [subs(eqn, point) for eqn in eqns]
            except ZeroDivisionError:
                continue
            except ValueError as err:
                if "division by zero" not in str(err):
                    raise
                continue
            real_rows, imag_rows = [], []
            for val in evaluated:
                real_row, imag_row = [], []
                for var in variables:
                    entry = subs(val, zeros | {var: 1})
                    a, b = exact_fraction(re(entry)), exact_fraction(im(entry))
                    if a is None or b is None:
                        return False
                    real_row.append(a)
                    imag_row.append(b)
                real_rows.append(real_row)
                imag_rows.append(imag_row)
            rows = [a + [-x for x in b] for a, b in zip(real_rows, imag_rows)]
            rows += [b + a for a, b in zip(real_rows, imag_rows)]
            return self._fraction_rank(rows) == 2 * len(variables)
        return False

    @staticmethod
    def _fraction_rank(rows):
        rows = [[Fraction(x) for x in row] for row in rows if any(row)]
        rank = 0
        for c in range(len(rows[0]) if rows else 0):
            piv = next((i for i in range(rank, len(rows)) if rows[i][c] != 0), None)
            if piv is None:
                continue
            rows[rank], rows[piv] = rows[piv], rows[rank]
            pivot_row = rows[rank]
            for i in range(len(rows)):
                if i != rank and rows[i][c] != 0:
                    f = rows[i][c] / pivot_row[c]
                    rows[i] = [a - f * b for a, b in zip(rows[i], pivot_row)]
            rank += 1
        return rank

    def _Freeman_descent_legacy(self, levels, local_vars):
        antihol = self.antiholomorphic_CR_distribution

        def descent(level_list, pair, variables):
            if len(level_list[0]) == 0:
                return level_list
            level = level_list[0]
            v_trunc = variables[: len(level)]
            general_elem = sum(v * elem for v, elem in zip(v_trunc, level))
            new_c_form = wedge(pair.anticanonical_bundle, *level)
            eqns = [
                wedge(LieDerivative(general_elem, vf), new_c_form)
                for vf in pair.vf_basis
            ]
            solution = solve_dgcv(
                eqns, v_trunc, method="linear_parametric", simplify_result=False
            )
            gen_solution = subs(general_elem, solution[0])
            free_vars, var_set = set(), set(v_trunc)
            for expr in solution[0].values():
                free_vars |= {
                    var for var in get_free_symbols(expr) if var in var_set
                }
            zeroing = {var: 0 for var in free_vars}
            new_level = [
                [
                    subs(
                        gen_solution, zeroing | {var: 1}
                    ).scale_to_polynomial_attempt()
                    for var in free_vars
                ]
            ]
            lnl = len(new_level[0])
            if len(level_list[0]) == lnl:
                return level_list
            if lnl == 0:
                return [[]] + level_list
            level_list = new_level + level_list
            return descent(level_list, pair, variables)

        return descent(levels, antihol, local_vars)

    @property
    def Freeman_filtration_list(self):
        if self._Freeman_filtration_list is None:
            self._Freeman_filtration_list = list(self.Freeman_filtration.distributions)[
                ::-1
            ]
        return self._Freeman_filtration_list

    @property
    def nondegeneracy_order(self):
        if len(self.Freeman_filtration.associated_graded_bases[0]) > 0:
            return "infinity"
        return self.Freeman_filtration.depth - 1

    def _recover_graph_equations(self):
        if self._recovered_graph_cache is not None:
            return self._recovered_graph_cache
        candidates = []
        for hv in self.holomorphic_coordinates:
            rel = vmf_lookup(hv, relatives=True).get("relatives", dict())
            for key in ("imag", "real"):
                c = rel.get(key)
                if c is not None and c not in candidates:
                    candidates.append(c)
        recovered = {}
        used = set()
        for rho in self._real_defining_equations:
            found = None
            for c in candidates:
                if c in used:
                    continue
                a = diff(rho, c)
                if get_free_symbols(a) or _scalar_is_zero(a):
                    continue
                rest = expand(rho - a * c)
                if c in get_free_symbols(rest):
                    continue
                found = (c, -rest / a)
                break
            if found is None:
                self._recovered_graph_cache = False
                return False
            used.add(found[0])
            recovered[found[0]] = found[1]
        eqns_data = [
            {"graph_coor": c, "fun_symbols": get_free_symbols(f)}
            for c, f in recovered.items()
        ]
        if _graph_coordinate_order(set(recovered), eqns_data) is None:
            self._recovered_graph_cache = False
            return False
        self._recovered_graph_cache = recovered
        return recovered

    def _admissible_graph_equations(self):
        if self.graph_format:
            return self.graph_equations
        recovered = self._recover_graph_equations()
        return recovered if recovered else None

    def compute_weighted_symmetries(
        self,
        target_weights,
        coordinate_weights=None,
        report_progress=False,
        verbose=False,
        degreeCap: int = 0,
        assume_polynomial: bool = False,
        simplify: bool = False,
    ):
        graph_eqns = self._admissible_graph_equations()
        if graph_eqns is None:
            raise RuntimeError(
                "`compute_weighted_symmetries` needs the defining equations in graph form: each equation solved for one real coordinate, given as a dictionary {coordinate: function}. The given equations could not be rewritten that way."
            )
        if isinstance(target_weights, expr_numeric_types()):
            target_weights = [target_weights]
        else:
            target_weights = list(target_weights)
        if not all(isinstance(elem, expr_numeric_types()) for elem in target_weights):
            dgcv_warning(
                "provided `target_weights` to compute symmetries for is not in a supported format. Should be either a simple scalar like an integer or rational or a list of them."
            )
            return {}
        caching = False
        if coordinate_weights is None:
            caching = True
            if self._weight_map is None:
                raise RuntimeError(
                    "This `CR_structure` instance was not initialized with valid coordinate weights so the optional `coordinate_weights` keyword needs to be provided instead."
                )
            else:
                coordinate_weights = [
                    self._weight_map[var]
                    for var in self.holomorphic_coordinates
                    if var in self.holomorphic_coordinates
                ]
            if len(coordinate_weights) != len(self.holomorphic_coordinates):
                raise RuntimeError(
                    "This `CR_structure` instance was not initialized with valid coordinate weights so the optional `coordinate_weights` keyword needs to be provided instead. A possible reason could be that the given `holomorphic_coordinates` list in the `CR_structure` initialization was incomplete, with additional variables appearing in the initialization defining equation system."
                )
        elif len(coordinate_weights) != len(self.holomorphic_coordinates):
            raise RuntimeError(
                "`coordinate_weights` must be an integer list of the same length as the CR_structure.holomorphic_coordinates list."
            )
        symmetries = dict()
        if verbose:
            print("Using variable weights:")
            print(
                f"{', '.join(['[' + str(var) + '] = ' + str(w) for var, w in zip(self.holomorphic_coordinates, coordinate_weights)])}"
            )
            print(" ")
        def _report_start(weight):
            if report_progress:
                print(f"Computing symmetries of weight {weight}:")

        def _report_done(weight, found):
            if report_progress:
                number_found = len(found)
                grammar_detail = (
                    ["symmetry", "was"] if number_found == 1 else ["symmetries", "were"]
                )
                print(
                    f"{number_found} {grammar_detail[0]} of weight {weight} {grammar_detail[1]} found."
                )
                print(" ")

        pending = []
        for weight in dict.fromkeys(target_weights):
            if caching is True and weight in self._symmetries:
                _report_start(weight)
                _report_done(weight, self._symmetries[weight])
            else:
                pending.append(weight)
        threaded = bool(thread_count())
        ordered = sorted(pending, key=lambda weight: -weight) if threaded else pending
        if threaded and report_progress and pending:
            print(f"Computing symmetries of weights {', '.join(str(w) for w in pending)} in parallel:")
        computed = dict(
            zip(
                ordered,
                parallel_map(
                    "cr_weights",
                    lambda weight: findWeightedCRSymmetries(
                        list(graph_eqns.values()),
                        holomorphic_coordinates=self.holomorphic_coordinates,
                        coordinate_weights=coordinate_weights,
                        symmetry_weight=weight,
                        graph_variable=list(graph_eqns.keys()),
                        returnVectorFieldBasis=True,
                        degreeCap=degreeCap,
                        assume_polynomial=assume_polynomial,
                        simplify=simplify,
                        parameters=self.parameters if self.parameters else None,
                    ),
                    ordered,
                    min_items=2,
                    chunk=1,
                    on_start=None if threaded else _report_start,
                    on_result=_report_done,
                ),
            )
        )
        for weight in target_weights:
            if weight in computed:
                symmetries[weight] = computed[weight]
                if caching is True:
                    self._symmetries[weight] = computed[weight]
            else:
                symmetries[weight] = self._symmetries[weight]
        return symmetries


class abstract_CR_structure(dgcv_class):
    """
    Represents local CR structure in terms of real coordinates. Initialization data
    should be the holomorphic component of the complexified CR distribution,
    i.e., the i-eigenspace bundle of the CR complex structure operator.
    """

    def __init__(self, holomorphic_distribution, validate_structure_data=False):
        super().__init__()


# -----------------------------------------------------------------------------
# CR tools
# -----------------------------------------------------------------------------
def tangencyObstruction(
    vf,
    CR_defining_expr,
    graph_variable,
    simplify: bool = False,
    data_already_in_real_coor: bool = False,
    *args,
):
    CR_defs = _as_list(CR_defining_expr)
    graph_vars = _as_list(graph_variable)

    if len(CR_defs) != len(graph_vars):
        raise ValueError(
            "`graph_variable` and `CR_defining_expr` must have the same length."
        )

    if not data_already_in_real_coor:
        graph_vars = [allToReal(v) for v in graph_vars]
        CR_defs = [allToReal(f) for f in CR_defs]
        vf = allToReal(vf)

    rVF = realPartOfVF(vf)
    real_eval = [rVF(gv - f) for gv, f in zip(graph_vars, CR_defs)]

    def mapping(x):
        out = x
        for var, val in zip(graph_vars, CR_defs):
            out = subs(out, {var: val})
        if simplify:
            return simplify_dgcv(out)
        return out

    out = [mapping(expr) for expr in real_eval]
    return out


def weightedHomogeneousVF(
    varSpace,
    weight,
    weights,
    varLabel,
    degreeCap: int = 0,
    _tempVar=None,
    polynomial_coeffs=False,
    assumptions: dict = None,
    **kwargs,
):
    if kwargs.get("assumeReal", False) is True:  # deprecated kwarg support
        if not isinstance(assumptions, dict):
            assumptions = {"real": True}
        else:
            assumptions["real"] = True
    if not _is_sequence(weight):
        weight = [weight]
    if not weights or not _is_sequence(weights[0]):
        weights = [weights]

    weight_list = list(weight)
    weight_systems = list(weights)

    if not all(len(ws) == len(varSpace) for ws in weight_systems):
        raise KeyError(
            "`weightedHomogeneousVF` expects each weight system to match len(varSpace)."
        )
    if len(weight_list) != len(weight_systems):
        raise KeyError(
            "`weightedHomogeneousVF` expects len(weight) == number of weight systems."
        )

    polys = []
    for j in range(len(varSpace)):
        degs = [d + ws[j] for d, ws in zip(weight_list, weight_systems)]
        newPoly = createMultigradedPolynomial(
            f"{varLabel}_{j}_",
            degs,
            varSpace,
            weight_systems,
            degreeCap=degreeCap,
            _tempVar=_tempVar,
            assumptions=assumptions,
            report_vmf_updates=False,
        )
        if polynomial_coeffs is False:
            newPoly = getattr(newPoly, "polyExpr", newPoly)
        polys.append(newPoly)

    Dz_list = get_VF(*varSpace)
    out = None
    for p, Dz in zip(polys, Dz_list):
        if not p:
            continue
        term = p * Dz
        out = term if out is None else (out + term)

    if out is None:
        out = type(Dz_list[0])(coeff_dict={tuple(): 0}, data_shape="all")

    return out



def _column_assembly(
    graph_function,
    graph_variable,
    holomorphic_coordinates,
    sym_weight_list,
    weight_systems,
    weight_tag,
    degreeCap,
    simplify,
    coord_ri,
    parameters,
):
    coords = tuple(holomorphic_coordinates)
    terms = []
    for j in range(len(coords)):
        degs = [int(d + ws[j]) for d, ws in zip(sym_weight_list, weight_systems)]
        exps_list = None
        for k0, w0 in enumerate(weight_systems):
            sign = 1 if all(w > 0 for w in w0) else -1 if all(w < 0 for w in w0) else 0
            if sign == 0:
                continue
            exps_list = [
                exps
                for exps in _weighted_exponent_tuples([sign * w for w in w0], sign * degs[k0])
                if all(
                    sum(e * wj for e, wj in zip(exps, weight_systems[k])) == degs[k]
                    for k in range(len(degs))
                    if k != k0
                )
            ]
            break
        if exps_list is None:
            return None
        for idx, exps in enumerate(exps_list, start=1):
            mono = 1
            for v, e in zip(coords, exps):
                if e:
                    mono = mono * v**e
            terms.append((j, idx, mono))
    graph_vars = [allToReal(v) for v in _as_list(graph_variable)]
    graph_defs = [allToReal(f) for f in _as_list(graph_function)]
    if len(graph_vars) != len(graph_defs):
        raise ValueError("`graph_variable` and `CR_defining_expr` must have the same length.")

    def restrict(expr):
        for var, val in zip(graph_vars, graph_defs):
            expr = subs(expr, {var: val})
        return expr

    Dz_list = get_VF(*coords)
    gens = _stable_tuple(sorted(coord_ri, key=str) + list(graph_vars))
    derivs = []
    try:
        for gv, f in zip(graph_vars, graph_defs):
            rho = gv - f
            row = []
            for Dz in Dz_list:
                d = restrict(Dz(rho))
                poly_terms(d, gens, assume_polynomial=True, parameters=parameters)
                row.append(d)
            derivs.append(row)
    except PolyBackendError:
        return None
    imu = imag_unit()
    unknowns = [
        (symbol(f"ALoc{weight_tag}_{j}_{idx}", assumptions={"real": True}), j, mono, False)
        for j, idx, mono in terms
    ] + [
        (symbol(f"BLoc{weight_tag}_{j}_{idx}", assumptions={"real": True}), j, mono, True)
        for j, idx, mono in terms
    ]

    split = engine_capability("complex_poly_terms")
    linear = engine_capability("linear_form")

    def term_columns(entry):
        j, _idx, mono = entry
        m_real = restrict(allToReal(mono))
        real_col = {}
        imag_col = {}
        for e, row in enumerate(derivs):
            prod = m_real * row[j]
            if simplify:
                prod = simplify_dgcv(prod)
            parts = split(prod, gens) if split is not None else None
            if parts is not None:
                real_part, imag_part = parts
                for exps, a in real_part.items():
                    real_col[(e, exps)] = a
                for exps, b in imag_part.items():
                    imag_col[(e, exps)] = -b
                continue
            prod = expand(prod)
            _g, monoms, cs = poly_terms(prod, gens, assume_polynomial=True, parameters=parameters)
            for exps, c in zip(monoms, cs):
                a = re(c)
                b = im(c)
                if not _scalar_is_zero(a):
                    real_col[(e, exps)] = a
                if not _scalar_is_zero(b):
                    imag_col[(e, exps)] = -b
        return real_col, imag_col

    try:
        pairs = parallel_map("cr_columns", term_columns, terms, min_items=16)
    except PolyBackendError:
        return None
    columns = [real for real, _imag in pairs] + [imag for _real, imag in pairs]
    rows = {}
    for (u, _j, _m, _i), col in zip(unknowns, columns):
        for key, c in col.items():
            rows.setdefault(key, []).append((c, u))
    eqns = []
    for key in sorted(rows):
        expr = linear(rows[key]) if linear is not None else None
        if expr is None:
            for c, u in rows[key]:
                term = c * u
                expr = term if expr is None else expr + term
        eqns.append(expr)
    unknowns_t = _stable_tuple(sorted((u for u, _j, _m, _i in unknowns), key=str))
    grouped = [[] for _ in coords]
    for u, j, mono, imaginary in unknowns:
        grouped[j].append((imu * u if imaginary else u) * mono)
    coeffs = [zero_form_class._dgcv_multiadd(group) if group else None for group in grouped]
    vf = None
    for c, Dz in zip(coeffs, Dz_list):
        if c is None:
            continue
        term = c * Dz
        vf = term if vf is None else vf + term
    if vf is None:
        vf = type(Dz_list[0])(coeff_dict={tuple(): 0}, data_shape="all")
    solutions = solve_dgcv(eqns, unknowns_t, method="linear_parametric", simplify_result=False)
    return vf, eqns, unknowns_t, solutions


def findWeightedCRSymmetries(
    graph_function,
    holomorphic_coordinates,
    coordinate_weights,
    symmetry_weight,
    graph_variable,
    coeff_label=None,
    degreeCap: int = 0,
    returnVectorFieldBasis: bool = False,
    returnAllformats: bool = False,
    simplifyingFactor=None,
    assume_polynomial: bool = False,
    simplify: bool = False,
    parameters=None,
):
    if not coordinate_weights or not _is_sequence(coordinate_weights[0]):
        coordinate_weights = [coordinate_weights]
    if not _is_sequence(symmetry_weight):
        symmetry_weight = [symmetry_weight]

    weight_systems = list(coordinate_weights)
    sym_weight_list = list(symmetry_weight)

    if len(sym_weight_list) != len(weight_systems):
        raise KeyError(
            "`findWeightedCRSymmetries` requires one symmetry weight per weight system."
        )

    if returnAllformats:
        returnVectorFieldBasis = True

    weight_tag = "_" + "_".join(str(w).replace("-", "m").replace("/", "o") for w in sym_weight_list)

    def _coord_ri_from_vmf(coords):
        out = set()
        for v in coords:
            info = vmf_lookup(v, relatives=True)
            if info.get("type") != "coordinate":
                continue
            rel = info.get("relatives") or {}
            r = rel.get("real", None)
            i = rel.get("imag", None)
            if r is not None:
                out.add(r)
            if i is not None:
                out.add(i)
            if r is None and i is None:
                out.add(v)
        return out

    free = get_free_symbols
    unwrap = _unwrap_symbolic
    coord_ri = _coord_ri_from_vmf(holomorphic_coordinates)

    column_data = (
        _column_assembly(
            graph_function,
            graph_variable,
            holomorphic_coordinates,
            sym_weight_list,
            weight_systems,
            weight_tag,
            degreeCap,
            simplify,
            coord_ri,
            parameters,
        )
        if engine_kind() == "builtin" and degreeCap == 0
        else None
    )
    if column_data is not None:
        vf_candidate_holo, tan_obst_list, unknowns_t, solutions = column_data
    else:
        vfA = weightedHomogeneousVF(
            holomorphic_coordinates,
            sym_weight_list,
            weight_systems,
            "ALoc" + weight_tag,
            _tempVar=retrieve_passkey(),
            degreeCap=degreeCap,
            assumptions={"real": True},
        )
        vfB = weightedHomogeneousVF(
            holomorphic_coordinates,
            sym_weight_list,
            weight_systems,
            "BLoc" + weight_tag,
            _tempVar=retrieve_passkey(),
            degreeCap=degreeCap,
            assumptions={"real": True},
        )

        vf_candidate_holo = vfA + imag_unit() * vfB

        tan_obst_list = tangencyObstruction(
            vf_candidate_holo,
            graph_function,
            graph_variable,
            simplify=simplify,
        )

        var_all = set()
        for TO in tan_obst_list:
            var_all.update(free(unwrap(TO)))

        unknowns = set(var_all) - set(coord_ri)
        if parameters is not None:
            unknowns -= set(parameters)

        if coeff_label is None:
            coeff_label = create_key(prefix="X", key_length=5)

        unknowns_t = _stable_tuple(sorted(unknowns, key=str))

        graph_vars = _as_list(graph_variable)
        poly_gens_vars = _stable_tuple(sorted(coord_ri, key=str) + list(graph_vars))

        coef_eqns = []
        for TO in tan_obst_list:
            expr = unwrap(TO)
            expr = allToReal(expr)
            if not assume_polynomial:
                expr = as_numer_denom(cancel(expr))[0]
            expr = expand(expr)
            P = (
                expr
                if query_dgcv_categories(expr, "polynomial")
                else polynomial_dgcv(expr, poly_gens_vars)
            )
            coef_eqns.extend([allToReal(c) for c in P.get_coeffs(formatting="unformatted")])

        solutions = solve_dgcv(
            coef_eqns, unknowns_t, method="linear_parametric", simplify_result=False
        )

    if not solutions:
        if len(tan_obst_list) == 0 or all(
            _scalar_is_zero(unwrap(x)) for x in tan_obst_list
        ):
            if not returnVectorFieldBasis:
                return [0 for _ in holomorphic_coordinates]
            if returnAllformats:
                return [0 for _ in holomorphic_coordinates], []
            return []
        raise ValueError(
            "findWeightedCRSymmetries: no solution to the coefficient system."
        )

    sol = solutions[0] if isinstance(solutions, (list, tuple)) else solutions

    vf_candidate_holo = vf_candidate_holo.apply(lambda x: getattr(x, "polyExpr", x))
    vf_solved = subs(vf_candidate_holo, sol)

    VFCLoc = holVF_coeffs(vf_solved, holomorphic_coordinates)

    sub_syms = set()
    for term in VFCLoc:
        sub_syms.update(free(unwrap(term)))

    sub_syms -= set(holomorphic_coordinates)
    sub_syms -= set(coord_ri)
    if parameters is not None:
        sub_syms.difference_update(parameters)

    if not sub_syms:
        if not returnVectorFieldBasis:
            return VFCLoc
        if returnAllformats:
            return VFCLoc, []
        return []

    sub_syms_t = _stable_tuple(sorted(sub_syms, key=str))

    coeff_vars = tuple(
        symbol(f"{coeff_label}{idx}", assumptions={"real": True})
        for idx in range(len(sub_syms_t))
    )
    repl = dict(zip(sub_syms_t, coeff_vars))
    vf_param = subs(vf_solved, repl)

    if not returnVectorFieldBasis:
        return holVF_coeffs(vf_param, holomorphic_coordinates)

    zeroing = {w: 0 for w in coeff_vars}
    VF_basis = []
    for v in coeff_vars:
        d = dict(zeroing)
        d[v] = 1
        VF_basis.append(subs(vf_param, d))

    if returnAllformats:
        return holVF_coeffs(vf_param, holomorphic_coordinates), VF_basis
    return VF_basis


def model2Nondegenerate(
    hermitian_matrix: Any,
    symmetric_matrix_function: Any,
    nondegenerate_coordinates: Sequence[Any],
    transverse_coordinate: Any,
    use_symbolic_conjugates: bool = False,
    return_matrices: bool = False,
    simplify: bool = True,
    return_CR_structure: bool = False,
    parameters: set = None,
    coordinates_to_weights_dict=None,
):
    """
    The parameter `coordinates_to_weights_dict` is relevant only if setting `return_CR_structure=True`. In this case it may be set to a dictionary with one key/value pair. The key should be a tuple of holomorphic coordinates, and the value should be a list/tuple of corresponding weights.
    """
    if return_CR_structure is True:
        use_symbolic_conjugates = True

    def _simp(x: Any) -> Any:
        return simplify_dgcv(x) if simplify else x

    def _as_md(mat: Any):
        return matrix_dgcv(mat)

    def _T(M: Any):
        try:
            return M.transpose()
        except Exception:
            pass
        fallback = M.to_symbolic_engine_class()
        try:
            transp = fallback.transpose
            if callable(transp):
                return transp()
            else:
                return transp
        except Exception:
            return fallback.T

    def _inv(M: Any):
        try:
            return M.inverse()
        except Exception:
            return M.symbolic_engine_method("__pow__", method_arguments=[-1])

    A = _as_md(hermitian_matrix)
    S = _as_md(symmetric_matrix_function)

    shp = getattr(A, "shape", None)
    if not (isinstance(shp, tuple) and len(shp) == 2 and shp[0] == shp[1]):
        raise TypeError("`hermitian_matrix` must be square.")
    n = int(shp[0])

    Iden = matrix_dgcv.eye(n)
    BARS = S.conjugate(symbolic=True)

    tc_info = vmf_lookup(transverse_coordinate, relatives=True)
    tc_holo = tc_info.get("relatives", dict()).get("holo")
    if tc_info.get("sub_type") == "holo" or tc_info.get("sub_type") == "anti":
        if use_symbolic_conjugates is False:
            transverse_coordinate = realToHol(
                tc_info.get("relatives", dict()).get("imag")
            )
        else:
            transverse_coordinate = realToSym(
                tc_info.get("relatives", dict()).get("imag")
            )
    elif tc_info.get("sub_type") not in {"real", "imag"}:
        dgcv_warning(
            "The given transverse coordinate is not associated with any complex coordinate system currently in the dgcv VMF."
        )

    base_coordinates = []
    seen = set()
    nonregistered_given = False
    transv_given = False
    minimal_not_given = False
    for var in nondegenerate_coordinates:
        holo = vmf_lookup(var, relatives=True).get("relatives", dict()).get("holo")
        if holo is not None and holo not in seen and tc_holo != holo:
            seen.add(holo)
            base_coordinates.append(holo)
        elif holo is None:
            nonregistered_given = True
        elif holo == tc_holo:
            transv_given = True
        else:
            minimal_not_given = True
    if nonregistered_given is True:
        dgcv_warning(
            "Some of the elements given in `nondegenerate_coordinates` are associated with a complex coordinate system in the dgcv VMF and were ignored."
        )
    if transv_given is True:
        dgcv_warning(
            "At least element in the given  `nondegenerate_coordinates` has the same holomorphic coordinate as the given transverse coordinate and was ignored."
        )
    if minimal_not_given is True:
        dgcv_warning(
            "Multiple elements given in `nondegenerate_coordinates` are associated with the same holomorphic coordinate and the repeats were ignored."
        )

    zVec = matrix_dgcv([[v] for v in base_coordinates])
    bzVec = zVec.conjugate(symbolic=True)

    half = rational(1, 2)

    At = _T(A)
    zt = _T(zVec)
    bzt = _T(bzVec)

    try:
        M1 = Iden - (BARS * At * S * A)
        M1_inv = _inv(M1)
        M1_inv_conj = M1_inv.conjugate(symbolic=True)
        M1_inv_conj_tran = M1_inv_conj.transpose()

        hFun = half * (A * M1_inv + M1_inv_conj_tran * A)
        sFun = A * M1_inv * BARS * At
        bsFun = At * M1_inv_conj * S * A
        expr = (zt * hFun * bzVec)[0, 0] + half * (
            (zt * sFun * zVec)[0, 0] + (bzt * bsFun * bzVec)[0, 0]
        )
    except Exception:
        raise RuntimeError(
            "`model2Nondegenerate` is unable to evaluate the model formula for the given parameters. Check that given matrices have compatible shapes."
        )
    graph = _simp(expr)
    if use_symbolic_conjugates is False:
        graph = symToHol(graph)
    result = graph - transverse_coordinate
    if return_CR_structure:
        kernel_total = get_free_symbols(S)
        seen = set()
        kernel_coordinates = []
        computed_parameters = set() if parameters is None else set(parameters)
        par_warning = False
        for var in kernel_total:
            if var not in computed_parameters:
                var_info = vmf_lookup(var, relatives=True).get("relatives", dict())
                holo = var_info.get("holo")
                if holo is None:
                    computed_parameters.add(var)
                    continue
                if (
                    var_info.get("real") in computed_parameters
                    or var_info.get("imag") in computed_parameters
                    or var_info.get("anti") in computed_parameters
                    or holo in computed_parameters
                ):
                    computed_parameters.add(var)
                    continue
                if var != holo:
                    par_warning = True
                if holo not in seen:
                    seen.add(holo)
                    kernel_coordinates.append(holo)
        if par_warning is True:
            dgcv_warning(
                "The provided symmetrix matrix has some complex coordinate elements other than holomorphic variables. They were not included among given parameters however. Since this matrix must be holomorphic in the underlying coordinates, you may have intended such elements to be included among the parameters. Use the optional `parameters` keyword to include them."
            )
        kernel_coordinates = order_coordinates(kernel_coordinates)
        holomorphic_coor = [tc_holo] + list(base_coordinates) + list(kernel_coordinates)
        if coordinates_to_weights_dict is not None:
            k, v = next(iter(coordinates_to_weights_dict.items()))
            cw_map = dict(zip(k, v))
            weights = [
                cw_map.get(var) for var in holomorphic_coor if var in holomorphic_coor
            ]
        else:
            weights = None
        return CR_structure(
            holomorphic_coor,
            {tc_info["relatives"]["imag"]: graph},
            parameters=parameters,
            weights=weights,
        )
    if return_matrices:
        return result, hFun, sFun
    return result


# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------
def _sym_engine_module():
    return engine_module()


def _is_sequence(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def _as_list(x: Any) -> List[Any]:
    return list(x) if _is_sequence(x) else [x]


def _stable_tuple(seq: Iterable[Any]) -> Tuple[Any, ...]:
    out: List[Any] = []
    seen = set()
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return tuple(out)


def _unwrap_symbolic(obj: Any) -> Any:
    if obj is None:
        return obj

    f = getattr(obj, "as_expr", None)
    if callable(f):
        try:
            return f()
        except TypeError:
            return f

    for attr in ("polyExpr", "_polyExpr", "expr", "_expr"):
        if hasattr(obj, attr):
            return getattr(obj, attr)

    P = getattr(obj, "poly_obj_unformatted", None)
    if P is not None and hasattr(P, "as_expr"):
        try:
            return P.as_expr()
        except Exception:
            pass

    return obj
