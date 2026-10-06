from __future__ import annotations

import numbers
from collections.abc import Iterable

from ..._aux._backends._polynomials import expr_union_primitives
from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    as_numer_denom,
    get_free_symbols,
)
from ..._aux._backends._types_and_constants import expr_numeric_types
from ..._aux._utilities._config import dgcv_warning
from ..._aux._vmf._safeguards import get_dgcv_category, retrieve_passkey
from ...algebras import (
    _extract_basis,
    algebra_subspace_class,
    intersection,
    subalgebra_class,
)
from ...core.arrays import array_dgcv, freeze_matrix, matrix_dgcv
from ...core.vector_fields_and_differential_forms.decomposition import decompose
from ._ds import _DS_component, _DS_realign, _DS_record, _DS_weight_list
from ._formatting import (
    _GAE_to_hom_formatting,
    _nonnegative_parts_weight,
    _to_subspace_format,
)
from ._tensor_products import _fast_tensor_products



def _one_hot_first(candidate):
    elem = candidate[2]
    cd = getattr(elem, "coeff_dict", None)
    values = cd.values() if isinstance(cd, dict) else elem.coeffs
    return 0 if sum(1 for x in values if not _scalar_is_zero(x)) == 1 else 1

class _symbol_core:
    def __init__(
        self,
        GLA,
        nonnegParts=[],
        assume_FGLA=False,
        subspace=None,
        distinguished_subspaces=None,
        prolongation_label_prefix: str = None,
        assume_linear_independence=False,
        assume_NNP_linear_indep=False,
        index_threshold=None,
        precompute_generators=True,
        compress_equation_systems=None,
        _validated=None,
        _internal_parameters=set(),
        _internal_singularities=None,
    ):
        class dynamic_dict(dict):
            def __init__(self, dict_data, initial_index=None):
                super().__init__(dict_data)
                self.index_threshold = initial_index

            def __getitem__(self, key):
                if isinstance(key, numbers.Integral) and (
                    self.index_threshold is None or key >= self.index_threshold
                ):
                    return super().get(key, [])
                return super().get(key, [])

            def _set_index_thr(self, new_threshold):
                if not (
                    isinstance(new_threshold, expr_numeric_types())
                    or new_threshold is None
                ):
                    raise TypeError("index_threshold must be an integer or None.")
                self.index_threshold = new_threshold

            def copy(self):
                return dynamic_dict(self, initial_index=self.index_threshold)

        # validation
        def valence_check(tp):
            for j in tp.coeff_dict:
                valence = tuple(factor[1] for factor in j)
                if valence[0] != 1:
                    return False
                if not all(w == 0 for w in valence[1:]):
                    return False
            return True

        if _validated != retrieve_passkey():
            if get_dgcv_category(GLA) not in {
                "algebra",
                "algebra_subspace",
                "subalgebra",
            }:
                raise TypeError(
                    "`Tanaka_symbol` expects `GLA` (which represents a generalized graded Lie algebra) to be an `algebra`, `sualgebra`, or `algebra_subspace_class`, and the first element of `GLA.grading` must contain negative weights (-depth,...,-1)."
                )
            elif not hasattr(GLA, "grading") or len(GLA.grading) == 0:
                raise TypeError(
                    "`Tanaka_symbol` expects `GLA` to be a graded Lie algebra, but the supplied `GLA` has no grading assigned."
                )
            if len(nonnegParts) != 0:  ###!!! patch __init__ to allow this branch
                if (
                    isinstance(GLA.grading[0], (list, tuple))
                    and max(GLA.grading[0]) >= 0
                ):
                    raise TypeError(
                        "While `Tanaka_symbol` supports two syntax formats for encoding non-negative weighted components, they cannot be combined. Either `GLA.grading` should have only non-negative weights or no value for the optional `nonnegParts` parameter should be given."
                    )
            if isinstance(nonnegParts, dict):
                NNPList = list(nonnegParts.values())
            elif (
                isinstance(nonnegParts, (list, tuple))
                and any(not isinstance(entry, (list, tuple)) for entry in nonnegParts)
            ) or len(nonnegParts) == 0:
                NNPList = [nonnegParts]
            else:
                raise TypeError(
                    "`Tanaka_symbol` expects `nonnegParts` to be a list of `tensorProduct` instances built from the `algebra_class` given for `GLA` with `valence` of the form (1,0,...,0). Or it can be a dictionary whose keys are non-negative weights, and whose key-values are such lists."
                )
            for NNP in NNPList:
                if not all(
                    get_dgcv_category(j) == "tensorProduct"
                    and j.vector_space == GLA
                    and valence_check(j)
                    for j in NNP
                ):
                    raise TypeError(
                        "`Tanaka_symbol` expects `nonnegParts` to be a list of `tensorProduct` instances built from the `algebra` given for `GLA` with `valence` of the form (1,0,...,0). Or it can be a dictionary whose keys are non-negative weights, and whose key-values are such lists."
                    )
        else:
            if isinstance(nonnegParts, dict):
                NNPList = list(nonnegParts.values())
            elif isinstance(nonnegParts, (list, tuple)):
                NNPList = [nonnegParts]
        if isinstance(GLA.grading[0], (list, tuple)):
            primary_grading = GLA.grading[0]
        else:
            if _validated != retrieve_passkey() and not any(
                j == -1 for j in GLA.grading
            ):  ###!!! review
                raise TypeError(
                    f"`Tanaka_symbol` expects `GLA` to be a Z-graded algebra (`algebra_class`, `algebra_subspace_class`, or `sualgebra_class` in particular) with the weight -1 among its weights in the first element of `GLA.grading`. Recieved grading data: {GLA.grading}"
                )
            primary_grading = GLA.grading
        non_neg_GLA = True if max(primary_grading) >= 0 else False

        raiseWarning = False
        if subspace is None:
            (
                subIndices,
                si_count,
                filtered_grading,
                truncateIndices,
                nonnegPartsTemp,
                index_map,
            ) = [], 0, [], dict(), dict(), dict()
            for count, weight in enumerate(primary_grading):
                if weight < 0:
                    truncateIndices[count] = si_count
                    subIndices.append(count)
                    index_map[count] = si_count
                    filtered_grading.append(weight)
                    si_count += 1
                else:
                    nonnegPartsTemp[weight] = nonnegPartsTemp.get(weight, []) + [
                        GLA.basis[count]
                    ]
            if si_count == 0:
                raise ValueError(
                    "`Tanaka_symbol` objects cannot be initialized from GLA data that has no negative weight components."
                )
            if len(nonnegPartsTemp) > 0:

                def truncateBySubInd(li):
                    return [li[j] for j in subIndices]

                def restrict_structure_data(data):
                    new_data = dict()
                    inner_shape = (si_count, 1)
                    for (i, j, k), v in data.items():
                        if i in subIndices and j in subIndices:
                            if k in subIndices:
                                outer_key = (index_map[i], index_map[j])
                                if outer_key in new_data:
                                    new_data[outer_key][index_map[k]] = v
                                else:
                                    new_data[outer_key] = matrix_dgcv(
                                        {index_map[k]: v}, shape=inner_shape
                                    )
                            elif v is not None and not _scalar_is_zero(v):
                                raise TypeError(
                                    "The GLA data given to the `Tanaka_symbol` initializer appears to not be compatible with its grading."
                                )
                    return array_dgcv(
                        new_data,
                        shape=(si_count, si_count),
                        null_return=freeze_matrix(matrix_dgcv.zeros(si_count, 1)),
                    )

                subspace = subalgebra_class(
                    truncateBySubInd(GLA.basis),
                    GLA,
                    grading=[filtered_grading],
                    _compressed_structure_data=restrict_structure_data(
                        GLA.structureDataDict
                    ),
                    _internal_lock=retrieve_passkey(),
                )
            else:
                subspace = GLA
            spans_full_negative_part = True
        else:
            spans_full_negative_part = False
            if not isinstance(subspace, Iterable):
                raise TypeError(
                    "`Tanaka_symbol` expects `subpsace` if given to be a list of algebra_element_class instances belonging to the algebra_class `GLA` or tensor products of such elements, or a similar subspace-like object."
                )
            typeCheck = {
                "subalgebra_element",
                "algebra_element",
            }
            negative_basis, filtered_grading, nonnegPartsTemp = [], [], {}
            for elem in subspace:
                dgcvType = get_dgcv_category(elem)
                if dgcvType in typeCheck and elem.vectorSpace == GLA:
                    w = elem.check_element_weight(
                        test_weights=[primary_grading], flatten_weights=True
                    )
                    if w == "NoW":
                        raise TypeError(
                            "`Tanaka_symbol` expects the spanning set of elements given to define `subpsace` to be weighted homogeneous w.r.t. the primary grading."
                        )
                    elif isinstance(w, numbers.Integral):
                        if w < 0:
                            negative_basis.append(elem)
                            filtered_grading.append(w)
                        else:
                            nonnegPartsTemp[w] = nonnegPartsTemp.get(w, []) + [elem]
                elif (
                    dgcvType == "tensorProduct"
                    and elem.vectorSpace == GLA
                    and valence_check(elem) is True
                ):
                    if raiseWarning is False and non_neg_GLA is True:
                        raiseWarning = True
                    w = elem.compute_weight(
                        test_weights=[primary_grading], flatten_weights=True
                    )
                    if w == "NoW":
                        raise TypeError(
                            "`Tanaka_symbol` expects the spanning set of elements given to define `subpsace` to be weighted homogeneous w.r.t. the primary grading."
                        )
                    elif (
                        elem.max_degree > 1 and isinstance(w, numbers.Number) and w < 0
                    ):
                        raise TypeError(
                            "negatively-graded elements among those given to define `subpsace` should be bare algebra_element_class/vecor_space_element instances, rather than tensor products of such."
                        )
                    elif isinstance(w, numbers.Integral):
                        if w < 0:
                            negative_basis.append(elem)
                            filtered_grading.append(w)
                        else:
                            nonnegPartsTemp[w] = nonnegPartsTemp.get(w, []) + [elem]
                    try:
                        subspace = subalgebra_class(
                            negative_basis, GLA, grading=[filtered_grading]
                        )
                    except ValueError:
                        raise TypeError(
                            "`Tanaka_symbol` expects `subpsace` if given to be have the subspace within its given `GLA` spanned by its negatively graded elements be closed under Lie brackets."
                        )
                    if raiseWarning is True:
                        dgcv_warning(
                            "The graded algebra `GLA` given to `Tanaka_symbol` has non-negative components, but the supplied `subspace` had some non-negative degree elements formatted has tensor products rather than elements of the provided `GLA`. This mixing of formatting results in slower prolongation algorithm, so it is recommended to instead either supply `subset` as formal elements in the `GLA` or give `GLA`  as just its negative component and then additionally supply non-negative components as tensor products via the optional `nonnegParts` parameter."
                        )
                        subspace = algebra_subspace_class(
                            negative_basis,
                            parent_algebra=GLA,
                            _grading=[filtered_grading],
                            _internal_lock=retrieve_passkey(),
                        )
                else:
                    raise TypeError(
                        "`Tanaka_symbol` expects `subpsace` if given to be a list of algebra_element_class instances belonging to the algebra_class `GLA` or tensor products of such elements, or a similar subspace-like object."
                    )

        if len(nonnegPartsTemp) > 0:
            if len(nonnegParts) > 0:
                dgcv_warning(
                    "The `GLA` or `subspace` parameter provided to `Tanaka_symbol` has nonnegatively weighted components. If providing such `GLA` or `subspace` data then the optional `nonnegParts` cannot be manually set. So the provided manual setting for `nonnegParts` is being ignored."
                )
            nonnegParts = nonnegPartsTemp
            for w in nonnegParts.keys():
                nonnegParts[w] = [
                    _GAE_to_hom_formatting(j, subspace, test_weights=[primary_grading])
                    for j in nonnegParts[w]
                ]

        subalgebra_format = get_dgcv_category(subspace) == "subalgebra"
        proper_subspace = subalgebra_format and not spans_full_negative_part
        pending_DS = []
        if distinguished_subspaces:
            if not isinstance(distinguished_subspaces, (list, tuple)):
                raise TypeError(
                    "`Tanaka_symbol` expects `distinguished_subspaces` to be a list of lists of algebra_element_class instances or tensor products belonging to the provided basis of the symbol."
                )
            ds_params = set()
            newDS = []
            processed_DS = []

            for subS in distinguished_subspaces:
                if not (
                    isinstance(subS, (list, tuple))
                    or get_dgcv_category(subS) == "algebra_subspace"
                ):
                    raise TypeError(
                        "`Tanaka_symbol` expects `distinguished_subspaces` to be a list of lists of algebra_element_class instances or tensor products built from the provided basis of the symbol, or some subspace class."
                    )
                supplied = []
                weight_lists = []
                graded = True
                for elem in subS:
                    ds_params |= get_free_symbols(elem)
                    weights = _DS_weight_list(elem, primary_grading)
                    if len(weights) == 0:
                        continue
                    supplied.append(elem)
                    weight_lists.append(weights)
                    if len(weights) > 1:
                        graded = False
                realigned = []
                if graded is False:
                    realigned = _DS_realign(supplied, primary_grading)
                    if all(low == high for (low, high), _, _ in realigned):
                        supplied = [elem for _, elem, _ in realigned]
                        weight_lists = [[low] for (low, _), _, _ in realigned]
                        realigned = []
                        graded = True
                if graded is False:
                    components = dict()
                    DSList = []
                    for (low, high), elem, pieces in realigned:
                        crossed = _GAE_to_hom_formatting(
                            elem, subspace, test_weights=[primary_grading]
                        )
                        DSList.append(crossed)
                        if high < 0:
                            spanner = crossed
                            parts = (
                                None
                                if low == high
                                else {
                                    w: _GAE_to_hom_formatting(
                                        piece, subspace, test_weights=[primary_grading]
                                    )
                                    for w, piece in pieces.items()
                                }
                            )
                        else:
                            spanner = _fast_tensor_products(crossed)
                            parts = None
                        if (low, high) in components:
                            components[(low, high)][0].append(spanner)
                            components[(low, high)][1].append(parts)
                        else:
                            components[(low, high)] = ([spanner], [parts])
                    newDS.append(DSList)
                    if len(components) > 0:
                        pending_DS.append(
                            _DS_record(
                                {
                                    key: _DS_component(
                                        held,
                                        None
                                        if all(p is None for p in held_parts)
                                        else held_parts,
                                    )
                                    for key, (held, held_parts) in components.items()
                                },
                                DSList,
                            )
                        )
                    continue
                bands = dict()
                legible = dict()
                for elem, weights in zip(supplied, weight_lists):
                    w = weights[0]
                    if w < 0:
                        reformElem = elem
                        banded = elem
                    else:
                        reformElem = _GAE_to_hom_formatting(
                            elem, subspace, test_weights=[primary_grading]
                        )
                        banded = _fast_tensor_products(reformElem)
                    bands[w] = bands.get(w, []) + [banded]
                    legible[w] = legible.get(w, []) + [reformElem]
                for w in [k for k in bands if k < 0]:
                    if proper_subspace:
                        bands[w] = intersection(
                            bands[w], [e.ambient_rep for e in subspace.basis]
                        )
                    elif assume_linear_independence is not True:
                        bands[w] = _extract_basis(bands[w])
                    if subalgebra_format:
                        bands[w] = [_to_subspace_format(e, subspace) for e in bands[w]]
                    legible[w] = bands[w]
                    if len(bands[w]) == 0:
                        del bands[w]
                        del legible[w]
                DSList = [e for w in sorted(legible) for e in legible[w]]
                newDS.append(DSList)
                if len(bands) > 0:
                    processed_DS.append(
                        _DS_record(
                            {(w, w): _DS_component(v) for w, v in bands.items()},
                            DSList,
                        )
                    )
            self._DS_records = processed_DS
        else:
            ds_params = set()
            newDS = []
            self._DS_records = []
        self._inadmissible_DS = []
        self._parameters = GLA._parameters | ds_params | _internal_parameters

        distinguished_subspaces = newDS

        self.negativePart = subspace
        self.ambientGLA = GLA
        self.assume_FGLA = assume_FGLA
        negWeights = sorted([j for j in set(primary_grading) if j < 0])
        self.negWeights = tuple(negWeights)
        if isinstance(nonnegParts, dict):
            nonneg_levels = {
                w: list(level) for w, level in nonnegParts.items() if len(level) != 0
            }
        else:
            nonneg_levels = dict()
            for position, elem in enumerate(nonnegParts):
                w = _nonnegative_parts_weight(elem, primary_grading, position)
                if w is None:
                    continue
                nonneg_levels[w] = nonneg_levels.get(w, []) + [elem]
        self.depth = negWeights[0]
        GLA_levels = dict()
        grad = (
            filtered_grading
            if get_dgcv_category(self.negativePart) == "subalgebra"
            else primary_grading
        )
        for elem in self.negativePart.basis:
            w = elem.check_element_weight(test_weights=[grad])[0]
            GLA_levels[w] = GLA_levels.get(w, []) + [elem]
        self.GLA_levels = GLA_levels
        self._dgcv_class_check = retrieve_passkey()
        self._dgcv_category = "Tanaka_symbol"

        self.nonneg_levels = nonneg_levels
        levels = dict(sorted((self.GLA_levels | self.nonneg_levels).items()))

        self._GLA_structure = dynamic_dict
        self.levels = dynamic_dict(levels, initial_index=index_threshold)
        self.dimension = sum(len(level) for level in self.levels.values())
        self.distinguished_subspaces = distinguished_subspaces
        self._test_commutators = None
        self._GLA_generators = None
        self._plp = (
            prolongation_label_prefix
            if isinstance(prolongation_label_prefix, str)
            else "eta"
        )
        self._aliasing = dict()
        self._nonneg_hom = self._derive_nonneg_hom(assume_NNP_linear_indep)
        nonNegWeights = sorted(self.nonneg_levels)
        self.height = nonNegWeights[-1] if len(nonNegWeights) > 0 else -1
        self.weights = negWeights + nonNegWeights
        if isinstance(nonnegParts, dict):
            self.nonnegParts = {w: list(v) for w, v in self.nonneg_levels.items()}
        else:
            self.nonnegParts = [
                elem for w in nonNegWeights for elem in self.nonneg_levels[w]
            ]
        self._singularities = (
            _internal_singularities if _internal_singularities else dict()
        )
        for record in pending_DS:
            if proper_subspace:
                self._inadmissible_DS.append(list(record.display))
                dgcv_warning(
                    "`Tanaka_symbol` received a distinguished subspace that is not spanned by weighted homogeneous elements, alongside a `subspace` spanning less than the full negative part. Restricting such a distinguished subspace to a proper subspace is not supported, so it is being disregarded."
                )
            elif max(hi for _, hi in record.components) > self.height:
                self._inadmissible_DS.append(list(record.display))
                dgcv_warning(
                    "`Tanaka_symbol` received a distinguished subspace that is not spanned by weighted homogeneous elements and whose weighted components reach above the height of the supplied symbol data. Such distinguished subspaces are only supported up to that height, so it is being disregarded."
                )
            else:
                self._DS_records.append(record)
        maxDSW = -1
        for record in self._DS_records:
            maxDSW = max(maxDSW, max(hi for _, hi in record.components))
            record.cap = max(self.height, max(hi for _, hi in record.components))
        self._default_to_characteristic_space_reductions = maxDSW >= 0
        self._compress_equation_systems = (
            precompute_generators is True and len(self._parameters) == 0
            if compress_equation_systems is None
            else compress_equation_systems is True
        )
        if precompute_generators is True or self._compress_equation_systems:
            _ = self.GLA_generators

    @property
    def test_commutators(self):
        if self._test_commutators is None:
            if self.assume_FGLA:
                deeper_levels = sum(
                    [self.GLA_levels[j] for j in self.negWeights[:-1]], []
                )
                f_level = self.GLA_levels[-1]
                first_commutators = [
                    (f_level[j], f_level[k], f_level[j] * f_level[k])
                    for j in range(len(f_level))
                    for k in range(j + 1, len(f_level))
                ]
                remaining_comm = [(j, k, j * k) for j in f_level for k in deeper_levels]
                self._test_commutators = first_commutators + remaining_comm
            elif self._GLA_generators is not None:
                generators = sum(self._GLA_generators["generators"].values(), [])
                gen_position = {id(g): n for n, g in enumerate(generators)}
                neg_basis = sum([list(j) for j in (self.GLA_levels).values()], [])
                commutators = []
                for n, g in enumerate(generators):
                    for b in neg_basis:
                        position = gen_position.get(id(b))
                        if position is not None and position <= n:
                            continue
                        commutators.append((g, b, g * b))
                self._test_commutators = commutators
            else:
                self._test_commutators = self._pairwise_test_commutators()
        return self._test_commutators

    def _pairwise_test_commutators(self):
        cached = getattr(self, "_test_commutators_full", None)
        if cached is None:
            neg_levels = sum([list(j) for j in (self.GLA_levels).values()], [])
            cached = [
                (neg_levels[j], neg_levels[k], neg_levels[j] * neg_levels[k])
                for j in range(len(neg_levels))
                for k in range(j + 1, len(neg_levels))
            ]
            self._test_commutators_full = cached
        return cached

    @property
    def GLA_generators(self):
        if self._GLA_generators is None:
            self._test_commutators = None
            self._GLA_generators = {"generators": {-1: self.levels[-1]}}
            self._GLA_generators["map"] = {-1: [(j, j, 1) for j in self.levels[-1]]}
            self._GLA_generators["triples"] = dict()
            elem_weight = {id(j): -1 for j in self.levels[-1]}
            nRange = range(-1, min(self.negWeights) - 1, -1)
            generated = []
            for w in nRange[1:]:
                w_level = []
                w_level_brackets = []
                w_level_triples = []
                brackets = self.ambientGLA.subspace()
                candidates = []
                for idx1 in range(-1, w // 2 - 1, -1):
                    idx2 = w - idx1
                    for c1, eT1 in enumerate(self._GLA_generators["map"].get(idx1, [])):
                        newC = c1 + 1 if idx1 == idx2 else 0
                        for eT2 in self._GLA_generators["map"].get(idx2, [])[newC:]:
                            tuple1, e1, dep1 = eT1
                            tuple2, e2, dep2 = eT2
                            dep3 = max(dep1, dep2) + 1
                            eProd = e1 * e2
                            elem_weight[id(eProd)] = w
                            if dep3 == 2:
                                w_level_triples.append((tuple1, tuple2, eProd))
                            candidates.append((tuple1, tuple2, eProd, dep3))
                candidates.sort(key=_one_hot_first)
                for tuple1, tuple2, eProd, dep3 in candidates:
                    d1 = brackets.dimension
                    brackets.append(eProd)
                    if brackets.dimension - d1 > 0:
                        generated.append(eProd)
                        w_level_brackets.append(([tuple1, tuple2], eProd, dep3))
                for elem in self.levels[w]:
                    d1 = brackets.dimension
                    brackets.append(elem)
                    elem_weight[id(elem)] = w
                    if brackets.dimension - d1 > 0:
                        w_level.append(elem)
                        w_level_brackets.append((elem, elem, 1))
                if len(w_level) > 0:
                    self._GLA_generators["generators"][w] = w_level
                self._GLA_generators["map"][w] = w_level_brackets
                if len(w_level_triples) > 0:
                    self._GLA_generators["triples"][w] = w_level_triples
            self._GLA_generators["generated"] = generated
            self._GLA_generators["elem_weight"] = elem_weight
            span_index = dict()
            node_elem = dict()
            expr = dict()
            spanning = dict()
            coords = dict()
            for w, entries in self._GLA_generators["map"].items():
                exprs = []
                S = []
                for node, elem, dep in entries:
                    span_index[id(elem)] = (w, len(exprs))
                    S.append(elem)
                    if dep == 1:
                        exprs.append(None)
                        node_elem[id(elem)] = elem
                    else:
                        exprs.append(node)
                        node_elem[id(node)] = elem
                expr[w] = exprs
                spanning[w] = S
                coords[w] = [
                    list(decompose(b, S, assume_basis=True)[0]) for b in self.levels[w]
                ]
            neg_positions = self._negative_basis_positions()
            self._GLA_generators["span_index"] = span_index
            self._GLA_generators["node_elem"] = node_elem
            self._GLA_generators["expr"] = expr
            self._GLA_generators["spanning"] = spanning
            self._GLA_generators["coords"] = coords
            self._GLA_generators["level_positions"] = {
                w: [neg_positions[(w, i)] for i in range(len(self.levels[w]))]
                for w in self._GLA_generators["map"]
            }
            if self._parameters:
                divisors = []
                for rows in coords.values():
                    for row in rows:
                        for c in row:
                            if _scalar_is_zero(c):
                                continue
                            _, d = as_numer_denom(c)
                            if get_free_symbols(d):
                                divisors.append(d)
                if divisors:
                    self._singularities["prolongation"] = expr_union_primitives(
                        list(self._singularities.get("prolongation", [])) + divisors,
                        self._parameters,
                        process_rationals=True,
                        fail_quietly=True,
                    )
            if self._compress_equation_systems:
                self._generator_data()
            if (
                self.assume_FGLA is True
                and min(self._GLA_generators["generators"]) < -1
            ):
                self.assume_FGLA = False
                dgcv_warning(
                    "The parameter setting `assume_FGLA=True` has been overwritten because a diognostic has shown the symbol is not fundamental."
                )
        return self._GLA_generators

    @property
    def basis(self):
        return sum(list(self.levels.values()), [])

    def __iter__(self):
        return iter(self.basis)
