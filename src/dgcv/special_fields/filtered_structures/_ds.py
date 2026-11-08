from __future__ import annotations

from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    clear_denominators,
    get_free_symbols,
    subs,
)
from ..._aux._backends._types_and_constants import _disposable_symbols
from ..._aux._vmf._safeguards import create_key, get_dgcv_category
from ...algebras import _extract_basis
from ...core.solvers import solve_dgcv


class _DS_component:
    __slots__ = ("parts", "spanners", "coords")

    def __init__(self, spanners, parts=None, coords=None):
        self.spanners = spanners
        self.parts = parts
        self.coords = coords

    def truncated_spanners(self, max_weight):
        if self.parts is None:
            return self.spanners
        truncated = []
        for part_map in self.parts:
            pieces = [v for w, v in sorted(part_map.items()) if w <= max_weight]
            if len(pieces) == 0:
                continue
            elem = pieces[0]
            for piece in pieces[1:]:
                elem = elem + piece
            truncated.append(elem)
        return truncated


class _DS_record:
    """
    distinguished subspace data

    Attributes
    ----------
    components : dict
        Maps weight intervals `(low, high)` to `_DS_component`. Weighted
        homogeneous components use `(w, w)`. Keys record minimal supports of
        a basis adapted to the weight filtration from both ends, so an element
        set for any weight range is recovered by key containment.
    cap : int
        Represents degrees above which DS are unconstrained
    display : list
        Reformatted elements retained for reporting and reconstruction.
    """

    __slots__ = ("cap", "components", "display")

    def __init__(self, components, display, cap=None):
        self.components = components
        self.cap = cap
        self.display = display

    def component(self, weight):
        return self.components.get((weight, weight), None)

    def target_spanners(self, low, high):
        gathered = []
        for (w1, w2), component in self.components.items():
            if low <= w1 and w2 <= high:
                gathered += list(component.spanners)
        return gathered


def _DS_weight_list(elem, primary_grading):
    if get_dgcv_category(elem) in {
        "algebra_element",
        "subalgebra_element",
    }:
        return sorted(
            elem.weighted_decomposition(
                test_weights=[primary_grading], flatten_weights=True
            ).keys()
        )
    if get_dgcv_category(elem) == "tensorProduct":
        try:
            reported = elem.compute_weight(
                test_weights=[primary_grading], _return_mixed_weight_list=True
            )
        except Exception:
            raise ValueError(
                "Unable to process given distinguished subspaces. Cannot infer weights of components in some of the given elements "
            )
        flattened = set()
        for entry in reported:
            if isinstance(entry, (list, tuple)):
                flattened |= set(entry)
            else:
                flattened.add(entry)
        return sorted(flattened)
    return []


def _DS_is_zero(elem):
    return _scalar_is_zero(elem)


def _DS_homogeneous_parts(elem, primary_grading):
    if get_dgcv_category(elem) in {
        "algebra_element",
        "subalgebra_element",
    }:
        return dict(
            elem.weighted_decomposition(
                test_weights=[primary_grading], flatten_weights=True
            )
        )
    if get_dgcv_category(elem) == "tensorProduct":
        return {
            w: elem.get_weighted_components([w], test_weights=[primary_grading])
            for w in _DS_weight_list(elem, primary_grading)
        }
    return {}


def _DS_realign(spanners, primary_grading):
    """
    Rebuild a spanning set as a basis adapted to the weight filtration.

    Parameters
    ----------
    spanners : list
        Elements spanning one distinguished subspace.
    primary_grading : list
        Weight vector the supports are measured against.

    Returns
    -------
    list
        Triples `(interval, element, parts)`, where `interval` is the minimal
        weight support of `element` and `parts` maps weights to its weighted
        homogeneous components.
    """
    parts = [_DS_homogeneous_parts(elem, primary_grading) for elem in spanners]
    live = [n for n, part_map in enumerate(parts) if len(part_map) > 0]
    spanners = [spanners[n] for n in live]
    parts = [parts[n] for n in live]
    if len(spanners) == 0:
        return []
    if len(spanners) > 1:
        _, independent = _extract_basis(spanners, return_indices=True)
        spanners = [spanners[n] for n in independent]
        parts = [parts[n] for n in independent]
    weights = sorted({w for part_map in parts for w in part_map})

    label = create_key(prefix="_dsRealign")
    avars = _disposable_symbols(label, len(spanners))
    var_set = set(avars)

    blocks = dict()
    for w in weights:
        terms = [a * part_map[w] for a, part_map in zip(avars, parts) if w in part_map]
        block = terms[0]
        for term in terms[1:]:
            block = block + term
        blocks[w] = block

    def build(vector):
        terms = [c * s for c, s in zip(vector, spanners) if not _scalar_is_zero(c)]
        elem = terms[0]
        for term in terms[1:]:
            elem = elem + term
        pieces = dict()
        for c, part_map in zip(vector, parts):
            if _scalar_is_zero(c):
                continue
            for w, piece in part_map.items():
                pieces[w] = pieces[w] + c * piece if w in pieces else c * piece
        return elem, {w: v for w, v in sorted(pieces.items()) if not _DS_is_zero(v)}

    def support_kernel(low, high):
        eqns = []
        for w in weights:
            if low <= w <= high:
                continue
            eqns += list(getattr(blocks[w], "coeff_dict", dict()).values())
        if len(eqns) == 0:
            return [
                tuple(1 if m == n else 0 for m in range(len(avars)))
                for n in range(len(avars))
            ]
        resolved = solve_dgcv(
            eqns, avars, method="linear_parametric", simplify_result=False
        )
        if len(resolved) == 0:
            return []
        resolved = resolved[0]
        values = [resolved.get(a, a) for a in avars]
        free = set()
        for value in values:
            free |= get_free_symbols(value)
        free &= var_set
        ordered = [a for a in avars if a in free]
        zeroing = {a: 0 for a in ordered}
        return [
            tuple(clear_denominators([subs(x, {**zeroing, a: 1}) for x in values]))
            for a in ordered
        ]

    assigned = []
    for span in range(len(weights)):
        if len(assigned) == len(spanners):
            break
        for position in range(len(weights) - span):
            if len(assigned) == len(spanners):
                break
            low, high = weights[position], weights[position + span]
            candidates = support_kernel(low, high)
            if len(candidates) == 0:
                continue
            prior = [
                entry
                for entry in assigned
                if low <= entry[0][0] and entry[0][1] <= high
            ]
            if len(prior) >= len(candidates):
                continue
            fresh = [build(vector) for vector in candidates]
            _, kept = _extract_basis(
                [entry[2] for entry in prior] + [elem for elem, _ in fresh],
                return_indices=True,
            )
            offset = len(prior)
            for idx in kept:
                if idx >= offset:
                    elem, pieces = fresh[idx - offset]
                    assigned.append(((low, high), pieces, elem))
    return [(key, elem, pieces) for key, pieces, elem in assigned]
