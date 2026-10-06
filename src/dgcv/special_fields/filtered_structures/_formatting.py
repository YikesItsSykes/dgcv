from __future__ import annotations

from ..._aux._vmf._safeguards import get_dgcv_category


def _to_subspace_format(elem, subspace):
    if (
        get_dgcv_category(subspace) != "subalgebra"
        or get_dgcv_category(elem) != "algebra_element"
    ):
        return elem
    coeffs = subspace.contains(elem, return_basis_coeffs=True)
    if not isinstance(coeffs, dict):
        return elem
    return subspace._class_builder(coeffs, elem.valence)


def _nonnegative_parts_weight(elem, primary_grading, position):
    stored = getattr(elem, "_properties", {}).get("_weight")
    if stored is not None:
        stored_grading, weight = stored
        if stored_grading is not None and list(stored_grading) == list(primary_grading):
            return weight
    weight = elem.compute_weight(test_weights=[primary_grading])[0]
    if isinstance(weight, str):
        if weight == "AllW":
            return None
        raise TypeError(
            f"`Tanaka_symbol` expects every element in provided nonnegative parts to be weighted homogeneous w.r.t. the symbol's primary grading, but element {position + 1} is not. Supply weighted homogeneous elements instead."
        )
    return weight


def _GAE_to_hom_formatting(elem, nilradical, test_weights=None, return_weights=False):
    if get_dgcv_category(elem) not in {
        "algebra_element",
        "subalgebra_element",
    }:
        if return_weights:
            if get_dgcv_category(elem) == "tensorProduct":
                try:
                    return elem, elem.compute_weight(
                        test_weights=test_weights, _return_mixed_weight_list=True
                    )[0]
                except Exception:
                    raise ValueError(
                        "Unable to process given distinguished subspaces. Cannot infer weights of components in some of the given elements "
                    )
            else:
                return elem, []
        return elem
    test_switch = get_dgcv_category(nilradical) == "subalgebra"
    if test_weights is None:
        test_weights = [elem.algebra.grading[0]]
    wd = elem.weighted_decomposition(test_weights=test_weights, flatten_weights=True)
    weights = list(wd.keys())
    if all(w < 0 for w in weights):
        if test_switch and get_dgcv_category(elem) == "algebra_element":
            elem = nilradical._class_builder(
                nilradical.contains(elem, return_basis_coeffs=True), elem.valence
            )
        if return_weights:
            return elem, weights
        return elem
    terms = []
    for weight, term in wd.items():
        if weight < 0:
            if test_switch and get_dgcv_category(term) == "algebra_element":
                term = nilradical._class_builder(
                    nilradical.contains(term, return_basis_coeffs=True), term.valence
                )
            terms.append(term)
        else:
            for testEl in nilradical:
                terms.append(
                    _GAE_to_hom_formatting(
                        term * testEl, nilradical, test_weights=test_weights
                    )
                    @ testEl.dual()
                )
    if return_weights:
        return sum(terms[1:], terms[0]), weights
    return sum(terms[1:], terms[0])
