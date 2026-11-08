from collections import Counter

from ..._aux._vmf._safeguards import get_dgcv_category
from ..combinatorics.combinatorics import shufflings
from ..dgcv_core.spaces.spaces import card_root
from ._protocol import _is_scalar, _is_vs_element, _vs_bracket
from .promotion import _resolve_binary_promotions


class _tp_contraction_product:
    def _contraction_product(self, other, include_Lie_brackets=True):
        if len(self.terms) > 1:
            tl = [term._contraction_product(other) for term in self.terms]
            return sum(tl)
        k, v = next(iter(self.coeff_dict.items()))
        deg = len(k)
        if deg == 0:
            return v * other
        if deg == 1:
            idx0, val0, card0 = k[0]
            if get_dgcv_category(other) == "subalgebra_element":
                if val0 == other.valence:
                    elem = (
                        v * (card0.space.basis[idx0])
                        if val0 == 1
                        else v * (card0.space.basis[idx0].dual())
                    )
                    if card0 == other.card:
                        return elem * other
                    elif card0 == other.ambient_rep.card:
                        return elem * (other.ambient_rep)
                    elif getattr(elem, "ambient", elem).card == other.ambient_rep.card:
                        return (elem.ambient_rep) * (other.ambient_rep)
                    else:
                        return 0 * other
                else:
                    if card0 == other.card:
                        return v * other.coeffs[idx0]
                    elif card0 == other.ambient_rep.card:
                        return v * other.ambient_rep.coeffs[idx0]
                    else:
                        elem = (
                            v * (card0.space.basis[idx0])
                            if val0 == 1
                            else v * (card0.space.basis[idx0].dual())
                        )
                        if (
                            getattr(elem, "ambient", elem).card
                            == other.ambient_rep.card
                        ):
                            return (
                                elem.ambient_rep._convert_to_tp()._contraction_product(
                                    other.ambient_rep
                                )
                            )
                        else:
                            return 0 * self
            elif _is_vs_element(other):
                if val0 == other.valence:
                    elem = (
                        v * (card0.space.basis[idx0])
                        if val0 == 1
                        else v * (card0.space.basis[idx0].dual())
                    )
                    if card0 == other.card:
                        return elem * other
                    else:
                        if getattr(elem, "ambient", elem).card == other.card:
                            return (elem.ambient_rep) * (other)
                        else:
                            return 0 * self
                else:
                    if card0 == other.card:
                        return v * other.coeffs[idx0]
                    else:
                        elem = (
                            v * (card0.space.basis[idx0])
                            if val0 == 1
                            else v * (card0.space.basis[idx0].dual())
                        )
                        if getattr(elem, "ambient", elem).card == other.card:
                            return (
                                elem.ambient_rep._convert_to_tp()._contraction_product(
                                    other
                                )
                            )
                        else:
                            return 0 * self
            elif get_dgcv_category(other) == "tensorProduct":
                newDict = Counter()
                for k2, v2 in other.coeff_dict.items():
                    deg2 = len(k2)
                    if deg2 == 0:
                        newDict[k] += v * v2
                    elif deg2 == 1:
                        idx1, val1, card1 = k2[0]
                        if card0 == card1:
                            if val0 != val1:
                                if idx0 == idx1:
                                    newDict[tuple()] += v * v2
                            else:
                                elem1 = (
                                    v * (card0.space.basis[idx0])
                                    if val0 == 1
                                    else v * (card0.space.basis[idx0].dual())
                                )
                                elem2 = (
                                    v2 * (card1.space.basis[idx1])
                                    if val1 == 1
                                    else v2 * (card1.space.basis[idx1].dual())
                                )
                                nd = _vs_bracket(elem1, elem2)
                                if hasattr(nd, "_convert_to_tp"):
                                    nd = nd._convert_to_tp().coeff_dict
                                elif hasattr(nd, "coeff_dict"):
                                    nd = nd.coeff_dict
                                elif _is_scalar(nd):
                                    nd = {tuple(): nd}
                                else:
                                    raise RuntimeError(
                                        "unanticipated edge case in tensorProduct._contraction_product algo"
                                    )
                                for nk, nv in nd.items():
                                    newDict[nk] += nv
                        elif card_root(card0) == card_root(card1):
                            if card0 == card_root(card0):
                                nd = self._contraction_product(
                                    tensorProduct({k2: v2}, _amb_prom=True)
                                )
                            else:
                                nd = self.ambient_rep._contraction_product(
                                    tensorProduct({k2: v2}, _amb_prom=True)
                                )
                            if hasattr(nd, "_convert_to_tp"):
                                nd = nd._convert_to_tp().coeff_dict
                            elif hasattr(nd, "coeff_dict"):
                                nd = nd.coeff_dict
                            elif _is_scalar(nd):
                                nd = {tuple(): nd}
                            else:
                                raise RuntimeError(
                                    "unanticipated edge case in tensorProduct._contraction_product algo"
                                )
                            for nk, nv in nd.items():
                                newDict[nk] += nv
                    else:  # deg2>1
                        if val0 != k2[0][1]:
                            if card0 == k2[0][2]:
                                if idx0 == k2[0][0]:
                                    newDict[k2[1:]] += v * v2
                            elif card_root(card0) == card_root(k2[0][2]):
                                if card0 == card_root(card0):
                                    nd = self._contraction_product(
                                        tensorProduct({k2: v2}, _amb_prom=True)
                                    )
                                else:
                                    nd = self.ambient_rep._contraction_product(
                                        tensorProduct({k2: v2}, _amb_prom=True)
                                    )
                                if hasattr(nd, "_convert_to_tp"):
                                    nd = nd._convert_to_tp().coeff_dict
                                elif hasattr(nd, "coeff_dict"):
                                    nd = nd.coeff_dict
                                elif _is_scalar(nd):
                                    nd = {tuple(): nd}
                                else:
                                    raise RuntimeError(
                                        "unanticipated edge case in tensorProduct._contraction_product algo"
                                    )
                                for nk, nv in nd.items():
                                    newDict[nk] += nv
                        if val0 != k2[-1][1]:
                            if card0 == k2[-1][2]:
                                if idx0 == k2[-1][0]:
                                    newDict[k2[:-1]] += -v * v2
                            elif card_root(card0) == card_root(k2[-1][2]):
                                if card0 == card_root(card0):
                                    nd = self._contraction_product(
                                        tensorProduct({k2: v2}, _amb_prom=True)
                                    )
                                else:
                                    nd = self.ambient_rep._contraction_product(
                                        tensorProduct({k2: v2}, _amb_prom=True)
                                    )
                                if hasattr(nd, "_convert_to_tp"):
                                    nd = nd._convert_to_tp().coeff_dict
                                elif hasattr(nd, "coeff_dict"):
                                    nd = nd.coeff_dict
                                elif _is_scalar(nd):
                                    nd = {tuple(): nd}
                                else:
                                    raise RuntimeError(
                                        "unanticipated edge case in tensorProduct._contraction_product algo"
                                    )
                                for nk, nv in nd.items():
                                    newDict[nk] += nv
                return tensorProduct(newDict)
            else:
                raise ValueError(
                    f"Unsupported operation for * between the given object types: {type(self)} and {type(other)}; `other` dgcv type is {get_dgcv_category(other)}."
                )
        else:
            if _is_vs_element(other):
                if (
                    card_root(k[-1][2]) != k[-1][2]
                    and k[-1][2] != other.card
                    and k[-1][1] != other.valence
                    and getattr(other, "ambient_rep", other).card == card_root(k[-1][2])
                ):
                    return self.ambient_rep._contraction_product(other)
                if (
                    card_root(k[0][2]) != k[0][2]
                    and k[0][1] != other.valence
                    and k[0][2] != other.card
                    and getattr(other, "ambient_rep", other).card == card_root(k[0][2])
                ):
                    return self.ambient_rep._contraction_product(other)
                newDict = Counter()

                if k[-1][1] != other.valence:
                    locElem = (
                        other.ambient_rep
                        if k[-1][2] == other.card.root
                        else other
                    )
                    if k[-1][2] == locElem.card:
                        newDict[k[:-1]] += locElem.coeffs[k[-1][0]] * v
                if k[0][1] != other.valence:
                    locElem = (
                        other.ambient_rep
                        if k[0][2] == other.card.root
                        else other
                    )
                    if k[0][2] == locElem.card:
                        newDict[k[1:]] += -locElem.coeffs[k[0][0]] * v
                return tensorProduct(newDict)
            elif get_dgcv_category(other) == "tensorProduct":
                newDict = Counter()
                targets_self, targets_other = _resolve_binary_promotions(
                    self._card_by_root, other._card_by_root
                )
                if targets_self or targets_other:
                    return self._promoted(targets_self)._contraction_product(
                        other._promoted(targets_other)
                    )

                for k2, v2 in other.coeff_dict.items():
                    deg2 = len(k2)
                    if deg2 == 0:
                        newDict[k] += v * v2
                    elif deg2 == 1:
                        if k2[0][1] == 0:
                            nd = self._contraction_product(
                                v2 * k2[0][2].space.basis[k2[0][0]].dual()
                            )
                        else:
                            nd = self._contraction_product(
                                v2 * k2[0][2].space.basis[k2[0][0]]
                            )
                        if hasattr(nd, "coeff_dict"):
                            nd = nd.coeff_dict
                        elif hasattr(nd, "_convert_to_tp"):
                            nd = nd._convert_to_tp().coeff_dict
                        elif _is_scalar(nd):
                            nd = {tuple(): nd}
                        else:
                            raise RuntimeError(
                                "unanticipated edge case in tensorProduct._contraction_product algo"
                            )
                        for nk, nv in nd.items():
                            newDict[nk] += nv
                    else:
                        for idx in range(1, deg - 1):
                            if (
                                k[idx][0] == k2[0][0]
                                and k[idx][1] != k2[0][1]
                                and k[idx][2] == k2[0][2]
                            ):
                                new_value = v * v2
                                k1_start = k[:idx]
                                k2_start = k2[1:2]
                                k1_tail_inputs = k[idx + 1 :]
                                k2_inputs = k2[2:]
                                new_tails = shufflings(k1_tail_inputs, k2_inputs)
                                new_keys = [
                                    k1_start + k2_start + tuple(tail)
                                    for tail in new_tails
                                ]
                                for key in new_keys:
                                    newDict[key] += new_value
                        if (
                            k[-1][0] == k2[0][0]
                            and k[-1][1] != k2[0][1]
                            and k[-1][2] == k2[0][2]
                        ):
                            new_value = v * v2
                            k1_start = k[: deg - 1]
                            k2_inputs = k2[1:]
                            newDict[k1_start + k2_inputs] += new_value
                        if (
                            k2[-1][0] == k[0][0]
                            and k2[-1][1] != k[0][1]
                            and k2[-1][2] == k[0][2]
                        ):
                            new_value = -v2 * v
                            k2_start = k2[: deg2 - 1]
                            k1_inputs = k[1:]
                            newDict[k2_start + k1_inputs] += new_value
                        for idx in range(1, deg2 - 1):
                            if (
                                k2[idx][0] == k[0][0]
                                and k2[idx][1] != k[0][1]
                                and k2[idx][2] == k[0][2]
                            ):
                                new_value = -v2 * v
                                k2_start = k2[:idx]
                                k1_start = k[1:2]
                                k2_tail_inputs = k2[idx + 1 :]
                                k1_inputs = k[2:]
                                new_tails = shufflings(k2_tail_inputs, k1_inputs)
                                new_keys = [
                                    k2_start + k1_start + tuple(tail)
                                    for tail in new_tails
                                ]
                                for key in new_keys:
                                    newDict[key] += new_value
            return tensorProduct(newDict)

    def __mul__(self, other):
        """Overload * to compute the contraction product, with special logic for algebra_element."""
        if _is_scalar(other):
            new_coeff_dict = {
                key: value * other for key, value in self.coeff_dict.items()
            }
            return tensorProduct(new_coeff_dict)
        if _is_vs_element(other) or get_dgcv_category(other) == "tensorProduct":
            return self._contraction_product(other)
        else:
            raise ValueError(
                f"Unsupported operation for * between the given object types: {type(self)} and {type(other)}; `other` dgcv type is {get_dgcv_category(other)}."
            )

    def __rmul__(self, other):
        if _is_scalar(other):
            new_coeff_dict = {
                key: value * other for key, value in self.coeff_dict.items()
            }
            return tensorProduct(new_coeff_dict)
        if _is_vs_element(other):
            return other._convert_to_tp()._contraction_product(self)
        else:
            raise ValueError(
                f"Unsupported operation for * between the given object types: {type(self)} and {type(other)}; `other` dgcv type is {get_dgcv_category(other)}."
            )
