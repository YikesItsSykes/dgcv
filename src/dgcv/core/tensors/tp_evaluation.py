from ..._aux._backends._symbolic_router import _scalar_is_zero
from ..._aux._vmf._safeguards import get_dgcv_category
from ..dgcv_core.spaces.spaces import _vs_card, card_root
from ._protocol import _is_scalar, _is_vs_element


class _tp_evaluation:
    def dual_pairing(self, other):
        if _is_vs_element(other):
            other = other._convert_to_tp()
        if get_dgcv_category(other) != "tensorProduct":
            raise TypeError(f"cannot apply dual_pairing to type {type(other)}")
        terms1, terms2 = self.terms, other.terms
        result = 0
        for t1 in terms1:
            for t2 in terms2:
                cd1, cd2 = next(iter(t1.coeff_dict)), next(iter(t2.coeff_dict))
                if len(cd1) != len(cd2):
                    continue
                if all(
                    f1[0] == f2[0] and f1[1] + f2[1] == 1 and f1[2] == f2[2]
                    for f1, f2 in zip(cd1, cd2)
                ):
                    result += t1.coeff_dict[cd1] * t2.coeff_dict[cd2]
        return result

    def __call__(
        self, *args, contract_from_left=False, demote_to_VS_when_possible=True
    ):
        if len(args) > self.min_degree:
            return "UNDEF"
        if len(args) == 0:
            return self
        if len(self.terms) > 1:
            return sum(
                term(
                    *args,
                    contract_from_left=contract_from_left,
                    demote_to_VS_when_possible=demote_to_VS_when_possible,
                )
                for term in self.terms
            )
        if len(args) > 1:
            partial = self.__call__(
                args[0],
                contract_from_left=contract_from_left,
                demote_to_VS_when_possible=demote_to_VS_when_possible,
            )
            if _is_scalar(partial):
                if _scalar_is_zero(partial):
                    return 0
                raise TypeError("tensorProduct.__call__() received more arguments than the tensor has slots")
            return partial.__call__(
                *args[1:],
                contract_from_left=contract_from_left,
                demote_to_VS_when_possible=demote_to_VS_when_possible,
            )
        parents = [root for root in self._vs_spring if root not in self.vs_id]
        for arg in args:
            if get_dgcv_category(arg) == "tensorProduct":
                tidx = arg.vs_id
            else:
                tidx = [getattr(arg, "card", None)]
            for t in tidx:
                if t not in self.vs_id and card_root(t) in parents:
                    return tensorProduct(self.coeff_dict, _amb_prom=True).__call__(
                        *args,
                        contract_from_left=contract_from_left,
                        demote_to_VS_when_possible=demote_to_VS_when_possible,
                    )
        k, v = next(iter(self.coeff_dict.items()))
        deg = len(k)
        factor = 1
        elem = args[0]
        if contract_from_left is True:
            idx = 0
        else:
            idx = deg - 1
        target = k[idx]
        if get_dgcv_category(elem) == "tensorProduct":
            if elem.max_degree == 1 and elem.min_degree == 1:
                if any(
                    (card_root(c) is not c and card_root(c) in self.vs_id)
                    for c in elem.vs_id
                ):
                    elem = tensorProduct(elem.coeff_dict, _amb_prom=True)
                accu = 0
                for kappa, nu in elem.coeff_dict.items():
                    if kappa[0][2] == target[2]:
                        if kappa[0][1] != target[1] and kappa[0][0] == target[0]:
                            accu += nu
                factor = factor * accu
            else:
                raise TypeError(
                    "tensorProduct.__call__() can only operatate on lists of vector space like elements, but it was applied to a tensorProduct element of degree≠1."
                )
        elif target[1] == getattr(elem, "valence", target[1]):
            factor = 0
        elif type(getattr(elem, "card", None)) is _vs_card:
            if target[2] == elem.card.root and elem.card.root is not elem.card:
                elem = elem.ambient_rep
            if target[2] == elem.card:
                coeffs = getattr(elem, "coeffs", [])
                if isinstance(coeffs, (list, tuple)) and len(coeffs) > target[0]:
                    factor = factor * coeffs[target[0]]
            else:
                factor = 0
        else:
            factor = 0
        if _scalar_is_zero(factor):
            return 0
        if contract_from_left is True:
            nk = k[1:]
        else:
            nk = k[:-1]
        nv = v * factor
        if demote_to_VS_when_possible is True and len(nk) == 1:
            ne = nv * (nk[0][2].space.basis[nk[0][0]])
            if nk[0][1] == 0:
                ne = ne.dual()
            return ne
        else:
            return tensorProduct({nk: nv})
