from ..._aux._vmf._safeguards import get_dgcv_category
from ._protocol import _is_vs_element


class _tp_contraction:
    def contract_call(self, other):
        """
        Contract the last index of self with the first index of other or handle algebra_element.
        """
        if self.is_zero:
            return self
        if other.is_zero:
            return 0 * self
        if isinstance(other, tensorProduct):
            if len(self.vs_id) != 1 or self.vs_id != other.vs_id:
                raise ValueError(
                    "Both tensors must be defined w.r.t. the same single vector space."
                )  ### generalize

            if self.trailing_valence + other.leading_valence != 1:
                raise ValueError(
                    "Contraction requires the first tensor factor of every term in other to have leading valence different from the last entry tensor factor from terms of self."
                )

            new_dict = {}
            for key1, value1 in self.coeff_dict.items():
                for key2, value2 in other.coeff_dict.items():
                    if key1[-1][0] == key2[0][0]:
                        new_key = key1[:-1] + key2[1:]
                        new_value = value1 * value2
                        new_dict[new_key] = new_dict.get(new_key, 0) + new_value

            return tensorProduct(new_dict)

        if (
            get_dgcv_category(other) == "subalgebra_element"
            and other.vectorSpace != self.vector_space
        ):
            other = other.ambient_rep

        elif _is_vs_element(other) and self.vector_space == other.vectorSpace:
            if self.trailing_valence != 0:
                raise ValueError(
                    f"Operating on algebra_element requires all terms in self to end in covariant tensor factor. Recieved self: {self} and other: {other}"
                )
            other_as_tensor = other._convert_to_tp()
            return self.contract_call(other_as_tensor)
        else:
            raise ValueError(
                "The other object must be a tensorProduct or an algebra_element with matching algebra."
            )

    def _recursion_contract_hom(self, other):
        if self.is_zero:
            return self
        if self.leading_valence == 0:
            vs = tuple([j.dual() for j in self.vector_space.basis])
            vsDual = self.vector_space.basis
        elif self.leading_valence == 1:
            vs = self.vector_space.basis
            vsDual = tuple([j.dual() for j in self.vector_space.basis])
        else:
            raise TypeError(
                f"`_recursion_contract_hom` does not operate on arguments with mixed `leading_valence` e.g., {self} has type {self.leading_valence}"
            )
        if self.max_degree == 1 or other.max_degree == 1:
            return self * other
        otherContract = other * vs[0]
        if hasattr(otherContract, "_convert_to_tp"):
            otherContract = otherContract._convert_to_tp()
        image_part = (self * vs[0])._recursion_contract_hom(
            other
        ) + self._recursion_contract_hom(otherContract)
        domain_part = vsDual[0]
        contraction = image_part @ domain_part
        for vec, vecD in zip(vs[1:], vsDual[1:]):
            otherContract = other * vec
            if hasattr(otherContract, "_convert_to_tp"):
                otherContract = otherContract._convert_to_tp()
            contraction += (
                (self * vec)._recursion_contract_hom(other)
                + self._recursion_contract_hom(otherContract)
            ) @ vecD
        return contraction

    def _recursion_contract(self, other):
        if self.is_zero:
            return self
        if other.is_zero:
            return 0 * self
        if isinstance(other, tensorProduct):
            if self.leading_valence != other.leading_valence:
                raise TypeError(
                    f"`tensorProduct` contraction is only supported between instances with matching `leading_valence`, not types: {self.leading_valence} and {other.leading_valence}"
                )
            if self.vs_id != other.vs_id:
                return 0 * self  ### critical logic
            hc1 = self.homogeneous_components
            hc2 = other.homogeneous_components
            terms = [t1._recursion_contract_hom(t2) for t1, t2 in zip(hc1, hc2)]
            return sum(terms[1:], terms[0])
