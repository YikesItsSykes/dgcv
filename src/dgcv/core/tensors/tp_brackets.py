from ..._aux._vmf._safeguards import get_dgcv_category
from ..combinatorics.combinatorics import shufflings
from ._protocol import _is_vs_element, _vs_bracket


class _tp_brackets:
    def _bracket(self, other):
        if self.is_zero:
            return self
        if other.is_zero:
            return 0 * self
        if (
            get_dgcv_category(other) == "subalgebra_element"
            and other.vectorSpace != self.vector_space
        ):
            other = other.ambient_rep
        if isinstance(other, tensorProduct):
            if len(self.vs_id) != 1 or self.vs_id != other.vs_id:
                raise ValueError(
                    "In `tensorProduct._bracket` both tensors must be defined w.r.t. the same vector_space."
                )

            if (
                self.leading_valence != other.leading_valence
                or self.leading_valence == -1
            ):
                raise ValueError(
                    "`tensorProduct._bracket` requires bracket components to have matching leading valences."
                )

            complimentType = 1 if self.leading_valence == 0 else 0
            card = self.vs_id[0]

            new_dict = {}
            for key1, value1 in self.coeff_dict.items():
                degree1 = len(key1)
                for key2, value2 in other.coeff_dict.items():
                    degree2 = len(key2)
                    for idx in range(1, degree1 - 1):  # double check degree-1
                        if (
                            key1[idx][0] == key2[0][0] and degree2 > 1
                        ):  # Check index matching before contraction
                            new_value = value1 * value2
                            k1_start = key1[:idx]
                            k2_start = key2[1:2]
                            k1_tail_inputs = key1[idx + 1 :]
                            k2_inputs = key2[2:]
                            new_tails = shufflings(k1_tail_inputs, k2_inputs)
                            valence = (self.leading_valence,) + (complimentType,) * (
                                degree1 + degree2 - 3
                            )
                            new_keys = [
                                tuple(
                                    (f[0], w, card)
                                    for f, w in zip(
                                        k1_start + k2_start + tuple(tail), valence
                                    )
                                )
                                for tail in new_tails
                            ]
                            for key in new_keys:
                                new_dict[key] = (
                                    new_dict.get(key, 0) + new_value
                                )  # Accumulate values for duplicate keys
                    if (
                        key1[degree1 - 1][0] == key2[0][0]
                    ):  # Check index matching before contraction
                        new_value = value1 * value2
                        k1_start = key1[: degree1 - 1]
                        k2_inputs = key2[1:]
                        valence = (self.leading_valence,) + (complimentType,) * (
                            degree1 + degree2 - 3
                        )
                        new_key = tuple(
                            (f[0], w, card)
                            for f, w in zip(k1_start + k2_inputs, valence)
                        )
                        new_dict[new_key] = new_dict.get(new_key, 0) + new_value

                    for idx in range(1, degree2 - 1):
                        if (
                            key2[idx][0] == key1[0][0] and degree1 > 1
                        ):  # Check index matching before contraction
                            new_value = -value1 * value2
                            k2_start = key2[:idx]
                            k1_start = key1[1:2]
                            k2_tail_inputs = key2[idx + 1 :]
                            k1_inputs = key1[2:]
                            new_tails = shufflings(k2_tail_inputs, k1_inputs)
                            valence = (self.leading_valence,) + (complimentType,) * (
                                degree1 + degree2 - 3
                            )
                            new_keys = [
                                tuple(
                                    (f[0], w, card)
                                    for f, w in zip(
                                        k2_start + k1_start + tuple(tail), valence
                                    )
                                )
                                for tail in new_tails
                            ]
                            for key in new_keys:
                                new_dict[key] = (
                                    new_dict.get(key, 0) + new_value
                                )  # Accumulate values for duplicate keys
                    if (
                        key2[degree2 - 1][0] == key1[0][0]
                    ):  # Check index matching before contraction
                        new_value = -value1 * value2
                        k2_start = key2[: degree2 - 1]
                        k1_inputs = key1[1:]
                        valence = (self.leading_valence,) + (complimentType,) * (
                            degree2 + degree1 - 3
                        )
                        new_key = tuple(
                            (f[0], w, card)
                            for f, w in zip(k2_start + k1_inputs, valence)
                        )
                        new_dict[new_key] = new_dict.get(new_key, 0) + new_value
            return tensorProduct(new_dict)

        elif _is_vs_element(other) and self.vector_space == other.vectorSpace:
            if self.vector_space != other.vectorSpace:
                raise ValueError(
                    "In `tensorProduct._bracket` both tensors must be defined w.r.t. the same single vector_space."
                )
            if self.leading_valence != other.valence:
                raise ValueError(
                    "`tensorProduct._bracket` operating on algebra_element requires all terms in self to end in covariant tensor factor."
                )
            other_as_tensor = other._convert_to_tp()
            other_index, other_value = list(other_as_tensor.coeff_dict.items())[
                0
            ]  ###!!! review
            new_dict = {}
            for key, value in self.coeff_dict.items():
                if key[-1] == other_index[0]:  # Matching indices for contraction
                    new_value = value * other_value
                    key_truncated = key[:-1]
                    new_dict[key_truncated] = new_dict.get(key_truncated, 0) + new_value
            return tensorProduct(new_dict)
        else:
            raise ValueError(
                "In `tensorProduct._bracket` the second factor must be a tensorProduct or an algebra_element with matching algebra."
            )

    def _bracket_gen(self, other):
        if self.is_zero:
            return self
        if other.is_zero:
            return 0 * self
        if get_dgcv_category(other) == "subalgebra_element":
            other = other.ambient_rep
        if isinstance(other, tensorProduct):
            new_dict = {}
            for key1, value1 in self.coeff_dict.items():
                if len(key1) == 0:
                    continue
                degree1 = len(key1)
                for key2, value2 in other.coeff_dict.items():
                    if len(key2) == 0:
                        continue
                    degree2 = len(key2)
                    for idx in range(1, degree1 - 1):
                        if (
                            key1[idx][0] == key2[0][0]
                            and key1[idx][1] != key2[0][1]
                            and key1[idx][2] != key2[0][2]
                            and degree2 > 1
                        ):
                            new_value = value1 * value2
                            k1_start = key1[:idx]
                            k2_start = key2[1:2]
                            k1_tail_inputs = key1[idx + 1 :]
                            k2_inputs = key2[2:]
                            new_tails = shufflings(k1_tail_inputs, k2_inputs)
                            new_keys = [
                                k1_start + k2_start + tuple(tail) for tail in new_tails
                            ]
                            for key in new_keys:
                                new_dict[key] = new_dict.get(key, 0) + new_value
                    if degree1 == 1 and degree2 == 1 and key1[0][1:] == key2[0][1:]:
                        algB = key1[-1][2].space.basis
                        if key1[0][1] == 0:
                            elem1, elem2 = (
                                algB[key1[0][0]].dual(),
                                algB[key2[0][0]].dual(),
                            )
                        else:
                            elem1, elem2 = (
                                algB[key1[0][0]],
                                algB[key2[0][0]],
                            )
                        newElem = _vs_bracket(elem1, elem2)
                        for k, v in newElem._convert_to_tp().coeff_dict.items():
                            new_dict[k] = new_dict.get(k, 0) + v
                    else:
                        if (
                            key1[-1][0] == key2[0][0]
                            and key1[-1][1] != key2[0][1]
                            and key1[-1][2] == key2[0][2]
                        ):
                            new_value = value1 * value2
                            k1_start = key1[: degree1 - 1]
                            k2_inputs = key2[1:]
                            new_key = k1_start + k2_inputs
                            new_dict[new_key] = new_dict.get(new_key, 0) + new_value
                        if (
                            key2[-1][0] == key1[0][0]
                            and key2[-1][1] != key1[0][1]
                            and key2[-1][2] == key1[0][2]
                        ):
                            new_value = -value2 * value1
                            k2_start = key2[: degree2 - 1]
                            k1_inputs = key1[1:]
                            new_key = k2_start + k1_inputs
                            new_dict[new_key] = new_dict.get(new_key, 0) + new_value

                    for idx in range(1, degree2 - 1):
                        if (
                            key2[idx][0] == key1[0][0]
                            and key2[idx][1] != key1[0][1]
                            and key2[idx][2] != key1[0][2]
                            and degree1 > 1
                        ):
                            new_value = -value2 * value1
                            k2_start = key2[:idx]
                            k1_start = key1[1:2]
                            k2_tail_inputs = key2[idx + 1 :]
                            k1_inputs = key1[2:]
                            new_tails = shufflings(k2_tail_inputs, k1_inputs)
                            new_keys = [
                                k2_start + k1_start + tuple(tail) for tail in new_tails
                            ]
                            for key in new_keys:
                                new_dict[key] = new_dict.get(key, 0) + new_value
            return tensorProduct(new_dict)

        elif _is_vs_element(other):
            other_as_tensor = other._convert_to_tp()
            new_dict = {}
            for other_index, other_value in other_as_tensor.coeff_dict.items():
                for key, value in self.coeff_dict.items():
                    if key[-1] == other_index[0]:
                        new_value = value * other_value
                        key_truncated = key[:-1]
                        new_dict[key_truncated] = (
                            new_dict.get(key_truncated, 0) + new_value
                        )
                    if key[0] == other_index[0]:
                        new_value = -value * other_value
                        key_truncated = key[1:]
                        new_dict[key_truncated] = (
                            new_dict.get(key_truncated, 0) + new_value
                        )

            return tensorProduct(new_dict)
        else:
            raise ValueError(
                "In `tensorProduct._bracket` the second factor must be a tensorProduct or an algebra_element with matching algebra."
            )
