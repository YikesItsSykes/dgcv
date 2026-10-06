from __future__ import annotations

from ..._aux._backends._types_and_constants import (
    imag_unit,
    rational,
)
from ..._aux._vmf._safeguards import retrieve_passkey
from ...core.arrays import array_dgcv, freeze_matrix, matrix_dgcv
from ...core.combinatorics.combinatorics import carProd
from ..creators import createAlgebra
from ._exceptional import _chevalley_structure_data, _exceptional_cartan_matrix


def createSimpleLieAlgebra(
    series: str,
    label: str = None,
    basis_labels: list = None,
    build_standard_mat_rep=False,
    return_created_object=False,
    forgo_vmf_registry=False,
    **kwargs,
):
    """
    Creates a simple complex Lie algebra (or nonsimples for D1 & D2) specified from the classical
    series
        - A_n = sl(n+1)     for n>0
        - B_n = so(2n+1)    for n>0
        - C_n = sp(2n)      for n>0
        - D_n = so(2n)      for n>0
    or one of the exceptional types G2, F4, E6, E7, E8.


    Parameters
    ----------
    series : str
        The type and rank of the Lie algebra, e.g., "A1", "A2", ..., "Dn", "G2", "F4", "E6", "E7", "E8".
    label : str, optional
        Label for the Lie algebra. If not provided, defaults to a standard notation,
        like sl3 for A2, or g2, f4, e6, e7, e8 for the exceptional types.
    basis_labels : list, optional
        Labels for the basis elements. If not provided, default labels will be generated.

    Returns
    -------
    simple_Lie_algebra
        The resulting Lie algebra as an simple_Lie_algebra instance

    Notes
    -----
    - The classical series are built as matrix algebras following
      S. Helgason, Differential Geometry, Lie Groups, and Symmetric Spaces (Academic Press, 1978).
    - The exceptional algebras are built in a Chevalley basis from their Cartan matrices: the
      structure constants are N_{a,b} = +-(p+1) (J. E. Humphreys, Introduction to Lie Algebras and
      Representation Theory, Springer, 1972, Section 25), with signs fixed on extraspecial pairs and
      propagated by the identities in R. W. Carter, Simple Groups of Lie Type (Wiley, 1972), Chapter 4.
      Simple roots are numbered as in the plates of N. Bourbaki, Lie Groups and Lie Algebras,
      Chapters 4-6. The basis lists the Cartan subalgebra (simple coroots), then root vectors for
      the positive roots by height, then for their negatives; the grading records each basis
      element's coefficients on the simple roots. `build_standard_mat_rep` does not apply to them.
    """
    try:
        series_type, rank = series[0], int(series[1:])
        series_type = "".join(c.upper() if c.islower() else c for c in series_type)
    except (IndexError, ValueError):
        raise ValueError(
            f"Invalid series format: {series}. Expected a letter 'A', 'B', 'C', 'D', 'E', 'F', or 'G' followed by a positive integer, like 'A1', 'B5', etc."
        ) from None
    if rank <= 0:
        raise ValueError(
            f"Sequence index must be a positive integer, but got: {rank}."
        ) from None
    if kwargs.get("return_created_obj"):  # old keyword support
        return_created_object = kwargs.get("return_created_object")

    def _generate_A_series_structure_data(n):
        matrix_dim = n + 1

        # Basis elements
        hBasis = {"elems": dict(), "grading": dict()}
        offDiag = {"elems": dict(), "grading": dict()}

        def elemWeights(idx1, idx2):
            wVec = []
            for idx in range(n):
                if idx1 <= idx:
                    if idx2 <= idx:
                        wVec.append(0)
                    else:
                        wVec.append(1)
                else:
                    if idx2 <= idx:
                        wVec.append(-1)
                    else:
                        wVec.append(0)
            return wVec

        for j in range(n + 1):
            for k in range(j, n + 1):
                # Diagonal (Cartan) element
                if j == k and j < n:
                    M = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                    for idx in range(n + 1):
                        if idx > j:
                            M[idx, idx] = -rational(j + 1, n + 1)
                        else:
                            M[idx, idx] = 1 - rational(j + 1, n + 1)
                    hBasis["elems"][(j, k, 0)] = M
                    hBasis["grading"][(j, k, 0)] = [0] * n
                elif j != k:
                    # off diagonal generators
                    MPlus = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                    MMinus = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                    MPlus[j, k] = 1
                    MMinus[k, j] = 1
                    offDiag["elems"][(j, k, 1)] = MPlus
                    offDiag["grading"][(j, k, 1)] = elemWeights(j, k)
                    offDiag["elems"][(k, j, 1)] = MMinus
                    offDiag["grading"][(k, j, 1)] = elemWeights(k, j)

        indexingKey = dict(
            enumerate(list(hBasis["grading"].keys()) + list(offDiag["grading"].keys()))
        )
        indexingKeyRev = {j: k for k, j in indexingKey.items()}
        LADimension = len(indexingKey)

        def _structureCoeffs(idx1, idx2):
            coeffs = matrix_dgcv({}, shape=(LADimension, 1))
            if idx2 == idx1:
                return coeffs
            if idx2 < idx1:
                reSign = -1
                idx2, idx1 = idx1, idx2
            else:
                reSign = 1
            p10, p11, p12 = indexingKey[idx1]  # (first index, second index, type index)
            p20, p21, p22 = indexingKey[idx2]
            if p12 == 0:  # implies p10 = p11
                if p22 == 1:
                    if p20 <= p10 and p21 > p10:
                        coeffs[idx2] = reSign
                    elif p21 <= p10 and p20 > p10:
                        coeffs[idx2] = -reSign
            elif p12 == 1:
                if p22 == 1:
                    if p11 == p20:
                        if p10 == p21:
                            if p10 < p11:
                                if 0 < p10:
                                    coeffs[
                                        indexingKeyRev[(p10 - 1, p10 - 1, 0)]
                                    ] = -reSign
                                if p10 == p11 - 1:
                                    coeffs[indexingKeyRev[(p10, p10, 0)]] = 2 * reSign
                                else:
                                    coeffs[indexingKeyRev[(p10, p10, 0)]] = reSign
                                    coeffs[indexingKeyRev[(p11 - 1, p11 - 1, 0)]] = (
                                        reSign
                                    )
                                if p11 < n:
                                    coeffs[indexingKeyRev[(p11, p11, 0)]] = -reSign
                            else:
                                if 0 < p11:
                                    coeffs[indexingKeyRev[(p11 - 1, p11 - 1, 0)]] = (
                                        reSign
                                    )
                                if p11 == p10 - 1:
                                    coeffs[indexingKeyRev[(p11, p11, 0)]] = -2 * reSign
                                else:
                                    coeffs[indexingKeyRev[(p11, p11, 0)]] = -reSign
                                    coeffs[
                                        indexingKeyRev[(p10 - 1, p10 - 1, 0)]
                                    ] = -reSign
                                if p10 < n:
                                    coeffs[indexingKeyRev[(p10, p10, 0)]] = reSign
                        else:
                            coeffs[indexingKeyRev[(p10, p21, 1)]] = reSign
                    elif p10 == p21:
                        coeffs[indexingKeyRev[(p20, p11, 1)]] = -reSign
            return coeffs

        _structure_data = array_dgcv(
            dict(),
            shape=(LADimension, LADimension),
            null_return=freeze_matrix(matrix_dgcv.zeros(LADimension, 1)),
        )
        for j in range(LADimension):
            for k in range(j + 1, LADimension):
                scoeffs = _structureCoeffs(j, k)
                if len(scoeffs._data) > 0:
                    _structure_data[(j, k)] = scoeffs
                    _structure_data[(k, j)] = -scoeffs

        CartanSubalg = list(hBasis["elems"].values())
        matrixBasis = CartanSubalg + list(offDiag["elems"].values())
        gradingVecs = list(hBasis["grading"].values()) + list(
            offDiag["grading"].values()
        )
        return _structure_data, list(zip(*gradingVecs)), CartanSubalg, matrixBasis

    def _generate_B_series_structure_data(n):
        matrix_dim = 2 * n + 1

        # Basis elements
        hBasis = {"elems": dict(), "grading": dict()}
        GPlus = {"elems": dict(), "grading": dict()}
        GMinus = {"elems": dict(), "grading": dict()}
        DPlus = {"elems": dict(), "grading": dict()}
        DMinus = {"elems": dict(), "grading": dict()}

        def gPlusWeights(idx1, idx2):
            wVec = []
            for idx in range(n - 1):
                if (idx1 <= idx and idx2 <= idx) or (idx1 > idx and idx2 > idx):
                    wVec.append(0)
                elif idx1 <= idx:
                    wVec.append(1)
                else:
                    wVec.append(-1)
            wVec.append(0)
            return wVec

        def gMinusWeights(idx1, idx2):
            wVec = []
            sign = 1 if idx2 < idx1 else -1
            for idx in range(n - 1):
                if idx1 <= idx and idx2 <= idx:
                    if idx == 0:
                        wVec.append(sign)
                    else:
                        wVec.append(2 * sign)
                elif idx1 > idx and idx2 > idx:
                    wVec.append(0)
                elif idx1 <= idx:
                    wVec.append(-1)
                elif idx2 <= idx:
                    wVec.append(1)
                else:  # should never trigger
                    wVec.append(0)
            wVec.append(2 * sign)
            return wVec

        def DWeights(idx1, sign):
            wVec = []
            for idx in range(n - 1):
                if idx1 <= idx:
                    wVec.append(-sign)
                else:
                    wVec.append(0)
            wVec.append(-sign)
            return wVec

        for j, k in carProd(range(n), range(n)):
            # Diagonal (Cartan) element
            if j == k and j < n - 1:
                M = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                for idx in range(n):
                    if idx < j + 1:
                        M[2 * idx, 2 * idx + 1] = imag_unit()
                        M[2 * idx + 1, 2 * idx] = -imag_unit()
                hBasis["elems"][(j, k, 0)] = M
                hBasis["grading"][(j, k, 0)] = [0] * n
                if j + 2 == n:
                    M = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                    for idx in range(n):
                        M[2 * idx, 2 * idx + 1] = imag_unit()
                        M[2 * idx + 1, 2 * idx] = -imag_unit()
                    hBasis["elems"][(j + 1, k + 1, 0)] = M
                    hBasis["grading"][(j + 1, k + 1, 0)] = [0] * n
            elif j != k:
                MPlus = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                MPlus[2 * j, 2 * k] = 1
                MPlus[2 * k, 2 * j] = -1
                MPlus[2 * j + 1, 2 * k + 1] = 1
                MPlus[2 * k + 1, 2 * j + 1] = -1
                MPlus[2 * j, 2 * k + 1] = imag_unit()
                MPlus[2 * k + 1, 2 * j] = -imag_unit()
                MPlus[2 * j + 1, 2 * k] = -imag_unit()
                MPlus[2 * k, 2 * j + 1] = imag_unit()
                GPlus["elems"][(j, k, 1)] = MPlus
                GPlus["grading"][(j, k, 1)] = gPlusWeights(j, k)

                if j < k:
                    MMinus = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                    MMinus[2 * j, 2 * k] = 1
                    MMinus[2 * k, 2 * j] = -1
                    MMinus[2 * j + 1, 2 * k + 1] = -1
                    MMinus[2 * k + 1, 2 * j + 1] = 1
                    MMinus[2 * j, 2 * k + 1] = imag_unit()
                    MMinus[2 * k + 1, 2 * j] = -imag_unit()
                    MMinus[2 * j + 1, 2 * k] = imag_unit()
                    MMinus[2 * k, 2 * j + 1] = -imag_unit()
                    GMinus["elems"][(j, k, -1)] = MMinus
                    GMinus["grading"][(j, k, -1)] = gMinusWeights(j, k)
                else:  # k<j
                    MMinus = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                    MMinus[2 * k, 2 * j] = 1
                    MMinus[2 * j, 2 * k] = -1
                    MMinus[2 * k + 1, 2 * j + 1] = -1
                    MMinus[2 * j + 1, 2 * k + 1] = 1
                    MMinus[2 * k, 2 * j + 1] = -imag_unit()
                    MMinus[2 * j + 1, 2 * k] = imag_unit()
                    MMinus[2 * k + 1, 2 * j] = -imag_unit()
                    MMinus[2 * j, 2 * k + 1] = imag_unit()
                    GMinus["elems"][(j, k, -1)] = MMinus
                    GMinus["grading"][(j, k, -1)] = gMinusWeights(j, k)
        for j in range(n):
            MPlus = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
            MMinus = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
            MPlus[2 * j, 2 * n] = 1
            MPlus[2 * n, 2 * j] = -1
            MPlus[2 * j + 1, 2 * n] = imag_unit()
            MPlus[2 * n, 2 * j + 1] = -imag_unit()
            MMinus[2 * j, 2 * n] = 1
            MMinus[2 * n, 2 * j] = -1
            MMinus[2 * j + 1, 2 * n] = -imag_unit()
            MMinus[2 * n, 2 * j + 1] = imag_unit()
            DPlus["elems"][(j, 2 * n, 2)] = MPlus
            DPlus["grading"][(j, 2 * n, 2)] = DWeights(j, 1)
            DMinus["elems"][(j, 2 * n, -2)] = MMinus
            DMinus["grading"][(j, 2 * n, -2)] = DWeights(j, -1)

        indexingKey = dict(
            enumerate(
                list(hBasis["grading"].keys())
                + list(GPlus["grading"].keys())
                + list(GMinus["grading"].keys())
                + list(DPlus["grading"].keys())
                + list(DMinus["grading"].keys())
            )
        )
        indexingKeyRev = {j: k for k, j in indexingKey.items()}
        LADimension = len(indexingKey)
        CSDict = {
            idx: {0: 1} if idx == 0 else {idx: 1, idx - 1: -1} for idx in range(n)
        }  # Cartan subalgebra basis transform indexing
        CSDictInv = {
            idx: {j: 0 if j > idx else 1 for j in range(n)} for idx in range(n - 1)
        } | {n - 1: {j: 1 for j in range(n)}}

        def _structureCoeffs(idx1, idx2):
            coeffs = matrix_dgcv({}, shape=(LADimension, 1))
            if idx2 == idx1:
                return coeffs
            if idx2 < idx1:
                reSign = -1
                idx2, idx1 = idx1, idx2
            else:
                reSign = 1
            p10, p11, p12 = indexingKey[idx1]
            p20, p21, p22 = indexingKey[idx2]
            if p12 == 0:
                for term, scale in CSDictInv[p10].items():
                    if p22 == 1:
                        coeffs[idx2] += (
                            scale * reSign * (int(term == p20) - int(term == p21))
                        )
                    elif p22 == -1:
                        sign = -reSign if p20 < p21 else reSign
                        coeffs[idx2] += (
                            scale * sign * (int(term == p20) + int(term == p21))
                        )
                    elif p22 == 2:
                        coeffs[idx2] += -scale * (int(term == p20)) * reSign
                    elif p22 == -2:
                        coeffs[idx2] += scale * (int(term == p20)) * reSign
            elif p12 == 1:
                if p22 == 1:
                    if p11 == p20:
                        if p10 == p21:
                            # l(p10)-l(p11)
                            for t, s in CSDict[p10].items():
                                coeffs[t] += reSign * 4 * s
                            for t, s in CSDict[p11].items():
                                coeffs[t] += -reSign * 4 * s
                        else:
                            coeffs[indexingKeyRev[(p10, p21, 1)]] += 2 * reSign
                    elif p10 == p21:
                        coeffs[indexingKeyRev[(p20, p11, 1)]] += -2 * reSign
                elif p22 == -1:
                    slope1 = 1 if p10 < p11 else -1
                    slope2 = 1 if p20 < p21 else -1
                    if p10 == p20:
                        if not (slope1 == -1 and slope2 == -1):
                            if p11 < p21:
                                coeffs[indexingKeyRev[(p11, p21, -1)]] += -2 * reSign
                            elif p21 < p11:
                                if not (slope1 == 1 and slope2 == -1):
                                    coeffs[indexingKeyRev[(p21, p11, -1)]] += 2 * reSign
                    elif p11 == p21:
                        if not (slope1 == 1 and slope2 == 1):
                            if p10 < p20:
                                coeffs[indexingKeyRev[(p20, p10, -1)]] += 2 * reSign
                            elif p20 < p10:
                                if not (slope1 == -1 and slope2 == 1):
                                    coeffs[indexingKeyRev[(p10, p20, -1)]] += (
                                        -2 * reSign
                                    )
                    elif p11 == p20:
                        if not (slope1 == 1 and slope2 == 1) and not (
                            slope1 == -1 and slope2 == 1
                        ):
                            if p10 < p21:
                                coeffs[indexingKeyRev[(p21, p10, -1)]] = -2 * reSign
                            elif p21 < p10:
                                coeffs[indexingKeyRev[(p10, p21, -1)]] = 2 * reSign
                    elif p10 == p21:
                        if not (slope1 == -1 and slope2 == -1) and not (
                            slope1 == 1 and slope2 == -1
                        ):
                            if p11 < p20:
                                coeffs[indexingKeyRev[(p11, p20, -1)]] = 2 * reSign
                            elif p20 < p11:
                                coeffs[indexingKeyRev[(p20, p11, -1)]] = -2 * reSign
                elif p22 == 2:
                    if p10 == p20:
                        coeffs[indexingKeyRev[(p11, p21, 2)]] = -2 * reSign
                elif p22 == -2:
                    if p11 == p20:
                        coeffs[indexingKeyRev[(p10, p21, -2)]] = 2 * reSign
            elif p12 == -1:
                slope1 = 1 if p10 < p11 else -1
                slope2 = 1 if p20 < p21 else -1
                if p22 == -1:
                    sign2 = 1 if p10 < p11 else -1
                    if (p10 < p11 and p20 < p21) or (p10 > p11 and p20 > p21):
                        pass
                    elif p11 == p20:
                        if p10 == p21:
                            # plus/minus (l(p10)+l(p11))
                            for t, s in CSDict[p10].items():
                                coeffs[t] += sign2 * reSign * 4 * s
                            for t, s in CSDict[p11].items():
                                coeffs[t] += sign2 * reSign * 4 * s
                        else:
                            if sign2 == 1:
                                coeffs[indexingKeyRev[(p21, p10, 1)]] += (
                                    2 * reSign * sign2
                                )
                            else:
                                coeffs[indexingKeyRev[(p10, p21, 1)]] += (
                                    2 * reSign * sign2
                                )
                    elif p10 == p21:
                        if sign2 == 1:
                            coeffs[indexingKeyRev[(p20, p11, 1)]] += 2 * reSign * sign2
                        else:
                            coeffs[indexingKeyRev[(p11, p20, 1)]] += 2 * reSign * sign2
                    elif p10 == p20 and p21 != p11:
                        if sign2 == 1:
                            coeffs[indexingKeyRev[(p21, p11, 1)]] += -2 * reSign * sign2
                        else:
                            coeffs[indexingKeyRev[(p11, p21, 1)]] += -2 * reSign * sign2
                    elif p11 == p21 and p10 != p20:
                        if sign2 == 1:
                            coeffs[indexingKeyRev[(p20, p10, 1)]] += -2 * reSign * sign2
                        else:
                            coeffs[indexingKeyRev[(p10, p20, 1)]] += -2 * reSign * sign2
                elif p22 == 2:
                    if slope1 == -1:
                        if p11 == p20:
                            coeffs[indexingKeyRev[(p10, p21, -2)]] = -2 * reSign
                        elif p10 == p20:
                            coeffs[indexingKeyRev[(p11, p21, -2)]] = 2 * reSign
                elif p22 == -2:
                    if p10 == p20 and slope1 == 1:
                        coeffs[indexingKeyRev[(p11, p21, 2)]] = -2 * reSign
                    if p11 == p20:
                        if slope1 == 1:
                            coeffs[indexingKeyRev[(p10, p21, 2)]] = 2 * reSign
            elif p12 == 2:
                if p22 == 2:
                    if p10 < p20:
                        coeffs[indexingKeyRev[(p10, p20, -1)]] = -reSign
                    elif p20 < p10:
                        coeffs[indexingKeyRev[(p20, p10, -1)]] = reSign
                if p22 == -2:
                    if p10 == p20:
                        for term, scale in CSDict[p10].items():
                            coeffs[term] = 2 * scale * reSign
                    elif p10 < p20:
                        coeffs[indexingKeyRev[(p20, p10, 1)]] = reSign
                    else:
                        coeffs[indexingKeyRev[(p20, p10, 1)]] = reSign
            elif p12 == -2:
                if p22 == -2:
                    if p10 < p20:
                        coeffs[indexingKeyRev[(p20, p10, -1)]] = -reSign
                    elif p20 < p10:
                        coeffs[indexingKeyRev[(p10, p20, -1)]] = reSign
            return coeffs

        _structure_data = array_dgcv(
            dict(),
            shape=(LADimension, LADimension),
            null_return=freeze_matrix(matrix_dgcv.zeros(LADimension, 1)),
        )
        for j in range(LADimension):
            for k in range(j + 1, LADimension):
                scoeffs = _structureCoeffs(j, k)
                if len(scoeffs._data) > 0:
                    _structure_data[(j, k)] = scoeffs
                    _structure_data[(k, j)] = -scoeffs

        CartanSubalg = list(hBasis["elems"].values())
        matrixBasis = (
            CartanSubalg
            + list(GPlus["elems"].values())
            + list(GMinus["elems"].values())
            + list(DPlus["elems"].values())
            + list(DMinus["elems"].values())
        )
        gradingVecs = (
            list(hBasis["grading"].values())
            + list(GPlus["grading"].values())
            + list(GMinus["grading"].values())
            + list(DPlus["grading"].values())
            + list(DMinus["grading"].values())
        )
        return _structure_data, list(zip(*gradingVecs)), CartanSubalg, matrixBasis

    def _generate_C_series_structure_data(n):
        matrix_dim = 2 * n

        # Basis elements
        hBasis = {"elems": dict(), "grading": dict()}
        offDiag = {"elems": dict(), "grading": dict()}

        def gPlusWeights(idx1, idx2):
            wVec = []
            for idx in range(idx1):
                wVec.append(0)
            for idx in range(idx1, idx2):
                wVec.append(1)
            for idx in range(idx2, n - 1):
                wVec.append(2)
            wVec.append(1)
            return wVec

        def gMinusWeights(idx1, idx2):
            wVec = []
            for idx in range(idx1):
                wVec.append(0)
            for idx in range(idx1, idx2):
                wVec.append(-1)
            for idx in range(idx2, n - 1):
                wVec.append(-2)
            wVec.append(-1)
            return wVec

        def GLWeights(idx1, idx2):
            if idx1 < idx2:
                wVec = [1 if idx1 <= idx and idx < idx2 else 0 for idx in range(n)]
            else:
                wVec = [-1 if idx2 <= idx and idx < idx1 else 0 for idx in range(n)]
            return wVec

        for j in range(n):
            for k in range(j, n):
                if j == k:
                    M = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                    if j < n - 1:
                        for idx in range(j + 1):
                            M[idx, idx] = 1
                            M[n + idx, n + idx] = -1
                    else:
                        for idx in range(n):
                            M[idx, idx] = rational(1, 2)
                            M[n + idx, n + idx] = -rational(1, 2)
                    hBasis["elems"][(j, k, 0)] = M
                    hBasis["grading"][(j, k, 0)] = [0] * n

                    M = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                    M[j, n + j] = 1
                    offDiag["elems"][(j, k, 1)] = M
                    offDiag["grading"][(j, k, 1)] = gPlusWeights(j, k)

                    M = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                    M[n + j, j] = 1
                    offDiag["elems"][(j, k, -1)] = M
                    offDiag["grading"][(j, k, -1)] = gMinusWeights(j, k)
                else:
                    M = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                    M[j, k] = 1
                    M[n + k, n + j] = -1
                    offDiag["elems"][(j, k, 2)] = M
                    offDiag["grading"][(j, k, 2)] = GLWeights(j, k)

                    M = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                    M[k, j] = 1
                    M[n + j, n + k] = -1
                    offDiag["elems"][(k, j, 2)] = M
                    offDiag["grading"][(k, j, 2)] = GLWeights(k, j)

                    M = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                    M[j, n + k] = 1
                    M[k, n + j] = 1
                    offDiag["elems"][(j, k, 1)] = M
                    offDiag["grading"][(j, k, 1)] = gPlusWeights(j, k)

                    M = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                    M[n + j, k] = 1
                    M[n + k, j] = 1
                    offDiag["elems"][(j, k, -1)] = M
                    offDiag["grading"][(j, k, -1)] = gMinusWeights(j, k)

        indexingKey = dict(
            enumerate(list(hBasis["grading"].keys()) + list(offDiag["grading"].keys()))
        )
        indexingKeyRev = {j: k for k, j in indexingKey.items()}
        LADimension = len(indexingKey)
        CSDict = {
            idx: {0: 1}
            if idx == 0
            else {idx: 2, idx - 1: -1}
            if idx == n - 1
            else {idx: 1, idx - 1: -1}
            for idx in range(n)
        }  # Cartan subalgebra basis transform indexing

        def minmaxtuple(id1, id2, id3):
            if id1 < id2:
                return (id1, id2, id3)
            return (id2, id1, id3)

        def _structureCoeffs(idx1, idx2):
            coeffs = matrix_dgcv({}, shape=(LADimension, 1))
            if idx2 == idx1:
                return coeffs
            if idx2 < idx1:
                reSign = -1
                idx2, idx1 = idx1, idx2
            else:
                reSign = 1
            p10, p11, p12 = indexingKey[idx1]
            p20, p21, p22 = indexingKey[idx2]
            if p12 == 0:
                if p22 == 1:
                    coeffs[idx2] += reSign * offDiag["grading"][indexingKey[idx2]][p10]
                elif p22 == -1:
                    coeffs[idx2] += reSign * offDiag["grading"][indexingKey[idx2]][p10]
                elif p22 == 2:
                    coeffs[idx2] += reSign * offDiag["grading"][indexingKey[idx2]][p10]
            elif p12 == 1:
                if p22 == -1:
                    if p11 == p20:
                        if p10 != p21:
                            coeffs[indexingKeyRev[(p10, p21, 2)]] += reSign
                        else:
                            rk = indexingKeyRev[(p10, p21, 0)]
                            for term, scale in CSDict[rk].items():
                                coeffs[term] += scale * reSign
                    elif p11 == p21:
                        if p10 != p20:
                            coeffs[indexingKeyRev[(p10, p20, 2)]] += reSign
                        else:
                            for rk in [
                                indexingKeyRev[(p10, p20, 0)],
                                indexingKeyRev[(p11, p21, 0)],
                            ]:
                                for term, scale in CSDict[rk].items():
                                    coeffs[term] += scale * reSign

                    if p11 != p10:
                        if p10 == p20 and p11 != p21:
                            coeffs[indexingKeyRev[(p11, p21, 2)]] += reSign
                        elif p10 == p21 and p11 != p20:
                            coeffs[indexingKeyRev[(p11, p20, 2)]] += reSign
                elif p22 == 2:
                    if p11 == p21 and p10 == p20:  ###!!! check second condition
                        coeffs[indexingKeyRev[minmaxtuple(p10, p20, 1)]] += -reSign
                    if p10 == p21:
                        coeffs[indexingKeyRev[minmaxtuple(p11, p20, 1)]] += -reSign
                    if p11 != p10:
                        if p10 == p21 and p11 == p20:  ###!!! check second condition
                            coeffs[indexingKeyRev[minmaxtuple(p11, p20, 1)]] += -reSign
                        if p11 == p21:
                            coeffs[indexingKeyRev[minmaxtuple(p10, p20, 1)]] += -reSign

            elif p12 == -1:
                if p22 == 1:
                    if p11 == p20:
                        if p10 != p21:
                            coeffs[indexingKeyRev[(p21, p10, 2)]] += -reSign
                        else:
                            rk = indexingKeyRev[(p21, p10, 0)]
                            for term, scale in CSDict[rk].items():
                                coeffs[term] += -scale * reSign
                    elif p11 == p21:
                        if p10 != p20:
                            coeffs[indexingKeyRev[(p20, p10, 2)]] += -reSign
                        else:
                            rk = indexingKeyRev[(p20, p10, 0)]
                            for term, scale in CSDict[rk].items():
                                coeffs[term] += -scale * reSign
                    if p11 != p10:
                        if p10 == p20:
                            coeffs[
                                indexingKeyRev[(p21, p11, 2 * int(p11 != p21))]
                            ] += -reSign
                        elif p10 == p21:
                            coeffs[
                                indexingKeyRev[(p20, p11, 2 * int(p11 != p20))]
                            ] += -reSign
                elif p22 == 2:
                    if p11 == p20 and p10 == p21:  ###!!! check second condition
                        coeffs[indexingKeyRev[minmaxtuple(p10, p21, -1)]] += reSign
                    if p10 == p20:
                        coeffs[indexingKeyRev[minmaxtuple(p11, p21, -1)]] += reSign
                    if p11 != p10:
                        if p10 == p20 and p11 == p21:  ###!!! check second condition
                            coeffs[indexingKeyRev[minmaxtuple(p11, p21, -1)]] += reSign
                        if p11 == p20:
                            coeffs[indexingKeyRev[minmaxtuple(p10, p21, -1)]] += reSign

            elif p12 == 2:
                if p22 == 1:
                    if p21 == p11 and p20 == p10:  ###!!! check second condition
                        coeffs[indexingKeyRev[minmaxtuple(p20, p10, 1)]] += reSign
                    if p20 == p11:
                        coeffs[indexingKeyRev[minmaxtuple(p21, p10, 1)]] += reSign
                    if p20 != p21:
                        if p20 == p11 and p21 == p10:  ###!!! check second condition
                            coeffs[indexingKeyRev[minmaxtuple(p21, p10, 1)]] += reSign
                        if p21 == p11:
                            coeffs[indexingKeyRev[minmaxtuple(p20, p10, 1)]] += reSign
                elif p22 == -1:
                    if p21 == p10 and p20 == p11:  ###!!! check second condition
                        coeffs[indexingKeyRev[minmaxtuple(p20, p11, -1)]] += -reSign
                    if p20 == p10:
                        coeffs[indexingKeyRev[minmaxtuple(p21, p11, -1)]] += -reSign
                    if p20 != p21:
                        if p20 == p10 and p21 == p11:  ###!!! check second condition
                            coeffs[indexingKeyRev[minmaxtuple(p21, p11, -1)]] += -reSign
                        if p21 == p10:
                            coeffs[indexingKeyRev[minmaxtuple(p20, p11, -1)]] += -reSign

                elif p22 == 2:
                    if p11 == p20:
                        if p10 != p21:
                            coeffs[indexingKeyRev[(p10, p21, 2)]] += reSign
                        else:
                            rk = indexingKeyRev[(p10, p21, 0)]
                            for term, scale in CSDict[rk].items():
                                coeffs[term] += scale * reSign
                    if p10 == p21:
                        if p20 != p11:
                            coeffs[indexingKeyRev[(p20, p11, 2)]] += -reSign
                        else:
                            rk = indexingKeyRev[(p20, p11, 0)]
                            for term, scale in CSDict[rk].items():
                                coeffs[term] += -scale * reSign
            return coeffs

        _structure_data = array_dgcv(
            dict(),
            shape=(LADimension, LADimension),
            null_return=freeze_matrix(matrix_dgcv.zeros(LADimension, 1)),
        )
        for j in range(LADimension):
            for k in range(j + 1, LADimension):
                scoeffs = _structureCoeffs(j, k)
                if len(scoeffs._data) > 0:
                    _structure_data[(j, k)] = scoeffs
                    _structure_data[(k, j)] = -scoeffs
        CartanSubalg = list(hBasis["elems"].values())
        matrixBasis = CartanSubalg + list(offDiag["elems"].values())
        gradingVecs = list(hBasis["grading"].values()) + list(
            offDiag["grading"].values()
        )
        return _structure_data, list(zip(*gradingVecs)), CartanSubalg, matrixBasis

    def _generate_D_series_structure_data(n):
        matrix_dim = 2 * n

        # Basis elements
        hBasis = {"elems": dict(), "grading": dict()}
        GPlus = {"elems": dict(), "grading": dict()}
        GMinus = {"elems": dict(), "grading": dict()}

        def gPlusWeights(idx1, idx2):
            wVec = []
            for idx in range(n - 2):
                if (idx1 <= idx and idx2 <= idx) or (idx1 > idx and idx2 > idx):
                    wVec.append(0)
                elif idx1 <= idx:
                    wVec.append(1)
                else:
                    wVec.append(-1)
            if (idx1 < n - 1 and idx2 < n - 1) or (idx1 > n - 2 and idx2 > n - 2):
                wVec.append(0)
            elif idx1 < n - 1:
                wVec.append(1)
            else:
                wVec.append(-1)
            wVec.append(0)
            return wVec

        def gMinusWeights(idx1, idx2):
            wVec = []
            sign = 1 if idx2 < idx1 else -1
            for idx in range(n - 2):
                if idx1 <= idx and idx2 <= idx:
                    wVec.append(2 * sign)
                elif idx1 > idx and idx2 > idx:
                    wVec.append(0)
                else:
                    wVec.append(sign)
            if idx1 < n - 1 and idx2 < n - 1:
                wVec.append(sign)
            elif idx1 > n - 2 and idx2 > n - 2:
                wVec.append(-sign)
            else:
                wVec.append(0)
            wVec.append(sign)
            return wVec

        for j, k in carProd(range(n), range(n)):
            # Diagonal (Cartan) element
            if j == k and j < n - 1:
                M = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                if j < n - 2:
                    for idx in range(j + 1):
                        M[2 * idx, 2 * idx + 1] = imag_unit()
                        M[2 * idx + 1, 2 * idx] = -imag_unit()
                    hBasis["elems"][(j, k, 0)] = M
                    hBasis["grading"][(j, k, 0)] = [0] * n
                else:
                    for idx in range(n):
                        if idx > j:
                            M[2 * idx, 2 * idx + 1] = -imag_unit() / 2
                            M[2 * idx + 1, 2 * idx] = imag_unit() / 2
                        else:
                            M[2 * idx, 2 * idx + 1] = imag_unit() / 2
                            M[2 * idx + 1, 2 * idx] = -imag_unit() / 2
                    hBasis["elems"][(j, k, 0)] = M
                    hBasis["grading"][(j, k, 0)] = [0] * n
                    M = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                    for idx in range(n):
                        M[2 * idx, 2 * idx + 1] = imag_unit() / 2
                        M[2 * idx + 1, 2 * idx] = -imag_unit() / 2
                    hBasis["elems"][(j + 1, k + 1, 0)] = M
                    hBasis["grading"][(j + 1, k + 1, 0)] = [0] * n
            elif j != k:
                MPlus = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                MPlus[2 * j, 2 * k] = 1
                MPlus[2 * k, 2 * j] = -1
                MPlus[2 * j + 1, 2 * k + 1] = 1
                MPlus[2 * k + 1, 2 * j + 1] = -1
                MPlus[2 * j, 2 * k + 1] = imag_unit()
                MPlus[2 * k + 1, 2 * j] = -imag_unit()
                MPlus[2 * j + 1, 2 * k] = -imag_unit()
                MPlus[2 * k, 2 * j + 1] = imag_unit()
                GPlus["elems"][(j, k, 1)] = MPlus
                GPlus["grading"][(j, k, 1)] = gPlusWeights(j, k)

                if j < k:
                    MMinus = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                    MMinus[2 * j, 2 * k] = 1
                    MMinus[2 * k, 2 * j] = -1
                    MMinus[2 * j + 1, 2 * k + 1] = -1
                    MMinus[2 * k + 1, 2 * j + 1] = 1
                    MMinus[2 * j, 2 * k + 1] = imag_unit()
                    MMinus[2 * k + 1, 2 * j] = -imag_unit()
                    MMinus[2 * j + 1, 2 * k] = imag_unit()
                    MMinus[2 * k, 2 * j + 1] = -imag_unit()
                    GMinus["elems"][(j, k, -1)] = MMinus
                    GMinus["grading"][(j, k, -1)] = gMinusWeights(j, k)
                else:  # k<j
                    MMinus = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                    MMinus[2 * k, 2 * j] = 1
                    MMinus[2 * j, 2 * k] = -1
                    MMinus[2 * k + 1, 2 * j + 1] = -1
                    MMinus[2 * j + 1, 2 * k + 1] = 1
                    MMinus[2 * k, 2 * j + 1] = -imag_unit()
                    MMinus[2 * j + 1, 2 * k] = imag_unit()
                    MMinus[2 * k + 1, 2 * j] = -imag_unit()
                    MMinus[2 * j, 2 * k + 1] = imag_unit()
                    GMinus["elems"][(j, k, -1)] = MMinus
                    GMinus["grading"][(j, k, -1)] = gMinusWeights(j, k)

        indexingKey = dict(
            enumerate(
                list(hBasis["grading"].keys())
                + list(GPlus["grading"].keys())
                + list(GMinus["grading"].keys())
            )
        )
        indexingKeyRev = {j: k for k, j in indexingKey.items()}
        LADimension = len(indexingKey)
        if n == 1:
            CSDict = {0: {0: 2}}  # Cartan subalgebra basis transform indexing
        elif n == 2:
            CSDict = {0: {0: 1, 1: 1}, 1: {0: -1, 1: 1}}
        else:
            CSDict = {
                idx: {0: 1} if idx == 0 else {idx: 1, idx - 1: -1}
                for idx in range(n - 2)
            } | {n - 2: {n - 2: 1, n - 1: 1, n - 3: -1}, n - 1: {n - 2: -1, n - 1: 1}}
        CSDictInv = {
            idx: {j: 0 if j > idx else 1 for j in range(n)} for idx in range(n - 2)
        } | {n - 1: {j: rational(1, 2) for j in range(n)}}
        if n > 2:
            CSDictInv |= {
                n - 2: {
                    j: rational(-1, 2) if j > n - 2 else rational(1, 2)
                    for j in range(n)
                }
            }

        def _structureCoeffs(idx1, idx2):
            coeffs = matrix_dgcv({}, shape=(LADimension, 1))
            if idx2 == idx1:
                return coeffs
            if idx2 < idx1:
                reSign = -1
                idx2, idx1 = idx1, idx2
            else:
                reSign = 1
            p10, p11, p12 = indexingKey[idx1]
            p20, p21, p22 = indexingKey[idx2]
            if p12 == 0:
                for term, scale in CSDictInv[p10].items():
                    if p22 == 1:
                        coeffs[idx2] += (
                            scale * reSign * (int(term == p20) - int(term == p21))
                        )
                    elif p22 == -1:
                        if p20 < p21:
                            if p21 <= p10:
                                sign = -2
                            else:
                                sign = -1
                        else:
                            if p21 <= p10:
                                sign = -2
                            else:
                                sign = 2
                        sign = -reSign if p20 < p21 else reSign
                        coeffs[idx2] += (
                            scale * sign * (int(term == p20) + int(term == p21))
                        )
            elif p12 == 1:
                if p22 == 1:
                    if p11 == p20:
                        if p10 == p21:
                            # l(p10)-l(p11)
                            for t, s in CSDict[p10].items():
                                coeffs[t] += reSign * 4 * s
                            for t, s in CSDict[p11].items():
                                coeffs[t] += -reSign * 4 * s
                        else:
                            coeffs[indexingKeyRev[(p10, p21, 1)]] += 2 * reSign
                    elif p10 == p21:
                        coeffs[indexingKeyRev[(p20, p11, 1)]] += -2 * reSign
                else:
                    slope1 = 1 if p10 < p11 else -1
                    slope2 = 1 if p20 < p21 else -1
                    if p10 == p20:
                        if not (slope1 == -1 and slope2 == -1):
                            if p11 < p21:
                                coeffs[indexingKeyRev[(p11, p21, -1)]] += -2 * reSign
                            elif p21 < p11:
                                if not (slope1 == 1 and slope2 == -1):
                                    coeffs[indexingKeyRev[(p21, p11, -1)]] += 2 * reSign
                    elif p11 == p21:
                        if not (slope1 == 1 and slope2 == 1):
                            if p10 < p20:
                                coeffs[indexingKeyRev[(p20, p10, -1)]] += 2 * reSign
                            elif p20 < p10:
                                if not (slope1 == -1 and slope2 == 1):
                                    coeffs[indexingKeyRev[(p10, p20, -1)]] += (
                                        -2 * reSign
                                    )
                    elif p11 == p20:
                        if not (slope1 == 1 and slope2 == 1) and not (
                            slope1 == -1 and slope2 == 1
                        ):
                            if p10 < p21:
                                coeffs[indexingKeyRev[(p21, p10, -1)]] = -2 * reSign
                            elif p21 < p10:
                                coeffs[indexingKeyRev[(p10, p21, -1)]] = 2 * reSign
                    elif p10 == p21:
                        if not (slope1 == -1 and slope2 == -1) and not (
                            slope1 == 1 and slope2 == -1
                        ):
                            if p11 < p20:
                                coeffs[indexingKeyRev[(p11, p20, -1)]] = 2 * reSign
                            elif p20 < p11:
                                coeffs[indexingKeyRev[(p20, p11, -1)]] = -2 * reSign
            else:
                sign2 = 1 if p10 < p11 else -1
                if (p10 < p11 and p20 < p21) or (p10 > p11 and p20 > p21):
                    pass
                elif p11 == p20:
                    if p10 == p21:
                        # plus/minus (l(p10)+l(p11))
                        for t, s in CSDict[p10].items():
                            coeffs[t] += sign2 * reSign * 4 * s
                        for t, s in CSDict[p11].items():
                            coeffs[t] += sign2 * reSign * 4 * s
                    else:
                        if sign2 == 1:
                            coeffs[indexingKeyRev[(p21, p10, 1)]] += 2 * reSign * sign2
                        else:
                            coeffs[indexingKeyRev[(p10, p21, 1)]] += 2 * reSign * sign2
                elif p10 == p21:
                    if sign2 == 1:
                        coeffs[indexingKeyRev[(p20, p11, 1)]] += 2 * reSign * sign2
                    else:
                        coeffs[indexingKeyRev[(p11, p20, 1)]] += 2 * reSign * sign2
                elif p10 == p20 and p21 != p11:
                    if sign2 == 1:
                        coeffs[indexingKeyRev[(p21, p11, 1)]] += -2 * reSign * sign2
                    else:
                        coeffs[indexingKeyRev[(p11, p21, 1)]] += -2 * reSign * sign2
                elif p11 == p21 and p10 != p20:
                    if sign2 == 1:
                        coeffs[indexingKeyRev[(p20, p10, 1)]] += -2 * reSign * sign2
                    else:
                        coeffs[indexingKeyRev[(p10, p20, 1)]] += -2 * reSign * sign2
            return coeffs

        _structure_data = array_dgcv(
            dict(),
            shape=(LADimension, LADimension),
            null_return=freeze_matrix(matrix_dgcv.zeros(LADimension, 1)),
        )
        for j in range(LADimension):
            for k in range(j + 1, LADimension):
                scoeffs = _structureCoeffs(j, k)
                if len(scoeffs._data) > 0:
                    _structure_data[(j, k)] = scoeffs
                    _structure_data[(k, j)] = -scoeffs

        CartanSubalg = list(hBasis["elems"].values())
        matrixBasis = (
            CartanSubalg
            + list(GPlus["elems"].values())
            + list(GMinus["elems"].values())
        )
        gradingVecs = (
            list(hBasis["grading"].values())
            + list(GPlus["grading"].values())
            + list(GMinus["grading"].values())
        )
        return _structure_data, list(zip(*gradingVecs)), CartanSubalg, matrixBasis

    if series_type == "A":
        default_label = f"sl{rank + 1}" if label is None else label
        structure_data, grading, CartanSubalgebra, matrixBasis = (
            _generate_A_series_structure_data(rank)
        )
        passkey = retrieve_passkey()
        if build_standard_mat_rep is True:
            return createAlgebra(
                matrixBasis,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                process_matrix_rep=True,
                preferred_representation=matrixBasis,
                _simple={
                    "lockKey": passkey,
                    "CartanSubalgebra": CartanSubalgebra,
                    "type": [series_type, rank],
                },
                return_created_object=return_created_object,
                forgo_vmf_registry=forgo_vmf_registry,
            )
        else:
            return createAlgebra(
                structure_data,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                preferred_representation=matrixBasis,
                _simple={
                    "lockKey": passkey,
                    "CartanSubalgebra": CartanSubalgebra,
                    "type": [series_type, rank],
                },
                return_created_object=return_created_object,
                forgo_vmf_registry=forgo_vmf_registry,
            )

    elif series_type == "B":
        default_label = f"so{2 * rank + 1}" if label is None else label
        structure_data, grading, CartanSubalgebra, matrixBasis = (
            _generate_B_series_structure_data(rank)
        )
        passkey = retrieve_passkey()
        if build_standard_mat_rep is True:
            return createAlgebra(
                matrixBasis,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                process_matrix_rep=True,
                preferred_representation=matrixBasis,
                _simple={
                    "lockKey": passkey,
                    "CartanSubalgebra": CartanSubalgebra,
                    "type": [series_type, rank],
                },
                return_created_object=return_created_object,
                forgo_vmf_registry=forgo_vmf_registry,
            )
        else:
            return createAlgebra(
                structure_data,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                preferred_representation=matrixBasis,
                _simple={
                    "lockKey": passkey,
                    "CartanSubalgebra": CartanSubalgebra,
                    "type": [series_type, rank],
                },
                return_created_object=return_created_object,
                forgo_vmf_registry=forgo_vmf_registry,
            )

    elif series_type == "C":
        default_label = f"sp{2 * rank}" if label is None else label
        structure_data, grading, CartanSubalgebra, matrixBasis = (
            _generate_C_series_structure_data(rank)
        )

        if build_standard_mat_rep is True:
            return createAlgebra(
                matrixBasis,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                process_matrix_rep=True,
                preferred_representation=matrixBasis,
                _simple={
                    "lockKey": retrieve_passkey(),
                    "CartanSubalgebra": CartanSubalgebra,
                    "type": [series_type, rank],
                },
                return_created_object=return_created_object,
                forgo_vmf_registry=forgo_vmf_registry,
            )
        else:
            return createAlgebra(
                structure_data,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                preferred_representation=matrixBasis,
                _simple={
                    "lockKey": retrieve_passkey(),
                    "CartanSubalgebra": CartanSubalgebra,
                    "type": [series_type, rank],
                },
                return_created_object=return_created_object,
                forgo_vmf_registry=forgo_vmf_registry,
            )

    elif series_type == "D":
        default_label = f"so{2 * rank}" if label is None else label
        structure_data, grading, CartanSubalgebra, matrixBasis = (
            _generate_D_series_structure_data(rank)
        )
        passkey = retrieve_passkey()
        if build_standard_mat_rep is True:
            return createAlgebra(
                matrixBasis,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                process_matrix_rep=True,
                preferred_representation=matrixBasis,
                _simple={
                    "lockKey": passkey,
                    "CartanSubalgebra": CartanSubalgebra,
                    "type": [series_type, rank],
                },
                return_created_object=return_created_object,
                forgo_vmf_registry=forgo_vmf_registry,
            )
        else:
            return createAlgebra(
                structure_data,
                label=default_label,
                basis_labels=basis_labels,
                grading=grading,
                preferred_representation=matrixBasis,
                _simple={
                    "lockKey": passkey,
                    "CartanSubalgebra": CartanSubalgebra,
                    "type": [series_type, rank],
                },
                return_created_object=return_created_object,
                forgo_vmf_registry=forgo_vmf_registry,
            )

    elif series_type + str(rank) in {"G2", "F4", "E6", "E7", "E8"}:
        default_label = f"{series_type.lower()}{rank}" if label is None else label
        structure_data, grading = _chevalley_structure_data(
            _exceptional_cartan_matrix(series_type, rank)
        )
        passkey = retrieve_passkey()
        return createAlgebra(
            structure_data,
            label=default_label,
            basis_labels=basis_labels,
            grading=grading,
            assume_skew=True,
            _simple={
                "lockKey": passkey,
                "type": [series_type, rank],
            },
            return_created_object=return_created_object,
            forgo_vmf_registry=forgo_vmf_registry,
        )

    else:
        raise ValueError(
            f"Invalid series parameter format: {series}. Expected a letter 'A', 'B', 'C', 'D', 'E', 'F', or 'G' followed by a positive integer, like 'A1', 'B5', etc. For the exceptional LA labels 'E', 'F', and 'G' the integer must be among the classified types (i.e., only 'G2', 'F4', 'E6', 'E7', and 'E8' are admissible)."
        ) from None
