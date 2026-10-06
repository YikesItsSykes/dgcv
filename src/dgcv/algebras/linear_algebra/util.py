from __future__ import annotations

import numbers

from ..._aux._backends._polynomials import expr_union_primitives
from ..._aux._backends._symbolic_router import (
    _scalar_is_zero,
    as_numer_denom,
    get_free_symbols,
)
from ..._aux._backends._types_and_constants import rational
from ..._aux._utilities._config import dgcv_warning, get_dgcv_settings_registry
from ..._aux._vmf._safeguards import (
    create_key,
    get_dgcv_category,
    query_dgcv_categories,
    retrieve_passkey,
)
from ..._aux._vmf.vmf import order_coordinates
from ...core.arrays import _as_matrix_dgcv, array_dgcv, freeze_matrix, matrix_dgcv
from ...core.base import dgcv_class
from ...core.morphisms.morphisms import homomorphism
from ..threads import adjointRepresentation


def _structure_array(data, dim):
    return array_dgcv(
        data,
        shape=(dim, dim),
        null_return=freeze_matrix(matrix_dgcv.zeros(dim, 1)),
    )


def _is_canonical_structure_array(obj):
    if getattr(obj, "_dgcv_category", None) != "array":
        return False
    shape = obj.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        return False
    if getattr(obj.null_return, "shape", None) != (shape[0], 1):
        return False
    data = obj._data
    if not data:
        return True
    entry = next(iter(data.values()))
    return getattr(entry, "shape", None) == (shape[0], 1) and hasattr(entry, "_data")


def _flatten_structure_data(structure_data, _source="algebra_class"):
    sdd = dict()
    unspool = structure_data._unspool
    for idx, val in structure_data._data.items():
        val_data = getattr(val, "_data", None)
        if val_data is None:
            raise TypeError(
                f"The `{_source}` initializer received data in an unsupported format."
            )
        idx1, idx2 = unspool(idx)
        for idx3, v in val_data.items():
            sdd[(idx1, idx2, idx3)] = v
    return sdd


def _gather_structure_singularities(structure_data, parameters):
    struct_sing = set()
    for slot in structure_data._data.values():
        for v in slot._data.values():
            _, d = as_numer_denom(v)
            if get_free_symbols(d):
                struct_sing.add(d)
    if get_dgcv_settings_registry().get("simplify_singularity_ideals_by_default", True):
        return expr_union_primitives(
            struct_sing,
            order_coordinates(parameters),
            process_rationals=True,
            fail_quietly=True,
        )
    return list(struct_sing)


def _generate_gl_structure_data(vs):
    n = len(vs.basis) - 1
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
            if j == k and j < n:
                M = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                for idx in range(n + 1):
                    if idx > j:
                        M[idx, idx] = -rational(j + 1, n + 1)
                    else:
                        M[idx, idx] = 1 - rational(j + 1, n + 1)
                hBasis["elems"][(j, k, 0)] = M
                hBasis["grading"][(j, k, 0)] = [0] * n
            elif j == n and k == n:
                M = matrix_dgcv({}, shape=(matrix_dim, matrix_dim))
                for idx in range(n + 1):
                    M[idx, idx] = 1
                hBasis["elems"][(j, k, 0)] = M
                hBasis["grading"][(j, k, 0)] = [0] * n
            elif j != k:
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
        p10, p11, p12 = indexingKey[idx1]
        p20, p21, p22 = indexingKey[idx2]
        if p12 == 0:
            if p22 == 1 and (p10 != n or p11 != n):
                val = reSign * (
                    int(p10 == p20)
                    - int(p10 == p21)
                    + int(p10 + 1 == p21)
                    - int(p10 + 1 == p20)
                )
                if val != 0:
                    coeffs[idx2] += val
        elif p12 == 1:
            if p22 == 1:
                if p11 == p20:
                    if p10 == p21:
                        if p10 < p11:
                            for idx in range(p10, p11):
                                coeffs[indexingKeyRev[(idx, idx, 0)]] = reSign
                        else:
                            for idx in range(p11, p10):
                                coeffs[indexingKeyRev[(idx, idx, 0)]] = -reSign
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

    CartanSubalg = list(hBasis["elems"].values())
    matrixBasis = CartanSubalg + list(offDiag["elems"].values())

    def obGen(j, k):
        if j == k:
            if j < n:
                tp = (1 - rational(j + 1, n + 1)) * vs.basis[0] @ (vs.basis[0].dual())
                for idx in range(1, n + 1):
                    if idx > j:
                        tp += (
                            -rational(j + 1, n + 1)
                            * vs.basis[idx]
                            @ (vs.basis[idx].dual())
                        )
                    else:
                        tp += (
                            (1 - rational(j + 1, n + 1))
                            * vs.basis[idx]
                            @ (vs.basis[idx].dual())
                        )
                return tp
            return sum(
                [vs.basis[j] @ (vs.basis[j].dual()) for j in range(n)],
                vs.basis[n] @ (vs.basis[n].dual()),
            )
        else:
            return vs.basis[j] @ (vs.basis[k].dual())

    operatorBasis = [
        obGen(indexingKey[idx][0], indexingKey[idx][1]) for idx in range(LADimension)
    ]
    gradingVecs = list(hBasis["grading"].values()) + list(offDiag["grading"].values())
    return (
        _structure_data,
        list(zip(*gradingVecs)),
        CartanSubalg,
        matrixBasis,
        operatorBasis,
    )


class linear_representation(dgcv_class):
    def __init__(target_alg, hom: homomorphism):
        target_alg.structureData, target_alg.antihomomorphism, params = (
            target_alg._validate_hom(hom)
        )
        target_alg.homomorphism = hom
        target_alg.domain = hom.domain
        target_alg._parameters = params | (
            set(hom._parameters) if getattr(hom, "_parameters") else set()
        )
        target_alg.representation_space = hom.codomain.domain

    @classmethod
    def _validate_hom(cls, hom):
        params = set()
        assert query_dgcv_categories(
            hom.codomain, {"endomorphism_space", "tensor_proxy"}
        )
        skew = getattr(hom.domain, "is_skew_symmetric", False)
        amb_dim = hom.domain.dimension + hom.codomain.domain.dimension
        dom_dim = hom.domain.dimension
        anti = None
        is_zero_map = getattr(hom, "_zero_map", False)

        def _equal(a, b):
            return _scalar_is_zero(a - b)

        for c, e1 in enumerate(hom.domain.basis):
            lidx = c + 1 if skew else 0
            for e2 in hom.domain.basis[lidx:]:
                if is_zero_map:
                    anti = False
                    break
                else:
                    p1 = hom(e1 * e2)
                    p2 = hom(e1) * hom(e2)
                    if anti is None and _equal(p1, p2) and not _equal(p1, 0 * p1):
                        anti = False
                    if not _equal(p1, p2):
                        if anti is None and _equal(p1, -p2):
                            anti = True
                        elif anti is True and _equal(p1, -p2):
                            pass
                        else:
                            raise ValueError(
                                f"The `hom` parameter given to the `linear_representation` initializer does not define an algebra homomorphism. The identity hom(v*w)=hom(v)*hom(w) fails for basis elements {e1} and {e2}, producing hom(v*w)={p1} and hom(v)*hom(w)={p2}"
                            )
        if anti is None:
            anti = False

        columns = dict()
        for k, v in getattr(hom.domain, "structureDataDict", dict()).items():
            columns.setdefault((k[0], k[1]), dict())[k[2]] = v
        for k, v in getattr(hom.codomain.domain, "structureDataDict", dict()).items():
            columns.setdefault((k[0] + dom_dim, k[1] + dom_dim), dict())[
                k[2] + dom_dim
            ] = v
        if not is_zero_map:
            for j in range(dom_dim):
                action = hom(hom.domain.basis[j])
                if not callable(action):
                    continue
                for k in range(amb_dim - dom_dim):
                    image = action(hom.codomain.domain.basis[k])
                    if _scalar_is_zero(image):
                        continue
                    column = {
                        idx + dom_dim: value
                        for idx, value in image.coeff_dict.items()
                        if not _scalar_is_zero(value)
                    }
                    if not column:
                        continue
                    columns[(j, k + dom_dim)] = column
                    columns[(k + dom_dim, j)] = {
                        idx: -value for idx, value in column.items()
                    }

        out_sd = matrix_dgcv(
            dict(),
            shape=(amb_dim, amb_dim),
            null_return=matrix_dgcv({}, shape=(amb_dim, 1)),
        )
        for key, column in columns.items():
            out_sd[key] = matrix_dgcv(column, shape=(amb_dim, 1))

        return out_sd, anti, params

    def semidirect_sum(
        target_alg,
        grading=None,
        label=None,
        basis_labels=None,
        register_in_vmf=False,
        initial_basis_index=None,
        simplify_products_by_default=None,
        _markers=None,
    ):
        if target_alg.antihomomorphism:
            raise ValueError(
                "This representation reverses products (it is an antihomomorphism), so its "
                "semidirect sum would not be a Lie algebra. If it was built from matrices, "
                "transpose them: matrices act on column vectors."
            )
        if simplify_products_by_default is None:
            simplify_products_by_default = getattr(
                target_alg.domain, "simplify_products_by_default", False
            )
        if grading is None:
            g1 = tuple(next(iter(target_alg.domain.grading)))
            g2 = tuple(next(iter(target_alg.representation_space.grading)))
            grading = [g1 + g2]
        if isinstance(basis_labels, (tuple, list)):
            if (
                not all(isinstance(elem, str) for elem in basis_labels)
                or len(basis_labels)
                != target_alg.domain.dimension
                + target_alg.representation_space.dimension
            ):
                dgcv_warning(
                    f"`basis_labels` is in an unsupported format and was ignored. Recieved {basis_labels}, types: {[type(lab) for lab in basis_labels]}, target length {target_alg.domain.dimension}+{target_alg.representation_space.dimension}"
                )
                basis_labels = None

        def _pref(el):
            if el[0] == "_":
                return "_I" + el
            return "_" + el

        def _preftex(el):
            if el[:2] == r"\_":
                return "\\_|" + el
            return "\\_" + el

        if _markers is None:
            _markers = {"sum": True, "lockKey": retrieve_passkey()}
            if label is None:
                label = f"{target_alg.domain.label}_semidir_{target_alg.representation_space.label}"
                _markers["_tex_label"] = (
                    f"{target_alg.domain._repr_latex_(raw=True, abbrev=True)}\\ltimes {target_alg.representation_space._repr_latex_(raw=True, abbrev=True)}"
                )
            if basis_labels is None:
                basis_labels = [elem.__repr__() for elem in target_alg.domain.basis] + [
                    elem.__repr__() for elem in target_alg.representation_space.basis
                ]
                _markers["_tex_basis_labels"] = [
                    elem._repr_latex_(raw=True) for elem in target_alg.domain.basis
                ] + [
                    elem._repr_latex_(raw=True)
                    for elem in target_alg.representation_space.basis
                ]
        elif not isinstance(basis_labels, (tuple, list)):
            if not isinstance(basis_labels, str):
                basis_labels = [elem.__repr__() for elem in target_alg.domain.basis] + [
                    elem.__repr__() for elem in target_alg.representation_space.basis
                ]
            else:
                pref = basis_labels
                IIdx = (
                    initial_basis_index
                    if isinstance(initial_basis_index, numbers.Integral)
                    else 1
                )
                basis_labels = [
                    f"{pref}{i + IIdx}"
                    for i in range(
                        target_alg.domain.dimension
                        + target_alg.representation_space.dimension
                    )
                ]
        if not isinstance(label, str) or label == "":
            label = "Alg_" + create_key()

        _markers["semidirect_decomposition"] = (
            target_alg.domain,
            target_alg.representation_space,
            target_alg.homomorphism,
        )
        _markers["_parameters"] = target_alg._parameters
        if register_in_vmf is True:
            from ..subspaces import createAlgebra

            return createAlgebra(
                target_alg.structureData,
                label,
                basis_labels=basis_labels,
                grading=grading,
                return_created_object=True,
                simplify_products_by_default=simplify_products_by_default,
                _markers=_markers,
            )
        else:
            _markers["registered"] = False
            from ..algebras import algebra_class

            return algebra_class(
                target_alg.structureData,
                grading=grading,
                simplify_products_by_default=simplify_products_by_default,
                _label=label,
                _basis_labels=basis_labels,
                _calledFromCreator=retrieve_passkey(),
                _markers=_markers,
            )

    def __call__(target_alg, *args, **kwds):
        return target_alg.homomorphism.__call__(*args, **kwds)


def _mat_to_tensor(mat, domain, codomain):
    mat_m = _as_matrix_dgcv(mat)
    if mat_m is None:
        return mat

    if domain.dimension != mat_m.ncols or codomain.dimension != mat_m.nrows:
        raise TypeError(
            "`mat` should be a s-by-r matrix where domain and codomain have dimensions r and s."
        )

    tp = 0
    for j in range(domain.dimension):
        for k in range(codomain.dimension):
            tp += mat_m[k, j] * codomain.basis[k] @ domain.basis[j]
    return tp


def _representation(
    target_alg,
    rep_space=None,
    representation_basis=None,
    use_matrix_rep_instead_of_tensor=None,
):
    if rep_space is None:
        rep_space = target_alg
    elif get_dgcv_category(rep_space) not in {
        "vector_space",
        "algebra",
        "subalgebra",
    }:
        raise TypeError(
            "`rep_space` must be a `dgcv` class type representing a vector space or algebra."
        ) from None
    if representation_basis is not None and any(
        isinstance(elem, matrix_dgcv) for elem in representation_basis
    ):
        use_matrix_rep_instead_of_tensor = True
    if use_matrix_rep_instead_of_tensor is None and representation_basis is None:
        representation_basis = target_alg.preferred_representation
        use_matrix_rep_instead_of_tensor = (
            True if target_alg._preferred_rep_type == "matrix" else False
        )
    if use_matrix_rep_instead_of_tensor is True:
        if representation_basis is None:
            if isinstance(target_alg._mat_rep, (list, tuple)):
                representation_basis = target_alg.matrix_representation
            elif target_alg._preferred_rep_type == "matrix":
                representation_basis = target_alg.preferred_representation
            else:
                representation_basis = adjointRepresentation(target_alg)
        elif isinstance(representation_basis, (list, tuple)):
            if len(representation_basis) != target_alg.dimension:
                raise TypeError(
                    "`representation_basis` should be a list of matrix/tensor elements matching the length of the represented algebra's basis."
                )
            for elem in representation_basis:
                if not isinstance(elem, matrix_dgcv):
                    raise TypeError(
                        f"If setting `use_matrix_rep_instead_of_tensor==True` and providing `representation_basis`, it should be a list of matrices. But an element in the given list was of type {type(elem)}"
                    )
                if elem.shape[0] != elem.shape[1]:
                    raise TypeError(
                        f"If setting `use_matrix_rep_instead_of_tensor==True` and providing `representation_basis`, it should be a list of square matrices. Received a matrix of shape {elem.shape}"
                    )
                if rep_space.dimension != elem.shape[0]:
                    raise TypeError(
                        f"If setting `use_matrix_rep_instead_of_tensor==True` and providing `representation_basis`, it should be a list of (d,d) matrices where d is the dimension of the reprentation space (defaults to `target_alg`). Received a matrix of shape {elem.shape} and rep. space of dimension {rep_space.dimension}"
                    )
        t_rep = [
            _mat_to_tensor(j, rep_space.dual(), rep_space) for j in representation_basis
        ]
    else:
        if representation_basis is None:
            if isinstance(target_alg._tensor_rep, (list, tuple)):
                representation_basis = target_alg.tensor_representation
            elif target_alg._preferred_rep_type == "tensor":
                representation_basis = target_alg.preferred_representation
            else:
                raise TypeError(
                    "`representation_basis` was not provided and no cached representation is currently stored in the algebra to fall back to."
                )
        if len(representation_basis) != target_alg.dimension:
            raise TypeError(
                "`representation_basis` should be a list of matrix/tensor elements matching the length of the represented algebra's basis."
            )
        for elem in representation_basis:
            if not get_dgcv_category(elem) == "tensorProduct":
                raise TypeError(
                    f"If not setting `representation_basis` to a list of matrices or setting `use_matrix_rep_instead_of_tensor==True` then `representation_basis` should be a list of tensor products. But an element in the given list was of type {type(elem)}"
                )
        t_rep = representation_basis
    hom = homomorphism(target_alg, [rep_space, rep_space.dual()], t_rep)
    return linear_representation(hom)
