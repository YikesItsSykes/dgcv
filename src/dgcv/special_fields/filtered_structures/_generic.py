from __future__ import annotations

from contextlib import contextmanager

from ..._aux._backends._engine import engine_capability, engine_kind
from ..._aux._backends._symbolic_router import _scalar_is_zero
from ..._aux._utilities._config import get_dgcv_settings_registry
from ..._aux._vmf._safeguards import create_key, get_dgcv_category
from ...core.arrays import array_dgcv, freeze_matrix, matrix_dgcv
from ._tensor_products import _fast_tensor_products

_UNSET = object()


class _generic_record:
    __slots__ = ("constants", "structure_data", "commutators")

    def __init__(self, constants, structure_data):
        self.constants = constants
        self.structure_data = structure_data
        self.commutators = None


class _symbol_generic_constants:
    def _generic_mode(self, with_characteristic_space_reductions, absorb_DS, DS_records):
        setting = get_dgcv_settings_registry().get("generic_structure_constants")
        if setting is False or engine_kind() != "builtin":
            return None
        if (
            not self._parameters
            or self._compress_equation_systems
            or with_characteristic_space_reductions is True
            or absorb_DS
            or any(record.cap > self.height for record in DS_records)
            or get_dgcv_category(self.negativePart) != "algebra"
        ):
            return None
        return self._generic_data()

    def _generic_data(self):
        data = getattr(self, "_generic_constants", _UNSET)
        if data is not _UNSET:
            return data
        data = None
        factory = engine_capability("generic_constants")
        if factory is not None:
            data = self._build_generic_data(factory)
        self._generic_constants = data
        return data

    def _build_generic_data(self, factory):
        alg = self.negativePart
        constants = factory(create_key(prefix="_s"))
        n = alg.dimension
        sd = alg.structureData
        new_data = {}
        for i in range(n):
            for j in range(n):
                col = {}
                for k, v in sd[i, j]._data.items():
                    g, _ = constants.substitute(v)
                    if g is None:
                        return None
                    col[k] = g
                if col:
                    new_data[(i, j)] = matrix_dgcv(col, shape=(n, 1))
        if constants.size == 0:
            return None
        structure_data = array_dgcv(
            new_data,
            shape=(n, n),
            null_return=freeze_matrix(matrix_dgcv.zeros(n, 1)),
        )
        return _generic_record(constants, structure_data)

    @contextmanager
    def _generic_structure(self, record):
        alg = self.negativePart
        original = alg.structureData
        alg.structureData = record.structure_data
        try:
            yield record
        finally:
            alg.structureData = original

    def _generic_test_commutators(self, record):
        if record.commutators is None:
            with self._generic_structure(record):
                record.commutators = [
                    (t0, t1, t0 * t1) for t0, t1, _ in self.test_commutators
                ]
        return record.commutators

    def _expand_generic_levels(self, record):
        aliasing = self._aliasing
        idxs = [idx for idx in aliasing if idx >= 0]
        expand = record.constants.expand
        for idx in idxs:
            alias = aliasing[idx]
            if alias.get("_pending_S"):
                self._materialize_S_atom(alias)
            for key in ("expanded", "operator"):
                ftp = alias.get(key)
                if isinstance(ftp, _fast_tensor_products):
                    new_dict = {}
                    for k, v in ftp.coeff_dict.items():
                        e = expand(v)
                        if not _scalar_is_zero(e):
                            new_dict[k] = e
                    alias[key] = _fast_tensor_products(
                        new_dict, ftp.algebra, _validated=ftp.degree
                    )
            hom = alias.get("hom")
            if hom:
                alias["hom"] = {k: expand(v) for k, v in hom.items()}
