from __future__ import annotations

import operator
from collections.abc import Sequence

from ..._aux._backends._symbolic_router import _scalar_is_one
from ..._aux._utilities._config import get_dgcv_settings_registry, parse_basis_minting


def basis_minting_policy():
    return parse_basis_minting(
        get_dgcv_settings_registry().get("basis_minting", "auto")
    )


def mints_unregistered_basis():
    return basis_minting_policy()[0] == "always"


def mints_registered_basis(dimension, override=None):
    if override is not None:
        return bool(override)
    kind, threshold = basis_minting_policy()
    if kind == "threshold":
        return dimension < threshold
    return kind == "always"


class lazy_sequence(Sequence):
    __slots__ = ("_cache", "_n")

    def __len__(self):
        return self._n

    def __getitem__(self, j):
        try:
            return self._cache[j]
        except (KeyError, TypeError):
            return self._miss(j)

    def _miss(self, j):
        if isinstance(j, slice):
            return tuple(self[i] for i in range(*j.indices(self._n)))
        j = operator.index(j)
        n = self._n
        if j < 0:
            j += n
        if not 0 <= j < n:
            raise IndexError("index out of range")
        cache = self._cache
        if j in cache:
            return cache[j]
        value = cache[j] = self._build(j)
        return value

    def _build(self, j):
        raise NotImplementedError

    def __iter__(self):
        cache, build = self._cache, self._build
        for j in range(self._n):
            if j in cache:
                yield cache[j]
            else:
                value = cache[j] = build(j)
                yield value

    def __bool__(self):
        return self._n > 0

    def __add__(self, other):
        return tuple(self) + tuple(other)

    def __radd__(self, other):
        return tuple(other) + tuple(self)

    @property
    def minted(self):
        return len(self._cache)

    def __repr__(self):
        n = self._n
        if n <= 20:
            return repr(tuple(self))
        head = ", ".join(repr(self[j]) for j in range(3))
        return f"({head}, ..., {self[n - 1]!r}; {n} entries)"


class _element_sequence(lazy_sequence):
    __slots__ = ("_alg", "_valence", "_tuple")

    def _position_of(self, item):
        if (
            getattr(item, "algebra", None) is not self._alg
            or getattr(item, "valence", None) != self._valence
        ):
            return None
        coeffs = getattr(item, "coeff_dict", None)
        if not isinstance(coeffs, dict) or len(coeffs) != 1:
            return None
        ((j, c),) = coeffs.items()
        return j if _scalar_is_one(c) else None

    def __contains__(self, item):
        return self._position_of(item) is not None

    def index(self, item, start=0, stop=None):
        j = self._position_of(item)
        if j is None or not start <= j < (self._n if stop is None else stop):
            raise ValueError(f"{item} is not in the basis")
        return j

    def count(self, item):
        return 1 if item in self else 0

    def __eq__(self, other):
        if isinstance(other, _element_sequence):
            return self._alg is other._alg and self._valence == other._valence
        if isinstance(other, (list, tuple)):
            return self._n == len(other) and all(a == b for a, b in zip(self, other))
        return NotImplemented

    def __hash__(self):
        return hash((type(self), id(self._alg), self._valence))

    def __iter__(self):
        return iter(self.materialize())

    def materialize(self):
        if self._tuple is None:
            cache, build = self._cache, self._build
            items = []
            for j in range(self._n):
                value = cache.get(j)
                if value is None:
                    value = cache[j] = build(j)
                items.append(value)
            self._tuple = tuple(items)
            self._on_materialized()
        return self._tuple

    def _on_materialized(self):
        pass


class lazy_basis(_element_sequence):
    __slots__ = ()

    def __init__(self, alg):
        self._alg = alg
        self._valence = 1
        self._n = alg.dimension
        self._cache = {}
        self._tuple = None

    def _build(self, j):
        return self._alg._class_builder({j: 1}, 1)

    def _miss(self, j):
        value = lazy_sequence._miss(self, j)
        if self._tuple is None and len(self._cache) == self._n:
            self.materialize()
        return value

    def _on_materialized(self):
        alg = self._alg
        if alg.__dict__.get("basis") is self:
            alg.basis = self._tuple

    def dual_view(self):
        return lazy_dual_basis(self)


class lazy_dual_basis(_element_sequence):
    __slots__ = ("_base",)

    def __init__(self, base):
        self._base = base
        self._alg = base._alg
        self._valence = 0
        self._n = base._n
        self._cache = {}
        self._tuple = None

    def _build(self, j):
        return self._base[j].dual()
