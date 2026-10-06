from types import MappingProxyType

from ..._aux._backends._engine import engine_kind, engine_module
from ._indexing import _spool, _unspool
from ._validation import _is_array_like


class _frozen_matrix:
    def __init__(
        self, array_data=None, *, shape=None, entry_rule=None, null_return=None
    ):
        super().__init__(
            array_data=array_data,
            shape=shape,
            entry_rule=entry_rule,
            null_return=null_return,
        )
        self._freeze()

    def _freeze(self):
        object.__setattr__(self, "_data", MappingProxyType(dict(self._data)))
        rep = getattr(self, "_engine_representation", None)
        if isinstance(rep, dict):
            object.__setattr__(
                self, "_engine_representation", MappingProxyType(dict(rep))
            )
        else:
            object.__setattr__(self, "_engine_representation", MappingProxyType({}))
        object.__setattr__(self, "_is_frozen", True)

    @classmethod
    def _freeze_existing(cls, M):
        out = cls.__new__(cls)
        object.__setattr__(out, "_dgcv_class_check", M._dgcv_class_check)
        object.__setattr__(out, "_dgcv_categories", set(M._dgcv_categories))
        object.__setattr__(out, "shape", tuple(M.shape))
        object.__setattr__(out, "ndim", M.ndim)
        object.__setattr__(out, "_data", MappingProxyType(dict(M._data)))
        object.__setattr__(out, "null_return", M.null_return)
        object.__setattr__(out, "_engine_representation", MappingProxyType({}))
        object.__setattr__(out, "_data_unspooled_cache", None)
        object.__setattr__(out, "_is_frozen", True)
        return out

    def __getstate__(self):
        return {
            "_data": dict(self._data),
            "_is_frozen": self._is_frozen,
            "shape": self.shape,
            "ndim": self.ndim,
            "null_return": self.null_return,
            "_dgcv_class_check": self._dgcv_class_check,
            "_dgcv_categories": set(self._dgcv_categories),
            "_engine_representation": dict(self._engine_representation),
            "_data_unspooled_cache": (
                dict(self._data_unspooled_cache)
                if self._data_unspooled_cache is not None
                else None
            ),
        }

    def __setstate__(self, state):
        object.__setattr__(self, "_is_frozen", False)
        object.__setattr__(self, "shape", state["shape"])
        object.__setattr__(self, "ndim", state["ndim"])
        object.__setattr__(self, "null_return", state["null_return"])
        object.__setattr__(self, "_dgcv_class_check", state["_dgcv_class_check"])
        object.__setattr__(self, "_dgcv_categories", state["_dgcv_categories"])
        object.__setattr__(self, "_data", MappingProxyType(state["_data"]))
        object.__setattr__(
            self,
            "_engine_representation",
            MappingProxyType(state["_engine_representation"]),
        )
        object.__setattr__(
            self, "_data_unspooled_cache", state["_data_unspooled_cache"]
        )
        object.__setattr__(self, "_is_frozen", state["_is_frozen"])

    def __setattr__(self, name, value):
        if getattr(self, "_is_frozen", False):
            raise TypeError(f"{self.__class__.__name__} is immutable")
        object.__setattr__(self, name, value)

    def __setitem__(self, key, value):
        raise TypeError(f"{self.__class__.__name__} is immutable")

    def __hash__(self):
        return hash(tuple(self.shape))

    def copy(self):
        return self.__class__(
            dict(self._data),
            shape=tuple(self.shape),
            null_return=self.null_return,
        )

    @property
    def _data_unspooled(self):
        cache = getattr(self, "_data_unspooled_cache", None)
        if cache is None:
            object.__setattr__(
                self,
                "_data_unspooled_cache",
                {self._unspool(key): value for key, value in self._data.items()},
            )
            cache = self._data_unspooled_cache
        return MappingProxyType(cache)

    def apply(self, func, *, in_place=False, skip_none=True, default=None, **kwargs):
        if in_place:
            raise TypeError(f"{self.__class__.__name__} does not support in_place=True")

        target = self.__class__.__new__(self.__class__)
        object.__setattr__(target, "_data", {})
        object.__setattr__(target, "shape", tuple(self.shape))
        object.__setattr__(target, "ndim", self.ndim)
        object.__setattr__(target, "null_return", self.null_return)
        object.__setattr__(target, "_dgcv_class_check", self._dgcv_class_check)
        object.__setattr__(target, "_dgcv_categories", set(self._dgcv_categories))
        object.__setattr__(target, "_engine_representation", {})
        object.__setattr__(target, "_data_unspooled_cache", None)
        object.__setattr__(target, "_is_frozen", False)

        if default is None:
            default = self.null_return
        if skip_none:
            for k, v in self._data.items():
                if v is not None:
                    target._data[k] = func(v)
        else:
            n = self.nrows * self.ncols
            for k in range(n):
                v = self._data.get(k, default)
                target._data[k] = func(v)

        target._freeze()
        return target

    def transpose(self):
        def _entry_transpose(x):
            if not _is_array_like(x):
                return x

            t = getattr(x, "transpose", None)
            if callable(t):
                return t()
            elif t is not None:
                return t

            rt = getattr(x, "T", None)
            if callable(rt):
                return rt()
            elif rt is not None:
                return rt
            return x

        out = self.__class__.__new__(self.__class__)
        object.__setattr__(out, "_dgcv_class_check", self._dgcv_class_check)
        object.__setattr__(out, "_dgcv_categories", set(self._dgcv_categories))
        object.__setattr__(out, "shape", (self.ncols, self.nrows))
        object.__setattr__(out, "ndim", 2)
        object.__setattr__(out, "_data", {})
        object.__setattr__(out, "null_return", self.null_return)
        object.__setattr__(out, "_engine_representation", {})
        object.__setattr__(out, "_data_unspooled_cache", None)
        object.__setattr__(out, "_is_frozen", False)

        for k, v in self._data.items():
            i, j = _unspool(k, self.shape)
            out._data[_spool((j, i), out.shape)] = _entry_transpose(v)

        out._freeze()
        return out

    def __repr__(self):
        nr = (
            f",\n null_return={repr(self.null_return)}" if self.null_return != 0 else ""
        )
        return f"frozen_matrix_dgcv(\n {repr(dict(self._data))},\n shape = {self.shape}{nr}\n )"

    def _to_engine_matrix(self, kind: str | None = None):
        if kind is None:
            kind = engine_kind()
        if kind not in ("sage", "sympy"):
            raise NotImplementedError(
                f"engine matrices (eigenvalues, engine matrix methods) are not available in the {kind!r} symbolic engine; install sympy or sage and set `default_engine` accordingly"
            )

        rep = self._engine_representation.get(kind, None)
        if rep is not None:
            return rep

        mod = engine_module()
        if mod is None:
            raise RuntimeError("No symbolic engine is available.")

        rows = [[self[i, j] for j in range(self.ncols)] for i in range(self.nrows)]

        if kind == "sage":
            M = mod.matrix(rows)
        else:
            M = mod.Matrix(rows)

        new_rep = dict(self._engine_representation)
        new_rep[kind] = M
        object.__setattr__(self, "_engine_representation", MappingProxyType(new_rep))
        return M

    def symmetric_inertia(self):
        """
        Cached Sylvester inertia of a frozen symmetric matrix.

        Returns
        -------
        tuple of int
            `(p, n, z)`, the counts of positive, negative, and zero
            eigenvalues.

        Notes
        -----
        Computed once and cached on `self`, since a `frozen_matrix_dgcv`
        is immutable. See `matrix_dgcv.symmetric_inertia` for the
        underlying algorithm and preconditions.
        """
        if self._inertia_cache is None:
            object.__setattr__(self, "_inertia_cache", super().symmetric_inertia())
        return self._inertia_cache


def freeze_matrix(M):
    if isinstance(M, frozen_matrix_dgcv):
        return M
    if not isinstance(M, matrix_dgcv):
        raise TypeError("freeze_matrix expects a matrix_dgcv instance")
    return frozen_matrix_dgcv._freeze_existing(M)
