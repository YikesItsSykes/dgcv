from ..._aux._backends._symbolic_router import subs
from ._indexing import _spool


class _matrix_core:
    def __init__(
        self, array_data=None, *, shape=None, entry_rule=None, null_return=None
    ):
        super().__init__(
            array_data=array_data,
            shape=shape,
            entry_rule=entry_rule,
            null_return=null_return,
        )

        if self.ndim == 1:
            n = self.shape[0]
            new_data = {}
            for k, v in self._data.items():
                if v is None:
                    continue
                i = k
                new_data[_spool((i, 0), (n, 1))] = v
            self._data = new_data
            self.shape = (n, 1)
            self.ndim = 2

        if self.ndim != 2:
            raise ValueError("matrix_dgcv requires 2-dimensional data")

        self._dgcv_categories = {"matrix"}
        self._engine_representation = dict()

    @property
    def nrows(self):
        return self.shape[0]

    @property
    def ncols(self):
        return self.shape[1]

    def row(self, i):
        return [self[i, j] for j in range(self.ncols)]

    def col(self, j):
        return [self[i, j] for i in range(self.nrows)]

    def columnspace(self, as_plain_lists=False):
        """
        Use `as_plain_lists=True` in hot loops to avoid class coersion costs
        """
        return (
            [self.col(j) for j in range(self.ncols)]
            if as_plain_lists
            else [
                matrix_dgcv(self.col(j), null_return=self.null_return)
                for j in range(self.ncols)
            ]
        )

    def rowspace(self, as_plain_lists=False):
        """
        Use `as_plain_lists=True` in hot loops to avoid class coersion costs
        """
        return (
            [self.row(j) for j in range(self.nrows)]
            if as_plain_lists
            else [
                matrix_dgcv([self.row(j)], null_return=self.null_return)
                for j in range(self.nrows)
            ]
        )

    def __getitem__(self, key):
        v = super().__getitem__(key)
        return self.null_return if v is None else v

    def __setitem__(self, key, value):
        rep = getattr(self, "_engine_representation", None)
        if isinstance(rep, dict):
            rep.clear()
        else:
            self._engine_representation = {}
        idx = _spool(key, self.shape) if isinstance(key, tuple) else key
        self._data[idx] = value
        self._data_unspooled_cache = None

    def copy(self):
        out = self.__class__.__new__(self.__class__)
        out._dgcv_class_check = self._dgcv_class_check
        out._dgcv_categories = set(self._dgcv_categories)
        out.shape = tuple(self.shape)
        out.ndim = 2
        out._data = dict(self._data)
        out.null_return = self.null_return
        out._engine_representation = dict()
        out._data_unspooled_cache = None
        return out

    def apply(self, func, *, in_place=False, skip_none=True, default=None, **kwargs):
        if default is None:
            default = self.null_return
        if skip_none:
            structure = {k: func(v, **kwargs) for k, v in self._data.items() if v is not None}
        else:
            n = self.nrows * self.ncols
            structure = {k: func(self._data.get(k, default), **kwargs) for k in range(n)}
        if in_place:
            self._data = structure
            self._data_unspooled_cache = None
            self._engine_representation = dict()
            return self
        return matrix_dgcv(structure, shape=self.shape)

    def subs(self, rules):
        return matrix_dgcv(
            {k: subs(v, rules) for k, v in self._data.items()}, shape=self.shape
        )

    def trace(self):
        if self.nrows != self.ncols:
            raise ValueError("trace is only defined for square matrices")
        s = 0
        for i in range(self.nrows):
            s += self[i, i]
        return s

    def tolist(self):
        return [[self[i, j] for j in range(self.ncols)] for i in range(self.nrows)]

    def augment_col(self, b):
        bM = b if isinstance(b, matrix_dgcv) else matrix_dgcv(b)
        if bM.ncols != 1 or bM.nrows != self.nrows:
            raise ValueError("b must be a column vector with matching nrows")
        rows = [
            [self[i, j] for j in range(self.ncols)] + [bM[i, 0]]
            for i in range(self.nrows)
        ]
        return matrix_dgcv(rows)

    def _dense_copy(self):
        return [[self[i, j] for j in range(self.ncols)] for i in range(self.nrows)]
